# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Tests for fsdp_vocab_parallel_linear_cross_entropy.

Each test simulates FSDP by:
  - broadcasting full data to all ranks (so all have the same reference)
  - slicing e and targets per rank (simulating different local batches)
  - running fsdp_vocab_parallel_linear_cross_entropy
  - comparing forward loss and gradients against a single-rank reference CE
    computed on the full combined batch.
"""

import contextlib
import socket

import pytest
import torch
import torch.distributed
from torch.multiprocessing.spawn import spawn as mp_spawn

from cut_cross_entropy import VocabParallelOptions, linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.vocab_parallel import fsdp_vocab_parallel_linear_cross_entropy
from cut_cross_entropy.vocab_parallel.utils import partition_n_into_range


def find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return port


def _setup_dist(rank: int, world_size: int, port: int) -> torch.device:
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda", rank % torch.cuda.device_count())
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        backend = "cpu:gloo,cuda:nccl"
    else:
        backend = "gloo"

    store = torch.distributed.TCPStore(
        "localhost", port, world_size=world_size, is_master=rank == 0
    )
    torch.distributed.init_process_group(
        backend=backend, store=store, world_size=world_size, rank=rank
    )
    return device


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _target_fn_test_fsdp_vp(
    rank: int,
    world_size: int,
    port: int,
    dtype: torch.dtype,
    error_tol: float,
    invalids: bool,
    reduction: str,
):
    device = _setup_dist(rank, world_size, port)

    # Total tokens must be divisible by world_size
    S_total, V, D = 256, 508, 64
    assert S_total % world_size == 0
    local_n = S_total // world_size

    # Build full data on rank 0, broadcast to all ranks
    torch.manual_seed(0)
    e_full = torch.randn((S_total, D), device=device, dtype=dtype) / (D**0.5)
    c_full = torch.randn((V, D), device=device, dtype=dtype)
    targets_full = torch.randint(0, V, (S_total,), device=device)

    if invalids:
        n_invalid = S_total // 5
        invalid_idx = torch.randperm(S_total, device=device)[:n_invalid]
        targets_full[invalid_idx] = IGNORE_INDEX

    torch.distributed.broadcast(e_full, src=0)
    torch.distributed.broadcast(c_full, src=0)
    torch.distributed.broadcast(targets_full, src=0)

    # ---- FSDP+VP: each rank has a different local slice ----
    vp_opts = VocabParallelOptions.from_vocab(V)

    e_local = e_full[rank * local_n : (rank + 1) * local_n].clone().requires_grad_(True)
    targets_local = targets_full[rank * local_n : (rank + 1) * local_n].clone()
    c_shard = c_full[vp_opts.start : vp_opts.stop].clone().requires_grad_(True)

    fsdp_loss = fsdp_vocab_parallel_linear_cross_entropy(
        e_local,
        c_shard,
        targets_local,
        vp_opts,
        reduction=reduction,
    )

    if reduction == "none":
        fsdp_loss.sum().backward()
    else:
        fsdp_loss.backward()

    fsdp_e_grad = e_local.grad.clone()
    fsdp_c_grad = c_shard.grad.clone()

    # ---- Reference: full CE on the full combined batch (single-rank semantics) ----
    e_ref = e_full.clone().requires_grad_(True)
    c_ref = c_full.clone().requires_grad_(True)

    ref_loss = linear_cross_entropy(
        e_ref,
        c_ref,
        targets_full,
        impl="torch_compile",
        reduction=reduction if reduction != "none" else "mean",
    )
    ref_loss.backward()

    ref_e_local_grad = e_ref.grad[rank * local_n : (rank + 1) * local_n]
    ref_c_shard_grad = c_ref.grad[vp_opts.start : vp_opts.stop]

    # ---- Compare losses ----
    if reduction != "none":
        assert torch.allclose(fsdp_loss, ref_loss, atol=error_tol), (
            f"rank={rank} loss mismatch: "
            f"fsdp={fsdp_loss.item():.6f} ref={ref_loss.item():.6f}"
        )

    # ---- Compare grad w.r.t. weight shard ----
    assert torch.allclose(fsdp_c_grad, ref_c_shard_grad, atol=error_tol), (
        f"rank={rank} c_shard grad mismatch: "
        f"max_diff={( fsdp_c_grad - ref_c_shard_grad).abs().max().item():.2e}"
    )

    # ---- Compare grad w.r.t. local hidden states ----
    assert torch.allclose(fsdp_e_grad, ref_e_local_grad, atol=error_tol), (
        f"rank={rank} e_local grad mismatch: "
        f"max_diff={(fsdp_e_grad - ref_e_local_grad).abs().max().item():.2e}"
    )

    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Test entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize(
    "dtype,error_tol",
    [
        # float32: tight tolerance — verifies mathematical correctness
        (torch.float32, 1e-5),
        # float16/bfloat16: looser tolerance — our float32 matmuls are more
        # accurate than the reference's native-dtype matmuls, so the two
        # differ by ~3–4 × ε_bf16 due to floating-point non-associativity
        # across the N split matmuls in reduce_scatter.
        (torch.float16, 2e-2),
        (torch.bfloat16, 5e-2),
    ],
)
@pytest.mark.parametrize("nprocs", [4])
def test_fsdp_vocab_parallel(
    nprocs: int,
    dtype: torch.dtype,
    error_tol: float,
    invalids: bool,
    reduction: str,
):
    mp_spawn(
        _target_fn_test_fsdp_vp,
        args=(nprocs, find_free_port(), dtype, error_tol, invalids, reduction),
        nprocs=nprocs,
        join=True,
    )
