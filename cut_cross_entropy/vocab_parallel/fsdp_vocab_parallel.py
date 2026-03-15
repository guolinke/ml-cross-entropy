# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""FSDP-compatible vocab-parallel fused linear + cross-entropy.

Supports the case where each rank has *different* hidden states (FSDP data
parallel) while the LM head is sharded across vocab (vocab parallel).

Design
------
After all_gather of the local hidden states, each rank holds the *full*
batch of hidden states and the same targets — exactly the pre-condition
required by the existing VP CE implementations (CCE Triton kernel or
torch_compile).  We therefore delegate entirely to those implementations
and only add:

    1. ``_AllGatherReduceScatter`` — all_gather in forward, reduce_scatter
       in backward (more efficient than all_reduce + slice).
    2. A wrapper that creates ``VocabParallelOptions(reduce_e_grad=False)``
       so that the existing VP implementations do *not* perform their own
       all_reduce on the e gradient; that reduction is instead absorbed
       into the reduce_scatter of step 1.

Communication schedule per forward+backward pass
-------------------------------------------------
Forward:
    1. all_gather(e_local)                     O(S·d)   bandwidth
    2. all_reduce(local_max, MAX)              O(S)
    3. all_reduce(local_sumexp, SUM)           O(S)
    4. all_reduce(local_target_logit, SUM)     O(S)
Backward:
    5. reduce_scatter(grad_full_e)             O(S·d)   bandwidth
       2× cheaper than the plain all_reduce used in standard VP, because
       only [S/N, d] is produced per rank rather than [S, d].

Note: ``S`` must be divisible by the process-group world size so that
``reduce_scatter_tensor`` can split tokens evenly across ranks.
"""

from __future__ import annotations

import torch
import torch.distributed

from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.vocab_parallel.utils import VocabParallelOptions


# ---------------------------------------------------------------------------
# Primitive: all_gather forward / reduce_scatter backward
# ---------------------------------------------------------------------------


class _AllGatherReduceScatter(torch.autograd.Function):
    """
    Forward : all_gather(e_local [S/N, d])  -> full_e [S, d]
    Backward: reduce_scatter(grad [S, d])   -> grad_local [S/N, d]

    reduce_scatter is 2× more bandwidth-efficient than all_reduce+slice:
    it produces [S/N, d] output directly instead of [S, d].
    The reduction in float32 prevents precision loss from bfloat16/fp16
    accumulation across vocab shards.
    """

    @staticmethod
    def forward(
        ctx,
        e_local: torch.Tensor,
        group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.local_n = e_local.shape[0]
        ctx.orig_dtype = e_local.dtype
        world_size = torch.distributed.get_world_size(group)
        full_e = torch.empty(
            world_size * ctx.local_n,
            *e_local.shape[1:],
            dtype=e_local.dtype,
            device=e_local.device,
        )
        torch.distributed.all_gather_into_tensor(full_e, e_local.contiguous(), group=group)
        return full_e

    @staticmethod
    def backward(
        ctx,
        grad_full_e: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        # Upcast to float32 for the reduction to avoid precision loss, then
        # cast back to the original dtype afterwards.
        grad_fp32 = grad_full_e.float()
        local_grad_fp32 = torch.empty(
            ctx.local_n,
            *grad_full_e.shape[1:],
            dtype=torch.float32,
            device=grad_full_e.device,
        )
        torch.distributed.reduce_scatter_tensor(
            local_grad_fp32, grad_fp32.contiguous(), group=ctx.group
        )
        return local_grad_fp32.to(dtype=ctx.orig_dtype), None


# ---------------------------------------------------------------------------
# Helper: all_gather for integer targets (no gradient)
# ---------------------------------------------------------------------------


def _all_gather_targets(
    targets_local: torch.Tensor,
    group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    world_size = torch.distributed.get_world_size(group)
    full = torch.empty(
        world_size * targets_local.numel(),
        dtype=targets_local.dtype,
        device=targets_local.device,
    )
    torch.distributed.all_gather_into_tensor(full, targets_local.contiguous(), group=group)
    return full


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fsdp_vocab_parallel_linear_cross_entropy(
    e_local: torch.Tensor,
    weight_shard: torch.Tensor,
    targets_local: torch.Tensor,
    vocab_parallel_options: VocabParallelOptions,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    return_lse: bool = False,
    impl: str = "auto",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused linear + cross-entropy for FSDP + vocab-parallel training.

    Unlike the standard vocab-parallel CE (which assumes all ranks hold the
    same hidden states), this function supports FSDP where each rank processes
    a *different* subset of the batch.

    After all_gathering hidden states, this function delegates entirely to
    the existing CCE Triton kernel or torch_compile implementations — all of
    their optimisations (no logit materialisation for CCE, fusion, etc.) apply
    unchanged.

    The key difference from calling those implementations directly is that
    ``reduce_e_grad`` is forced to ``False`` in the ``VocabParallelOptions``
    passed downstream.  The e gradient reduction is instead performed by the
    ``reduce_scatter`` in ``_AllGatherReduceScatter.backward``, which is 2×
    more bandwidth-efficient than the plain ``all_reduce`` used in standard VP.

    Args:
        e_local:                Local hidden states, shape ``[S/N, d]``.
            Must be 2-D; flatten batch × sequence dimensions before calling.
            ``S`` (= ``S/N × world_size``) must be divisible by the
            process-group world size.
        weight_shard:           LM-head weight shard, shape ``[V/N, d]``.
            Must be a plain ``torch.Tensor`` (not a DTensor); passing a
            DTensor would cause ``linear_cross_entropy`` to gather it into
            a full-vocab weight, undoing the vocab sharding.
            Must correspond to ``[vocab_start, vocab_end)`` in
            *vocab_parallel_options*.
        targets_local:          Local target token ids, shape ``[S/N]``.
        vocab_parallel_options: Defines ``[start, stop)`` vocab range and the
            process group.
        ignore_index:           Targets equal to this value contribute zero.
        softcap:                Optional logit soft-capping value.
        reduction:              ``"mean"`` (default), ``"sum"``, or ``"none"``.
            When ``"none"``, the returned tensor has shape ``[S]`` (all
            gathered tokens, not just the local ``[S/N]`` slice), because
            hidden states are all-gathered before computing the loss.
            Calling ``.sum().backward()`` on that result is equivalent to
            ``reduction="sum"`` and is correct.
        return_lse:             If ``True``, also return the log-sum-exp.
            The returned LSE has shape ``[S]`` (all gathered tokens).
        impl:                   Backend: ``"auto"`` (CCE on CUDA,
            torch_compile otherwise), ``"torch_compile"``, or any CCE preset
            accepted by ``linear_cross_entropy``.

    Returns:
        Loss scalar (``"mean"``/``"sum"``), loss tensor of shape ``[S]``
        (``"none"``), or ``(loss, lse)`` when ``return_lse=True``.
    """
    if e_local.dim() != 2:
        raise ValueError(
            f"e_local must be 2-D [S/N, d], got shape {tuple(e_local.shape)}. "
            "Flatten batch and sequence dimensions before calling."
        )

    if targets_local.numel() != e_local.shape[0]:
        raise ValueError(
            f"targets_local.numel()={targets_local.numel()} must equal "
            f"e_local.shape[0]={e_local.shape[0]}."
        )

    group = vocab_parallel_options.group

    expected = vocab_parallel_options.stop - vocab_parallel_options.start
    if weight_shard.shape[0] != expected:
        raise ValueError(
            f"weight_shard.shape[0]={weight_shard.shape[0]} does not match "
            f"vocab shard size {expected} "
            f"([{vocab_parallel_options.start}, {vocab_parallel_options.stop}))"
        )

    # Step 1: all_gather hidden states.
    # Backward automatically performs reduce_scatter (see _AllGatherReduceScatter).
    full_e = _AllGatherReduceScatter.apply(e_local, group)

    # Step 2: all_gather targets — integer tensor, no gradient.
    with torch.no_grad():
        full_targets = _all_gather_targets(targets_local.flatten(), group)

    # Step 3: build VP options with reduce_e_grad=False.
    # The existing VP implementations would normally all_reduce the e gradient
    # within the VP group.  We disable that here because _AllGatherReduceScatter
    # already performs a reduce_scatter in its backward, which correctly sums
    # contributions from all vocab shards and scatters to the right token chunks.
    # Doing both would double-reduce the gradient.
    vp_opts = VocabParallelOptions(
        start=vocab_parallel_options.start,
        stop=vocab_parallel_options.stop,
        group=group,
        reduce_e_grad=False,
    )

    # Step 4: delegate to the existing CCE / torch_compile implementations.
    # Lazy import breaks the circular dependency:
    #   torch_compile.py -> vocab_parallel/__init__.py -> fsdp_vocab_parallel.py
    from cut_cross_entropy.linear_cross_entropy import linear_cross_entropy  # noqa: PLC0415

    if impl == "auto":
        from cut_cross_entropy.linear_cross_entropy import LCE_IMPL_DEFAULT  # noqa: PLC0415

        impl = LCE_IMPL_DEFAULT

    return linear_cross_entropy(
        full_e,
        weight_shard,
        full_targets,
        ignore_index=ignore_index,
        softcap=softcap,
        reduction=reduction,
        return_lse=return_lse,
        impl=impl,
        vocab_parallel_options=vp_opts,
    )
