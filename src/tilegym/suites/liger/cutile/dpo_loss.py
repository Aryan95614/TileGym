# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused linear + DPO preference loss (CuTile backend).

Computes the Direct Preference Optimization loss over stacked chosen/rejected
sequence pairs without materializing the full (B*T, V) logit tensor, following
the chunked backward-in-forward structure of fused_linear_cross_entropy.py.

Every DPO loss variant is elementwise over pairs, so chunks that hold whole
pairs (chosen sequence i with rejected sequence i + n_pairs) contribute
independently to the loss and every gradient. This keeps the chunked path
single-pass: within a chunk the logits buffer stays live between the row-stats
kernel and the in-place d_logits kernel, and no recompute pass is needed.

Two execution paths, selected by B*T*V*sizeof vs _MAX_LOGIT_MEMORY_BYTES:

Single-pass (fits in _MAX_LOGIT_MEMORY_BYTES):
  One GEMM, row stats, pair math, d_logits in place. d_logits is saved and the
  two large gradient GEMMs run in backward, scaled by grad_output.

Chunked (larger, or chunk_size forced):
  Per pair-chunk: GEMM, row stats, pair math, d_logits in place, then
  grad_input / grad_weight_f32 / grad_bias accumulated immediately and the
  chunk logits discarded. Backward is an elementwise scale.

Per chunk, cuTile owns the two row-wise kernels (per-row logsumexp, target
log-prob and row sum; per-row d_logits write), torch/cuBLAS owns the GEMMs,
and the O(n_pairs) preference math runs as device-resident torch with a tiny
autograd graph that supplies dLoss/d(seq_logp) for every loss_type.

NOTE: the two cuTile kernels are currently torch stand-ins (_row_stats and
_apply_dlogits_) that fix the kernel contracts; they are swapped for
@ct.kernel implementations during GPU bring-up and nothing else changes.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from tilegym.backend import register_impl

# Single-pass threshold: if the full (B*T, V) logit tensor fits within this
# limit, run one chunk and defer the gradient GEMMs to backward.
_MAX_LOGIT_MEMORY_BYTES = 4 * 1024**3  # 4 GB

# Chunked path: largest power-of-2 pair count whose logits fit this budget.
_MAX_CHUNK_LOGIT_BYTES = 1 * 1024**3  # 1 GB


def _row_stats(logits, target, ignore_index):
    """Per-row statistics over the vocab. Kernel A contract.

    Args:
        logits: (R, V) logits in compute dtype.
        target: (R,) int64 target ids; rows with ignore_index are masked.
        ignore_index: masked target value.

    Returns:
        logp: (R,) float32 target log-prob, 0.0 for masked rows.
        lse: (R,) float32 logsumexp of the row.
        rowsum: (R,) float32 plain sum of the row (masked rows included).
    """
    logits_f = logits.float()
    lse = torch.logsumexp(logits_f, dim=-1)
    rowsum = logits_f.sum(dim=-1)
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    target_logit = logits_f.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    logp = torch.where(valid, target_logit - lse, torch.zeros_like(lse))
    return logp, lse, rowsum


def _apply_dlogits_(logits, target, lse, coeff, ignore_index):
    """In-place d_logits write. Kernel B contract.

    Overwrites logits with coeff[r] * (softmax(logits[r]) - onehot(target[r])),
    in the logits dtype. Rows with coeff == 0 come out zero.
    """
    p = torch.exp(logits.float() - lse.unsqueeze(-1))
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    p.scatter_add_(-1, safe_target.unsqueeze(-1), -torch.ones_like(safe_target, dtype=p.dtype).unsqueeze(-1))
    logits.copy_((coeff.unsqueeze(-1) * p).to(logits.dtype))


def _preference_loss_terms(
    chosen_logps,
    rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    n_pairs_total,
    beta,
    loss_type,
    label_smoothing,
    discopop_tau,
):
    """DPO preference loss variants over one chunk's pairs, normalized by the
    GLOBAL pair count so chunk contributions sum to the full-batch loss.
    Formulas follow Liger-Kernel chunked_loss/dpo_loss.py (ead96b618e5c)."""
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps

    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    logits_diff = beta * (chosen_logratios - rejected_logratios)

    if loss_type == "sigmoid":
        losses = -F.logsigmoid(logits_diff)
    elif loss_type == "hinge":
        losses = torch.relu(1 - logits_diff)
    elif loss_type == "exo_pair":
        epsilon = torch.tensor(label_smoothing, device=chosen_logps.device)
        qw = torch.sigmoid(logits_diff)
        ql = torch.sigmoid(-logits_diff)
        losses = qw * (F.logsigmoid(logits_diff) - torch.log1p(-epsilon)) + ql * (
            F.logsigmoid(-logits_diff) - torch.log(epsilon)
        )
    elif loss_type == "nca_pair":
        losses = (
            -F.logsigmoid(chosen_rewards) - 0.5 * F.logsigmoid(-chosen_rewards) - 0.5 * F.logsigmoid(-rejected_rewards)
        )
    elif loss_type == "robust":
        clean_loss_term = -(1 - label_smoothing) * F.logsigmoid(logits_diff)
        flipped_loss_term = -label_smoothing * F.logsigmoid(-logits_diff)
        losses = (clean_loss_term - flipped_loss_term) / (1 - 2 * label_smoothing)
    elif loss_type == "bco_pair":
        losses = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)
    elif loss_type == "sppo_hard":
        losses = (chosen_logratios - 0.5 / beta) ** 2 + (rejected_logratios + 0.5 / beta) ** 2
    elif loss_type == "apo_zero":
        losses = (1 - F.sigmoid(beta * chosen_logratios)) + F.sigmoid(beta * rejected_logratios)
    elif loss_type == "apo_down":
        losses = F.sigmoid(beta * chosen_logratios) + (1 - F.sigmoid(beta * (chosen_logratios - rejected_logratios)))
    elif loss_type == "discopop":
        log_ratio_modulation = torch.sigmoid(logits_diff / discopop_tau)
        logistic_component = -F.logsigmoid(logits_diff)
        exp_component = torch.exp(-logits_diff)
        losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = losses.sum() / n_pairs_total
    return loss, chosen_rewards, rejected_rewards


def _pair_math(
    chosen_logps,
    rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    n_pairs_total,
    beta,
    loss_type,
    label_smoothing,
    discopop_tau,
):
    """Preference loss for one chunk plus dLoss/d(seq_logp) via a tiny
    autograd graph over the (chunk_pairs,) log-prob vectors."""
    with torch.enable_grad():
        c = chosen_logps.detach().requires_grad_(True)
        r = rejected_logps.detach().requires_grad_(True)
        loss, chosen_rewards, rejected_rewards = _preference_loss_terms(
            c,
            r,
            ref_chosen_logps,
            ref_rejected_logps,
            n_pairs_total,
            beta,
            loss_type,
            label_smoothing,
            discopop_tau,
        )
        g_chosen, g_rejected = torch.autograd.grad(loss, (c, r))
    return loss.detach(), chosen_rewards.detach(), rejected_rewards.detach(), g_chosen, g_rejected


def _gather_pair_chunk(tensor, p0, p1, n_pairs):
    """Rows for pairs [p0, p1): chosen block then the matching rejected block."""
    return torch.cat([tensor[p0:p1], tensor[n_pairs + p0 : n_pairs + p1]], dim=0)


class DPOLossCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias,
        ref_input,
        ref_weight,
        ref_bias,
        ignore_index,
        beta,
        alpha,
        compute_nll_loss,
        use_ref_model,
        average_log_prob,
        loss_type,
        label_smoothing,
        discopop_tau,
        chunk_size,
    ):
        B, T, H = _input.shape
        V = weight.shape[0]
        assert B % 2 == 0, "batch must stack chosen then rejected halves, so B must be even"
        n_pairs = B // 2
        device = _input.device
        elsize = _input.element_size()

        if chunk_size is not None:
            single_pass = False
            chunk_pairs = max(1, min(int(chunk_size), n_pairs))
        elif B * T * V * elsize <= _MAX_LOGIT_MEMORY_BYTES:
            single_pass = True
            chunk_pairs = n_pairs
        else:
            single_pass = False
            chunk_pairs = 1
            while 2 * chunk_pairs * 2 * T * V * elsize <= _MAX_CHUNK_LOGIT_BYTES and chunk_pairs < n_pairs:
                chunk_pairs *= 2

        mask = target != ignore_index
        tok_w = mask.to(torch.float32)
        if average_log_prob:
            tok_w = tok_w / mask.sum(dim=-1, keepdim=True).to(torch.float32)
        n_chosen_valid = mask[:n_pairs].sum()

        loss_acc = torch.zeros((), device=device, dtype=torch.float32)
        nll_acc = torch.zeros((), device=device, dtype=torch.float32)
        chosen_mean_acc = torch.zeros((), device=device, dtype=torch.float32)
        rejected_mean_acc = torch.zeros((), device=device, dtype=torch.float32)
        chosen_logps_parts = []
        rejected_logps_parts = []
        chosen_rewards_parts = []
        rejected_rewards_parts = []

        if single_pass:
            d_logits_saved = None
        else:
            grad_input = torch.empty_like(_input)
            grad_weight_f32 = torch.zeros(V, H, device=device, dtype=torch.float32)
            grad_bias_f32 = torch.zeros(V, device=device, dtype=torch.float32) if bias is not None else None

        for p0 in range(0, n_pairs, chunk_pairs):
            p1 = min(p0 + chunk_pairs, n_pairs)
            cp = p1 - p0
            rows = 2 * cp * T

            x_c = _gather_pair_chunk(_input, p0, p1, n_pairs)
            x_2d = x_c.reshape(rows, H)
            target_c = _gather_pair_chunk(target, p0, p1, n_pairs)
            target_flat = target_c.reshape(rows)
            tok_w_c = _gather_pair_chunk(tok_w, p0, p1, n_pairs)

            logits_c = x_2d @ weight.t()
            if bias is not None:
                logits_c = logits_c + bias

            logp, lse, rowsum = _row_stats(logits_c, target_flat, ignore_index)

            if use_ref_model:
                with torch.no_grad():
                    ref_x_2d = _gather_pair_chunk(ref_input, p0, p1, n_pairs).reshape(rows, H)
                    ref_logits_c = ref_x_2d @ ref_weight.t()
                    if ref_bias is not None:
                        ref_logits_c = ref_logits_c + ref_bias
                    ref_logp, _, _ = _row_stats(ref_logits_c, target_flat, ignore_index)
                    del ref_logits_c
                ref_seq_logp = (ref_logp.view(2 * cp, T) * tok_w_c).sum(dim=-1)
                ref_chosen_logps = ref_seq_logp[:cp]
                ref_rejected_logps = ref_seq_logp[cp:]
            else:
                ref_chosen_logps = torch.zeros(cp, device=device, dtype=torch.float32)
                ref_rejected_logps = torch.zeros(cp, device=device, dtype=torch.float32)

            seq_logp = (logp.view(2 * cp, T) * tok_w_c).sum(dim=-1)
            chosen_logps_c = seq_logp[:cp]
            rejected_logps_c = seq_logp[cp:]

            pref_loss_c, chosen_rewards_c, rejected_rewards_c, g_chosen, g_rejected = _pair_math(
                chosen_logps_c,
                rejected_logps_c,
                ref_chosen_logps,
                ref_rejected_logps,
                n_pairs,
                beta,
                loss_type,
                label_smoothing,
                discopop_tau,
            )
            loss_acc += pref_loss_c
            chosen_logps_parts.append(chosen_logps_c)
            rejected_logps_parts.append(rejected_logps_c)
            chosen_rewards_parts.append(chosen_rewards_c)
            rejected_rewards_parts.append(rejected_rewards_c)

            rowsum_2d = rowsum.view(2 * cp, T)
            chosen_mean_acc += rowsum_2d[:cp].sum() / (n_pairs * T * V)
            rejected_mean_acc += rowsum_2d[cp:].sum() / (n_pairs * T * V)

            g_seq = torch.cat([g_chosen, g_rejected], dim=0)
            coeff = -(g_seq.unsqueeze(-1) * tok_w_c)
            if compute_nll_loss:
                logp_2d = logp.view(2 * cp, T)
                nll_acc += -logp_2d[:cp].sum() / n_chosen_valid
                coeff[:cp] += (alpha / n_chosen_valid) * mask[p0:p1].to(torch.float32)

            _apply_dlogits_(logits_c, target_flat, lse, coeff.reshape(rows), ignore_index)

            if single_pass:
                d_logits_saved = logits_c
            else:
                grad_x_c = (logits_c @ weight).view(2 * cp, T, H)
                grad_input[p0:p1] = grad_x_c[:cp]
                grad_input[n_pairs + p0 : n_pairs + p1] = grad_x_c[cp:]
                grad_weight_f32 += logits_c.float().t() @ x_2d.float()
                if bias is not None:
                    grad_bias_f32 += logits_c.float().sum(dim=0)
                del logits_c

        loss = loss_acc + alpha * nll_acc

        chosen_logps = torch.cat(chosen_logps_parts, dim=0)
        rejected_logps = torch.cat(rejected_logps_parts, dim=0)
        chosen_rewards = torch.cat(chosen_rewards_parts, dim=0)
        rejected_rewards = torch.cat(rejected_rewards_parts, dim=0)

        ctx.single_pass = single_pass
        ctx.has_bias = bias is not None
        if single_pass:
            ctx.save_for_backward(d_logits_saved, _input, weight)
            ctx.shape = (B, T, H, n_pairs)
        else:
            ctx.save_for_backward(
                grad_input,
                grad_weight_f32,
                grad_bias_f32 if bias is not None else torch.empty(0),
            )
        ctx.weight_dtype = weight.dtype
        ctx.input_dtype = _input.dtype

        outputs = (
            loss,
            chosen_logps,
            rejected_logps,
            chosen_mean_acc,
            rejected_mean_acc,
            nll_acc,
            chosen_rewards,
            rejected_rewards,
        )
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    def backward(ctx, grad_loss, *aux_grads):
        if ctx.single_pass:
            d_logits, _input, weight = ctx.saved_tensors
            B, T, H, n_pairs = ctx.shape
            # The single-pass chunk spans all pairs, so row order equals input order.
            x_2d = _input.reshape(B * T, H)
            grad_x = (d_logits @ weight).view(B, T, H) * grad_loss
            grad_input = grad_x.to(ctx.input_dtype)
            grad_weight = (d_logits.float().t() @ x_2d.float()) * grad_loss
            grad_weight = grad_weight.to(ctx.weight_dtype)
            grad_bias = (d_logits.float().sum(dim=0) * grad_loss).to(ctx.weight_dtype) if ctx.has_bias else None
        else:
            grad_input_saved, grad_weight_f32, grad_bias_f32 = ctx.saved_tensors
            grad_input = (grad_input_saved.float() * grad_loss).to(ctx.input_dtype)
            grad_weight = (grad_weight_f32 * grad_loss).to(ctx.weight_dtype)
            grad_bias = (grad_bias_f32 * grad_loss).to(ctx.weight_dtype) if ctx.has_bias else None

        return (
            grad_input,
            grad_weight,
            None,  # target
            grad_bias,
            None,  # ref_input
            None,  # ref_weight
            None,  # ref_bias
            None,  # ignore_index
            None,  # beta
            None,  # alpha
            None,  # compute_nll_loss
            None,  # use_ref_model
            None,  # average_log_prob
            None,  # loss_type
            None,  # label_smoothing
            None,  # discopop_tau
            None,  # chunk_size
        )


@register_impl("liger.dpo_loss", backend="cutile")
def dpo_loss_cutile(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ref_input: Optional[torch.Tensor] = None,
    ref_weight: Optional[torch.Tensor] = None,
    ref_bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    beta: float = 0.1,
    alpha: float = 1.0,
    compute_nll_loss: bool = False,
    use_ref_model: bool = True,
    average_log_prob: bool = False,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
    discopop_tau: float = 0.05,
    chunk_size: Optional[int] = None,
):
    return DPOLossCuTileFunction.apply(
        input,
        weight,
        target,
        bias,
        ref_input,
        ref_weight,
        ref_bias,
        ignore_index,
        beta,
        alpha,
        compute_nll_loss,
        use_ref_model,
        average_log_prob,
        loss_type,
        label_smoothing,
        discopop_tau,
        chunk_size,
    )
