from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vereda.config import ModelConfig
from vereda.memory.vsa import VSAEngine
from vereda.models import build_model
from vereda.models.vereda_final import VeredaFinalBackbone, VeredaFinalBlock
from vereda.models.vereda_v2 import (
    AdaptiveGammaQuant,
    DendriticNeuronLayer,
    LearnableCurvatureLorentz,
    TernaryQuantSTE,
    VeredaV2Block,
    combine_states,
    compute_stp_sequence,
    lorentz_inner_product,
)
from vereda.training.trainer import compute_topology_loss, masked_language_loss


def _read_json(path: str | Path) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_path = ROOT / path
    if not full_path.exists():
        return rows
    for line in full_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_monotonic_segment(rows: list[dict[str, Any]], key: str) -> tuple[list[dict[str, Any]], int]:
    selected = [row for row in rows if key in row and "step" in row]
    if not selected:
        return [], 0
    starts = [0]
    previous = selected[0]["step"]
    for index, row in enumerate(selected[1:], start=1):
        current = row["step"]
        if current < previous:
            starts.append(index)
        previous = current
    return selected[starts[-1] :], len(starts) - 1


def _max_abs(*values: torch.Tensor) -> float:
    return max(float(value.detach().abs().max().cpu()) for value in values)


def probe_lorentz_projection() -> dict[str, Any]:
    torch.manual_seed(1001)
    geo = LearnableCurvatureLorentz(d_model=32).double()
    spatial = torch.randn(128, 32, dtype=torch.float64)
    projected, curvature = geo(spatial)
    product = lorentz_inner_product(projected, projected)
    target = -1.0 / curvature.squeeze(-1)
    error = float((product - target).detach().abs().max())
    return {
        "name": "lorentz_projection_identity",
        "max_abs_error": error,
        "threshold": 1e-10,
        "pass": error <= 1e-10,
    }


def probe_lorentz_retraction_error_bound() -> dict[str, Any]:
    torch.manual_seed(1006)
    dim = 32
    curvature = 1.0
    direction = torch.randn(dim, dtype=torch.float64)
    direction = direction / direction.norm()
    origin = torch.zeros(dim + 1, dtype=torch.float64)
    origin[0] = math.sqrt(1.0 / curvature)
    norms = torch.tensor([1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2], dtype=torch.float64)
    errors = []
    constraint_errors = []
    for norm in norms:
        tangent = torch.zeros(dim + 1, dtype=torch.float64)
        tangent[1:] = direction * norm
        lorentz_norm = torch.sqrt(lorentz_inner_product(tangent, tangent))
        z = math.sqrt(curvature) * lorentz_norm
        exponential = torch.cosh(z) * origin + (torch.sinh(z) / z) * tangent
        gamma = torch.rsqrt(1.0 - curvature * lorentz_norm.square())
        retraction = gamma * (origin + tangent)
        errors.append(float((retraction - exponential).norm()))
        target = torch.tensor(-1.0 / curvature, dtype=torch.float64)
        constraint_errors.append(float((lorentz_inner_product(retraction, retraction) - target).abs()))
    slope = float(np.polyfit(np.log(norms.numpy()), np.log(np.array(errors)), deg=1)[0])
    return {
        "name": "lorentz_retraction_cubic_error",
        "max_abs_error": max(errors),
        "max_constraint_error": max(constraint_errors),
        "loglog_error_slope": slope,
        "threshold": 1e-5,
        "pass": slope >= 2.8 and max(constraint_errors) <= 1e-12,
    }


def probe_dendritic_manifold() -> dict[str, Any]:
    torch.manual_seed(1002)
    layer = DendriticNeuronLayer(d_in=32, d_out=16, n_branches=4).double()
    values = torch.randn(64, 32, dtype=torch.float64)
    output = layer(values)
    product = lorentz_inner_product(output, output)
    target = torch.full_like(product, -1.0)
    error = float((product - target).detach().abs().max())
    return {
        "name": "dendritic_lorentz_identity",
        "max_abs_error": error,
        "threshold": 1e-10,
        "pass": error <= 1e-10,
    }


def probe_dual_state_delta_exact_recall() -> dict[str, Any]:
    torch.manual_seed(1007)
    max_error = 0.0
    for _ in range(256):
        head = 32
        memory = torch.randn(head, head)
        key = torch.randn(head)
        key = key / key.norm()
        value = torch.randn(head)
        recalled = memory @ key
        error = value - recalled
        updated = memory + error.unsqueeze(-1) @ key.unsqueeze(0)
        new_recalled = updated @ key
        max_error = max(max_error, float((new_recalled - value).abs().max()))
    return {
        "name": "dual_state_delta_exact_recall_ideal_key",
        "max_abs_error": max_error,
        "threshold": 1e-5,
        "pass": max_error <= 1e-5,
    }


def probe_cbh_associativity() -> dict[str, Any]:
    torch.manual_seed(1003)
    max_error = 0.0
    for _ in range(128):
        dim = 32
        batch = 4
        values = [torch.randn(batch, dim) for _ in range(9)]
        b1 = (torch.rand(batch, 1) > 0.7).float()
        b2 = (torch.rand(batch, 1) > 0.7).float()
        b3 = (torch.rand(batch, 1) > 0.7).float()
        u1, w1, c1, u2, w2, c2, u3, w3, c3 = values
        left_temp = combine_states(u1, w1, c1, b1, u2, w2, c2, b2)
        left = combine_states(*left_temp, u3, w3, c3, b3)
        right_temp = combine_states(u2, w2, c2, b2, u3, w3, c3, b3)
        right = combine_states(u1, w1, c1, b1, *right_temp)
        max_error = max(max_error, _max_abs(*(a - b for a, b in zip(left, right))))
    return {
        "name": "cbh_step2_associativity",
        "max_abs_error": max_error,
        "threshold": 1e-5,
        "pass": max_error <= 1e-5,
    }


def probe_orthogonal_reset() -> dict[str, Any]:
    torch.manual_seed(1008)
    dim = 64
    block = VeredaV2Block(ModelConfig(architecture="vereda_v2", d_model=dim)).double()
    values = torch.randn(128, dim, dtype=torch.float64)
    base = block.v_base.double()
    projected = values - ((values * base).sum(dim=-1, keepdim=True) / ((base**2).sum() + 1e-12)) * base
    error = float((projected * base).sum(dim=-1).detach().abs().max())
    return {
        "name": "orthogonal_reset_projection",
        "max_abs_error": error,
        "threshold": 1e-10,
        "pass": error <= 1e-10,
    }


def probe_cif_parallel_equivalence() -> dict[str, Any]:
    torch.manual_seed(1004)
    config = ModelConfig(architecture="vereda_v2", d_model=32, n_layers=1, ffn_multiplier=1.5)
    block = VeredaV2Block(config).double()
    max_error = 0.0
    lengths = [2, 4, 8, 16]
    for seq_len in lengths:
        batch = 2
        x = torch.randn(batch, seq_len, config.d_model, dtype=torch.float64)
        u = torch.zeros(batch, config.d_model, dtype=torch.float64)
        w = torch.zeros(batch, config.d_model, dtype=torch.float64)
        c = torch.zeros(batch, config.d_model, dtype=torch.float64)
        b = torch.zeros(batch, 1, dtype=torch.float64)
        f = torch.zeros(batch, config.d_model, dtype=torch.float64)
        stp_x = torch.ones(batch, config.d_model, dtype=torch.float64)
        cif = torch.zeros(batch, 1, dtype=torch.float64)
        parallel = block(x, u, w, c, b, f, stp_x, cif)
        current_u, current_w, current_c, current_b = u, w, c, b
        current_f, current_x, current_cif = f, stp_x, cif
        for step in range(seq_len):
            (
                _,
                current_u,
                current_w,
                current_c,
                current_b,
                current_f,
                current_x,
                current_cif,
                _,
            ) = block.step(
                x[:, step],
                current_u,
                current_w,
                current_c,
                current_b,
                current_f,
                current_x,
                current_cif,
            )
        sequential = (
            current_u,
            current_w,
            current_c,
            current_b,
            current_f,
            current_x,
            current_cif,
        )
        max_error = max(max_error, _max_abs(*(p - s for p, s in zip(parallel[1:8], sequential))))
    return {
        "name": "cif_parallel_vs_sequential",
        "max_abs_error": max_error,
        "threshold": 1e-7,
        "pass": max_error <= 1e-7,
        "sequence_lengths": lengths,
    }


def probe_stp_boundedness() -> dict[str, Any]:
    torch.manual_seed(1009)
    batch, seq_len, dim = 8, 64, 32
    values = torch.randn(batch, seq_len, dim)
    previous_f = torch.zeros(batch, dim)
    previous_x = torch.ones(batch, dim)
    tau_f = torch.tensor(0.35)
    tau_d = torch.tensor(0.75)
    usage = torch.tensor(0.2)
    f_seq, x_seq, last_f, last_x = compute_stp_sequence(values, previous_f, previous_x, tau_f, tau_d, usage)
    lower_violation = min(float(f_seq.min()), float(x_seq.min()), float(last_f.min()), float(last_x.min()))
    upper_violation = max(float(f_seq.max()), float(x_seq.max()), float(last_f.max()), float(last_x.max()))
    error = max(abs(min(lower_violation, 0.0)), abs(max(upper_violation - 1.0, 0.0)))
    return {
        "name": "stp_state_boundedness",
        "max_abs_error": error,
        "threshold": 0.0,
        "pass": error == 0.0,
        "min_value": lower_violation,
        "max_value": upper_violation,
    }


def probe_ternary_ste() -> dict[str, Any]:
    weights = torch.tensor([0.2, -0.6, 1.2], requires_grad=True)
    beta = torch.tensor(0.5)
    quantized = TernaryQuantSTE.apply(weights, beta, 1.0)
    quantized.backward(torch.ones_like(weights))
    expected = torch.tensor([math.tanh(0.2), 0.0, 0.0])
    error = float((weights.grad - expected).abs().max())
    return {
        "name": "ternary_ste_masked_gradient",
        "max_abs_error": error,
        "threshold": 1e-7,
        "pass": error <= 1e-7,
    }


def probe_adaptive_gamma_quant_finite() -> dict[str, Any]:
    torch.manual_seed(1010)
    quantizer = AdaptiveGammaQuant()
    values = torch.randn(64, 32, requires_grad=True)
    quantized, gamma = quantizer(values)
    loss = quantized.square().mean()
    loss.backward()
    finite = bool(torch.isfinite(quantized).all() and torch.isfinite(values.grad).all())
    gamma_value = float(gamma.detach())
    grad_max = float(values.grad.detach().abs().max())
    return {
        "name": "adaptive_gamma_quant_finite_gradient",
        "max_abs_error": 0.0 if finite else float("inf"),
        "threshold": 0.0,
        "pass": finite and 0.4 <= gamma_value <= 1.0,
        "gamma": gamma_value,
        "max_grad_abs": grad_max,
    }


def probe_rls_spd_convergence() -> dict[str, Any]:
    torch.manual_seed(1011)
    dim = 8
    theta = torch.randn(dim, 1, dtype=torch.float64)
    estimate = torch.zeros(dim, 1, dtype=torch.float64)
    covariance = torch.eye(dim, dtype=torch.float64) * 100.0
    forgetting = 0.995
    for _ in range(256):
        features = torch.randn(dim, 1, dtype=torch.float64)
        target = features.T @ theta
        denominator = forgetting + features.T @ covariance @ features
        gain = covariance @ features / denominator
        residual = target - features.T @ estimate
        estimate = estimate + gain * residual
        covariance = (covariance - gain @ features.T @ covariance) / forgetting
        covariance = 0.5 * (covariance + covariance.T)
    eig_min = float(torch.linalg.eigvalsh(covariance).min())
    param_error = float((estimate - theta).norm() / theta.norm())
    return {
        "name": "rls_covariance_spd_and_convergence",
        "max_abs_error": param_error,
        "threshold": 1e-4,
        "pass": eig_min > 0.0 and param_error <= 1e-4,
        "min_covariance_eigenvalue": eig_min,
    }


def probe_vsa_random_orthogonality() -> dict[str, Any]:
    torch.manual_seed(1005)
    dimension = 1024
    count = 384
    engine = VSAEngine(dimension=dimension, device="cpu")
    vectors = engine.random_hypervector(count)
    similarities = vectors @ vectors.T
    off_diagonal = similarities[~torch.eye(count, dtype=torch.bool)]
    mean_abs = float(off_diagonal.abs().mean())
    max_abs = float(off_diagonal.abs().max())
    expected_std = 1.0 / math.sqrt(dimension)
    return {
        "name": "vsa_random_near_orthogonality",
        "mean_abs_similarity": mean_abs,
        "max_abs_similarity": max_abs,
        "expected_std": expected_std,
        "threshold": 0.14,
        "pass": max_abs <= 0.14,
    }


def probe_vereda_final_state_bounds() -> dict[str, Any]:
    torch.manual_seed(1012)
    config = ModelConfig(
        architecture="vereda_final",
        d_model=32,
        n_layers=1,
        head_size=16,
        ffn_multiplier=1.0,
        dropout=0.0,
    )
    block = VeredaFinalBlock(config)
    batch = 3
    heads = config.d_model // config.head_size
    u = torch.zeros(batch, config.d_model)
    w = torch.zeros(batch, config.d_model)
    c = torch.zeros(batch, config.d_model)
    b = torch.zeros(batch, 1)
    stp_f = torch.zeros(batch, config.d_model)
    stp_x = torch.ones(batch, config.d_model)
    cif = torch.zeros(batch, 1)
    fast = torch.zeros(batch, config.d_model)
    slow = torch.zeros(batch, heads, config.head_size, config.head_size)
    kalman = torch.ones(batch, heads, config.head_size)
    finite = True
    for _ in range(16):
        x = torch.randn(batch, config.d_model)
        (
            _,
            u,
            w,
            c,
            b,
            stp_f,
            stp_x,
            cif,
            fast,
            slow,
            kalman,
            metrics,
        ) = block.step(x, u, w, c, b, stp_f, stp_x, cif, fast, slow, kalman)
        tensors = [u, w, c, b, stp_f, stp_x, cif, fast, slow, kalman]
        finite = finite and all(bool(torch.isfinite(value).all()) for value in tensors)
    violations = [
        float(torch.relu(-stp_f).detach().max()),
        float(torch.relu(stp_f - 1.0).detach().max()),
        float(torch.relu(-stp_x).detach().max()),
        float(torch.relu(stp_x - 1.0).detach().max()),
        float(torch.relu(-cif).detach().max()),
        float(torch.relu(cif - 1.0).detach().max()),
        float(torch.relu(1e-5 - kalman).detach().max()),
        float(torch.relu(kalman - 100.0).detach().max()),
        float(torch.relu(-metrics["kalman_gain"]).detach().max()),
        float(torch.relu(metrics["kalman_gain"] - 1.0).detach().max()),
    ]
    error = max(violations) if finite else float("inf")
    return {
        "name": "vereda_final_state_bounds",
        "max_abs_error": error,
        "threshold": 0.0,
        "pass": finite and error == 0.0,
        "cif_min": float(cif.detach().min()),
        "cif_max": float(cif.detach().max()),
        "kalman_cov_min": float(kalman.detach().min()),
        "kalman_cov_max": float(kalman.detach().max()),
        "stp_f_min": float(stp_f.detach().min()),
        "stp_f_max": float(stp_f.detach().max()),
        "stp_x_min": float(stp_x.detach().min()),
        "stp_x_max": float(stp_x.detach().max()),
    }


def probe_vereda_final_backward_and_vsa_grad() -> dict[str, Any]:
    torch.manual_seed(1013)
    config = ModelConfig(
        architecture="vereda_final",
        d_model=32,
        n_layers=2,
        head_size=16,
        local_dim=16,
        ffn_multiplier=1.0,
        dropout=0.0,
    )
    model = build_model(config)
    byte_ids = torch.randint(0, 256, (2, 24))
    memory = SimpleNamespace(
        embedding=torch.randn(config.d_model).tolist(),
        text="Memoria textual deterministica: o codigo VEREDA e Z-731.",
    )
    logits, _, auxiliary = model(byte_ids, memories=[memory])
    loss = masked_language_loss(logits, byte_ids, auxiliary["loss_mask"])
    loss.backward()
    grad_targets = {
        "p_u": model.backbone.blocks[0].p_u.weight.grad,
        "p_w": model.backbone.blocks[0].p_w.weight.grad,
        "p_c": model.backbone.blocks[0].p_c.weight.grad,
        "slow_write": model.backbone.blocks[0].slow_write.weight.grad,
        "memory_projection": model.memory_projection.weight.grad,
        "memory_vsa_projection": model.memory_vsa_projection.weight.grad,
    }
    grad_sums = {
        key: float(value.detach().abs().sum()) if value is not None else 0.0
        for key, value in grad_targets.items()
    }
    finite = bool(torch.isfinite(logits).all()) and all(math.isfinite(value) for value in grad_sums.values())
    min_grad = min(grad_sums.values())
    return {
        "name": "vereda_final_backward_and_vsa_grad",
        "max_abs_error": 0.0 if finite and min_grad > 0.0 else float("inf"),
        "threshold": 0.0,
        "pass": finite and min_grad > 0.0,
        "loss": float(loss.detach()),
        "grad_sums": grad_sums,
    }


def probe_vereda_final_checkpoint_equivalence() -> dict[str, Any]:
    torch.manual_seed(1014)
    config = ModelConfig(
        architecture="vereda_final",
        d_model=32,
        n_layers=2,
        head_size=16,
        ffn_multiplier=1.0,
        dropout=0.0,
    )
    backbone = VeredaFinalBackbone(config).train()
    patch_embeddings = torch.randn(2, 5, config.d_model, requires_grad=True)
    state = backbone.init_state(2, patch_embeddings.device, patch_embeddings.dtype)
    backbone.activation_checkpointing = False
    hidden_plain, _, _ = backbone(patch_embeddings, {key: value.clone() for key, value in state.items()})
    loss_plain = hidden_plain.square().mean()
    grad_plain = torch.autograd.grad(loss_plain, patch_embeddings, retain_graph=False)[0]

    patch_embeddings_ckpt = patch_embeddings.detach().clone().requires_grad_(True)
    backbone.activation_checkpointing = True
    hidden_ckpt, _, _ = backbone(
        patch_embeddings_ckpt,
        {key: value.clone() for key, value in state.items()},
    )
    loss_ckpt = hidden_ckpt.square().mean()
    grad_ckpt = torch.autograd.grad(loss_ckpt, patch_embeddings_ckpt, retain_graph=False)[0]

    hidden_error = float((hidden_plain.detach() - hidden_ckpt.detach()).abs().max())
    grad_error = float((grad_plain.detach() - grad_ckpt.detach()).abs().max())
    error = max(hidden_error, grad_error)
    return {
        "name": "vereda_final_checkpoint_equivalence",
        "max_abs_error": error,
        "threshold": 1e-6,
        "pass": error <= 1e-6,
        "hidden_error": hidden_error,
        "grad_error": grad_error,
    }


def probe_topology_loss_tearing_order() -> dict[str, Any]:
    smooth = torch.zeros(2, 8, 6)
    for step in range(8):
        smooth[:, step, 0] = step * 0.02
        smooth[:, step, 1] = step * 0.01
    torn = smooth.clone()
    torn[:, 4:, :] += torch.tensor([4.0, -3.0, 2.0, -1.0, 0.5, -0.5])
    smooth_loss = float(compute_topology_loss(smooth))
    torn_loss = float(compute_topology_loss(torn))
    margin = torn_loss - smooth_loss
    return {
        "name": "topology_loss_tearing_order",
        "max_abs_error": 0.0 if margin > 1e-3 else float("inf"),
        "threshold": 0.0,
        "pass": margin > 1e-3 and math.isfinite(smooth_loss) and math.isfinite(torn_loss),
        "smooth_loss": smooth_loss,
        "torn_loss": torn_loss,
        "margin": margin,
    }


def lab_contribution_summary() -> dict[str, Any]:
    summary = _read_json("research/experiments/lab-comparison/summary.json")
    rows = []
    for item in summary["comparisons"]:
        comparison = item["comparison"]
        rows.append(
            {
                "seed": item["seed"],
                "bpb_ratio_dual_over_baseline": comparison["bpb_ratio_dual_over_baseline"],
                "latency_ratio_dual_over_baseline": comparison["latency_ratio_dual_over_baseline"],
                "memory_gain_points": comparison["memory_gain_points"],
                "contribution_pass": comparison["contribution_pass"],
            }
        )
    return {
        "criteria": {
            "max_bpb_ratio": 1.03,
            "max_latency_ratio": 1.5,
            "min_memory_gain_points": 0.15,
        },
        "rows": rows,
        "mean": summary["mean"],
        "all_pass": all(row["contribution_pass"] for row in rows),
    }


def run_metrics_summary() -> dict[str, Any]:
    runs = [
        "runs/lab-2m-rwkv7/metrics.jsonl",
        "runs/lab-2m-dual-state/metrics.jsonl",
        "runs/lab-2m-vereda-v2/metrics.jsonl",
        "runs/vereda-7m-mac/metrics.jsonl",
    ]
    result: dict[str, Any] = {}
    for run in runs:
        rows = _read_jsonl(run)
        train_rows = [row for row in rows if "train_bpb" in row]
        val_rows = [row for row in rows if "validation_bpb" in row]
        latest_train_segment, resets = _latest_monotonic_segment(rows, "train_bpb")
        result[run] = {
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "non_monotonic_step_resets": resets,
            "last_train": train_rows[-1] if train_rows else None,
            "last_validation": val_rows[-1] if val_rows else None,
            "best_validation_bpb": min((row["validation_bpb"] for row in val_rows), default=None),
            "latest_train_segment": latest_train_segment,
        }
    return result


def patching_cost_summary() -> dict[str, Any]:
    sequence_bytes = 4096
    patch_sizes = [1, 2, 4, 8, 16, 32]
    rows = []
    for patch_size in patch_sizes:
        recurrent_steps = math.ceil(sequence_bytes / patch_size)
        rows.append(
            {
                "patch_size": patch_size,
                "recurrent_steps": recurrent_steps,
                "relative_recurrent_steps": recurrent_steps / sequence_bytes,
            }
        )
    sample = (
        "O projeto VEREDA mede acentua\u00e7\u00e3o, concord\u00e2ncia, mem\u00f3ria "
        "e a\u00e7\u00e3o em portugu\u00eas brasileiro."
    )
    return {
        "sequence_bytes": sequence_bytes,
        "rows": rows,
        "pt_br_sample_chars": len(sample),
        "pt_br_sample_bytes": len(sample.encode("utf-8")),
        "pt_br_bytes_per_char": len(sample.encode("utf-8")) / len(sample),
    }


def plot_contribution(contribution: dict[str, Any]) -> str:
    seeds = [str(row["seed"]) for row in contribution["rows"]]
    bpb = [row["bpb_ratio_dual_over_baseline"] for row in contribution["rows"]]
    latency = [row["latency_ratio_dual_over_baseline"] for row in contribution["rows"]]
    memory = [row["memory_gain_points"] for row in contribution["rows"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].bar(seeds, bpb, color="#4C78A8")
    axes[0].axhline(contribution["criteria"]["max_bpb_ratio"], color="#C44E52", linestyle="--")
    axes[0].set_title("BPB ratio")
    axes[0].set_xlabel("seed")
    axes[0].set_ylim(0, max(max(bpb), 1.1) * 1.1)
    axes[1].bar(seeds, latency, color="#55A868")
    axes[1].axhline(contribution["criteria"]["max_latency_ratio"], color="#C44E52", linestyle="--")
    axes[1].set_title("Latency ratio")
    axes[1].set_xlabel("seed")
    axes[1].set_ylim(0, max(1.6, max(latency) * 1.1))
    axes[2].bar(seeds, memory, color="#8172B3")
    axes[2].axhline(contribution["criteria"]["min_memory_gain_points"], color="#C44E52", linestyle="--")
    axes[2].set_title("Memory gain")
    axes[2].set_xlabel("seed")
    axes[2].set_ylim(0, 0.2)
    path = OUT / "contribution_criteria.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_math_invariants(probes: list[dict[str, Any]]) -> str:
    labels = [probe["name"].replace("_", "\n") for probe in probes if "max_abs_error" in probe]
    errors = [max(probe["max_abs_error"], 1e-16) for probe in probes if "max_abs_error" in probe]
    thresholds = [probe["threshold"] for probe in probes if "max_abs_error" in probe]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, errors, color="#4C78A8", label="measured error")
    ax.scatter(x, thresholds, color="#C44E52", label="threshold", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("absolute error, log scale")
    ax.set_title("Local mathematical invariants")
    ax.legend()
    path = OUT / "math_invariants.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_training_bpb(metrics: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for run, summary in metrics.items():
        segment = summary["latest_train_segment"]
        if not segment:
            continue
        name = Path(run).parent.name
        steps = [row["step"] for row in segment]
        bpb = [row["train_bpb"] for row in segment]
        ax.plot(steps, bpb, marker="o", markersize=3, label=name)
    ax.set_xlabel("step in latest monotonic log segment")
    ax.set_ylabel("train BPB")
    ax.set_title("Latest visible training segments")
    ax.legend(fontsize=8)
    path = OUT / "training_bpb_latest_segments.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_patching_cost(patching: dict[str, Any]) -> str:
    rows = patching["rows"]
    labels = [str(row["patch_size"]) for row in rows]
    values = [row["relative_recurrent_steps"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(labels, values, color="#55A868")
    ax.set_xlabel("patch size in bytes")
    ax.set_ylabel("relative recurrent steps")
    ax.set_title("Fixed patching recurrence cost")
    path = OUT / "patching_cost_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probes = [
        probe_lorentz_projection(),
        probe_lorentz_retraction_error_bound(),
        probe_dendritic_manifold(),
        probe_dual_state_delta_exact_recall(),
        probe_cbh_associativity(),
        probe_orthogonal_reset(),
        probe_cif_parallel_equivalence(),
        probe_stp_boundedness(),
        probe_ternary_ste(),
        probe_adaptive_gamma_quant_finite(),
        probe_rls_spd_convergence(),
        probe_vsa_random_orthogonality(),
        probe_vereda_final_state_bounds(),
        probe_vereda_final_backward_and_vsa_grad(),
        probe_vereda_final_checkpoint_equivalence(),
        probe_topology_loss_tearing_order(),
    ]
    contribution = lab_contribution_summary()
    metrics = run_metrics_summary()
    patching = patching_cost_summary()
    figures = {
        "contribution_criteria": plot_contribution(contribution),
        "math_invariants": plot_math_invariants(probes),
        "training_bpb_latest_segments": plot_training_bpb(metrics),
        "patching_cost_curve": plot_patching_cost(patching),
    }
    report = {
        "generated_by": "research/experiments/math-validation/validate_math.py",
        "math_probes": probes,
        "lab_contribution": contribution,
        "run_metrics": {
            key: {k: v for k, v in value.items() if k != "latest_train_segment"}
            for key, value in metrics.items()
        },
        "patching_cost": patching,
        "figures": figures,
        "overall": {
            "local_math_invariants_pass": all(probe["pass"] for probe in probes),
            "lab_contribution_pass": contribution["all_pass"],
            "pretraining_ready": all(probe["pass"] for probe in probes) and contribution["all_pass"],
        },
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
