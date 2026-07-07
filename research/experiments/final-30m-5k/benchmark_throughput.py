from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from vereda.config import load_config
from vereda.data import BatchStream, build_source
from vereda.models import build_model
from vereda.patching import bits_per_byte
from vereda.training.trainer import (
    batch_memory_embeddings,
    compute_curvature_loss,
    compute_syntax_loss,
    compute_topology_loss,
    masked_language_loss,
    select_device,
    seed_everything,
)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _sample_batch(config_path: str, seq_len: int, batch_size: int) -> dict[str, torch.Tensor]:
    config = load_config(config_path)
    source = build_source(config.data)
    stream = BatchStream(source, seq_len, config.data.seed + 900_001 + seq_len)
    return stream.next_batch(batch_size)


def _loss_for_chunk(model, byte_ids: torch.Tensor, source_mask: torch.Tensor, memories) -> tuple[torch.Tensor, dict[str, Any]]:
    logits, _, auxiliary = model(byte_ids, memories=memories)
    mask = source_mask & auxiliary["loss_mask"]
    loss = masked_language_loss(logits, byte_ids, mask)
    if "hidden" in auxiliary:
        loss_curv = compute_curvature_loss(auxiliary["hidden"])
        loss = loss + 0.1 * loss_curv
        auxiliary["loss_curvature"] = loss_curv

        loss_syn = compute_syntax_loss(
            byte_ids,
            auxiliary["hidden"],
            patch_size=model.config.patch_size,
        )
        loss = loss + 0.2 * loss_syn
        auxiliary["loss_syntax"] = loss_syn

        loss_topology = compute_topology_loss(auxiliary["hidden"])
        loss = loss + 0.05 * loss_topology
        auxiliary["loss_topology"] = loss_topology
    return loss, auxiliary


def _scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().float().cpu())
    return None


def _measure_chunk(
    config_path: str,
    chunk_bytes: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    config = load_config(config_path)
    config.model.validate()
    seed_everything(config.train.seed)
    model = build_model(config.model).to(device).train()
    parameter_count = model.parameter_count()
    values = []
    losses = []
    metrics: dict[str, list[float]] = {}
    for repeat in range(repeats):
        batch = _sample_batch(config_path, chunk_bytes, batch_size)
        byte_ids = batch["input_ids"].to(device)
        source_mask = batch["loss_mask"].to(device)
        memories = batch_memory_embeddings(model, batch, device)
        model.zero_grad(set_to_none=True)
        _sync(device)
        started = time.perf_counter()
        loss, auxiliary = _loss_for_chunk(model, byte_ids, source_mask, memories)
        loss.backward()
        _sync(device)
        elapsed = time.perf_counter() - started
        losses.append(float(loss.detach().cpu()))
        values.append(elapsed)
        for key, value in auxiliary.items():
            scalar = _scalar(value)
            if scalar is not None:
                metrics.setdefault(key, []).append(scalar)
    mean_seconds = sum(values) / len(values)
    seq_len = config.data.seq_len
    official_chunks_per_microbatch = math.ceil(seq_len / chunk_bytes)
    estimated_step_seconds = mean_seconds * official_chunks_per_microbatch * config.train.grad_accum_steps
    estimated_5000_hours = estimated_step_seconds * 5000 / 3600.0
    trainable_bytes = batch_size * max(chunk_bytes - config.model.patch_size, 0)
    result = {
        "chunk_bytes": chunk_bytes,
        "batch_size": batch_size,
        "repeats": repeats,
        "device": str(device),
        "parameters": parameter_count,
        "mean_seconds_per_chunk": mean_seconds,
        "min_seconds_per_chunk": min(values),
        "max_seconds_per_chunk": max(values),
        "bytes_per_second": trainable_bytes / mean_seconds if mean_seconds > 0 else 0.0,
        "patches_per_second": (batch_size * math.ceil(chunk_bytes / config.model.patch_size)) / mean_seconds,
        "mean_loss": sum(losses) / len(losses),
        "mean_bpb": bits_per_byte(sum(losses) / len(losses)),
        "estimated_official_step_seconds": estimated_step_seconds,
        "estimated_5000_hours": estimated_5000_hours,
        "official_seq_len": seq_len,
        "official_grad_accum_steps": config.train.grad_accum_steps,
        "official_chunks_per_microbatch": official_chunks_per_microbatch,
        "auxiliary_means": {
            key: sum(items) / len(items)
            for key, items in sorted(metrics.items())
            if items
        },
    }
    return result


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["chunk_bytes"]) for row in rows]
    step_seconds = [row["estimated_official_step_seconds"] for row in rows]
    hours = [row["estimated_5000_hours"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].bar(labels, step_seconds, color="#4C78A8")
    axes[0].set_xlabel("chunk bytes")
    axes[0].set_ylabel("estimated seconds / optimizer step")
    axes[0].set_title("30M step estimate")
    axes[1].bar(labels, hours, color="#55A868")
    axes[1].set_xlabel("chunk bytes")
    axes[1].set_ylabel("estimated hours / 5000 steps")
    axes[1].set_title("5000-step estimate")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VEREDA-Final 30M chunk throughput")
    parser.add_argument("--config", default="configs/vereda_final_30m_5k.yaml")
    parser.add_argument("--chunks", default="64,128,256")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="research/experiments/final-30m-5k/throughput-report.json")
    parser.add_argument("--figure", default="research/experiments/final-30m-5k/throughput-estimate.png")
    args = parser.parse_args()

    device = select_device(args.device)
    rows = [
        _measure_chunk(args.config, chunk_bytes, args.batch_size, args.repeats, device)
        for chunk_bytes in _parse_ints(args.chunks)
    ]
    report = {
        "config": args.config,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "rows": rows,
        "note": (
            "Each row is a real forward/backward chunk with the full model. "
            "The 5000-step estimate scales by official seq_len/chunk and grad_accum_steps."
        ),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(rows, ROOT / args.figure)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
