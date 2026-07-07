from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vereda.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from vereda.eval.benchmark import benchmark_checkpoint
from vereda.eval.memory import evaluate_episodic_memory, evaluate_recurrent_memory
from vereda.training.trainer import Trainer


EXPERIMENT_DIR = ROOT / "research" / "experiments" / "article-ablations"
RESULTS_SYNTHESIS = ROOT / "research" / "results" / "article_ablation_synthesis.md"
DEFAULT_RUN_ROOT = ROOT / "runs" / "article-ablations"


@dataclass(frozen=True)
class Variant:
    slug: str
    group: str
    question: str
    hypothesis: str
    changed_variable: str
    falsification: str
    articles: tuple[str, ...]
    data_kind: str = "synthetic"
    data_path: str = ""
    model_updates: dict[str, Any] = field(default_factory=dict)
    train_updates: dict[str, Any] = field(default_factory=dict)
    baseline_slug: str | None = None
    run_memory_eval: bool = False


GROUP_TITLES = {
    "memory-vsa-kalman": "Memoria VSA Kalman",
    "portuguese-explicit": "Portugues Explicito",
    "topology-geometry": "Topologia Geometria",
    "hyperbolic-quantization": "Quantizacao Hiperbolica",
    "patching-compression": "Patching Compressao",
}


def local_critical_variants() -> list[Variant]:
    memory_articles = (
        "Projeto VEREDA_ Nucleo SSM Cognitivo",
        "VEREDA_ Alfabetizacao Radical em Portugues",
        "Arquitetura de Linguagem Byte-Level Inovadora",
    )
    geometry_articles = (
        "Estabilizacao Geometrica de SSMs Ternarios",
        "VEREDA_ Hyperbolic SSM Derivation",
        "VEREDA-v2_ Arquitetura Byte-Level Revolucionaria",
    )
    return [
        Variant(
            slug="vereda_final_full",
            group="memory-vsa-kalman",
            question="VSA textual e Kalman/RLS-style ajudam memoria sem destruir BPB?",
            hypothesis="O candidato completo melhora memoria recorrente ou episodica sem piorar BPB e latencia.",
            changed_variable="Baseline completo: VSA textual ligado, Kalman write ligado.",
            falsification="Rejeitar se uma variante sem VSA/Kalman for igual ou melhor em memoria e BPB.",
            articles=memory_articles,
            data_kind="conversation_jsonl",
            data_path="data/long_memory_curriculum.jsonl",
            run_memory_eval=True,
        ),
        Variant(
            slug="no_kalman_write",
            group="memory-vsa-kalman",
            question="A modulacao Kalman/RLS-style da escrita lenta e necessaria?",
            hypothesis="Remover Kalman deve reduzir recall ou estabilidade da memoria lenta.",
            changed_variable="`use_kalman_write=false`; VSA textual permanece ligado.",
            falsification="Rejeitar Kalman se esta variante empatar ou melhorar BPB/memoria contra o completo.",
            articles=memory_articles,
            data_kind="conversation_jsonl",
            data_path="data/long_memory_curriculum.jsonl",
            model_updates={"use_kalman_write": False},
            baseline_slug="vereda_final_full",
            run_memory_eval=True,
        ),
        Variant(
            slug="no_vsa_memory",
            group="memory-vsa-kalman",
            question="A assinatura textual VSA melhora a memoria externa treinavel?",
            hypothesis="Remover VSA deve reduzir recuperacao factual quando memorias textuais existem.",
            changed_variable="`use_vsa_memory=false`; embeddings densos continuam ativos.",
            falsification="Rejeitar VSA se esta variante empatar ou melhorar BPB/memoria contra o completo.",
            articles=memory_articles,
            data_kind="conversation_jsonl",
            data_path="data/long_memory_curriculum.jsonl",
            model_updates={"use_vsa_memory": False},
            baseline_slug="vereda_final_full",
            run_memory_eval=True,
        ),
        Variant(
            slug="syntax_on",
            group="portuguese-explicit",
            question="Regularizacao sintatica explicita ajuda concordancia PT-BR?",
            hypothesis="`syntax_loss_weight=0.2` melhora concordancia sem piorar BPB alem do ruido local.",
            changed_variable="Syntax loss ligada com peso 0.2.",
            falsification="Rejeitar se `syntax_off` igualar ou superar concordancia e BPB.",
            articles=(
                "Evolucao do VEREDA para Ensino Explicito de Portugues",
                "VEREDA_ Alfabetizacao Radical em Portugues",
            ),
            data_kind="conversation_jsonl",
            data_path="data/morphology_curriculum.jsonl",
            train_updates={"syntax_loss_weight": 0.2},
        ),
        Variant(
            slug="syntax_off",
            group="portuguese-explicit",
            question="Qual e o controle sem loss sintatica explicita?",
            hypothesis="Sem loss sintatica, o treino depende apenas de cross-entropy.",
            changed_variable="`syntax_loss_weight=0.0`.",
            falsification="Se empatar ou melhorar, a loss sintatica fica sem evidencia local.",
            articles=(
                "Evolucao do VEREDA para Ensino Explicito de Portugues",
                "VEREDA_ Alfabetizacao Radical em Portugues",
            ),
            data_kind="conversation_jsonl",
            data_path="data/morphology_curriculum.jsonl",
            train_updates={"syntax_loss_weight": 0.0},
            baseline_slug="syntax_on",
        ),
        Variant(
            slug="geometry_full",
            group="topology-geometry",
            question="Geometria leve estabiliza o candidato final em treino curto?",
            hypothesis="Curvatura, topologia e dendrito Lorentz reduzem tearing sem custo proibitivo.",
            changed_variable="Baseline geometrico: curvatura 0.1, topologia 0.05, Lorentz/dendrito ligado.",
            falsification="Rejeitar se variantes sem esses termos tiverem BPB/estado melhores sem falhas.",
            articles=geometry_articles,
        ),
        Variant(
            slug="topology_off",
            group="topology-geometry",
            question="A topology loss tem efeito mensuravel no treino local?",
            hypothesis="Desligar topology loss aumenta tearing ou piora estabilidade.",
            changed_variable="`topology_loss_weight=0.0`.",
            falsification="Rejeitar topology loss se BPB/estado melhorarem sem falhas.",
            articles=geometry_articles,
            train_updates={"topology_loss_weight": 0.0},
            baseline_slug="geometry_full",
        ),
        Variant(
            slug="curvature_off",
            group="topology-geometry",
            question="A curvature loss ajuda alem da topology loss?",
            hypothesis="Desligar curvatura piora suavidade ou estado.",
            changed_variable="`curvature_loss_weight=0.0`.",
            falsification="Rejeitar curvature loss se a variante melhorar BPB/estado.",
            articles=geometry_articles,
            train_updates={"curvature_loss_weight": 0.0},
            baseline_slug="geometry_full",
        ),
        Variant(
            slug="no_lorentz_dendrite",
            group="topology-geometry",
            question="O dendrito Lorentz contribui em linguagem local curta?",
            hypothesis="Remover o caminho Lorentz/dendrito deve piorar BPB ou estabilidade.",
            changed_variable="`use_lorentz_dendrite=false`.",
            falsification="Rejeitar dendrito Lorentz se a variante empatar ou melhorar.",
            articles=geometry_articles,
            model_updates={"use_lorentz_dendrite": False},
            baseline_slug="geometry_full",
        ),
        Variant(
            slug="quant_on",
            group="hyperbolic-quantization",
            question="Adaptive gamma quantization e estavel e util em treino curto?",
            hypothesis="Quantizacao adaptativa mantem gradientes finitos sem piorar BPB/latencia.",
            changed_variable="Quantizacao adaptativa ligada.",
            falsification="Rejeitar se `quant_off` melhorar BPB/latencia sem custo de estabilidade.",
            articles=(
                "Hyperbolic Quantization for VEREDA Architecture",
                "Estabilizacao Geometrica de SSMs Ternarios",
            ),
        ),
        Variant(
            slug="quant_off",
            group="hyperbolic-quantization",
            question="Qual e o controle sem adaptive gamma quantization?",
            hypothesis="Sem quantizacao adaptativa, a arquitetura mede capacidade sem esse ruído.",
            changed_variable="`use_adaptive_gamma_quant=false`.",
            falsification="Se melhorar, a quantizacao fica apenas como candidato de compressao futura.",
            articles=(
                "Hyperbolic Quantization for VEREDA Architecture",
                "Estabilizacao Geometrica de SSMs Ternarios",
            ),
            model_updates={"use_adaptive_gamma_quant": False},
            baseline_slug="quant_on",
        ),
        *[
            Variant(
                slug=f"patch_size_{patch_size}",
                group="patching-compression",
                question="Como o tamanho fixo de patch muda custo recorrente e qualidade local?",
                hypothesis="Patch maior reduz passos recorrentes, mas pode piorar BPB ou UTF-8.",
                changed_variable=f"`patch_size={patch_size}`; decoder ainda fixo por tamanho.",
                falsification="Rejeitar ganho de compressao se BPB/UTF-8 piorarem alem do controle local.",
                articles=(
                    "VEREDA_ Hierarchical Temporal Sub-sampling Mechanism",
                    "VEREDA-v2_ Arquitetura Byte-Level Revolucionaria",
                    "Arquitetura de Linguagem Byte-Level Inovadora",
                ),
                model_updates={"patch_size": patch_size},
                baseline_slug="patch_size_4" if patch_size != 4 else None,
            )
            for patch_size in (1, 2, 4, 8)
        ],
    ]


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def markdown_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def base_config(variant: Variant, seed: int, steps: int, run_dir: Path, device: str) -> ExperimentConfig:
    model_values = {
        "architecture": "vereda_final",
        "d_model": 64,
        "n_layers": 2,
        "head_size": 16,
        "patch_size": 4,
        "local_dim": 32,
        "ffn_multiplier": 1.0,
        "dropout": 0.0,
    }
    model_values.update(variant.model_updates)
    patch_size = int(model_values["patch_size"])
    data_path = ROOT / variant.data_path if variant.data_path else Path()
    if variant.data_kind != "synthetic" and (not data_path.exists() or data_path.stat().st_size == 0):
        raise FileNotFoundError(f"Required ablation data is unavailable: {data_path}")
    data = DataConfig(
        kind=variant.data_kind,
        path=variant.data_path,
        seq_len=max(64, patch_size * 8),
        seed=seed,
    )
    train_values = {
        "output_dir": str(run_dir),
        "batch_size": 2,
        "steps": steps,
        "learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "grad_accum_steps": 1,
        "checkpoint_every": max(steps + 1, 1),
        "eval_every": max(steps, 1),
        "log_every": max(1, min(10, steps or 1)),
        "device": device,
        "dtype": "fp32",
        "seed": seed,
        "tbptt_chunk_bytes": 0,
        "sample_every": 0,
    }
    train_values.update(variant.train_updates)
    return ExperimentConfig(
        name=f"{variant.slug}-seed-{seed}",
        model=ModelConfig(**model_values),
        data=data,
        train=TrainConfig(**train_values),
    )


def load_train_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return {}
    latest_train = next((row for row in reversed(rows) if "train_bpb" in row), {})
    latest_validation = next((row for row in reversed(rows) if "validation_bpb" in row), {})
    return {
        "latest_train": latest_train,
        "latest_validation": latest_validation,
        "log_rows": len(rows),
    }


def flattened_metric(report: dict[str, Any], key: str) -> float | None:
    value: Any = report
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def metric_mean(reports: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [value for report in reports if (value := flattened_metric(report, key)) is not None]
    return mean(values) if values else None


def run_one(
    variant: Variant,
    seed: int,
    steps: int,
    suite: str,
    run_root: Path,
    experiment_dir: Path,
    device: str,
    generation_bytes: int,
    quality_batches: int,
    memory_horizons: tuple[int, ...],
    overwrite: bool,
) -> dict[str, Any]:
    run_dir = run_root / suite / f"{variant.slug}-seed-{seed}"
    report_path = experiment_dir / "reports" / "json" / f"{variant.slug}-seed-{seed}.json"
    if report_path.exists() and (run_dir / "final.pt").exists() and not overwrite:
        return json.loads(report_path.read_text(encoding="utf-8"))
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                f"Run directory already exists without a reusable report: {run_dir}. "
                "Use --overwrite to replace it."
            )

    config = base_config(variant, seed, steps, run_dir, device)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "suite": suite,
        "variant": variant.slug,
        "group": variant.group,
        "seed": seed,
        "run_dir": str(run_dir),
        "status": "running",
        "config": config.to_dict(),
    }
    try:
        checkpoint = Trainer(config).train(max_steps=steps)
        report.update(
            benchmark_checkpoint(
                checkpoint,
                generation_bytes=generation_bytes,
                quality_batches=quality_batches,
                data=DataConfig(kind="conversation_jsonl", path="data/pt_teste_5k.jsonl", seed=seed + 9000),
                seq_len=config.data.seq_len,
            )
        )
        if variant.run_memory_eval:
            report["episodic_memory"] = evaluate_episodic_memory(checkpoint, horizons=memory_horizons)
            report["recurrent_memory"] = evaluate_recurrent_memory(checkpoint, horizons=memory_horizons)
        report["train_metrics"] = load_train_metrics(run_dir)
        report["checkpoint"] = str(checkpoint)
        report["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - experiment failures are evidence.
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(report, report_path)
    return report


def read_probe_bytes(limit: int = 4096) -> bytes:
    preferred = ROOT / "data" / "pt_teste_5k.jsonl"
    if preferred.exists() and preferred.stat().st_size:
        return preferred.read_bytes()[:limit]
    return (ROOT / "README.md").read_bytes()[:limit]


def dynamic_patching_poc(values: bytes) -> dict[str, Any]:
    if not values:
        return {"causal": True, "segments": 0, "steps_per_byte": 0.0, "average_patch_length": 0.0}
    counts = [1] * 256
    total = 256
    patch_lengths = []
    current = 0
    surprises = []
    for byte in values:
        probability = counts[byte] / total
        surprise = -math.log2(probability)
        surprises.append(surprise)
        current += 1
        counts[byte] += 1
        total += 1
        prefix_mean = mean(surprises)
        variance = mean((item - prefix_mean) ** 2 for item in surprises)
        threshold = prefix_mean + math.sqrt(variance)
        if current >= 8 or (current >= 2 and surprise >= threshold):
            patch_lengths.append(current)
            current = 0
    if current:
        patch_lengths.append(current)
    return {
        "causal": True,
        "segments": len(patch_lengths),
        "bytes": len(values),
        "steps_per_byte": len(patch_lengths) / len(values),
        "average_patch_length": mean(patch_lengths),
        "min_patch_length": min(patch_lengths),
        "max_patch_length": max(patch_lengths),
    }


def patching_poc() -> dict[str, Any]:
    values = read_probe_bytes()
    fixed = [
        {
            "patch_size": patch_size,
            "bytes": len(values),
            "segments": math.ceil(len(values) / patch_size),
            "steps_per_byte": math.ceil(len(values) / patch_size) / max(len(values), 1),
        }
        for patch_size in (1, 2, 4, 8)
    ]
    return {
        "note": "Dynamic segmentation is an offline causal PoC; the model decoder is still fixed-size.",
        "fixed": fixed,
        "dynamic_entropy_poc": dynamic_patching_poc(values),
    }


def write_figures(summary: dict[str, Any], experiment_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, str] = {}
    runs = [run for run in summary["runs"] if run.get("status") == "ok"]
    for group, title in GROUP_TITLES.items():
        group_runs = [run for run in runs if run["group"] == group]
        if not group_runs:
            continue
        variants = sorted({run["variant"] for run in group_runs})
        metrics = [
            ("validation_bpb", "validation_bpb"),
            ("bytes_per_second", "bytes_per_second"),
            ("recurrent_memory", "recurrent_memory.recurrent_memory_accuracy"),
        ]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3), constrained_layout=True)
        if len(metrics) == 1:
            axes = [axes]
        for axis, (label, key) in zip(axes, metrics):
            values = [
                metric_mean([run for run in group_runs if run["variant"] == variant], key)
                for variant in variants
            ]
            axis.bar(variants, [0.0 if value is None else value for value in values])
            axis.set_title(label)
            axis.tick_params(axis="x", rotation=45)
            for tick, value in zip(axis.get_xticklabels(), values):
                if value is None:
                    tick.set_alpha(0.35)
        fig.suptitle(title)
        path = experiment_dir / "figures" / f"{group}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures[group] = display_path(path)

    poc = summary.get("patching_poc", {})
    if poc:
        labels = [str(item["patch_size"]) for item in poc["fixed"]] + ["dynamic"]
        values = [item["steps_per_byte"] for item in poc["fixed"]] + [
            poc["dynamic_entropy_poc"]["steps_per_byte"]
        ]
        fig, axis = plt.subplots(figsize=(6, 3), constrained_layout=True)
        axis.bar(labels, values)
        axis.set_title("Patching recurrent steps per byte")
        axis.set_xlabel("patch policy")
        axis.set_ylabel("steps / byte")
        path = experiment_dir / "figures" / "patching_cost_poc.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures["patching_poc"] = display_path(path)
    return figures


def summarize_group(group: str, variants: list[Variant], runs: list[dict[str, Any]]) -> dict[str, Any]:
    group_runs = [run for run in runs if run["group"] == group and run.get("status") == "ok"]
    means = {}
    for variant in sorted({variant.slug for variant in variants if variant.group == group}):
        reports = [run for run in group_runs if run["variant"] == variant]
        means[variant] = {
            "validation_bpb": metric_mean(reports, "validation_bpb"),
            "bytes_per_second": metric_mean(reports, "bytes_per_second"),
            "utf8_valid_rate": metric_mean(reports, "utf8_valid_rate"),
            "repetition_rate": metric_mean(reports, "repetition_rate"),
            "rough_agreement_accuracy": metric_mean(reports, "rough_agreement_accuracy"),
            "recurrent_memory_accuracy": metric_mean(
                reports, "recurrent_memory.recurrent_memory_accuracy"
            ),
            "state_norm": metric_mean(reports, "train_metrics.latest_validation.validation_state_norm"),
        }
    failed = [run for run in runs if run["group"] == group and run.get("status") == "failed"]
    result = "pendente"
    learned = "Ainda sem execucao completa; o dossie registra o protocolo."
    if failed:
        result = "inconclusivo"
        learned = "Houve falha de execucao; a falha deve ser tratada como evidencia de engenharia antes de promover a tese."
    elif group_runs:
        result = "inconclusivo"
        learned = "A rodada local curta produz sinais comparativos, mas nao valida a tese como resultado final."
    return {"group": group, "means": means, "result": result, "learned": learned}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown_reports(summary: dict[str, Any], variants: list[Variant], experiment_dir: Path) -> None:
    reports_dir = experiment_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    group_summaries = {
        item["group"]: item for item in summary.get("groups", [])
    }
    for group, title in GROUP_TITLES.items():
        group_variants = [variant for variant in variants if variant.group == group]
        if not group_variants:
            continue
        group_summary = group_summaries.get(group, {})
        rows = []
        for variant in group_variants:
            metrics = group_summary.get("means", {}).get(variant.slug, {})
            rows.append(
                [
                    variant.slug,
                    metrics.get("validation_bpb"),
                    metrics.get("bytes_per_second"),
                    metrics.get("utf8_valid_rate"),
                    metrics.get("recurrent_memory_accuracy"),
                ]
            )
        figures = summary.get("figures", {})
        figure_line = f"\n![{title}]({markdown_path(figures[group])})\n" if group in figures else ""
        baseline = next((variant.slug for variant in group_variants if variant.baseline_slug is None), group_variants[0].slug)
        content = f"""# {title}

## Pergunta
{group_variants[0].question}

## Hipotese
{group_variants[0].hypothesis}

## Baseline
`{baseline}`

## Variavel alterada
{'; '.join(f'`{variant.slug}`: {variant.changed_variable}' for variant in group_variants)}

## Artigos-alvo
{chr(10).join(f'- {article}' for article in sorted(set(article for variant in group_variants for article in variant.articles)))}

## Seeds e metricas
Seeds: `{', '.join(str(seed) for seed in summary.get('seeds', []))}`.

{markdown_table(['Variante', 'BPB', 'bytes/s', 'UTF-8', 'memoria recorrente'], rows)}
{figure_line}
## Criterio de falsificacao
{group_variants[0].falsification}

## Resultado
{group_summary.get('result', 'pendente')}

## O que aprendemos
{group_summary.get('learned', 'Ainda sem execucao completa.')}
"""
        (reports_dir / f"{group}.md").write_text(content, encoding="utf-8")


def write_synthesis(summary: dict[str, Any], experiment_dir: Path) -> None:
    rows = [
        [GROUP_TITLES.get(item["group"], item["group"]), item["result"], item["learned"]]
        for item in summary.get("groups", [])
    ]
    figures = summary.get("figures", {})
    figure_lines = "\n".join(
        f"- {name}: `{path}`" for name, path in sorted(figures.items())
    )
    summary_json = display_path(experiment_dir / "summary.json")
    reports_dir = display_path(experiment_dir / "reports")
    content = f"""# Sintese das ablations criticas por artigo

Suite: `{summary.get('suite')}`
Seeds: `{', '.join(str(seed) for seed in summary.get('seeds', []))}`
Steps por run: `{summary.get('steps')}`

{markdown_table(['Dossie', 'Resultado', 'O que aprendemos'], rows)}

## Artefatos
- Summary JSON: `{summary_json}`
- Relatorios Markdown: `{reports_dir}`
- Runs brutos: `runs/article-ablations/{summary.get('suite')}`

## Figuras
{figure_lines or '- Nenhuma figura gerada ainda.'}

## Nota
Esta rodada e evidencia local curta. Resultado positivo aqui promove uma tese
para ablation maior; nao valida o modelo final por si so.
"""
    destination = (
        RESULTS_SYNTHESIS
        if experiment_dir.resolve() == EXPERIMENT_DIR.resolve()
        else experiment_dir / "article_ablation_synthesis.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def build_dry_run(
    variants: list[Variant],
    seeds: tuple[int, ...],
    steps: int,
    suite: str,
    run_root: Path,
    device: str,
) -> dict[str, Any]:
    entries = []
    for variant in variants:
        for seed in seeds:
            run_dir = run_root / suite / f"{variant.slug}-seed-{seed}"
            config = base_config(variant, seed, steps, run_dir, device)
            entries.append(
                {
                    "variant": variant.slug,
                    "group": variant.group,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "config": config.to_dict(),
                }
            )
    return {
        "suite": suite,
        "dry_run": True,
        "seeds": list(seeds),
        "steps": steps,
        "runs": entries,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    if args.suite != "local-critical":
        raise ValueError(f"Unknown suite: {args.suite}")
    variants = local_critical_variants()
    seeds = parse_ints(args.seeds)
    run_root = Path(args.run_root)
    experiment_dir = Path(args.output)

    if args.dry_run:
        return build_dry_run(variants, seeds, args.steps, args.suite, run_root, args.device)

    runs = []
    for variant in variants:
        for seed in seeds:
            print(json.dumps({"event": "start", "variant": variant.slug, "seed": seed}, ensure_ascii=False))
            runs.append(
                run_one(
                    variant,
                    seed,
                    args.steps,
                    args.suite,
                    run_root,
                    experiment_dir,
                    args.device,
                    args.generation_bytes,
                    args.quality_batches,
                    parse_ints(args.memory_horizons),
                    args.overwrite,
                )
            )
    summary = {
        "suite": args.suite,
        "seeds": list(seeds),
        "steps": args.steps,
        "variants": [
            {
                "slug": variant.slug,
                "group": variant.group,
                "baseline_slug": variant.baseline_slug,
                "articles": list(variant.articles),
                "changed_variable": variant.changed_variable,
            }
            for variant in variants
        ],
        "runs": runs,
        "patching_poc": patching_poc(),
    }
    summary["groups"] = [
        summarize_group(group, variants, runs)
        for group in GROUP_TITLES
        if any(variant.group == group for variant in variants)
    ]
    summary["figures"] = write_figures(summary, experiment_dir)
    write_json(summary, experiment_dir / "summary.json")
    write_markdown_reports(summary, variants, experiment_dir)
    write_synthesis(summary, experiment_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run article-level VEREDA critical ablations.")
    parser.add_argument("--suite", default="local-critical")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", default=str(EXPERIMENT_DIR))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generation-bytes", type=int, default=64)
    parser.add_argument("--quality-batches", type=int, default=1)
    parser.add_argument("--memory-horizons", default="10,100")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_suite(args)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
