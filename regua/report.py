# -*- coding: utf-8 -*-
"""Gera e valida o relatório M0 a partir dos JSONs já medidos.

Uso:
  python3 -m regua.report
  python3 -m regua.report --require-complete
"""
import argparse
import json
import os
import sys


OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")
REPORT = os.path.join(OUT_DIR, "RELATORIO_M0.md")

# A não depende de k na implementação atual: mede-se uma vez com perguntas k=8.
REQUIRED = [
    ("A", 8),
    ("B", 2), ("B", 8), ("B", 32), ("B", 100),
    ("C", 2), ("C", 8), ("C", 32), ("C", 100),
]


def _load_results(out_dir):
    rows = []
    if not os.path.isdir(out_dir):
        return rows
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
            row = json.load(f)
        row["_filename"] = fn
        rows.append(row)
    return rows


def _missing(rows):
    have = {(r["baseline"], int(r["k"])) for r in rows}
    return [(b, k) for b, k in REQUIRED if (b, k) not in have]


def _fmt(x, ndigits=3):
    if x is None:
        return "—"
    return ("%." + str(ndigits) + "f") % x


def write_report(out_dir=OUT_DIR):
    rows = _load_results(out_dir)
    rows.sort(key=lambda r: (r["baseline"], int(r["k"])))
    missing = _missing(rows)

    md = [
        "# Relatório M0 — baselines congelados",
        "",
        "Juiz: valor-ouro normalizado, com fronteira de palavra, na resposta",
        "gerada (greedy, determinística). Dataset: held pools, seed fixa.",
        "",
        "Nota: o baseline A não depende de `k`; por isso é medido uma vez",
        "com o conjunto de perguntas `k=8`.",
        "",
        "| baseline | k | n | acc | retr top-1 | tok prompt | s/item | acc pos 1º/2º/3º terço |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        thirds = r["acc_por_terco_posicao"]
        third_text = "/".join(
            _fmt(thirds[x], 2) if thirds[x] is not None else "-"
            for x in ("0", "1", "2")
        )
        md.append("| %s | %d | %d | %s | %s | %.0f | %.2f | %s |" % (
            r["baseline"], int(r["k"]), int(r["n"]), _fmt(r["acc"]),
            _fmt(r["retrieval_acc_top1"]),
            float(r["tokens_prompt_medio"]), float(r["s_por_item"]),
            third_text,
        ))

    md += [
        "",
        "## Portão M0",
        "",
    ]
    if missing:
        md.append("**Status: incompleto.** Faltam células:")
        for b, k in missing:
            md.append("- `%s_k%d_n1024.json`" % (b, k))
    else:
        md.append("**Status: completo.** Todas as células obrigatórias existem.")

    md += [
        "",
        "Sentinelas aplicáveis no M0: S3 (valores variáveis — ver",
        "`dataset_stats` nos JSONs), S4 (posição embaralhada + acc por terço),",
        "S5 (n reportado), S6 (n/a — nada treinado). S1/S2 aplicam-se a",
        "partir do M1 (módulos treinados).",
        "",
    ]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RELATORIO_M0.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    missing = write_report()
    print("Relatório: %s" % REPORT, flush=True)
    if missing:
        print("M0 incompleto: faltam %s" % ", ".join(
            "%s_k%d" % (b, k) for b, k in missing), flush=True)
        if args.require_complete:
            sys.exit(1)
    else:
        print("M0 completo.", flush=True)


if __name__ == "__main__":
    main()
