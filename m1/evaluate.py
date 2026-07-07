# -*- coding: utf-8 -*-
"""Avalia o portão M1 e grava relatório auditável.

Uso oficial no Colab:
  python3 -m m1.evaluate --device cuda

Uso rápido local, reaproveitando cache smoke:
  python3 -m m1.evaluate --n-train 300 --n-held 100 --eval-n 128 --out-dir /tmp
"""
import argparse
import json
import os
import random
import time

import torch

from m1.head import Head
from m1.train import CKPT, build_cache, raw_keys, retrieval_acc


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "research", "results")
REPORT_MD = os.path.join(OUT_DIR, "m1_retrieval_report.md")
REPORT_JSON = os.path.join(OUT_DIR, "m1_retrieval_report.json")


def _load_head(path, device):
    head = Head().to(device)
    head.load_state_dict(torch.load(path, map_location=device))
    head.eval()
    return head


def _eval_table(d, checkpoint, device, ks, eval_n):
    hef, hefm = d["hef"].float().to(device), d["hefm"].to(device)
    heq, heqm = d["heq"].float().to(device), d["heqm"].to(device)

    rng = random.Random(99)
    qwen_f, qwen_q = raw_keys(hef, hefm), raw_keys(heq, heqm)

    head = _load_head(checkpoint, device)
    with torch.no_grad():
        learned_f, learned_q = head(hef, hefm), head(heq, heqm)

        torch.manual_seed(7)
        random_head = Head().to(device)
        random_f, random_q = random_head(hef, hefm), random_head(heq, heqm)

    rows = []
    for k in ks:
        if k > hef.size(0):
            rows.append({
                "k": k,
                "status": "skipped",
                "reason": "k maior que n_held",
            })
            continue
        rows.append({
            "k": k,
            "status": "ok",
            "qwen_mean_pool": retrieval_acc(qwen_f, qwen_q, rng, k, n=eval_n),
            "head_learned": retrieval_acc(learned_f, learned_q, rng, k, n=eval_n),
            "head_random_ablation": retrieval_acc(random_f, random_q, rng, k, n=eval_n),
        })
    return rows


def _verdict(rows):
    by_k = {r["k"]: r for r in rows if r.get("status") == "ok"}
    needed = [8, 32, 100]
    if any(k not in by_k for k in needed):
        return "incompleto", ["faltam medições para k=8,32,100"]

    reasons = []
    if by_k[8]["head_learned"] < 0.95:
        reasons.append("k=8 abaixo de 0.95")
    if by_k[32]["head_learned"] < 0.90:
        reasons.append("k=32 abaixo de 0.90")
    if by_k[100]["head_random_ablation"] >= by_k[100]["head_learned"] * 0.5:
        reasons.append("ablação aleatória não caiu o suficiente em k=100")

    # O requisito "vence baseline B em k=100" depende do M0 já existir.
    b100_path = os.path.join(os.path.dirname(__file__), "..", "regua",
                             "resultados", "B_k100_n1024.json")
    if os.path.exists(b100_path):
        with open(b100_path, encoding="utf-8") as f:
            b100 = json.load(f)["acc"]
        if by_k[100]["head_learned"] <= b100:
            reasons.append("não venceu baseline B k=100 (%.3f)" % b100)
    else:
        reasons.append("baseline B_k100 ausente para comparação")

    return ("passou" if not reasons else "falhou"), reasons


def _write_reports(rows, meta, out_dir):
    verdict, reasons = _verdict(rows)
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "meta": meta,
        "rows": rows,
        "verdict": verdict,
        "reasons": reasons,
    }
    json_path = os.path.join(out_dir, "m1_retrieval_report.json")
    md_path = os.path.join(out_dir, "m1_retrieval_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = [
        "# Relatório M1 — cabeça de memória aprendida",
        "",
        "Avaliação held-out de retrieval top-1. Qwen está congelado; só a cabeça",
        "de memória é treinada. A ablação usa uma cabeça aleatória com a mesma",
        "arquitetura.",
        "",
        "| k | Qwen mean-pool | cabeça aprendida | ablação aleatória |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            md.append("| %d | — | — | — |" % r["k"])
            continue
        md.append("| %d | %.3f | %.3f | %.3f |" % (
            r["k"], r["qwen_mean_pool"], r["head_learned"],
            r["head_random_ablation"]))

    md += [
        "",
        "## Portão M1",
        "",
        "**Status: %s.**" % verdict,
    ]
    if reasons:
        md.append("")
        md.append("Motivos:")
        for reason in reasons:
            md.append("- %s" % reason)

    md += [
        "",
        "Critérios do roadmap: `>=0.95 @ k=8`, `>=0.90 @ k=32`, vencer",
        "baseline B em `k=100`, e ablação aleatória cair fortemente.",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return verdict, md_path, json_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CKPT)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-held", type=int, default=1200)
    p.add_argument("--eval-n", type=int, default=1024)
    p.add_argument("--k-list", default="8,32,100")
    p.add_argument("--device", default="cpu")
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="reavalia mesmo se o relatório já existir")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    report_json = os.path.join(args.out_dir, "m1_retrieval_report.json")
    if os.path.exists(report_json) and not args.force:
        print("relatório M1 já existe (%s), pulando avaliação (use --force "
              "para refazer)" % report_json, flush=True)
        return

    if not os.path.exists(args.checkpoint):
        raise SystemExit("checkpoint ausente: %s" % args.checkpoint)

    t0 = time.time()
    d = build_cache(args.n_train, args.n_held, force=args.force_cache,
                    device=args.device, batch_size=args.encode_batch)
    rows = _eval_table(
        d, args.checkpoint, args.device,
        [int(x) for x in args.k_list.split(",")],
        args.eval_n,
    )
    meta = {
        "checkpoint": args.checkpoint,
        "n_train": d.get("n_train"),
        "n_held": d.get("n_held"),
        "eval_n": args.eval_n,
        "device": args.device,
        "elapsed_s": time.time() - t0,
    }
    verdict, md_path, json_path = _write_reports(rows, meta, args.out_dir)
    print("Relatório M1: %s" % md_path, flush=True)
    print("JSON M1: %s" % json_path, flush=True)
    print("Portão M1: %s" % verdict, flush=True)


if __name__ == "__main__":
    main()
