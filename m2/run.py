# -*- coding: utf-8 -*-
"""M2 — run oficial end-to-end: pergunta → chave M1 → injetor → resposta → juiz.

Braços (DESIGN.md): T (texto, controle), S1 (soft v1), S2 (soft v2 +recon),
SR (ablação: projetor aleatório). Mesmos itens do M0 (seed 1000+k, held).

Uso:
  python3 -m m2.run --smoke                          # mecânica, números não contam
  python3 -m m2.run --device cuda --dtype float16 --batch 32   # oficial

Um JSON por célula em m2/resultados/ (retomável). Ao final, RELATORIO_M2.md
com o portão avaliado (EM ≥ retrieval M1 − 0.05 por k).
"""
import argparse
import json
import os
import time

import torch

from m2.pipeline import (M1_CKPT, generate_with_prefix, load_m1_head,
                         retrieve_chunk)
from m2.projector import Projector
from m2.train_soft import ckpt_path
from regua import dataset
from regua.judge import is_correct
from regua.qwen_io import generate_batch, load

OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")
M1_REPORT = os.path.join(os.path.dirname(__file__), "..", "research",
                         "results", "m1_retrieval_report.json")
# congelados no DESIGN.md; o JSON do M1, se existir, é a fonte da verdade
GATE_FALLBACK = {8: 0.939, 32: 0.914, 100: 0.869}


def gate_thresholds():
    if os.path.exists(M1_REPORT):
        with open(M1_REPORT, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        return {r["k"]: r["head_learned"] - 0.05 for r in rows
                if r.get("status") == "ok"}
    return dict(GATE_FALLBACK)


def _prompt_T(fato, pergunta):
    return ("Considere o fato abaixo.\n- %s\n\nPergunta: %s\n"
            "Responda só com a informação pedida." % (fato, pergunta))


def _acc_por_terco(items, corretos):
    k = len(items[0]["fatos"])
    buckets = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for it, c in zip(items, corretos):
        t = min(2, it["target_idx"] * 3 // k)
        buckets[t][1] += 1
        buckets[t][0] += int(c)
    return {str(t): (b[0] / float(b[1]) if b[1] else None)
            for t, b in buckets.items()}


def load_projector(arm, device, emb_weight, m_soft=8):
    torch.manual_seed(7 if arm == "SR" else 0)
    proj = Projector(m=m_soft).to(device)
    if arm == "SR":
        proj.calibrate(emb_weight)  # arquitetura idêntica, pesos aleatórios
    else:
        path = ckpt_path(aux_recon=(arm == "S2"))
        if not os.path.exists(path):
            raise SystemExit("checkpoint do braço %s ausente: %s "
                             "(rode m2.train_soft antes)" % (arm, path))
        proj.load_state_dict(torch.load(path, map_location=device))
    proj.eval()
    return proj


def run_cell(tok, model, head, arm, k, n, args):
    items = dataset.build_items(k=k, n=n, seed=1000 + k)
    t0 = time.time()

    retr = []
    for i in range(0, len(items), args.chunk):
        retr += retrieve_chunk(tok, model, head, items[i:i + args.chunk],
                               model.device,
                               encode_batch_size=args.encode_batch)
        print("  [%s k=%d] retrieval %d/%d (%.1fs)"
              % (arm, k, len(retr), n, time.time() - t0), flush=True)
    retrieval_acc = sum(r["retr_ok"] for r in retr) / float(n)

    t_gen0 = time.time()

    def _prog(done):
        el = time.time() - t_gen0
        print("  [%s k=%d] geração %d/%d | %.2fs/item | ETA %.1f min"
              % (arm, k, done, n, el / done, el / done * (n - done) / 60),
              flush=True)

    if arm == "T":
        users = [_prompt_T(r["fato"], it["pergunta"])
                 for r, it in zip(retr, items)]
        answers, prompt_toks = generate_batch(
            tok, model, users, batch_size=args.batch,
            max_new_tokens=args.max_new, progress=_prog)
    else:
        proj = load_projector(arm, model.device,
                              model.get_input_embeddings().weight,
                              args.m_soft)
        softs = []
        with torch.no_grad():
            for i in range(0, n, 64):
                st = torch.stack([r["states"] for r in retr[i:i + 64]])
                mk = torch.stack([r["mask"] for r in retr[i:i + 64]])
                softs += list(proj(st.float().to(model.device),
                                   mk.to(model.device)).cpu())
        users = [it["pergunta"] for it in items]
        answers, prompt_toks = generate_with_prefix(
            tok, model, softs, users, batch_size=args.batch,
            max_new_tokens=args.max_new, progress=_prog)

    corretos = [is_correct(a, it["gold"]) for a, it in zip(answers, items)]
    dt = time.time() - t0
    n_ok = sum(1 for r in retr if r["retr_ok"])
    acc_cond = (sum(1 for r, c in zip(retr, corretos) if r["retr_ok"] and c)
                / float(n_ok)) if n_ok else None

    return {
        "marco": "M2", "arm": arm, "k": k, "n": n,
        "acc": sum(corretos) / float(n),
        "retrieval_acc_top1": retrieval_acc,
        "acc_cond_retr_ok": acc_cond,
        "acc_por_terco_posicao": _acc_por_terco(items, corretos),
        "tokens_prompt_medio": prompt_toks / float(n),
        "s_por_item": dt / float(n),
        "tempo_total_s": dt,
        "dataset_stats": dataset.stats(items),
        "config": {"modelo": "Qwen2.5-0.5B-Instruct", "greedy": True,
                   "max_new_tokens": args.max_new, "batch_size": args.batch,
                   "seed_dataset": 1000 + k, "split": "held",
                   "m_soft": args.m_soft if arm != "T" else None,
                   "cabeca_m1": os.path.basename(M1_CKPT),
                   "device": str(model.device)},
        "itens": [{"pergunta": it["pergunta"], "gold": it["gold"],
                   "pos": it["target_idx"], "retr_ok": bool(r["retr_ok"]),
                   "resposta": a, "correto": bool(c)}
                  for it, r, a, c in zip(items, retr, answers, corretos)],
    }


def escreve_relatorio(out_dir):
    cells = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(out_dir, fn)) as f:
                cells.append(json.load(f))
    if not cells:
        return
    gates = gate_thresholds()
    md = ["# Relatório M2 — a voz: fato recuperado vira resposta gerada",
          "",
          "Portão: EM fim-a-fim ≥ retrieval M1 − 0.05, mesmos itens do M0",
          "(seed 1000+k, held). Juiz da régua, geração greedy. Braços:",
          "T=texto (controle), S1=soft v1, S2=soft v2 (+recon), SR=ablação.",
          "",
          "| braço | k | n | EM | retr top-1 | EM se retr ok | portão | tok "
          "prompt | s/item | acc pos 1º/2º/3º terço |",
          "|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    by_arm = {}
    for r in cells:
        t = r["acc_por_terco_posicao"]
        terco = "/".join("%.2f" % t[x] if t[x] is not None else "-"
                         for x in ("0", "1", "2"))
        gate = gates.get(r["k"])
        if r["arm"] == "SR" or gate is None:
            portao = "—"
        else:
            portao = "✅ ≥%.3f" % gate if r["acc"] >= gate else "❌ <%.3f" % gate
        by_arm.setdefault(r["arm"], {})[r["k"]] = r
        md.append("| %s | %d | %d | %.3f | %.3f | %s | %s | %.0f | %.2f | %s |"
                  % (r["arm"], r["k"], r["n"], r["acc"],
                     r["retrieval_acc_top1"],
                     ("%.3f" % r["acc_cond_retr_ok"])
                     if r["acc_cond_retr_ok"] is not None else "—",
                     portao, r["tokens_prompt_medio"], r["s_por_item"], terco))

    md += ["", "## Portão M2", ""]
    ks_needed = sorted(gates)
    veredito = []
    for arm in ("S1", "S2", "T"):
        rs = by_arm.get(arm, {})
        if not all(kk in rs for kk in ks_needed):
            veredito.append("- %s: incompleto (faltam k)." % arm)
            continue
        falhas = [kk for kk in ks_needed if rs[kk]["acc"] < gates[kk]]
        if falhas:
            veredito.append("- %s: ❌ falhou em k=%s."
                            % (arm, ",".join(map(str, falhas))))
        else:
            veredito.append("- %s: ✅ passou o portão em todos os k." % arm)
    sr = by_arm.get("SR", {})
    if sr:
        s_ref = by_arm.get("S2") or by_arm.get("S1") or {}
        caiu = all(sr[kk]["acc"] < 0.5 * s_ref[kk]["acc"]
                   for kk in sr if kk in s_ref) if s_ref else None
        veredito.append("- SR (ablação): EM %s — %s."
                        % (", ".join("k=%d:%.3f" % (kk, sr[kk]["acc"])
                                     for kk in sorted(sr)),
                           "desabou (✅ CARTA §3.1)" if caiu
                           else "NÃO desabou (❌ investigar)" if caiu is not None
                           else "sem braço S para comparar"))
    md += veredito
    md += ["",
           "S6 (fluência): amostras de respostas auditáveis no campo `itens`",
           "de cada JSON. Baseline A (Θ sozinho, CARTA §3.2): 0.015 (M0).",
           "Caminho 100% aprendido (CARTA §3.3): sem parser/regex/banco — ver",
           "m2/pipeline.py.", ""]
    path = os.path.join(out_dir, "RELATORIO_M2.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("Relatório: %s" % path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k-list", default="8,32,100")
    p.add_argument("--arms", default="T,S1,S2,SR")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--m-soft", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n, args.k_list = 8, "8"
        if args.arms == "T,S1,S2,SR":
            args.arms = "T,SR"

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Carregando Θ (%s, %s)..." % (args.device, args.dtype), flush=True)
    tok, model = load(args.device, args.dtype)
    head = load_m1_head(model.device)

    for arm in args.arms.split(","):
        for k in [int(x) for x in args.k_list.split(",")]:
            tag = "M2_%s_k%d_n%d" % (arm, k, args.n)
            path = os.path.join(OUT_DIR, tag + ".json")
            if os.path.exists(path) and not args.force:
                print("célula %s pronta, pulando" % tag, flush=True)
                continue
            print("== célula %s ==" % tag, flush=True)
            r = run_cell(tok, model, head, arm, k, args.n, args)
            with open(path, "w") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
            print("célula %s: EM=%.3f retr=%.3f (%.2fs/item)"
                  % (tag, r["acc"], r["retrieval_acc_top1"],
                     r["s_por_item"]), flush=True)

    escreve_relatorio(OUT_DIR)


if __name__ == "__main__":
    main()
