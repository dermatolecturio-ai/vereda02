# -*- coding: utf-8 -*-
"""M0 — mede e CONGELA os baselines da régua (REGUA.md §3).

  A — Θ cru, sem fatos (piso; deve errar o que depende de fato ensinado)
  B — Θ cru com TODOS os k fatos no contexto (baseline FORTE, in-context)
  C — RAG ingênuo (embedding Qwen cru, top-1 no contexto) — referência externa

Uso (da raiz do repo):
  python3 -m regua.run --smoke            # validação + medição de tempo
  python3 -m regua.run --n 1024           # run oficial (congela baselines)

Um JSON por célula em regua/resultados/ (retomável: célula pronta é pulada).
Ao final escreve regua/resultados/RELATORIO_M0.md.
"""
import argparse
import json
import os
import time

import torch

from regua import dataset
from regua.judge import is_correct
from regua.qwen_io import load, generate_batch, embed_batch

OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")


def _prompt_A(item):
    return item["pergunta"]


def _prompt_B(item):
    fatos = "\n".join("- " + f for f in item["fatos"])
    return ("Considere os fatos abaixo.\n%s\n\nPergunta: %s\n"
            "Responda só com a informação pedida." % (fatos, item["pergunta"]))


def _prompt_C(item, fato_recuperado):
    return ("Considere o fato abaixo.\n- %s\n\nPergunta: %s\n"
            "Responda só com a informação pedida."
            % (fato_recuperado, item["pergunta"]))


def _acc_por_terco(items, corretos):
    k = len(items[0]["fatos"])
    buckets = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for it, c in zip(items, corretos):
        t = min(2, it["target_idx"] * 3 // k)
        buckets[t][1] += 1
        buckets[t][0] += int(c)
    return {str(t): (b[0] / float(b[1]) if b[1] else None)
            for t, b in buckets.items()}


def run_cell(tok, model, baseline, k, n, batch_size, max_new, emb_cache):
    items = dataset.build_items(k=k, n=n, seed=1000 + k)
    t0 = time.time()
    retrieval_acc = None

    if baseline == "A":
        users = [_prompt_A(it) for it in items]
    elif baseline == "B":
        users = [_prompt_B(it) for it in items]
    elif baseline == "C":
        users = []
        acertos_retr = 0
        for j, it in enumerate(items):
            embs = embed_batch(tok, model, it["fatos"], cache=emb_cache)
            q = embed_batch(tok, model, [it["pergunta"]], cache=emb_cache)[0]
            top1 = int(torch.argmax(embs @ q).item())
            acertos_retr += int(top1 == it["target_idx"])
            users.append(_prompt_C(it, it["fatos"][top1]))
            if (j + 1) % 50 == 0:
                print("  [C k=%d] retrieval %d/%d (%.1fs)"
                      % (k, j + 1, n, time.time() - t0), flush=True)
        retrieval_acc = acertos_retr / float(n)
    else:
        raise ValueError(baseline)

    t_gen0 = time.time()

    def _prog(done):
        el = time.time() - t_gen0
        print("  [%s k=%d] geração %d/%d | %.2fs/item | ETA %.1f min"
              % (baseline, k, done, n, el / done, el / done * (n - done) / 60),
              flush=True)

    answers, prompt_toks = generate_batch(
        tok, model, users, batch_size=batch_size, max_new_tokens=max_new,
        progress=_prog)
    corretos = [is_correct(a, it["gold"]) for a, it in zip(answers, items)]
    dt = time.time() - t0

    return {
        "baseline": baseline, "k": k, "n": n,
        "acc": sum(corretos) / float(n),
        "acc_por_terco_posicao": _acc_por_terco(items, corretos),
        "retrieval_acc_top1": retrieval_acc,
        "tokens_prompt_medio": prompt_toks / float(n),
        "s_por_item": dt / float(n),
        "tempo_total_s": dt,
        "dataset_stats": dataset.stats(items),
        "config": {"modelo": "Qwen2.5-0.5B-Instruct", "greedy": True,
                   "max_new_tokens": max_new, "batch_size": batch_size,
                   "seed_dataset": 1000 + k, "split": "held",
                   "device": str(model.device)},
        "itens": [{"pergunta": it["pergunta"], "gold": it["gold"],
                   "pos": it["target_idx"], "resposta": a, "correto": bool(c)}
                  for it, a, c in zip(items, answers, corretos)],
    }


def escreve_relatorio(out_dir):
    linhas = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(out_dir, fn)) as f:
                linhas.append(json.load(f))
    if not linhas:
        return
    md = ["# Relatório M0 — baselines congelados",
          "",
          "Juiz: valor-ouro normalizado, com fronteira de palavra, na resposta",
          "gerada (greedy, determinística). Dataset: held pools, seed fixa.",
          "",
          "| baseline | k | n | acc | retr top-1 | tok prompt | s/item | acc pos 1º/2º/3º terço |",
          "|---|---|---|---|---|---|---|---|"]
    for r in linhas:
        t = r["acc_por_terco_posicao"]
        terco = "/".join("%.2f" % t[x] if t[x] is not None else "-"
                         for x in ("0", "1", "2"))
        md.append("| %s | %d | %d | %.3f | %s | %.0f | %.2f | %s |" % (
            r["baseline"], r["k"], r["n"], r["acc"],
            ("%.3f" % r["retrieval_acc_top1"]) if r["retrieval_acc_top1"]
            is not None else "—",
            r["tokens_prompt_medio"], r["s_por_item"], terco))
    md += ["", "Sentinelas aplicáveis no M0: S3 (valores variáveis — ver",
           "dataset_stats nos JSONs), S4 (posição embaralhada + acc por terço),",
           "S5 (n reportado), S6 (n/a — nada treinado). S1/S2 aplicam-se a",
           "partir do M1 (módulos treinados)."]
    with open(os.path.join(out_dir, "RELATORIO_M0.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("Relatório: %s" % os.path.join(out_dir, "RELATORIO_M0.md"),
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k-list", default="2,8,32,100")
    p.add_argument("--baselines", default="A,B,C")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.n = 8
        args.k_list = "2,8"

    os.makedirs(OUT_DIR, exist_ok=True)
    ks = [int(x) for x in args.k_list.split(",")]
    baselines = args.baselines.split(",")

    print("Carregando Θ (%s, %s)..." % (args.device, args.dtype), flush=True)
    t0 = time.time()
    tok, model = load(args.device, args.dtype)
    print("Θ carregado em %.1fs" % (time.time() - t0), flush=True)

    emb_cache = {}
    for baseline in baselines:
        # A não depende de k: mede uma vez, com as perguntas do set k=8
        cell_ks = [8] if baseline == "A" else ks
        for k in cell_ks:
            tag = "%s_k%d_n%d" % (baseline, k, args.n)
            path = os.path.join(OUT_DIR, tag + ".json")
            if os.path.exists(path) and not args.force:
                print("célula %s pronta, pulando" % tag, flush=True)
                continue
            print("== célula %s ==" % tag, flush=True)
            r = run_cell(tok, model, baseline, k, args.n, args.batch,
                         args.max_new, emb_cache)
            with open(path, "w") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
            print("célula %s: acc=%.3f (%.2fs/item)"
                  % (tag, r["acc"], r["s_por_item"]), flush=True)

    escreve_relatorio(OUT_DIR)


if __name__ == "__main__":
    main()
