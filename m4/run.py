# -*- coding: utf-8 -*-
"""M4 — run oficial: override do prior + edição com vizinhos.

Uso:
  python3 -m m4.run --smoke                                    # mecânica
  python3 -m m4.run --device cuda --dtype float16 --batch 32   # oficial

Capacidade (4.A) roda pelo runner do M2: m2.run --arms T --k-list 200.
Persistência (4.D) roda por m4.persist (2 processos).
Um JSON por célula em m4/resultados/ (retomável); RELATORIO_M4.md ao final.
"""
import argparse
import json
import os
import time

import torch

from m1.head import Head
from m2.pipeline import M1_CKPT, retrieve_chunk
from m2.run import _prompt_T
from m4.dataset import build_edit_items, build_override_items
from regua.judge import is_correct
from regua.qwen_io import generate_batch, load

OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")
GATES = {"override_em": 0.90, "eco_prior": 0.05, "edicao_em": 0.90,
         "eco_antigo": 0.05, "integridade": 0.95}


def load_head(device, random_weights=False):
    torch.manual_seed(7 if random_weights else 0)
    head = Head().to(device)
    if not random_weights:
        head.load_state_dict(torch.load(M1_CKPT, map_location=device))
    head.eval()
    return head


def _gen(tok, model, users, args, progress_tag):
    def _prog(done):
        print("  [%s] geração %d/%d" % (progress_tag, done, len(users)),
              flush=True)
    return generate_batch(tok, model, users, batch_size=args.batch,
                          max_new_tokens=args.max_new, progress=_prog)


def cell_prior(tok, model, n, args):
    items = build_override_items(n, seed=6000, k=args.k)
    t0 = time.time()
    answers, _ = _gen(tok, model, [it["pergunta"] for it in items], args,
                      "prior")
    corr = [is_correct(a, it["prior"]) for a, it in zip(answers, items)]
    return {
        "marco": "M4", "fase": "prior", "n": n,
        "acc_prior": sum(corr) / float(n),
        "s_por_item": (time.time() - t0) / n,
        "itens": [{"entidade": it["entidade"], "prior": it["prior"],
                   "resposta": a, "correto_prior": bool(c)}
                  for it, a, c in zip(items, answers, corr)],
    }


def cell_override(tok, model, arm, n, args):
    items = build_override_items(n, seed=6000, k=args.k)
    head = load_head(model.device, random_weights=(arm == "SR"))
    t0 = time.time()
    retr = []
    for i in range(0, n, args.chunk):
        retr += retrieve_chunk(tok, model, head, items[i:i + args.chunk],
                               model.device,
                               encode_batch_size=args.encode_batch)
    users = [_prompt_T(r["fato"], it["pergunta"])
             for r, it in zip(retr, items)]
    answers, _ = _gen(tok, model, users, args, "override-%s" % arm)
    em_c = [is_correct(a, it["gold"]) for a, it in zip(answers, items)]
    eco = [is_correct(a, it["prior"]) for a, it in zip(answers, items)]
    return {
        "marco": "M4", "fase": "override", "arm": arm, "k": args.k, "n": n,
        "em_contrafactual": sum(em_c) / float(n),
        "eco_prior": sum(eco) / float(n),
        "retrieval_acc_top1": sum(r["retr_ok"] for r in retr) / float(n),
        "s_por_item": (time.time() - t0) / n,
        "itens": [{"entidade": it["entidade"], "gold": it["gold"],
                   "prior": it["prior"], "retr_ok": bool(r["retr_ok"]),
                   "resposta": a, "correto": bool(c), "ecoou_prior": bool(e)}
                  for it, r, a, c, e in zip(items, retr, answers, em_c, eco)],
    }


def cell_edicao(tok, model, n, args):
    items = build_edit_items(n, seed=7000, k=args.k_edicao)
    head = load_head(model.device)
    t0 = time.time()
    # memória JÁ editada: alvo trocado pelo fato com valor novo
    for it in items:
        it["fatos"] = list(it["fatos"])
        it["fatos"][it["target_idx"]] = it["fato_editado"]

    def _retrieve(pergunta_key, alvo_key):
        pseudo = [{"fatos": it["fatos"], "pergunta": it[pergunta_key],
                   "target_idx": it[alvo_key]} for it in items]
        out = []
        for i in range(0, n, args.chunk):
            out += retrieve_chunk(tok, model, head, pseudo[i:i + args.chunk],
                                  model.device,
                                  encode_batch_size=args.encode_batch)
        return out

    retr_t = _retrieve("pergunta", "target_idx")
    retr_v = _retrieve("pergunta_viz", "viz_idx")

    ans_t, _ = _gen(tok, model, [_prompt_T(r["fato"], it["pergunta"])
                                 for r, it in zip(retr_t, items)],
                    args, "edicao-alvo")
    ans_v, _ = _gen(tok, model, [_prompt_T(r["fato"], it["pergunta_viz"])
                                 for r, it in zip(retr_v, items)],
                    args, "edicao-vizinho")

    em_novo = [is_correct(a, it["valor_novo"]) for a, it in zip(ans_t, items)]
    eco_ant = [is_correct(a, it["valor_antigo"])
               for a, it in zip(ans_t, items)]
    integ = [is_correct(a, it["gold_viz"]) for a, it in zip(ans_v, items)]
    return {
        "marco": "M4", "fase": "edicao", "k": args.k_edicao, "n": n,
        "em_valor_novo": sum(em_novo) / float(n),
        "eco_valor_antigo": sum(eco_ant) / float(n),
        "integridade_vizinhos": sum(integ) / float(n),
        "retr_alvo": sum(r["retr_ok"] for r in retr_t) / float(n),
        "retr_viz": sum(r["retr_ok"] for r in retr_v) / float(n),
        "s_por_item": (time.time() - t0) / n,
        "itens": [{"pergunta": it["pergunta"],
                   "valor_antigo": it["valor_antigo"],
                   "valor_novo": it["valor_novo"], "resposta": a,
                   "correto": bool(c), "ecoou_antigo": bool(e),
                   "viz_ok": bool(v)}
                  for it, a, c, e, v in zip(items, ans_t, em_novo, eco_ant,
                                            integ)],
    }


def escreve_relatorio(out_dir):
    cells = {}
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(out_dir, fn)) as f:
                r = json.load(f)
            cells[(r["fase"], r.get("arm", "N"))] = r
    if not cells:
        return
    g = GATES
    md = ["# Relatório M4 — memória viva: override, edição, persistência",
          "",
          "Portões (DESIGN.md): override EM ≥%.2f e eco do prior ≤%.2f; "
          "edição EM ≥%.2f, eco antigo ≤%.2f, vizinhos ≥%.2f; persistência "
          "EM_ler ≥ EM_escrever − 0.02; capacidade k=200 no RELATORIO_M2 "
          "(células T k=200)." % (g["override_em"], g["eco_prior"],
                                  g["edicao_em"], g["eco_antigo"],
                                  g["integridade"]), ""]

    pr = cells.get(("prior", "N"))
    ov = cells.get(("override", "N"))
    sr = cells.get(("override", "SR"))
    ed = cells.get(("edicao", "N"))
    pe = cells.get(("persist_escrever", "N"))
    pl = cells.get(("persist_ler", "N"))

    if pr:
        md.append("- Prior baseline (Θ cru): %.3f das capitais respondidas "
                  "certas sem memória (n=%d)." % (pr["acc_prior"], pr["n"]))
    if ov:
        md.append("- Override: EM contrafactual %.3f (portão %s), eco do "
                  "prior %.3f (%s), retrieval %.3f."
                  % (ov["em_contrafactual"],
                     "✅" if ov["em_contrafactual"] >= g["override_em"]
                     else "❌",
                     ov["eco_prior"],
                     "✅" if ov["eco_prior"] <= g["eco_prior"] else "❌",
                     ov["retrieval_acc_top1"]))
        if pr:
            ids_ok = [i for i, it in enumerate(pr["itens"])
                      if it["correto_prior"]]
            if ids_ok:
                sub = [ov["itens"][i]["correto"] for i in ids_ok]
                md.append("  - No subconjunto DIFÍCIL (prior existia, "
                          "n=%d): EM contrafactual %.3f."
                          % (len(sub), sum(sub) / float(len(sub))))
    if sr and ov:
        caiu = sr["em_contrafactual"] < 0.5 * ov["em_contrafactual"]
        md.append("- Ablação SR (chave aleatória): EM %.3f, retrieval %.3f "
                  "— %s." % (sr["em_contrafactual"],
                             sr["retrieval_acc_top1"],
                             "desabou ✅" if caiu else "NÃO desabou ❌"))
    if ed:
        md.append("- Edição: EM valor novo %.3f (%s), eco do antigo %.3f "
                  "(%s), integridade vizinhos %.3f (%s)."
                  % (ed["em_valor_novo"],
                     "✅" if ed["em_valor_novo"] >= g["edicao_em"] else "❌",
                     ed["eco_valor_antigo"],
                     "✅" if ed["eco_valor_antigo"] <= g["eco_antigo"]
                     else "❌",
                     ed["integridade_vizinhos"],
                     "✅" if ed["integridade_vizinhos"] >= g["integridade"]
                     else "❌"))
    if pe and pl:
        ok = pl["em"] >= pe["em"] - 0.02
        md.append("- Persistência: EM escrever %.3f → EM ler (processo "
                  "novo) %.3f — %s." % (pe["em"], pl["em"],
                                        "✅" if ok else "❌"))
    md += ["", "Esquecer-restaura-o-prior é propriedade arquitetural nesta "
           "fase (Θ congelado + memória externa): remover o fato devolve o "
           "comportamento do prior baseline por construção (DESIGN.md §4.B).",
           ""]
    path = os.path.join(out_dir, "RELATORIO_M4.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("Relatório: %s" % path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--k-edicao", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n, args.k_edicao = 8, 8

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Carregando Θ (%s, %s)..." % (args.device, args.dtype), flush=True)
    tok, model = load(args.device, args.dtype)

    cells = [
        ("M4_prior_n%d" % args.n, lambda: cell_prior(tok, model, args.n,
                                                     args)),
        ("M4_override_N_k%d_n%d" % (args.k, args.n),
         lambda: cell_override(tok, model, "N", args.n, args)),
        ("M4_override_SR_k%d_n%d" % (args.k, args.n),
         lambda: cell_override(tok, model, "SR", args.n, args)),
        ("M4_edicao_k%d_n%d" % (args.k_edicao, args.n),
         lambda: cell_edicao(tok, model, args.n, args)),
    ]
    for tag, fn in cells:
        path = os.path.join(OUT_DIR, tag + ".json")
        if os.path.exists(path) and not args.force:
            print("célula %s pronta, pulando" % tag, flush=True)
            continue
        print("== célula %s ==" % tag, flush=True)
        r = fn()
        with open(path, "w") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        chave = [k for k in ("acc_prior", "em_contrafactual",
                             "em_valor_novo") if k in r]
        print("célula %s: %s=%.3f (%.2fs/item)"
              % (tag, chave[0], r[chave[0]], r["s_por_item"]), flush=True)

    escreve_relatorio(OUT_DIR)


if __name__ == "__main__":
    main()
