# -*- coding: utf-8 -*-
"""M3 — run oficial: portão de extração (held) + portão fim-a-fim (k=8).

Uso:
  python3 -m m3.run --smoke                                    # mecânica
  python3 -m m3.run --device cuda --dtype float16 --batch 32   # oficial

Um JSON por célula em m3/resultados/ (retomável). Ao final,
RELATORIO_M3.md com os dois portões do ROADMAP avaliados.
"""
import argparse
import json
import os
import time

import torch

from m1.head import Head
from m2.pipeline import M1_CKPT, retrieve_chunk
from m2.run import _prompt_T
from m3.dataset import build_items, build_k_items
from m3.encode import encode_with_offsets
from m3.extractor import Extractor
from m3.pipeline import M3_CKPT, extract_batch, fato_canonico
from regua.judge import is_correct, normalize
from regua.qwen_io import generate_batch, load

OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")
GATE_EXTRACAO = 0.90
GATE_E2E = 0.85
TETO_V1_BYTE_LEVEL = 0.59


def load_extractor_arm(arm, device):
    torch.manual_seed(7 if arm == "SR" else 0)
    model = Extractor().to(device)
    if arm != "SR":
        model.load_state_dict(torch.load(M3_CKPT, map_location=device))
    model.eval()
    return model


def load_m1_head(device):
    head = Head().to(device)
    head.load_state_dict(torch.load(M1_CKPT, map_location=device))
    head.eval()
    return head


def _encoded_cached(qwen_model, tok, texts, args, enc_cache, key):
    """Codificação Qwen 1x por conjunto de textos, reaproveitada entre braços
    (a codificação não depende do braço; só o extrator muda)."""
    if key not in enc_cache:
        st, mk, off, ids = encode_with_offsets(
            qwen_model, tok, texts, device=qwen_model.device,
            batch_size=args.encode_batch, return_ids=True)
        enc_cache[key] = (st.half(), mk, off, ids)
    else:
        print("  codificação reaproveitada do braço anterior", flush=True)
    return enc_cache[key]


def run_extraction_cell(qwen_model, tok, arm, n, args, enc_cache):
    items = build_items(n, seed=3000, split="held", marker_dropout_p=0.0)
    extractor = load_extractor_arm(arm, qwen_model.device)
    t0 = time.time()
    encoded = _encoded_cached(qwen_model, tok,
                              [it["frase"] for it in items], args,
                              enc_cache, ("extracao", n))
    extracted = extract_batch(qwen_model, tok, extractor,
                              [it["frase"] for it in items],
                              qwen_model.device,
                              encode_batch_size=args.encode_batch,
                              encoded=encoded)
    dt = time.time() - t0

    por_ordem = {"entidade_primeiro": [0, 0], "valor_primeiro": [0, 0]}
    itens_json = []
    corretos = 0
    for it, ex in zip(items, extracted):
        # juiz ESTRITO: igualdade normalizada, não containment — o portão do
        # ROADMAP é EM; um span sujo ("de João" p/ gold "João") NÃO conta
        ok_ent = normalize(ex["entidade"]) == normalize(it["nome"])
        ok_val = normalize(ex["valor"]) == normalize(it["valor"])
        ok = ok_ent and ok_val
        corretos += int(ok)
        b = por_ordem[it["order"]]
        b[1] += 1
        b[0] += int(ok)
        itens_json.append({
            "frase": it["frase"], "ordem": it["order"],
            "gold_entidade": it["nome"], "gold_valor": it["valor"],
            "extraido_entidade": ex["entidade"], "extraido_valor": ex["valor"],
            "correto": bool(ok),
        })
    return {
        "marco": "M3", "fase": "extracao", "arm": arm, "n": n,
        "em": corretos / float(n),
        "em_por_ordem": {k: (v[0] / float(v[1]) if v[1] else None)
                        for k, v in por_ordem.items()},
        "s_por_item": dt / float(n), "tempo_total_s": dt,
        "itens": itens_json,
    }


def run_e2e_cell(qwen_model, tok, arm, k, n, args, enc_cache):
    items = build_k_items(k, n, seed=4000, split="held", marker_dropout_p=0.0)
    extractor = load_extractor_arm(arm, qwen_model.device)
    head = load_m1_head(qwen_model.device)
    t0 = time.time()

    all_frases = [f for it in items for f in it["frases"]]
    encoded = _encoded_cached(qwen_model, tok, all_frases, args, enc_cache,
                              ("e2e", k, n))
    extracted = extract_batch(qwen_model, tok, extractor, all_frases,
                              qwen_model.device,
                              encode_batch_size=args.encode_batch,
                              encoded=encoded)
    tmp_items, p = [], 0
    for it in items:
        fatos = []
        for j in range(k):
            ex = extracted[p]
            p += 1
            fatos.append(fato_canonico(it["attrs"][j], ex["entidade"],
                                       ex["valor"]))
        tmp_items.append({"fatos": fatos, "target_idx": it["target_idx"],
                          "pergunta": it["pergunta"]})

    retr = []
    for i in range(0, len(tmp_items), args.chunk):
        retr += retrieve_chunk(tok, qwen_model, head,
                               tmp_items[i:i + args.chunk],
                               qwen_model.device,
                               encode_batch_size=args.encode_batch)
    retrieval_acc = sum(r["retr_ok"] for r in retr) / float(n)

    users = [_prompt_T(r["fato"], it["pergunta"])
            for r, it in zip(retr, items)]
    answers, prompt_toks = generate_batch(tok, qwen_model, users,
                                          batch_size=args.batch,
                                          max_new_tokens=args.max_new)
    corretos = [is_correct(a, it["gold"]) for a, it in zip(answers, items)]
    dt = time.time() - t0
    n_ok = sum(1 for r in retr if r["retr_ok"])
    acc_cond = (sum(1 for r, c in zip(retr, corretos) if r["retr_ok"] and c)
               / float(n_ok)) if n_ok else None

    return {
        "marco": "M3", "fase": "e2e", "arm": arm, "k": k, "n": n,
        "em": sum(corretos) / float(n),
        "retrieval_acc_top1": retrieval_acc,
        "em_cond_retr_ok": acc_cond,
        "tokens_prompt_medio": prompt_toks / float(n),
        "s_por_item": dt / float(n), "tempo_total_s": dt,
        "itens": [{"pergunta": it["pergunta"], "gold": it["gold"],
                   "retr_ok": bool(r["retr_ok"]), "fato_usado": r["fato"],
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
    md = ["# Relatório M3 — extração aprendida de texto cru", "",
          "Portão 1 (extração, ROADMAP): EM >= %.2f held-out, fraseados "
          "nunca vistos no treino, valores variáveis." % GATE_EXTRACAO, "",
          "Portão 2 (fim-a-fim, ROADMAP): EM >= %.2f @ k=8, texto cru -> "
          "extração -> M (chave M1) -> resposta (injetor T)." % GATE_E2E, "",
          "Morte: extração precisa bater CLARAMENTE o teto byte-level do "
          "V1 (~%.2f, variável, ver NEGATIVE_FINDINGS.md/ROADMAP.md)." %
          TETO_V1_BYTE_LEVEL, ""]

    extr = [c for c in cells if c["fase"] == "extracao"]
    e2e = [c for c in cells if c["fase"] == "e2e"]

    if extr:
        md += ["## Extração (held-out)", "",
              "| arm | n | EM | EM entidade-primeiro | EM valor-primeiro | "
              "s/item | portão |",
              "|---|---:|---:|---:|---:|---:|---|"]
        for r in extr:
            o = r["em_por_ordem"]
            gate = ("✅ ≥%.2f" % GATE_EXTRACAO if r["arm"] != "SR"
                   and r["em"] >= GATE_EXTRACAO else
                   "❌ <%.2f" % GATE_EXTRACAO if r["arm"] != "SR" else "—")
            md.append("| %s | %d | %.3f | %s | %s | %.2f | %s |" % (
                r["arm"], r["n"], r["em"],
                "%.3f" % o["entidade_primeiro"]
                if o["entidade_primeiro"] is not None else "—",
                "%.3f" % o["valor_primeiro"]
                if o["valor_primeiro"] is not None else "—",
                r["s_por_item"], gate))
        md.append("")

    if e2e:
        md += ["## Fim-a-fim (k=8, held-out)", "",
              "| arm | k | n | EM | retr top-1 | EM se retr ok | tok "
              "prompt | s/item | portão |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in e2e:
            gate = ("✅ ≥%.2f" % GATE_E2E if r["arm"] != "SR"
                   and r["em"] >= GATE_E2E else
                   "❌ <%.2f" % GATE_E2E if r["arm"] != "SR" else "—")
            md.append("| %s | %d | %d | %.3f | %.3f | %s | %.0f | %.2f | "
                      "%s |" % (
                          r["arm"], r["k"], r["n"], r["em"],
                          r["retrieval_acc_top1"],
                          "%.3f" % r["em_cond_retr_ok"]
                          if r["em_cond_retr_ok"] is not None else "—",
                          r["tokens_prompt_medio"], r["s_por_item"], gate))
        md.append("")

    md += ["## Veredito", ""]
    n_ok = next((r for r in extr if r["arm"] == "N"), None)
    sr = next((r for r in extr if r["arm"] == "SR"), None)
    if n_ok:
        p1 = n_ok["em"] >= GATE_EXTRACAO
        md.append("- Extração: %s (EM=%.3f, teto V1=%.2f)."
                 % ("✅ passou" if p1 else "❌ falhou", n_ok["em"],
                    TETO_V1_BYTE_LEVEL))
    if sr and n_ok:
        colapsou = sr["em"] < 0.5 * n_ok["em"]
        md.append("- Ablação (pesos aleatórios): EM=%.3f — %s."
                 % (sr["em"], "desabou (✅ CARTA §3.1)" if colapsou
                    else "NÃO desabou (❌ investigar)"))
    e2e_n = next((r for r in e2e if r["arm"] == "N"), None)
    if e2e_n:
        p2 = e2e_n["em"] >= GATE_E2E
        md.append("- Fim-a-fim @ k=8: %s (EM=%.3f)."
                 % ("✅ passou" if p2 else "❌ falhou", e2e_n["em"]))
    md.append("")

    path = os.path.join(out_dir, "RELATORIO_M3.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("Relatório: %s" % path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--arms", default="N,SR")
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
        args.n, args.k = 8, 8

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Carregando Θ (%s, %s)..." % (args.device, args.dtype), flush=True)
    tok, model = load(args.device, args.dtype)

    enc_cache = {}
    for arm in args.arms.split(","):
        tag = "M3_extracao_%s_n%d" % (arm, args.n)
        path = os.path.join(OUT_DIR, tag + ".json")
        if os.path.exists(path) and not args.force:
            print("célula %s pronta, pulando" % tag, flush=True)
        else:
            print("== célula %s ==" % tag, flush=True)
            r = run_extraction_cell(model, tok, arm, args.n, args, enc_cache)
            with open(path, "w") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
            print("célula %s: EM=%.3f (%.2fs/item)"
                  % (tag, r["em"], r["s_por_item"]), flush=True)

        tag2 = "M3_e2e_%s_k%d_n%d" % (arm, args.k, args.n)
        path2 = os.path.join(OUT_DIR, tag2 + ".json")
        if os.path.exists(path2) and not args.force:
            print("célula %s pronta, pulando" % tag2, flush=True)
            continue
        print("== célula %s ==" % tag2, flush=True)
        r2 = run_e2e_cell(model, tok, arm, args.k, args.n, args, enc_cache)
        with open(path2, "w") as f:
            json.dump(r2, f, ensure_ascii=False, indent=1)
        print("célula %s: EM=%.3f retr=%.3f (%.2fs/item)"
              % (tag2, r2["em"], r2["retrieval_acc_top1"], r2["s_por_item"]),
              flush=True)

    escreve_relatorio(OUT_DIR)


if __name__ == "__main__":
    main()
