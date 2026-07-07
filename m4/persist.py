# -*- coding: utf-8 -*-
"""M4 — portão 4.D: persistência entre PROCESSOS (2 fases, 2 processos).

  python3 -m m4.persist --fase escrever --device cuda   # processo 1
  python3 -m m4.persist --fase ler --device cuda        # processo 2 (novo)

A fase `escrever` constrói as memórias (fatos + chaves M1), responde no
mesmo processo (referência EM_escrever) e serializa TUDO em
m4/resultados/persistencia.vereda. A fase `ler`, num processo NOVO, só
recarrega o arquivo, refaz a chave da PERGUNTA e responde: se a memória
viva serializada funciona, EM_ler ≥ EM_escrever − 0.02.
"""
import argparse
import json
import os
import time

import torch

from m1.head import Head, encode_batch
from m2.pipeline import M1_CKPT
from m2.run import _prompt_T
from regua import dataset
from regua.judge import is_correct
from regua.qwen_io import generate_batch, load

OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")
VEREDA_FILE = os.path.join(OUT_DIR, "persistencia.vereda")


def load_head(device):
    head = Head().to(device)
    head.load_state_dict(torch.load(M1_CKPT, map_location=device))
    head.eval()
    return head


def keys_de(tok, model, head, textos, encode_bs):
    side = tok.padding_side
    tok.padding_side = "right"
    states, masks = encode_batch(model, tok, textos, device=model.device,
                                 batch_size=encode_bs)
    tok.padding_side = side
    with torch.no_grad():
        return head(states.float().to(model.device),
                    masks.to(model.device)).cpu()


def responder(tok, model, memorias, perguntas, retrs, args):
    users = [_prompt_T(m["fatos"][r], q)
             for m, q, r in zip(memorias, perguntas, retrs)]
    answers, _ = generate_batch(tok, model, users, batch_size=args.batch,
                                max_new_tokens=args.max_new)
    return answers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fase", required=True, choices=["escrever", "ler"])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n = 8

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR,
                            "M4_persist_%s_n%d.json" % (args.fase, args.n))
    if os.path.exists(out_json):
        print("fase %s pronta, pulando" % args.fase, flush=True)
        return

    print("Carregando Θ (%s, %s)..." % (args.device, args.dtype), flush=True)
    tok, model = load(args.device, args.dtype)
    head = load_head(model.device)
    t0 = time.time()

    if args.fase == "escrever":
        items = dataset.build_items(k=args.k, n=args.n, seed=5000)
        memorias = []
        for it in items:
            memorias.append({
                "fatos": it["fatos"],
                "keys": keys_de(tok, model, head, it["fatos"],
                                args.encode_batch),
                "pergunta": it["pergunta"], "gold": it["gold"],
                "target_idx": it["target_idx"],
            })
        qkeys = keys_de(tok, model, head, [m["pergunta"] for m in memorias],
                        args.encode_batch)
        retrs = [int(torch.argmax(m["keys"] @ qk).item())
                 for m, qk in zip(memorias, qkeys)]
        answers = responder(tok, model, memorias,
                            [m["pergunta"] for m in memorias], retrs, args)
        corr = [is_correct(a, m["gold"]) for a, m in zip(answers, memorias)]
        em = sum(corr) / float(args.n)
        torch.save({"memorias": memorias}, VEREDA_FILE)
        print("memória viva serializada: %s (%.1f MB)"
              % (VEREDA_FILE, os.path.getsize(VEREDA_FILE) / 1e6),
              flush=True)
    else:
        if not os.path.exists(VEREDA_FILE):
            raise SystemExit("rode --fase escrever antes (%s ausente)"
                             % VEREDA_FILE)
        memorias = torch.load(VEREDA_FILE)["memorias"]
        assert len(memorias) == args.n, "n difere da fase escrever"
        qkeys = keys_de(tok, model, head, [m["pergunta"] for m in memorias],
                        args.encode_batch)
        retrs = [int(torch.argmax(m["keys"] @ qk).item())
                 for m, qk in zip(memorias, qkeys)]
        answers = responder(tok, model, memorias,
                            [m["pergunta"] for m in memorias], retrs, args)
        corr = [is_correct(a, m["gold"]) for a, m in zip(answers, memorias)]
        em = sum(corr) / float(args.n)

    retr_acc = sum(int(r == m["target_idx"])
                   for r, m in zip(retrs, memorias)) / float(args.n)
    payload = {
        "marco": "M4", "fase": "persist_%s" % args.fase,
        "k": args.k, "n": args.n, "em": em,
        "retrieval_acc_top1": retr_acc,
        "s_por_item": (time.time() - t0) / args.n,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("fase %s: EM=%.3f retr=%.3f" % (args.fase, em, retr_acc),
          flush=True)


if __name__ == "__main__":
    main()
