# -*- coding: utf-8 -*-
"""M3 — treina o extrator (refinador bidirecional + 4 ponteiros), Qwen CONGELADO.

Uso:
  python3 -m m3.train_extract --smoke                 # valida mecânica
  python3 -m m3.train_extract --device cuda            # oficial

Termômetro (S5: não é manchete): EM de extração held-out (entidade E valor
exatos) a cada eval_every passos. Checkpoint = melhor termômetro (P6/lição
#12 do V1: early-stop é obrigatório, held-out despenca por overfitting se
deixar passar do pico).
"""
import argparse
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3.dataset import build_items
from m3.encode import build_targets, encode_with_offsets
from m3.extractor import Extractor, decode_spans, extraction_losses
from regua.qwen_io import MODEL_ID

CACHE = os.path.join(os.path.dirname(__file__), "..", "dados",
                     "cache_m3_qwen.pt")
CKPT = os.path.join(os.path.dirname(__file__), "..", "modelos",
                    "vereda2_m3_extractor.pt")


def build_cache(n_train, n_held, marker_dropout_p, force=False,
                device="cpu", batch_size=32):
    if os.path.exists(CACHE) and not force:
        d = torch.load(CACHE)
        if (d.get("n_train") == n_train and d.get("n_held") == n_held
                and d.get("marker_dropout_p") == marker_dropout_p):
            print("cache M3 encontrado (%d train + %d held)"
                  % (n_train, n_held), flush=True)
            return d
        print("cache M3 existe mas config difere, recacheando", flush=True)

    print("cacheando reps do Qwen p/ M3 (1x, device=%s)..." % device,
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32).to(device).eval()

    tr = build_items(n_train, seed=1000, split="train",
                     marker_dropout_p=marker_dropout_p)
    he = build_items(n_held, seed=2000, split="held", marker_dropout_p=0.0)

    t0 = time.time()
    tr_states, tr_masks, tr_offs = encode_with_offsets(
        model, tok, [it["frase"] for it in tr], device=device,
        batch_size=batch_size)
    he_states, he_masks, he_offs = encode_with_offsets(
        model, tok, [it["frase"] for it in he], device=device,
        batch_size=batch_size)
    tse, tee, tsv, tev, ok = build_targets(tr, tr_offs)
    hse, hee, hsv, hev, hok = build_targets(he, he_offs)
    print("  alinhamento train ok: %d/%d | held ok: %d/%d"
          % (ok.sum(), len(tr), hok.sum(), len(he)), flush=True)

    data = dict(
        tr_states=tr_states.half(), tr_masks=tr_masks,
        tse=tse, tee=tee, tsv=tsv, tev=tev, tr_ok=ok,
        he_states=he_states.half(), he_masks=he_masks,
        hse=hse, hee=hee, hsv=hsv, hev=hev, he_ok=hok,
        he_items=he,
        n_train=n_train, n_held=n_held, marker_dropout_p=marker_dropout_p)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(data, CACHE)
    print("cache M3 pronto em %.0fs" % (time.time() - t0), flush=True)
    return data


@torch.no_grad()
def extraction_em(model, states, masks, targets, ok_mask, device,
                  batch_size=256):
    tse, tee, tsv, tev = targets
    total, corretos = 0, 0
    for i in range(0, states.size(0), batch_size):
        s = states[i:i + batch_size].float().to(device)
        m = masks[i:i + batch_size].to(device)
        vok = ok_mask[i:i + batch_size]
        logits = model(s, m)
        pe_s, pe_e, pv_s, pv_e = decode_spans(logits, m)
        pe_s, pe_e = pe_s.cpu(), pe_e.cpu()
        pv_s, pv_e = pv_s.cpu(), pv_e.cpu()
        chunk = tse[i:i + batch_size].size(0)
        for j in range(chunk):
            if not vok[j]:
                continue
            total += 1
            ok_e = (pe_s[j] == tse[i + j] and pe_e[j] == tee[i + j])
            ok_v = (pv_s[j] == tsv[i + j] and pv_e[j] == tev[i + j])
            corretos += int(ok_e and ok_v)
    return corretos / float(total) if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-held", type=int, default=1200)
    p.add_argument("--marker-dropout-p", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam-excl", type=float, default=0.3)
    p.add_argument("--lam-cov", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--device", default="cpu")
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--force-retrain", action="store_true",
                   help="retreina mesmo se modelos/vereda2_m3_extractor.pt já existir")
    args = p.parse_args()
    if args.smoke:
        args.n_train, args.n_held = 300, 100
        args.steps, args.batch, args.eval_every = 40, 16, 10
    elif os.path.exists(CKPT) and not args.force_retrain:
        print("checkpoint já existe (%s), pulando treino (use "
              "--force-retrain para refazer)" % CKPT, flush=True)
        return

    dev = args.device
    d = build_cache(args.n_train, args.n_held, args.marker_dropout_p,
                    force=args.force_cache, device=dev,
                    batch_size=args.encode_batch)
    tr_states, tr_masks = d["tr_states"], d["tr_masks"]
    tr_targets = (d["tse"], d["tee"], d["tsv"], d["tev"])
    tr_ok = d["tr_ok"]
    he_states, he_masks = d["he_states"], d["he_masks"]
    he_targets = (d["hse"], d["hee"], d["hsv"], d["hev"])
    he_ok = d["he_ok"]

    idx_validos = tr_ok.nonzero(as_tuple=True)[0].tolist()
    print("treino: %d itens validos (de %d)" % (len(idx_validos),
                                                tr_states.size(0)),
          flush=True)

    torch.manual_seed(0)
    model = Extractor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(x.numel() for x in model.parameters())
    print("extrator: %d params | steps=%d batch=%d" % (
        n_params, args.steps, args.batch), flush=True)

    rng = random.Random(0)
    best_em, t0 = -1.0, time.time()
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.sample(idx_validos,
                                      min(args.batch, len(idx_validos))))
        s = tr_states[idx].float().to(dev)
        m = tr_masks[idx].to(dev)
        targets = tuple(t[idx].to(dev) for t in tr_targets)
        logits = model(s, m)
        loss, parts = extraction_losses(logits, targets, m,
                                        args.lam_excl, args.lam_cov)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            em = extraction_em(model, he_states, he_masks, he_targets,
                               he_ok, dev)
            el = time.time() - t0
            print("  step %d/%d loss=%.3f (ce=%.3f excl=%.3f cov=%.3f) "
                 "| EM held=%.3f | %.2fs/step | ETA %.1fmin"
                 % (step, args.steps, loss.item(), parts["ce"],
                    parts["excl"], parts["cov"], em, el / step,
                    el / step * (args.steps - step) / 60), flush=True)
            if em > best_em:
                best_em = em
                os.makedirs(os.path.dirname(CKPT), exist_ok=True)
                torch.save({k: v.cpu() for k, v in
                           model.state_dict().items()}, CKPT)

    print("\nmelhor EM de extração (held, n=%d): %.3f"
          % (he_states.size(0), best_em))
    print("checkpoint: %s" % CKPT)


if __name__ == "__main__":
    main()
