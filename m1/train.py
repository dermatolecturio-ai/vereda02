# -*- coding: utf-8 -*-
"""M1 — treina a cabeça de memória (InfoNCE, Qwen congelado).

Uso (da raiz do repo):
  python3 -m m1.train --smoke          # valida o pipeline, poucos passos
  python3 -m m1.train                  # treino oficial

Reps do Qwen são CACHEADAS 1x (`dados/cache_m1_qwen.pt`) — depois disso o
treino roda só sobre tensores em memória (rápido, medido, sem recompute).
"""
import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from m1.head import Head, encode_batch
from m1.pairs import build

CACHE = os.path.join(os.path.dirname(__file__), "..", "dados",
                     "cache_m1_qwen.pt")
CKPT = os.path.join(os.path.dirname(__file__), "..", "modelos",
                    "vereda2_m1_head.pt")


def build_cache(n_train, n_held, force=False, device="cpu", batch_size=32):
    if os.path.exists(CACHE) and not force:
        d = torch.load(CACHE)
        if d.get("n_train") == n_train and d.get("n_held") == n_held:
            print("cache encontrado (%d train + %d held pares)"
                  % (n_train, n_held), flush=True)
            return d
        print("cache existe mas com n diferente, recacheando", flush=True)

    print("cacheando reps do Qwen (1x, device=%s)..." % device, flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to(device).eval()
    tr, he = build(n_train, n_held)
    t0 = time.time()
    trf, trfm = encode_batch(model, tok, [f for f, _, _ in tr],
                             device=device, batch_size=batch_size)
    print("  fatos train: %.1fs" % (time.time() - t0), flush=True)
    trq, trqm = encode_batch(model, tok, [q for _, q, _ in tr],
                             device=device, batch_size=batch_size)
    hef, hefm = encode_batch(model, tok, [f for f, _, _ in he],
                             device=device, batch_size=batch_size)
    heq, heqm = encode_batch(model, tok, [q for _, q, _ in he],
                             device=device, batch_size=batch_size)
    data = dict(trf=trf.half(), trfm=trfm, trq=trq.half(), trqm=trqm,
               hef=hef.half(), hefm=hefm, heq=heq.half(), heqm=heqm,
               n_train=n_train, n_held=n_held,
               tr_valores=[v for _, _, v in tr], he_valores=[v for _, _, v in he])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(data, CACHE)
    print("cache pronto em %.0fs" % (time.time() - t0), flush=True)
    return data


def retrieval_acc(keyf, keyq, rng, k, n=1024):
    N = keyf.size(0)
    ok = 0
    for _ in range(n):
        idx = rng.sample(range(N), k)
        qi = rng.randrange(k)  # S4: posição alvo uniforme
        sims = keyf[idx] @ keyq[idx[qi]]
        ok += int(sims.argmax().item() == qi)
    return ok / float(n)


def raw_keys(states, mask):
    m = mask.unsqueeze(-1).float()
    return F.normalize(
        (states.float() * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-held", type=int, default=1200)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--device", default="cpu",
                   help="device para o Qwen no CACHE (cpu/cuda/mps); "
                        "a cabeça treina sempre no mesmo device")
    p.add_argument("--encode-batch", type=int, default=32)
    args = p.parse_args()
    if args.smoke:
        args.n_train, args.n_held, args.steps = 300, 100, 60
        args.eval_every = 20

    d = build_cache(args.n_train, args.n_held, force=args.force_cache,
                    device=args.device, batch_size=args.encode_batch)
    dev = args.device
    trf, trfm = d["trf"].float().to(dev), d["trfm"].to(dev)
    trq, trqm = d["trq"].float().to(dev), d["trqm"].to(dev)
    hef, hefm = d["hef"].float().to(dev), d["hefm"].to(dev)
    heq, heqm = d["heq"].float().to(dev), d["heqm"].to(dev)

    erng = random.Random(99)
    rkf, rkq = raw_keys(hef, hefm), raw_keys(heq, heqm)
    print("\nQwen CRU (mean-pool, sem treino) — held (n=1024 amostragens):")
    for k in (8, 32):
        print("  k=%d: %.3f" % (k, retrieval_acc(rkf, rkq, erng, k)))

    torch.manual_seed(0)
    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    N = trf.size(0)
    rng = random.Random(0)
    print("\ntreino da cabeça (InfoNCE, Qwen congelado):", flush=True)
    t0 = time.time()
    best_a8 = -1.0
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.sample(range(N), min(args.batch, N)), device=dev)
        kf = head(trf[idx], trfm[idx])
        kq = head(trq[idx], trqm[idx])
        logits = kq @ kf.T / 0.07
        loss = F.cross_entropy(logits, torch.arange(len(idx), device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                hkf = head(hef, hefm)
                hkq = head(heq, heqm)
                a8 = retrieval_acc(hkf, hkq, erng, 8, n=512)
            el = time.time() - t0
            print("  step %d/%d loss=%.3f held k=8=%.3f | %.3fs/step | ETA %.1fmin"
                  % (step, args.steps, loss.item(), a8, el / step,
                     el / step * (args.steps - step) / 60), flush=True)
            if a8 > best_a8:
                best_a8 = a8
                os.makedirs(os.path.dirname(CKPT), exist_ok=True)
                cpu_state = {k: v.cpu() for k, v in head.state_dict().items()}
                torch.save(cpu_state, CKPT)

    with torch.no_grad():
        hkf = head(hef, hefm)
        hkq = head(heq, heqm)
        torch.manual_seed(7)
        rndhead = Head().to(dev)
        akf = rndhead(hef, hefm)
        akq = rndhead(heq, heqm)
    print("\n=== RESULTADO (held: nomes/valores NOVOS, vocab ABERTO, n=1024) ===")
    print("%4s%11s%18s%17s" % ("k", "Qwen cru", "cabeça aprendida", "ablação(aleat.)"))
    for k in (8, 32, 100):
        print("%4d%11.3f%18.3f%17.3f" % (
            k, retrieval_acc(rkf, rkq, erng, k),
            retrieval_acc(hkf, hkq, erng, k),
            retrieval_acc(akf, akq, erng, k)))
    print("\ncheckpoint salvo (melhor held k=8): %s" % CKPT)


if __name__ == "__main__":
    main()
