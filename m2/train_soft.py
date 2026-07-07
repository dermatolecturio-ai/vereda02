# -*- coding: utf-8 -*-
"""M2 — treina o projetor do injetor S (Qwen CONGELADO, só o projetor aprende).

Braços (DESIGN.md, A/B P2/P5):
  S-v1: perda = CE na resposta-ouro (teacher forcing)      → _v1.pt
  S-v2: + CE de reconstrução do fato (--aux-recon, λ=1)    → _v2.pt

Uso:
  python3 -m m2.train_soft --smoke                 # valida mecânica (CPU ok)
  python3 -m m2.train_soft --device cuda           # S-v1 oficial
  python3 -m m2.train_soft --device cuda --aux-recon   # S-v2 oficial

Termômetro durante o treino (S5: não é manchete): EM com o fato-OURO injetado
(sem retrieval) em fatia held — mede só a VOZ, isola o endereçamento.
"""
import argparse
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from m1.pairs import build
from m1.train import build_cache
from m2.pipeline import generate_with_prefix
from m2.projector import Projector
from regua.judge import is_correct
from regua.qwen_io import MODEL_ID, chat_prompt

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "modelos")


def ckpt_path(aux_recon):
    return os.path.join(CKPT_DIR,
                        "vereda2_m2_projector_%s.pt"
                        % ("v2" if aux_recon else "v1"))


def lm_loss(model, soft, id_lists, label_starts, device):
    """CE nos tokens a partir de label_starts; soft tokens e prompt = -100."""
    emb = model.get_input_embeddings()
    msoft = soft.size(1)
    rows, labs = [], []
    for s, ids, ls in zip(soft, id_lists, label_starts):
        e = emb(torch.tensor(ids, device=device))
        rows.append(torch.cat([s, e.float()], 0))
        labs.append([-100] * (msoft + ls) + ids[ls:])
    L = max(r.size(0) for r in rows)
    pad_vec = emb(torch.tensor([model.config.eos_token_id], device=device))[0]
    batch = pad_vec.float().repeat(len(rows), L, 1)
    mask = torch.zeros(len(rows), L, dtype=torch.long, device=device)
    labels = torch.full((len(rows), L), -100, dtype=torch.long, device=device)
    for j, (r, lb) in enumerate(zip(rows, labs)):
        batch[j, :r.size(0)] = r
        mask[j, :r.size(0)] = 1
        labels[j, :len(lb)] = torch.tensor(lb, device=device)
    return model(inputs_embeds=batch.to(emb.weight.dtype),
                 attention_mask=mask, labels=labels).loss


@torch.no_grad()
def thermometer(tok, model, projector, hef, hefm, he, n_eval, device):
    """EM com fato-OURO injetado (voz isolada), fatia held."""
    idx = list(range(min(n_eval, len(he))))
    soft = projector(hef[idx].float().to(device), hefm[idx].to(device))
    users = [he[i][1] for i in idx]
    answers, _ = generate_with_prefix(tok, model, soft, users,
                                      batch_size=8, max_new_tokens=24)
    ok = sum(is_correct(a, he[i][2]) for a, i in zip(answers, idx))
    return ok / float(len(idx)), answers[:5], [he[i][2] for i in idx[:5]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-held", type=int, default=1200)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--m-soft", type=int, default=8)
    p.add_argument("--aux-recon", action="store_true")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-n", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--force-retrain", action="store_true",
                   help="retreina mesmo se o checkpoint do braço já existir")
    args = p.parse_args()
    if args.smoke:
        args.n_train, args.n_held = 300, 100
        args.steps, args.batch, args.eval_every, args.eval_n = 8, 4, 4, 16
    elif os.path.exists(ckpt_path(args.aux_recon)) and not args.force_retrain:
        print("checkpoint já existe (%s), pulando treino (use "
              "--force-retrain para refazer)" % ckpt_path(args.aux_recon),
              flush=True)
        return

    dev = args.device
    d = build_cache(args.n_train, args.n_held, force=args.force_cache,
                    device=dev, batch_size=args.encode_batch)
    tr, he = build(args.n_train, args.n_held)  # mesmos seeds do cache
    trf, trfm = d["trf"], d["trfm"]
    hef, hefm = d["hef"], d["hefm"]

    print("Carregando Θ para treino (%s, float32, CONGELADO)..." % dev,
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32).to(dev).eval()
    model.requires_grad_(False)

    # pré-tokeniza uma vez: [prompt da pergunta + valor-ouro] e [fato] p/ recon
    eos = [tok.eos_token_id]
    seq_ans, start_ans, seq_fato = [], [], []
    for fato, pergunta, valor in tr:
        pids = tok(chat_prompt(tok, pergunta)).input_ids
        aids = tok(valor, add_special_tokens=False).input_ids + eos
        seq_ans.append(pids + aids)
        start_ans.append(len(pids))
        seq_fato.append(tok(fato, add_special_tokens=False).input_ids + eos)

    torch.manual_seed(0)
    projector = Projector(m=args.m_soft).to(dev)
    projector.calibrate(model.get_input_embeddings().weight)
    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    n_params = sum(p_.numel() for p_ in projector.parameters())
    print("projetor: %d params | braço %s | steps=%d batch=%d"
          % (n_params, "S-v2 (+recon)" if args.aux_recon else "S-v1",
             args.steps, args.batch), flush=True)

    rng = random.Random(0)
    N = trf.size(0)
    best_em, path = -1.0, ckpt_path(args.aux_recon)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.sample(range(N), min(args.batch, N))
        soft = projector(trf[idx].float().to(dev), trfm[idx].to(dev))
        loss = lm_loss(model, soft, [seq_ans[i] for i in idx],
                       [start_ans[i] for i in idx], dev)
        if args.aux_recon:
            loss = loss + args.lam * lm_loss(
                model, soft, [seq_fato[i] for i in idx],
                [0] * len(idx), dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            projector.eval()
            em, amostras, golds = thermometer(
                tok, model, projector, hef, hefm, he, args.eval_n, dev)
            projector.train()
            el = time.time() - t0
            print("  step %d/%d loss=%.3f | termômetro EM(fato-ouro)=%.3f "
                  "| %.2fs/step | ETA %.1fmin"
                  % (step, args.steps, loss.item(), em, el / step,
                     el / step * (args.steps - step) / 60), flush=True)
            if step == args.eval_every:
                for a, g in zip(amostras, golds):
                    print("    ex: gold=%r resposta=%r" % (g, a), flush=True)
            if em > best_em:
                best_em = em
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save({k: v.cpu() for k, v in
                            projector.state_dict().items()}, path)

    print("\nmelhor termômetro EM(fato-ouro held, n=%d): %.3f"
          % (args.eval_n, best_em))
    print("checkpoint: %s" % path)


if __name__ == "__main__":
    main()
