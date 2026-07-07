# -*- coding: utf-8 -*-
"""M5 — teste de faca: canal não textual interno vs baselines (DESIGN.md).

Braços novos: K (kNN-LM/interp. de logits, saída) e I (memória delta lida na
camada 12, meio) + ablações IZ (memória zerada), IS (chaves embaralhadas),
IR (módulos aleatórios). T e S1 são REUSADOS das células oficiais do M2
(mesmos itens: seed 1008, held) — não se re-roda o que já é oficial.

Uso (fases resumíveis; cada uma pula o que já existe):
  python3 -m m5.knife_test --smoke                    # mecânica (CPU), números não contam
  python3 -m m5.knife_test --do grid  --device cuda --dtype float16
  python3 -m m5.knife_test --do train --device cuda   # float32
  python3 -m m5.knife_test --do run   --device cuda --dtype float16
  python3 -m m5.knife_test --do edit  --device cuda --dtype float16
  python3 -m m5.knife_test --do report
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from m1.head import encode_batch
from m2.pipeline import load_m1_head
from m5.delta import DeltaReader, Injector, L_INJ, ValueEncoder, write_facts
from regua import dataset
from regua.judge import is_correct, normalize
from regua.qwen_io import chat_prompt, load

OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")
M2_DIR = os.path.join(os.path.dirname(__file__), "..", "m2", "resultados")
CKPT = os.path.join(os.path.dirname(__file__), "..", "modelos",
                    "vereda2_m5_knife.pt")
GRID_PATH = os.path.join(OUT_DIR, "K_grid.json")

K_EVAL = 8                   # k primário do knife (DESIGN.md)
SEED_EVAL = 1000 + K_EVAL    # mesmos itens do M0/M2
GRID_LAM = (0.3, 0.5, 0.7)   # travados no DESIGN.md
GRID_TAU = (0.05, 0.2, 1.0)


# ---------------------------------------------------------------- memórias

def build_memories(tok, model, head, valenc, items, device,
                   encode_batch_size=64, shuffle_keys=False, grad=False):
    """Escreve os k fatos de cada item na memória delta S.

    Devolve S (B,dv,dk) e keys (B,k,dk) na ordem original. shuffle_keys=True
    (ablação IS) roda as chaves em círculo dentro do item: todo valor escrito
    sob a chave de OUTRO fato — se houver binding real, tem de desabar.
    """
    k = len(items[0]["fatos"])
    texts = []
    for it in items:
        texts += it["fatos"]
    side = tok.padding_side
    tok.padding_side = "right"  # consistente com o treino da cabeça M1
    states, masks = encode_batch(model, tok, texts, device=device,
                                 batch_size=encode_batch_size)
    tok.padding_side = side
    states = states.float().to(device)
    masks = masks.to(device)
    with torch.no_grad():
        keys = head(states, masks).view(len(items), k, -1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        values = valenc(states, masks).view(len(items), k, -1)
        if shuffle_keys:
            perm = ((torch.arange(k) + 1) % k).to(keys.device)
            S = write_facts(keys[:, perm, :], values)
        else:
            S = write_facts(keys, values)
    return S, keys


@torch.no_grad()
def enderecamento_implicito(tok, model, reader, items, keys, device,
                            batch_size=32):
    """top-1 de q (último token do prompt, camada L_INJ) contra as chaves.

    Diagnóstico do DESIGN.md: separa gargalo de leitura (q não acha a chave)
    de gargalo de voz (acha, mas a geração não entrega o valor).
    """
    prompts = [chat_prompt(tok, it["pergunta"]) for it in items]
    side = tok.padding_side
    tok.padding_side = "right"
    oks = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt",
                  padding=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states[L_INJ]
        last = enc.attention_mask.sum(1) - 1
        h = hs[torch.arange(hs.size(0)), last].float()
        q = reader.query(h)
        sim = torch.einsum("bkd,bd->bk", keys[i:i + batch_size], q)
        top1 = sim.argmax(-1)
        for j, it in enumerate(items[i:i + batch_size]):
            oks.append(int(top1[j].item()) == it["target_idx"])
    tok.padding_side = side
    return oks


@torch.no_grad()
def generate_with_memory(tok, model, injector, S, users, batch_size=8,
                         max_new_tokens=32, progress=None):
    """Geração gulosa com a leitura delta ativa (hook); S alinhado ao batch."""
    outs, total_toks = [], 0
    for i in range(0, len(users), batch_size):
        prompts = [chat_prompt(tok, u) for u in users[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        total_toks += int(enc.attention_mask.sum().item())
        injector.set_memory(S[i:i + batch_size].to(model.device))
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                 skip_special_tokens=True)
        if progress:
            progress(len(outs))
    injector.set_memory(None)
    return outs, total_toks


# ---------------------------------------------------------------- treino (I)

def lm_loss(tok, model, seqs, starts, device):
    """CE nos tokens a partir de starts (resposta); prompt/pad = -100."""
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    labels = torch.full((len(seqs), L), -100, dtype=torch.long, device=device)
    for j, (s, st) in enumerate(zip(seqs, starts)):
        ids[j, :len(s)] = torch.tensor(s, device=device)
        mask[j, :len(s)] = 1
        labels[j, st:len(s)] = torch.tensor(s[st:], device=device)
    return model(input_ids=ids, attention_mask=mask, labels=labels).loss


@torch.no_grad()
def thermometer(tok, model, head, valenc, reader, injector, items, device,
                batch=8, max_new=24):
    """EM held + EM com memória zerada (mini-IZ) + endereçamento implícito."""
    S, keys = build_memories(tok, model, head, valenc, items, device)
    users = [it["pergunta"] for it in items]
    with injector:
        ans, _ = generate_with_memory(tok, model, injector, S, users,
                                      batch_size=batch,
                                      max_new_tokens=max_new)
        ans0, _ = generate_with_memory(tok, model, injector,
                                       torch.zeros_like(S), users,
                                       batch_size=batch,
                                       max_new_tokens=max_new)
    em = sum(is_correct(a, it["gold"])
             for a, it in zip(ans, items)) / float(len(items))
    em0 = sum(is_correct(a, it["gold"])
              for a, it in zip(ans0, items)) / float(len(items))
    retr = enderecamento_implicito(tok, model, reader, items, keys, device)
    return em, em0, sum(retr) / float(len(retr)), ans[:4]


def do_train(args, tok, model, head):
    if os.path.exists(CKPT) and not args.force_retrain:
        print("checkpoint já existe (%s), pulando treino "
              "(--force-retrain para refazer)" % CKPT, flush=True)
        return
    dev = model.device
    eps = dataset.build_items(k=args.k_train, n=args.n_train, seed=500,
                              split="train")
    held = dataset.build_items(k=args.k_train, n=args.eval_n, seed=501,
                               split="held")
    eos = [tok.eos_token_id]
    seqs, starts = [], []
    for it in eps:
        pids = tok(chat_prompt(tok, it["pergunta"])).input_ids
        aids = tok(it["gold"], add_special_tokens=False).input_ids + eos
        seqs.append(pids + aids)
        starts.append(len(pids))

    torch.manual_seed(args.seed)
    valenc = ValueEncoder().to(dev)
    reader = DeltaReader().to(dev)
    params = list(valenc.parameters()) + list(reader.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    injector = Injector(model, reader)
    print("braço I: %d params treináveis | steps=%d batch=%d k_treino=%d "
          "seed=%d" % (sum(q.numel() for q in params), args.steps,
                       args.train_batch, args.k_train, args.seed), flush=True)

    rng = random.Random(args.seed)
    best_em, t0 = -1.0, time.time()
    for step in range(1, args.steps + 1):
        idx = rng.sample(range(len(eps)), min(args.train_batch, len(eps)))
        S, _ = build_memories(tok, model, head, valenc,
                              [eps[i] for i in idx], dev, grad=True)
        with injector:
            injector.set_memory(S)
            loss = lm_loss(tok, model, [seqs[i] for i in idx],
                           [starts[i] for i in idx], dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            valenc.eval(), reader.eval()
            em, em0, retr, amostras = thermometer(
                tok, model, head, valenc, reader, injector, held, dev,
                batch=args.batch)
            valenc.train(), reader.train()
            el = time.time() - t0
            print("  step %d/%d loss=%.3f | EM=%.3f | EM(S=0)=%.3f | "
                  "endereç=%.3f | %.2fs/step | ETA %.1fmin"
                  % (step, args.steps, loss.item(), em, em0, retr, el / step,
                     el / step * (args.steps - step) / 60), flush=True)
            if step == args.eval_every:
                for a, it in zip(amostras, held[:4]):
                    print("    ex: gold=%r resposta=%r" % (it["gold"], a),
                          flush=True)
            if em > best_em:
                best_em = em
                os.makedirs(os.path.dirname(CKPT), exist_ok=True)
                torch.save(
                    {"valenc": dict((k, v.cpu()) for k, v in
                                    valenc.state_dict().items()),
                     "reader": dict((k, v.cpu()) for k, v in
                                    reader.state_dict().items()),
                     "config": {"steps": args.steps, "step": step,
                                "lr": args.lr, "seed": args.seed,
                                "k_train": args.k_train,
                                "n_train": args.n_train, "em_thermo": em,
                                "em_zero": em0, "enderec": retr,
                                "camada_injecao": L_INJ}}, CKPT)
    print("\nmelhor termômetro EM (held, n=%d): %.3f | ckpt: %s"
          % (args.eval_n, best_em, CKPT))


def load_knife_modules(arm, device):
    if not os.path.exists(CKPT):
        raise SystemExit("checkpoint ausente: %s (rode --do train)" % CKPT)
    torch.manual_seed(7 if arm == "IR" else 0)
    valenc = ValueEncoder().to(device)
    reader = DeltaReader().to(device)
    d = torch.load(CKPT, map_location=device)
    if arm != "IR":
        valenc.load_state_dict(d["valenc"])
        reader.load_state_dict(d["reader"])
    else:
        # IR (ablação): arquitetura idêntica, pesos aleatórios. O W_m
        # zero-init tornaria a ablação trivial (leitor nulo == Θ sozinho);
        # recebe a ESCALA do W_m treinado, com direções aleatórias.
        with torch.no_grad():
            std = float(d["reader"]["wm.weight"].float().std())
            reader.wm.weight.normal_(0.0, max(std, 1e-3))
    valenc.eval(), reader.eval()
    return valenc, reader


# ---------------------------------------------------------------- braço K

@torch.no_grad()
def knn_datastore(tok, model, fatos, device):
    """Pares (estado final L2-norm, próximo token) dos k fatos do item."""
    side = tok.padding_side
    tok.padding_side = "right"
    enc = tok(fatos, return_tensors="pt", padding=True).to(device)
    tok.padding_side = side
    hs = model(**enc, output_hidden_states=True).hidden_states[-1]
    keys, nexts = [], []
    for j in range(len(fatos)):
        n_valid = int(enc.attention_mask[j].sum().item())
        keys.append(F.normalize(hs[j, :n_valid - 1].float(), dim=-1))
        nexts.append(enc.input_ids[j, 1:n_valid])
    return torch.cat(keys), torch.cat(nexts)


@torch.no_grad()
def knn_generate(tok, model, item, lam, tau, topk=8, max_new=32):
    """Geração gulosa com p = (1−λ)·p_LM + λ·p_kNN (kNN-LM episódico)."""
    ds_keys, ds_next = knn_datastore(tok, model, item["fatos"], model.device)
    ids = tok(chat_prompt(tok, item["pergunta"]),
              return_tensors="pt").input_ids.to(model.device)
    n_prompt = ids.size(1)
    past, cur, out_ids = None, ids, []
    for _ in range(max_new):
        o = model(input_ids=cur, past_key_values=past, use_cache=True,
                  output_hidden_states=True)
        past = o.past_key_values
        h = F.normalize(o.hidden_states[-1][0, -1].float(), dim=-1)
        p_lm = torch.softmax(o.logits[0, -1].float(), -1)
        top = torch.topk(ds_keys @ h, min(topk, ds_keys.size(0)))
        w = torch.softmax(top.values / tau, 0)
        p_knn = torch.zeros_like(p_lm)
        p_knn.index_add_(0, ds_next[top.indices], w)
        nxt = int(torch.argmax((1 - lam) * p_lm + lam * p_knn).item())
        if nxt == tok.eos_token_id:
            break
        out_ids.append(nxt)
        cur = torch.tensor([[nxt]], device=model.device)
    return tok.decode(out_ids, skip_special_tokens=True), n_prompt


def do_grid(args, tok, model):
    if os.path.exists(GRID_PATH) and not args.force:
        print("grid do K já existe (%s), pulando" % GRID_PATH, flush=True)
        return
    n = 8 if args.smoke else 96
    lams = (0.5,) if args.smoke else GRID_LAM
    taus = (0.2,) if args.smoke else GRID_TAU
    items = dataset.build_items(k=K_EVAL, n=n, seed=600, split="train")
    rows, best = [], None
    for lam in lams:
        for tau in taus:
            t0, ok = time.time(), 0
            for it in items:
                a, _ = knn_generate(tok, model, it, lam, tau,
                                    max_new=args.max_new)
                ok += int(is_correct(a, it["gold"]))
            em = ok / float(n)
            rows.append({"lam": lam, "tau": tau, "em": em,
                         "s_por_item": (time.time() - t0) / n})
            print("  grid K: lam=%.1f tau=%.2f EM=%.3f" % (lam, tau, em),
                  flush=True)
            if best is None or em > best["em"]:
                best = rows[-1]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GRID_PATH, "w") as f:
        json.dump({"n": n, "seed": 600, "split": "train",
                   "rows": rows, "best": best}, f, indent=1)
    print("grid K: melhor lam=%.1f tau=%.2f (EM=%.3f em train) → %s"
          % (best["lam"], best["tau"], best["em"], GRID_PATH), flush=True)


# ---------------------------------------------------------------- células

def _acc_por_terco(items, corretos):
    k = len(items[0]["fatos"])
    buckets = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for it, c in zip(items, corretos):
        t = min(2, it["target_idx"] * 3 // k)
        buckets[t][1] += 1
        buckets[t][0] += int(c)
    return dict((str(t), b[0] / float(b[1]) if b[1] else None)
                for t, b in buckets.items())


def cell_json(arm, n, items, answers, corretos, retr_oks, prompt_toks, dt,
              extra_cfg):
    n_ok = sum(retr_oks) if retr_oks else 0
    acc_cond = (sum(1 for r, c in zip(retr_oks, corretos) if r and c)
                / float(n_ok)) if n_ok else None
    cfg = {"modelo": "Qwen2.5-0.5B-Instruct", "greedy": True,
           "seed_dataset": SEED_EVAL, "split": "held"}
    cfg.update(extra_cfg)
    return {
        "marco": "M5-knife", "arm": arm, "k": K_EVAL, "n": n,
        "acc": sum(corretos) / float(n),
        "enderecamento_top1": (sum(retr_oks) / float(n)) if retr_oks
        else None,
        "acc_cond_enderec_ok": acc_cond,
        "acc_por_terco_posicao": _acc_por_terco(items, corretos),
        "tokens_prompt_medio": prompt_toks / float(n),
        "s_por_item": dt / float(n),
        "tempo_total_s": dt,
        "dataset_stats": dataset.stats(items),
        "config": cfg,
        "itens": [{"pergunta": it["pergunta"], "gold": it["gold"],
                   "pos": it["target_idx"],
                   "enderec_ok": (bool(r) if retr_oks else None),
                   "resposta": a, "correto": bool(c)}
                  for it, r, a, c in zip(items, retr_oks or [None] * n,
                                         answers, corretos)],
    }


def run_cell_K(tok, model, items, args):
    if not os.path.exists(GRID_PATH):
        raise SystemExit("K_grid.json ausente — rode --do grid antes")
    with open(GRID_PATH) as f:
        best = json.load(f)["best"]
    t0, answers, toks = time.time(), [], 0
    for i, it in enumerate(items):
        a, npt = knn_generate(tok, model, it, best["lam"], best["tau"],
                              max_new=args.max_new)
        answers.append(a)
        toks += npt
        if (i + 1) % 128 == 0:
            el = time.time() - t0
            print("  [K] %d/%d | %.2fs/item | ETA %.1fmin"
                  % (i + 1, len(items), el / (i + 1),
                     el / (i + 1) * (len(items) - i - 1) / 60), flush=True)
    corretos = [is_correct(a, it["gold"]) for a, it in zip(answers, items)]
    return cell_json("K", len(items), items, answers, corretos, None, toks,
                     time.time() - t0,
                     {"lam": best["lam"], "tau": best["tau"], "topk": 8,
                      "grid": "train seed 600 (K_grid.json)"})


def run_cell_I(arm, tok, model, head, items, args):
    """arm ∈ {I, IZ, IS, IR} — mesmo checkpoint, ablação por manipulação."""
    valenc, reader = load_knife_modules(arm, model.device)
    injector = Injector(model, reader)
    t0, answers, retr_oks, toks = time.time(), [], [], 0
    n = len(items)
    for i in range(0, n, args.chunk):
        chunk = items[i:i + args.chunk]
        S, keys = build_memories(tok, model, head, valenc, chunk,
                                 model.device, shuffle_keys=(arm == "IS"))
        if arm == "IZ":
            S = torch.zeros_like(S)
        retr_oks += enderecamento_implicito(tok, model, reader, chunk, keys,
                                            model.device)
        with injector:
            ans, tk = generate_with_memory(
                tok, model, injector, S, [it["pergunta"] for it in chunk],
                batch_size=args.batch, max_new_tokens=args.max_new)
        answers += ans
        toks += tk
        el = time.time() - t0
        print("  [%s] %d/%d | %.2fs/item | ETA %.1fmin"
              % (arm, len(answers), n, el / len(answers),
                 el / len(answers) * (n - len(answers)) / 60), flush=True)
    corretos = [is_correct(a, it["gold"]) for a, it in zip(answers, items)]
    return cell_json(arm, n, items, answers, corretos, retr_oks, toks,
                     time.time() - t0,
                     {"ckpt": os.path.basename(CKPT),
                      "camada_injecao": L_INJ,
                      "ablacao": arm if arm != "I" else None})


def do_run(args, tok, model, head):
    os.makedirs(OUT_DIR, exist_ok=True)
    items = dataset.build_items(k=K_EVAL, n=args.n, seed=SEED_EVAL,
                                split="held")
    for arm in args.arms.split(","):
        tag = "M5_%s_k%d_n%d" % (arm, K_EVAL, args.n)
        path = os.path.join(OUT_DIR, tag + ".json")
        if os.path.exists(path) and not args.force:
            print("célula %s pronta, pulando" % tag, flush=True)
            continue
        print("== célula %s ==" % tag, flush=True)
        if arm == "K":
            r = run_cell_K(tok, model, items, args)
        elif arm in ("I", "IZ", "IS", "IR"):
            r = run_cell_I(arm, tok, model, head, items, args)
        else:
            print("braço %s é reusado do M2 (não se roda aqui)" % arm,
                  flush=True)
            continue
        with open(path, "w") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print("célula %s: EM=%.3f (%.2fs/item)"
              % (tag, r["acc"], r["s_por_item"]), flush=True)


# ---------------------------------------------------------------- edição

def _novo_valor(rng, pools, pool_name, it, tries=200):
    """Valor novo do MESMO pool, ausente do episódio (fronteira de palavra,
    critério do juiz). Pools held são pequenos (ex.: 4 cores) — pode não
    existir candidato; devolve None e o item é PULADO (contado no JSON)."""
    for _ in range(tries):
        v = pools.valor(rng, pool_name)
        nv = " %s " % normalize(v)
        if nv.strip() == normalize(it["gold"]):
            continue
        if any(nv in (" %s " % normalize(f)) for f in it["fatos"]):
            continue
        return v
    return None


def do_edit(args, tok, model, head):
    """Fase 2 (F4): edição por re-write delta; esquecimento por rebuild."""
    tag = "M5_edicao_k%d_n%d" % (K_EVAL, args.n_edit)
    path = os.path.join(OUT_DIR, tag + ".json")
    if os.path.exists(path) and not args.force:
        print("célula %s pronta, pulando" % tag, flush=True)
        return
    valenc, reader = load_knife_modules("I", model.device)
    injector = Injector(model, reader)
    todos = dataset.build_items(k=K_EVAL, n=args.n_edit, seed=1508,
                                split="held")
    pools = dataset.Pools("held")
    rng = random.Random(1508)
    pool_por_attr = dict(dataset.ATRIBUTOS)
    items, novos = [], []
    for it in todos:
        v = _novo_valor(rng, pools, pool_por_attr[it["attr"]], it)
        if v is None:
            continue
        items.append(it)
        novos.append(v)
    pulados = len(todos) - len(items)
    if pulados:
        print("  [edicao] %d/%d itens sem valor novo possível (pools held "
              "pequenos), pulados" % (pulados, len(todos)), flush=True)

    t0 = time.time()
    em_novo, eco_antigo, eco_esq, itens_out = [], [], [], []
    for i in range(0, len(items), args.chunk):
        chunk = items[i:i + args.chunk]
        novos_val = novos[i:i + args.chunk]
        novos_txt = [dataset.FATO_TEMPLATES[0].format(
            attr=it["attr"], attr_cap=it["attr"].capitalize(),
            nome=it["nome"], valor=v)
            for it, v in zip(chunk, novos_val)]
        S, _ = build_memories(tok, model, head, valenc, chunk, model.device)
        side = tok.padding_side
        tok.padding_side = "right"
        st, mk = encode_batch(model, tok, novos_txt, device=model.device)
        tok.padding_side = side
        st, mk = st.float().to(model.device), mk.to(model.device)
        with torch.no_grad():
            k_new = head(st, mk)                     # (B,dk)
            v_new = valenc(st, mk)                   # (B,dv)
            pred = torch.einsum("bvk,bk->bv", S, k_new)
            S_edit = S + torch.einsum("bv,bk->bvk", v_new - pred, k_new)
        # esquecimento: rebuild determinístico SEM o fato-alvo (ledger)
        sem_alvo = [dict(it, fatos=[f for j, f in enumerate(it["fatos"])
                                    if j != it["target_idx"]])
                    for it in chunk]
        S_forget, _ = build_memories(tok, model, head, valenc, sem_alvo,
                                     model.device)
        users = [it["pergunta"] for it in chunk]
        with injector:
            ans_e, _ = generate_with_memory(tok, model, injector, S_edit,
                                            users, batch_size=args.batch,
                                            max_new_tokens=args.max_new)
            ans_f, _ = generate_with_memory(tok, model, injector, S_forget,
                                            users, batch_size=args.batch,
                                            max_new_tokens=args.max_new)
        for it, vnovo, ae, af in zip(chunk, novos_val, ans_e, ans_f):
            em_novo.append(is_correct(ae, vnovo))
            eco_antigo.append(is_correct(ae, it["gold"]))
            eco_esq.append(is_correct(af, it["gold"]))
            itens_out.append({"pergunta": it["pergunta"],
                              "gold_antigo": it["gold"], "gold_novo": vnovo,
                              "resp_edicao": ae, "resp_esquecimento": af})
        print("  [edicao] %d/%d" % (len(itens_out), len(items)), flush=True)
    n = float(len(items))
    r = {"marco": "M5-knife", "fase": "edicao", "k": K_EVAL,
         "n": len(items), "n_pedido": args.n_edit,
         "itens_pulados_sem_valor_novo": pulados,
         "em_valor_novo": sum(em_novo) / n,
         "eco_valor_antigo_apos_edicao": sum(eco_antigo) / n,
         "eco_apos_esquecimento_rebuild": sum(eco_esq) / n,
         "s_por_item": (time.time() - t0) / n,
         "config": {"ckpt": os.path.basename(CKPT), "seed_dataset": 1508},
         "itens": itens_out}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print("edição: EM novo=%.3f eco antigo=%.3f | esquecimento eco=%.3f"
          % (r["em_valor_novo"], r["eco_valor_antigo_apos_edicao"],
             r["eco_apos_esquecimento_rebuild"]), flush=True)


# ---------------------------------------------------------------- relatório

def _m2_cell(arm, n):
    p = os.path.join(M2_DIR, "M2_%s_k%d_n%d.json" % (arm, K_EVAL, n))
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return {"arm": {"T": "T", "S1": "S"}[arm], "acc": d["acc"],
            "endr": d.get("retrieval_acc_top1"),
            "cond": d.get("acc_cond_retr_ok"),
            "tok": d.get("tokens_prompt_medio"), "s": d.get("s_por_item"),
            "origem": "M2 (reuso, mesmos itens)"}


def do_report(args):
    cells = {}
    if os.path.isdir(OUT_DIR):
        for fn in sorted(os.listdir(OUT_DIR)):
            if fn.startswith("M5_") and fn.endswith("_n%d.json" % args.n):
                with open(os.path.join(OUT_DIR, fn)) as f:
                    d = json.load(f)
                if "arm" in d:
                    cells[d["arm"]] = d
    rows = []
    for m2arm in ("T", "S1"):
        c = _m2_cell(m2arm, args.n)
        if c:
            rows.append(c)
    for arm in ("K", "I", "IZ", "IS", "IR"):
        if arm in cells:
            d = cells[arm]
            rows.append({"arm": arm, "acc": d["acc"],
                         "endr": d.get("enderecamento_top1"),
                         "cond": d.get("acc_cond_enderec_ok"),
                         "tok": d.get("tokens_prompt_medio"),
                         "s": d.get("s_por_item"), "origem": "knife"})
    if not rows:
        print("sem células ainda — rode --do run antes do report", flush=True)
        return

    def _f(x, fmt="%.3f"):
        return (fmt % x) if x is not None else "—"

    md = ["# Relatório M5 — teste de faca (portões F1–F4 do DESIGN.md)",
          "",
          "Mesmos itens do M0/M2 (seed %d, held, n=%d). T e S reusados das"
          % (SEED_EVAL, args.n),
          "células oficiais do M2. Fato NUNCA no prompt em K/I/IZ/IS/IR.",
          "",
          "| braço | EM | endereç. top-1 | EM se endereç. ok | tok prompt | "
          "s/item | origem |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        md.append("| %s | %s | %s | %s | %s | %s | %s |"
                  % (r["arm"], _f(r["acc"]), _f(r["endr"]), _f(r["cond"]),
                     _f(r["tok"], "%.0f"), _f(r["s"], "%.2f"), r["origem"]))
    by = dict((r["arm"], r) for r in rows)
    md += ["", "## Portões", ""]
    if "I" in by:
        f1 = by["I"]["acc"] >= 0.80
        md.append("- F1 (canal: EM do braço I ≥ 0.80): %.3f %s"
                  % (by["I"]["acc"], "✅" if f1 else "❌"))
        nao_textuais = [by[a]["acc"] for a in ("K", "S") if a in by]
        base = max(nao_textuais) if nao_textuais else 0.0
        f2 = by["I"]["acc"] >= base + 0.05
        md.append("- F2 (margem ≥ +0.05 sobre melhor não textual, %.3f): "
                  "%+.3f %s" % (base, by["I"]["acc"] - base,
                                "✅" if f2 else "❌"))
        abls = [a for a in ("IZ", "IS", "IR") if a in by]
        f3 = bool(abls) and all(by[a]["acc"] <= 0.10 for a in abls)
        md.append("- F3 (ablações IZ/IS/IR ≤ 0.10): %s %s"
                  % (", ".join("%s=%.3f" % (a, by[a]["acc"]) for a in abls),
                     "✅ desabou" if f3 else "❌"))
        md += ["", "**Veredito: %s**" % (
            "F1–F3 ✅ — o canal interno existe; seguir m5/ANALYSIS.md "
            "(fase 2/3)" if (f1 and f2 and f3) else
            "MORTE da hipótese M5a nesta forma (DESIGN.md) — registrar em "
            "NEGATIVE_FINDINGS.md no dia, com o diagnóstico voz vs "
            "endereçamento (coluna endereç. top-1)")]
    ed_path = os.path.join(OUT_DIR, "M5_edicao_k%d_n%d.json"
                           % (K_EVAL, args.n_edit))
    if os.path.exists(ed_path):
        with open(ed_path) as f:
            ed = json.load(f)
        ok4 = (ed["em_valor_novo"] >= 0.80
               and ed["eco_valor_antigo_apos_edicao"] <= 0.20
               and ed["eco_apos_esquecimento_rebuild"] <= 0.20)
        md += ["", "- F4 (edição/esquecimento, n=%d): EM novo=%.3f, eco "
               "antigo=%.3f, eco pós-rebuild=%.3f %s"
               % (ed["n"], ed["em_valor_novo"],
                  ed["eco_valor_antigo_apos_edicao"],
                  ed["eco_apos_esquecimento_rebuild"],
                  "✅" if ok4 else "❌")]
    md += ["", "Sentinelas: S1 juiz da régua; S3/S4 por construção (terços "
           "nos JSONs); S5 n=%d; S6 amostras em `itens`. Baseline A "
           "(Θ sozinho): 0.015." % args.n, ""]
    path = os.path.join(OUT_DIR, "RELATORIO_M5_KNIFE.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("Relatório: %s" % path, flush=True)


# ---------------------------------------------------------------- main

def main():
    global CKPT, GRID_PATH
    p = argparse.ArgumentParser()
    p.add_argument("--do", default="all",
                   help="grid,train,run,edit,report ou all")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--n-edit", type=int, default=512)
    p.add_argument("--arms", default="K,I,IZ,IS,IR")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--max-new", type=int, default=32)
    # treino (números travados no DESIGN.md)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--train-batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--k-train", type=int, default=8)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n, args.n_edit, args.steps = 8, 8, 6
        args.train_batch, args.n_train = 2, 64
        args.eval_every, args.eval_n, args.chunk = 3, 8, 4
        args.max_new = 12
        # nunca contaminar checkpoint/grid oficiais
        CKPT = CKPT + ".smoke"
        GRID_PATH = GRID_PATH.replace(".json", ".smoke.json")

    fases = (["grid", "train", "run", "edit", "report"]
             if args.do == "all" else args.do.split(","))
    if "train" in fases and args.dtype != "float32":
        raise SystemExit("treino exige --dtype float32 (rode as fases "
                         "de avaliação em float16 separadamente)")
    tok = model = head = None
    if any(f in fases for f in ("grid", "train", "run", "edit")):
        print("Carregando Θ (%s, %s)..." % (args.device, args.dtype),
              flush=True)
        tok, model = load(args.device, args.dtype)
        model.requires_grad_(False)
        head = load_m1_head(model.device)
        n_layers = len(model.model.layers)
        assert n_layers == 24, "esperava 24 blocos, achei %d" % n_layers

    if "grid" in fases:
        do_grid(args, tok, model)
    if "train" in fases:
        do_train(args, tok, model, head)
    if "run" in fases:
        do_run(args, tok, model, head)
    if "edit" in fases:
        do_edit(args, tok, model, head)
    if "report" in fases:
        do_report(args)


if __name__ == "__main__":
    main()
