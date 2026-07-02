# -*- coding: utf-8 -*-
"""I/O com o substrato Θ (Qwen2.5-0.5B-Instruct, CONGELADO).

Geração gulosa (determinística, reprodutível) em lote com padding à esquerda.
Embedding = mean-pool do último hidden state do Qwen CRU — usado APENAS pelo
baseline C (RAG ingênuo de referência; proibido como mecanismo, CARTA §3).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "Responda em português do Brasil, de forma curta e direta."


def load(device="cpu", dtype="float32"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok, model


def chat_prompt(tok, user):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def generate_batch(tok, model, users, batch_size=8, max_new_tokens=32,
                   progress=None):
    outs = []
    total_prompt_toks = 0
    for i in range(0, len(users), batch_size):
        prompts = [chat_prompt(tok, u) for u in users[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        total_prompt_toks += int(enc.attention_mask.sum().item())
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        outs += tok.batch_decode(new, skip_special_tokens=True)
        if progress:
            progress(len(outs))
    return outs, total_prompt_toks


@torch.no_grad()
def embed_batch(tok, model, texts, batch_size=32, cache=None):
    res = [None] * len(texts)
    todo, idxs = [], []
    for j, t in enumerate(texts):
        if cache is not None and t in cache:
            res[j] = cache[t]
        else:
            todo.append(t)
            idxs.append(j)
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[-1]
        mask = enc.attention_mask.unsqueeze(-1).to(hs.dtype)
        emb = (hs * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu()
        for b in range(len(chunk)):
            res[idxs[i + b]] = emb[b]
            if cache is not None and len(cache) < 200000:
                cache[chunk[b]] = emb[b]
    return torch.stack(res)
