# -*- coding: utf-8 -*-
"""Caminho compartilhado do M2: chave M1 congelada + geração com prefixo soft.

Regras do caminho (CARTA §3): pergunta → chave M1 (aprendida) → top-1 sobre os
k fatos do item → carga útil (estados de token) do fato vencedor → injetor →
Θ gera. Nenhum parser/regex/banco em ponto algum.

Cuidado de consistência: a cabeça M1 foi treinada sobre estados codificados com
padding à DIREITA (RoPE muda os estados com left-pad). Quem chama encode aqui
deve garantir tok.padding_side == "right" durante a codificação.
"""
import os

import torch

from m1.head import Head, encode_batch
from regua.qwen_io import chat_prompt

M1_CKPT = os.path.join(os.path.dirname(__file__), "..", "modelos",
                       "vereda2_m1_head.pt")


def load_m1_head(device, path=M1_CKPT):
    head = Head().to(device)
    head.load_state_dict(torch.load(path, map_location=device))
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


@torch.no_grad()
def retrieve_chunk(tok, model, head, items, device, encode_batch_size=64,
                   key_batch=1024):
    """Para um pedaço de itens: codifica fatos+pergunta, top-1 pela chave M1.

    Devolve, por item: idx recuperado, acerto do retrieval, texto do fato
    recuperado e estados/máscara (CPU, half) da carga útil.
    """
    k = len(items[0]["fatos"])
    texts = []
    for it in items:
        texts += it["fatos"]
    texts += [it["pergunta"] for it in items]

    side = tok.padding_side
    tok.padding_side = "right"  # consistente com o treino da cabeça M1
    states, masks = encode_batch(model, tok, texts, device=device,
                                 batch_size=encode_batch_size)
    tok.padding_side = side

    keys = torch.zeros(states.size(0), 128)
    for i in range(0, states.size(0), key_batch):
        s = states[i:i + key_batch].float().to(device)
        m = masks[i:i + key_batch].to(device)
        keys[i:i + key_batch] = head(s, m).cpu()

    out = []
    nk = len(items) * k
    for j, it in enumerate(items):
        fk = keys[j * k:(j + 1) * k]
        qk = keys[nk + j]
        top1 = int(torch.argmax(fk @ qk).item())
        out.append({
            "top1": top1,
            "retr_ok": top1 == it["target_idx"],
            "fato": it["fatos"][top1],
            "states": states[j * k + top1].half(),
            "mask": masks[j * k + top1],
        })
    return out


@torch.no_grad()
def generate_with_prefix(tok, model, softs, users, batch_size=8,
                         max_new_tokens=32, progress=None):
    """Geração gulosa com prefixo de soft tokens via inputs_embeds.

    Cada linha = [soft tokens do fato] + [chat prompt da pergunta], com
    left-pad manual no espaço de embeddings. Conta os soft tokens no custo.
    """
    emb = model.get_input_embeddings()
    dev = model.device
    dtype = emb.weight.dtype
    pad_vec = emb(torch.tensor([tok.pad_token_id], device=dev))[0]
    outs, total_toks = [], 0
    for i in range(0, len(users), batch_size):
        rows = []
        for u, s in zip(users[i:i + batch_size], softs[i:i + batch_size]):
            ids = tok(chat_prompt(tok, u),
                      return_tensors="pt").input_ids.to(dev)[0]
            rows.append(torch.cat([s.to(dev, dtype), emb(ids)], 0))
        L = max(r.size(0) for r in rows)
        batch = pad_vec.repeat(len(rows), L, 1)
        mask = torch.zeros(len(rows), L, dtype=torch.long, device=dev)
        for j, r in enumerate(rows):
            batch[j, L - r.size(0):] = r
            mask[j, L - r.size(0):] = 1
            total_toks += r.size(0)
        gen = model.generate(inputs_embeds=batch, attention_mask=mask,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        # com inputs_embeds (sem input_ids), generate devolve só tokens novos
        outs += tok.batch_decode(gen, skip_special_tokens=True)
        if progress:
            progress(len(outs))
    return outs, total_toks
