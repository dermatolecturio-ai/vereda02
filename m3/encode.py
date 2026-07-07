# -*- coding: utf-8 -*-
"""Codificação Qwen (congelado) + alinhamento char→token para o M3.

Padding à DIREITA durante a codificação (RoPE muda os estados com left-pad —
mesma pegadinha documentada em m2/pipeline.py). Usa offset_mapping do
tokenizer FAST para converter spans de caractere (do dataset) em spans de
token (para supervisionar os ponteiros).
"""
import torch

MAXLEN = 64


@torch.no_grad()
def encode_with_offsets(model, tok, texts, device="cpu", batch_size=32,
                        maxlen=MAXLEN, return_ids=False):
    din = model.config.hidden_size
    states = torch.zeros(len(texts), maxlen, din)
    masks = torch.zeros(len(texts), maxlen, dtype=torch.bool)
    offsets = torch.zeros(len(texts), maxlen, 2, dtype=torch.long)
    ids = torch.zeros(len(texts), maxlen, dtype=torch.long)
    side = tok.padding_side
    tok.padding_side = "right"
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=maxlen,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        enc = enc.to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        states[i:i + batch_size] = h.float().cpu()
        masks[i:i + batch_size] = enc["attention_mask"].bool().cpu()
        offsets[i:i + batch_size] = offs
        ids[i:i + batch_size] = enc["input_ids"].cpu()
    tok.padding_side = side
    if return_ids:
        return states, masks, offsets, ids
    return states, masks, offsets


def char_to_token_span(offsets_row, char_start, char_end):
    """offsets_row: (L,2). Retorna (tok_start, tok_end) inclusive ou None."""
    tok_start = tok_end = None
    for i in range(offsets_row.size(0)):
        o0, o1 = int(offsets_row[i, 0]), int(offsets_row[i, 1])
        if o0 == o1:
            continue  # token especial/padding (span vazio)
        if tok_start is None and o1 > char_start:
            tok_start = i
        if o0 < char_end:
            tok_end = i
    if tok_start is None or tok_end is None or tok_end < tok_start:
        return None
    return tok_start, tok_end


def build_targets(items, offsets):
    """Retorna (tse,tee,tsv,tev) LongTensors e uma máscara de itens válidos
    (alinhamento pode falhar em casos raros de truncamento)."""
    n = len(items)
    tse = torch.zeros(n, dtype=torch.long)
    tee = torch.zeros(n, dtype=torch.long)
    tsv = torch.zeros(n, dtype=torch.long)
    tev = torch.zeros(n, dtype=torch.long)
    ok = torch.zeros(n, dtype=torch.bool)
    for j, it in enumerate(items):
        se, ee = it["span_entidade_char"]
        sv, ev = it["span_valor_char"]
        pe = char_to_token_span(offsets[j], se, ee)
        pv = char_to_token_span(offsets[j], sv, ev)
        if pe is None or pv is None:
            continue
        tse[j], tee[j] = pe
        tsv[j], tev[j] = pv
        ok[j] = True
    return tse, tee, tsv, tev, ok
