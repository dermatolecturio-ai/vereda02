# -*- coding: utf-8 -*-
"""M3 — caminho fim-a-fim: TEXTO cru → extração (entidade,valor) → M (chave
M1) → resposta (injetor T do M2).

Regra do caminho (CARTA §3): nenhum parser/regex decide span ou fato. O
único elemento "dado" (não aprendido) é o TIPO do atributo, escolhido por
correspondência a um conjunto FECHADO e conhecido de 6 frases (não é a
capacidade extraída/alegada — extração aprendida cobre entidade+valor, os
elementos de vocabulário ABERTO). Ver DESIGN.md.
"""
import os

import torch

from m3.encode import encode_with_offsets
from m3.extractor import Extractor, decode_spans

M3_CKPT = os.path.join(os.path.dirname(__file__), "..", "modelos",
                       "vereda2_m3_extractor.pt")


def load_extractor(device, path=M3_CKPT):
    model = Extractor().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _decode_span(tok, ids_row, s, e):
    piece_ids = ids_row[s:e + 1].tolist()
    return tok.decode(piece_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def extract_batch(qwen_model, tok, extractor, texts, device,
                  encode_batch_size=32, infer_batch_size=256):
    """Retorna lista de dicts {entidade, valor} — 100% decodido dos ponteiros
    aprendidos, nenhum parser decidindo o span."""
    states, masks, offsets, ids = encode_with_offsets(
        qwen_model, tok, texts, device=device, batch_size=encode_batch_size,
        return_ids=True)
    out = []
    for i in range(0, len(texts), infer_batch_size):
        s = states[i:i + infer_batch_size].float().to(device)
        m = masks[i:i + infer_batch_size].to(device)
        se, ee, sv, ev = decode_spans(extractor(s, m), m)
        se, ee, sv, ev = se.cpu(), ee.cpu(), sv.cpu(), ev.cpu()
        for j in range(s.size(0)):
            row = ids[i + j]
            out.append({
                "entidade": _decode_span(tok, row, se[j].item(), ee[j].item()),
                "valor": _decode_span(tok, row, sv[j].item(), ev[j].item()),
            })
    return out


def fato_canonico(attr, entidade, valor):
    """Reconstrói o registro de M a partir do extraído (S1: o atributo é
    lookup de conjunto fechado conhecido, não parte da capacidade extraída)."""
    return "%s de %s é %s." % (attr.capitalize(), entidade, valor)
