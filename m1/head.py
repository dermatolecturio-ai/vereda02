# -*- coding: utf-8 -*-
"""Cabeça de memória aprendida (herdada de herdados/vereda_m_qwen.py).

Atenção escalar aprendida sobre os estados de TOKEN do Qwen congelado + projeção
linear + L2-normalize. Qwen nunca é treinado; só a cabeça (poucos milhares de
parâmetros) aprende a focar a entidade em vez do mean-pool "sujo".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DIN = 896  # hidden do Qwen2.5-0.5B
MAXLEN = 40  # frases da régua são mais longas que as do V1 ("Anote: a senha...")


class Head(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.qk = nn.Parameter(torch.randn(DIN) * DIN ** -0.5)
        self.proj = nn.Linear(DIN, d, bias=False)

    def forward(self, states, mask):
        sc = (states @ self.qk).masked_fill(~mask, -1e4)
        a = torch.softmax(sc, -1)
        pooled = (a.unsqueeze(-1) * states).sum(1)
        return F.normalize(self.proj(pooled), dim=-1)


@torch.no_grad()
def encode_batch(model, tok, texts, device="cpu", batch_size=32, maxlen=MAXLEN):
    """Estados de token (não mean-pool) — para a cabeça atender sozinha.

    Buffer de saída sempre em CPU (independente de `device`), para o cache
    serializado em disco não ficar amarrado a CUDA.
    """
    states = torch.zeros(len(texts), maxlen, DIN)
    masks = torch.zeros(len(texts), maxlen, dtype=torch.bool)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=maxlen).to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        states[i:i + batch_size] = h.float().cpu()
        masks[i:i + batch_size] = enc.attention_mask.bool().cpu()
    return states, masks
