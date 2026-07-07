# -*- coding: utf-8 -*-
"""Projetor do injetor S (M2): carga útil do fato → m soft tokens.

Perceiver-style: m latentes aprendidos fazem cross-attention sobre os ESTADOS
DE TOKEN do fato (Qwen congelado, 40×896) e são projetados para o espaço de
embeddings do Qwen. Só isto treina no M2 (~1M params); Qwen e cabeça M1 ficam
congelados. Escala de saída calibrada pela RMS da matriz de embeddings — soft
token fora de escala desestabiliza o LM congelado.

Fundamentação: xRAG (arXiv:2405.13792) para a topologia; estados de token (e
não mean-pool) porque o juiz exige o valor LITERAL (senhas) e compressão
agressiva perde conteúdo literal (arXiv:2412.17483).
"""
import torch
import torch.nn as nn

from m1.head import DIN  # 896, hidden do Qwen2.5-0.5B


class Projector(nn.Module):
    def __init__(self, m=8, dlat=256, heads=4, dout=DIN):
        super().__init__()
        self.m = m
        self.latents = nn.Parameter(torch.randn(m, dlat) * dlat ** -0.5)
        self.attn = nn.MultiheadAttention(dlat, heads, kdim=DIN, vdim=DIN,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(dlat)
        self.ff = nn.Sequential(nn.Linear(dlat, 2 * dlat), nn.GELU(),
                                nn.Linear(2 * dlat, dlat))
        self.ln2 = nn.LayerNorm(dlat)
        self.out = nn.Linear(dlat, dout)
        self.register_buffer("emb_rms", torch.tensor(1.0))

    @torch.no_grad()
    def calibrate(self, embedding_weight):
        self.emb_rms.fill_(
            embedding_weight.detach().float().pow(2).mean().sqrt().item())

    def forward(self, states, mask):
        # states: (B, L, DIN) float; mask: (B, L) bool (True = token válido)
        B = states.size(0)
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        att, _ = self.attn(q, states, states, key_padding_mask=~mask)
        x = self.ln1(q + att)
        x = self.ln2(x + self.ff(x))
        y = self.out(x)
        rms = y.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return y / rms * self.emb_rms
