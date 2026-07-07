# -*- coding: utf-8 -*-
"""Memória delta + leitor de camada intermediária (M5 knife, braço I).

Escrita (por forward, regra delta pura): S ← S + (v_t − S·k_t)·k_tᵀ, com
k_t = chave M1 CONGELADA do texto do fato (128-d, L2-norm) e v_t = codificador
de valor aprendido sobre os estados de token do fato.

Leitura (aprendida, dentro do Θ CONGELADO): hook após o bloco L_INJ do Qwen;
q = L2norm(W_q·h) por posição; m = S·q; h̃ = h + gate·W_m·LN(m). W_m zero-init
e viés do gate em −2: no passo 0 o backbone fica intacto (h̃ ≡ h).

Números e porquês travados em m5/DESIGN.md (arXiv:2603.22329 — família com
prior de binding explícito; arXiv:2102.11174 — regra delta).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from m1.head import DIN  # 896, hidden do Qwen2.5-0.5B

DKEY = 128   # dimensão da chave M1 (m1/head.py, d=128)
DVAL = 256   # dimensão do valor na memória (risco de compressão declarado)
L_INJ = 12   # injeção após o 12º bloco (de 24) — TRAVADO no DESIGN.md


class ValueEncoder(nn.Module):
    """Estados de token do fato (L,896) → v (256).

    Mesma família da cabeça M1 (atenção escalar aprendida + projeção): o que
    provou focar (nome, atributo) pode aprender a focar o VALOR literal.
    """

    def __init__(self, d=DVAL):
        super().__init__()
        self.qk = nn.Parameter(torch.randn(DIN) * DIN ** -0.5)
        self.proj = nn.Linear(DIN, d, bias=False)

    def forward(self, states, mask):
        sc = (states @ self.qk).masked_fill(~mask, -1e4)
        a = torch.softmax(sc, -1)
        pooled = (a.unsqueeze(-1) * states).sum(1)
        return self.proj(pooled)


class DeltaReader(nn.Module):
    """q = L2norm(W_q h); m = S·q; h̃ = h + σ(gate)·W_m·LN(m)."""

    def __init__(self, dv=DVAL, dk=DKEY):
        super().__init__()
        self.wq = nn.Linear(DIN, dk, bias=False)
        self.ln = nn.LayerNorm(dv)
        self.wm = nn.Linear(dv, DIN, bias=False)
        nn.init.zeros_(self.wm.weight)            # passo 0: h̃ ≡ h
        self.gate = nn.Linear(DIN + dv, 1)
        nn.init.constant_(self.gate.bias, -2.0)   # gate abre devagar

    def query(self, h):
        return F.normalize(self.wq(h), dim=-1)

    def forward(self, h, S):
        # h: (B,L,896) float32; S: (B,dv,dk)
        q = self.query(h)                                  # (B,L,dk)
        m = torch.einsum("bvk,blk->blv", S, q)             # (B,L,dv)
        mn = self.ln(m)
        g = torch.sigmoid(self.gate(torch.cat([h, mn], -1)))
        return h + g * self.wm(mn)


def write_facts(keys, values):
    """Regra delta sobre o episódio inteiro (todos os k fatos escritos).

    keys: (B,k,dk) L2-norm (chave M1, congelada); values: (B,k,dv) (com
    gradiente no treino). Devolve S: (B,dv,dk). Ordem = ordem do item
    (posição do alvo já embaralhada, S4).
    """
    B, k, _ = keys.shape
    S = values.new_zeros(B, values.size(-1), keys.size(-1))
    for t in range(k):
        kt = keys[:, t, :]
        vt = values[:, t, :]
        pred = torch.einsum("bvk,bk->bv", S, kt)
        S = S + torch.einsum("bv,bk->bvk", vt - pred, kt)
    return S


class Injector(object):
    """Hook após o bloco L_INJ do Qwen; S por item do batch corrente.

    Uso:
        inj = Injector(model, reader)
        with inj:
            inj.set_memory(S)          # (B,dv,dk), alinhado ao batch
            model(...) / model.generate(...)
    Fora do `with`, o Qwen fica intacto. S=None desliga a leitura.
    """

    def __init__(self, model, reader, layer=L_INJ):
        self.reader = reader
        self.S = None
        self._layer = model.model.layers[layer - 1]
        self._handle = None

    def set_memory(self, S):
        self.S = S

    def _hook(self, module, args, output):
        if self.S is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.size(0) != self.S.size(0):
            raise RuntimeError("batch (%d) != memórias (%d)"
                               % (h.size(0), self.S.size(0)))
        out = self.reader(h.float(), self.S.float()).to(h.dtype)
        if isinstance(output, tuple):
            return (out,) + tuple(output[1:])
        return out

    def __enter__(self):
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
        self.S = None
