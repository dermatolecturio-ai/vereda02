# -*- coding: utf-8 -*-
"""Extrator aprendido do M3: entidade+valor via ponteiros sobre Θ (Qwen).

Qwen é CAUSAL (decoder-only): a representação de um token só viu tokens
anteriores. A lição herdada do V1 (04_aprendizados.md #10) mostrou que
causal quebra quando o valor aparece ANTES da entidade na frase. Correção
sem tocar em Θ: um refinador bidirecional PEQUENO e TREINÁVEL (self-attention
SEM máscara causal) sobre os estados congelados, antes dos ponteiros.

Só o refinador + os 4 ponteiros treinam (~1.3M params). Qwen nunca treina.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DIN = 896  # hidden do Qwen2.5-0.5B


class BidirectionalRefiner(nn.Module):
    def __init__(self, din=DIN, d=256, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(d, heads, batch_first=True),
                "ln1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                    nn.Linear(2 * d, d)),
                "ln2": nn.LayerNorm(d),
            }))
        self.d = d

    def forward(self, states, mask):
        # states: (B,L,DIN) float; mask: (B,L) bool, True=token válido
        x = self.inp(states)
        kpm = ~mask  # key_padding_mask: True = ignorar
        for blk in self.blocks:
            att, _ = blk["attn"](x, x, x, key_padding_mask=kpm)
            x = blk["ln1"](x + att)
            x = blk["ln2"](x + blk["ff"](x))
        return x  # (B,L,d) — SEM máscara causal: cada posição vê a frase toda


class PointerHeads(nn.Module):
    """4 ponteiros: início/fim × entidade/valor, sobre o refinador."""

    def __init__(self, d=256):
        super().__init__()
        self.start_e = nn.Linear(d, 1)
        self.end_e = nn.Linear(d, 1)
        self.start_v = nn.Linear(d, 1)
        self.end_v = nn.Linear(d, 1)

    def forward(self, refined, mask):
        neg = torch.finfo(refined.dtype).min
        def logits(head):
            l = head(refined).squeeze(-1)
            return l.masked_fill(~mask, neg)
        return (logits(self.start_e), logits(self.end_e),
               logits(self.start_v), logits(self.end_v))


class Extractor(nn.Module):
    def __init__(self, din=DIN, d=256, heads=4, layers=2):
        super().__init__()
        self.refiner = BidirectionalRefiner(din, d, heads, layers)
        self.pointers = PointerHeads(d)

    def forward(self, states, mask):
        refined = self.refiner(states, mask)
        return self.pointers(refined, mask)


def soft_span_membership(start_logits, end_logits):
    """m(pos) ~= P(início<=pos) * P(fim>=pos), aproximação independente.

    Usada só para as perdas de mutual-exclusivity/coverage (regularização),
    não para a predição final (essa usa argmax dos ponteiros).
    """
    p_start = F.softmax(start_logits, dim=-1)
    p_end = F.softmax(end_logits, dim=-1)
    cdf_start = torch.cumsum(p_start, dim=-1)          # P(início<=pos)
    cdf_end_rev = torch.cumsum(p_end.flip(-1), dim=-1).flip(-1)  # P(fim>=pos)
    return cdf_start * cdf_end_rev


def extraction_losses(logits, targets, mask, lam_excl=0.3, lam_cov=0.3):
    se, ee, sv, ev = logits
    tse, tee, tsv, tev = targets
    ce = (F.cross_entropy(se, tse) + F.cross_entropy(ee, tee)
         + F.cross_entropy(sv, tsv) + F.cross_entropy(ev, tev))

    m_ent = soft_span_membership(se, ee)
    m_val = soft_span_membership(sv, ev)
    excl = (m_ent * m_val).sum(-1).mean()  # penaliza sobreposição

    B, L = mask.shape
    pos = torch.arange(L, device=mask.device).unsqueeze(0)
    union_true = (((pos >= tse.unsqueeze(1)) & (pos <= tee.unsqueeze(1)))
                 | ((pos >= tsv.unsqueeze(1)) & (pos <= tev.unsqueeze(1))))
    union_pred = (m_ent + m_val).clamp(max=1.0)
    cov = F.binary_cross_entropy(
        union_pred[mask].clamp(1e-6, 1 - 1e-6), union_true[mask].float())

    return ce + lam_excl * excl + lam_cov * cov, {
        "ce": ce.item(), "excl": excl.item(), "cov": cov.item()}


def decode_spans(logits, mask, maxlen_span=12):
    """Argmax com fim>=início e fim-início<maxlen_span (evita spans absurdos)."""
    se, ee, sv, ev = [l.detach() for l in logits]
    B, L = se.shape
    tri = torch.full((L, L), float("-inf"), device=se.device)
    for i in range(L):
        tri[i, i:min(L, i + maxlen_span)] = 0.0

    def best(sl, el):
        score = sl.unsqueeze(2) + el.unsqueeze(1) + tri.unsqueeze(0)
        flat = score.view(B, -1)
        idx = flat.argmax(-1)
        return (idx // L), (idx % L)

    s_e, e_e = best(se, ee)
    s_v, e_v = best(sv, ev)
    return s_e, e_e, s_v, e_v
