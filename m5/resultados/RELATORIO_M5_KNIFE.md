# Relatório M5 — teste de faca (portões F1–F4 do DESIGN.md)

Mesmos itens do M0/M2 (seed 1008, held, n=1024). T e S reusados das
células oficiais do M2. Fato NUNCA no prompt em K/I/IZ/IS/IR.

| braço | EM | endereç. top-1 | EM se endereç. ok | tok prompt | s/item | origem |
|---|---:|---:|---:|---:|---:|---|
| T | 0.940 | 0.944 | 0.993 | 83 | 0.02 | M2 (reuso, mesmos itens) |
| S | 0.040 | 0.944 | 0.041 | 52 | 0.02 | M2 (reuso, mesmos itens) |
| K | 0.215 | — | — | 44 | 0.65 | knife |
| I | 0.002 | 0.114 | 0.000 | 44 | 0.04 | knife |
| IZ | 0.018 | 0.114 | 0.043 | 44 | 0.06 | knife |
| IS | 0.004 | 0.114 | 0.009 | 44 | 0.04 | knife |
| IR | 0.011 | 0.098 | 0.020 | 44 | 0.06 | knife |

## Portões

- F1 (canal: EM do braço I ≥ 0.80): 0.002 ❌
- F2 (margem ≥ +0.05 sobre melhor não textual, 0.215): -0.213 ❌
- F3 (ablações IZ/IS/IR ≤ 0.10): IZ=0.018, IS=0.004, IR=0.011 ✅ desabou

**Veredito: MORTE da hipótese M5a nesta forma (DESIGN.md) — registrar em NEGATIVE_FINDINGS.md no dia, com o diagnóstico voz vs endereçamento (coluna endereç. top-1)**

- F4 (edição/esquecimento, n=503): EM novo=0.000, eco antigo=0.000, eco pós-rebuild=0.004 ❌

Sentinelas: S1 juiz da régua; S3/S4 por construção (terços nos JSONs); S5 n=1024; S6 amostras em `itens`. Baseline A (Θ sozinho): 0.015.

