# Relatório M3 — extração aprendida de texto cru

Portão 1 (extração, ROADMAP): EM >= 0.90 held-out, fraseados nunca vistos no treino, valores variáveis.

Portão 2 (fim-a-fim, ROADMAP): EM >= 0.85 @ k=8, texto cru -> extração -> M (chave M1) -> resposta (injetor T).

Morte: extração precisa bater CLARAMENTE o teto byte-level do V1 (~0.59, variável, ver NEGATIVE_FINDINGS.md/ROADMAP.md).

## Extração (held-out)

| arm | n | EM | EM entidade-primeiro | EM valor-primeiro | s/item | portão |
|---|---:|---:|---:|---:|---:|---|
| N | 1024 | 0.871 | 0.900 | 0.834 | 0.00 | ❌ <0.90 |
| SR | 1024 | 0.001 | 0.002 | 0.000 | 0.00 | — |

## Fim-a-fim (k=8, held-out)

| arm | k | n | EM | retr top-1 | EM se retr ok | tok prompt | s/item | portão |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| N | 8 | 1024 | 0.853 | 0.920 | 0.925 | 80 | 0.08 | ✅ ≥0.85 |
| SR | 8 | 1024 | 0.171 | 0.503 | 0.320 | 86 | 0.06 | — |

## Veredito

- Extração: ❌ falhou (EM=0.871, teto V1=0.59).
- Ablação (pesos aleatórios): EM=0.001 — desabou (✅ CARTA §3.1).
- Fim-a-fim @ k=8: ✅ passou (EM=0.853).

