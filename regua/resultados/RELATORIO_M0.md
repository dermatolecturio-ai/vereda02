# Relatório M0 — baselines congelados

Juiz: valor-ouro normalizado, com fronteira de palavra, na resposta
gerada (greedy, determinística). Dataset: held pools, seed fixa.

Nota: o baseline A não depende de `k`; por isso é medido uma vez
com o conjunto de perguntas `k=8`.

| baseline | k | n | acc | retr top-1 | tok prompt | s/item | acc pos 1º/2º/3º terço |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 8 | 1024 | 0.015 | — | 44 | 0.54 | 0.01/0.02/0.02 |
| B | 2 | 1024 | 0.994 | — | 100 | 0.22 | 1.00/0.99/- |
| B | 8 | 1024 | 0.781 | — | 202 | 0.39 | 0.84/0.69/0.83 |
| B | 32 | 1024 | 0.581 | — | 617 | 1.18 | 0.55/0.52/0.67 |
| B | 100 | 1024 | 0.463 | — | 1865 | 33.95 | 0.44/0.42/0.53 |
| C | 2 | 1024 | 0.772 | 0.771 | 83 | 21.42 | 0.77/0.77/- |
| C | 8 | 1024 | 0.460 | 0.447 | 83 | 1.31 | 0.48/0.45/0.45 |
| C | 32 | 1024 | 0.371 | 0.356 | 84 | 0.08 | 0.37/0.38/0.36 |
| C | 100 | 1024 | 0.345 | 0.321 | 84 | 0.15 | 0.34/0.34/0.35 |

## Portão M0

**Status: completo.** Todas as células obrigatórias existem.

Sentinelas aplicáveis no M0: S3 (valores variáveis — ver
`dataset_stats` nos JSONs), S4 (posição embaralhada + acc por terço),
S5 (n reportado), S6 (n/a — nada treinado). S1/S2 aplicam-se a
partir do M1 (módulos treinados).
