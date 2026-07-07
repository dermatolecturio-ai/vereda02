# Relatório M1 — cabeça de memória aprendida

Avaliação held-out de retrieval top-1. Qwen está congelado; só a cabeça
de memória é treinada. A ablação usa uma cabeça aleatória com a mesma
arquitetura.

| k | Qwen mean-pool | cabeça aprendida | ablação aleatória |
|---:|---:|---:|---:|
| 8 | 0.528 | 0.989 | 0.187 |
| 32 | 0.357 | 0.964 | 0.101 |
| 100 | 0.241 | 0.919 | 0.038 |

## Portão M1

**Status: passou.**

Critérios do roadmap: `>=0.95 @ k=8`, `>=0.90 @ k=32`, vencer
baseline B em `k=100`, e ablação aleatória cair fortemente.
