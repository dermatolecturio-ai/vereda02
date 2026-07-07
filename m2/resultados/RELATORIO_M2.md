# Relatório M2 — a voz: fato recuperado vira resposta gerada

Portão: EM fim-a-fim ≥ retrieval M1 − 0.05, mesmos itens do M0
(seed 1000+k, held). Juiz da régua, geração greedy. Braços:
T=texto (controle), S1=soft v1, S2=soft v2 (+recon), SR=ablação.

| braço | k | n | EM | retr top-1 | EM se retr ok | portão | tok prompt | s/item | acc pos 1º/2º/3º terço |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| S1 | 100 | 1024 | 0.041 | 0.941 | 0.043 | ❌ <0.869 | 52 | 0.08 | 0.07/0.03/0.03 |
| S1 | 32 | 1024 | 0.025 | 0.937 | 0.026 | ❌ <0.914 | 52 | 0.04 | 0.01/0.02/0.04 |
| S1 | 8 | 1024 | 0.040 | 0.944 | 0.041 | ❌ <0.939 | 52 | 0.02 | 0.04/0.04/0.04 |
| S2 | 100 | 1024 | 0.023 | 0.941 | 0.024 | ❌ <0.869 | 52 | 0.09 | 0.02/0.02/0.02 |
| S2 | 32 | 1024 | 0.022 | 0.937 | 0.024 | ❌ <0.914 | 52 | 0.04 | 0.03/0.03/0.01 |
| S2 | 8 | 1024 | 0.021 | 0.944 | 0.022 | ❌ <0.939 | 52 | 0.02 | 0.01/0.03/0.02 |
| SR | 100 | 1024 | 0.018 | 0.941 | 0.019 | — | 52 | 0.11 | 0.02/0.01/0.02 |
| SR | 32 | 1024 | 0.013 | 0.937 | 0.014 | — | 52 | 0.06 | 0.01/0.01/0.02 |
| SR | 8 | 1024 | 0.019 | 0.944 | 0.020 | — | 52 | 0.04 | 0.02/0.03/0.01 |
| T | 100 | 1024 | 0.938 | 0.941 | 0.994 | ✅ ≥0.869 | 83 | 0.09 | 0.96/0.95/0.91 |
| T | 32 | 1024 | 0.931 | 0.937 | 0.992 | ✅ ≥0.914 | 83 | 0.04 | 0.92/0.95/0.92 |
| T | 8 | 1024 | 0.940 | 0.944 | 0.993 | ✅ ≥0.939 | 83 | 0.02 | 0.95/0.94/0.92 |

## Portão M2

- S1: ❌ falhou em k=8,32,100.
- S2: ❌ falhou em k=8,32,100.
- T: ✅ passou o portão em todos os k.
- SR (ablação): EM k=8:0.019, k=32:0.013, k=100:0.018 — NÃO desabou (❌ investigar).

S6 (fluência): amostras de respostas auditáveis no campo `itens`
de cada JSON. Baseline A (Θ sozinho, CARTA §3.2): 0.015 (M0).
Caminho 100% aprendido (CARTA §3.3): sem parser/regex/banco — ver
m2/pipeline.py.

