# VEREDA-Final 30M Implementation Report

Date: 2026-06-07

## Result

`vereda_final` foi implementado como arquitetura nova e treinavel, sem substituir `vereda_v2`.

Parametrizacao final:

- `architecture: vereda_final`
- `d_model: 384`
- `n_layers: 12`
- `head_size: 32`
- `patch_size: 4`
- `local_dim: 96`
- `ffn_multiplier: 1.0`
- `dropout: 0.05`
- parametros: `37,957,428`

## Modulos ativos por bloco

- patching byte-level fixo `P=4`
- projecoes ternarias `p_u`, `p_w`, `p_c`
- quantizacao adaptativa gamma
- CIF com acumulador e boundary reset
- CBH step-2 por scan sequencial treinavel
- reset ortogonal de `c`
- STP `stp_f * stp_x`
- projecao/fusao Lorentz via dendrito hiperbolico
- Dual-State fast vector + slow delta matrix memory
- ganho Kalman/RLS-style com covariancia diagonal
- memory injection com memoria densa + VSA textual deterministica
- residual output + channel FFN

## Dados

Manifesto final criado em `data/mixture_vereda_final.yaml`:

- `data/carolina_curated.bin`: `0.30`
- `data/gigaverbo.bin`: `0.20`
- `data/morphology_curriculum.jsonl`: `0.20`
- `data/long_memory_curriculum.jsonl`: `0.20`
- `data/conversation_curriculum.jsonl`: `0.10`

`data/pt_teste_5k.jsonl` ficou reservado para holdout externo.

## Validacao feita

- `python3 -m pytest`: `41 passed`
- `python3 research/experiments/math-validation/validate_math.py`:
  - `local_math_invariants_pass: true`
  - `lab_contribution_pass: false`
  - `pretraining_ready: false`
- probes locais adicionados para `vereda_final`:
  - state bounds CIF/STP/Kalman: passou
  - backward + VSA textual: passou
  - checkpoint equivalence: passou
  - topology tearing order: passou
- `python3 -m vereda.cli params --config configs/vereda_final_30m_5k.yaml`:
  - `parameters: 37957428`
- smoke tiny `vereda_final`: passou
- smoke com manifesto final + TBPTT: passou
- eval holdout override com `data/pt_teste_5k.jsonl`: passou
- memory-eval smoke: passou no caminho de execucao
- throughput benchmark 30M real por chunk: passou

## Bloqueio do treino 5000 steps nesta maquina

O probe 30M com a configuracao oficial falhou primeiro por OOM no MPS antes das otimizacoes de memoria.

Depois do checkpointing de ativacao, o modelo coube em memoria, mas 1 optimizer step oficial nao terminou em `583.99s` de relogio e foi encerrado manualmente.

Foi entao implementado backward streaming por subchunk TBPTT, para nao reter todos os grafos ate o fim do batch. Com checkpointing desligado por padrao, o probe 30M continuou sem OOM, mas ainda nao concluiu 1 optimizer step em `474.31s` de relogio e foi encerrado manualmente.

Como a configuracao oficial usa `batch_size=4`, `grad_accum_steps=8`, `seq_len=1024` e `tbptt_chunk_bytes=256`, extrapolar 5000 steps nessa maquina ainda nao e pratico para esta sessao. O bloqueio atual e tempo de computacao do loop patch-a-patch/bloco-a-bloco, nao completude matematica local.

## Benchmark de throughput 30M

Arquivo: `research/experiments/final-30m-5k/throughput-report.json`.

Figura: `research/experiments/final-30m-5k/throughput-estimate.png`.

Medicao: forward/backward real do modelo 30M em MPS, `batch_size=4`, `repeats=1`.

| TBPTT chunk bytes | segundos/chunk | segundos/optimizer step estimado | horas/5000 steps estimadas |
| --- | ---: | ---: | ---: |
| 32 | `2.26` | `579.06` | `804.25` |
| 64 | `3.90` | `499.43` | `693.65` |
| 128 | `6.44` | `411.94` | `572.13` |
| 256 | `25.12` | `803.97` | `1116.63` |

Decisao: `128` e o melhor chunk medido nesta maquina. A variante `configs/vereda_final_30m_5k_chunk128.yaml` foi criada para probes locais, mantendo a configuracao oficial original intacta.

## Melhorias prioritarias antes do treino oficial

1. Criar kernel/loop mais eficiente para `VeredaFinalBlock`, reduzindo overhead Python e projecoes ternarias repetidas.
2. Rodar probes locais com `configs/vereda_final_30m_5k_chunk128.yaml`.
3. Rodar o treino 5000 em GPU com mais memoria ou job persistente monitorado.
4. Gerar comparativo lab contra baseline para virar `lab_contribution_pass: true`.
