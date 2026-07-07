# Memoria VSA Kalman

## Pergunta
VSA textual e Kalman/RLS-style ajudam memoria sem destruir BPB?

## Hipotese
O candidato completo melhora memoria recorrente ou episodica sem piorar BPB e latencia.

## Baseline
`vereda_final_full`

## Variavel alterada
`vereda_final_full`: Baseline completo: VSA textual ligado, Kalman write ligado.; `no_kalman_write`: `use_kalman_write=false`; VSA textual permanece ligado.; `no_vsa_memory`: `use_vsa_memory=false`; embeddings densos continuam ativos.

## Artigos-alvo
- Arquitetura de Linguagem Byte-Level Inovadora
- Projeto VEREDA_ Nucleo SSM Cognitivo
- VEREDA_ Alfabetizacao Radical em Portugues

## Seeds e metricas
Seeds: `11, 22, 33`.

| Variante | BPB | bytes/s | UTF-8 | memoria recorrente |
| --- | --- | --- | --- | --- |
| vereda_final_full | 6.229 | 351.4 | 0.7293 | 0 |
| no_kalman_write | 6.2 | 445.3 | 1 | 0 |
| no_vsa_memory | 6.227 | 682 | 0.7585 | 0 |

![Memoria VSA Kalman](/Users/victorprudencio/veredassm2/research/experiments/article-ablations/figures/memory-vsa-kalman.png)

## Criterio de falsificacao
Rejeitar se uma variante sem VSA/Kalman for igual ou melhor em memoria e BPB.

## Resultado
inconclusivo

## O que aprendemos
A rodada local curta produz sinais comparativos, mas nao valida a tese como resultado final.
