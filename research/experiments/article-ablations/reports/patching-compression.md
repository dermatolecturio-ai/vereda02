# Patching Compressao

## Pergunta
Como o tamanho fixo de patch muda custo recorrente e qualidade local?

## Hipotese
Patch maior reduz passos recorrentes, mas pode piorar BPB ou UTF-8.

## Baseline
`patch_size_4`

## Variavel alterada
`patch_size_1`: `patch_size=1`; decoder ainda fixo por tamanho.; `patch_size_2`: `patch_size=2`; decoder ainda fixo por tamanho.; `patch_size_4`: `patch_size=4`; decoder ainda fixo por tamanho.; `patch_size_8`: `patch_size=8`; decoder ainda fixo por tamanho.

## Artigos-alvo
- Arquitetura de Linguagem Byte-Level Inovadora
- VEREDA-v2_ Arquitetura Byte-Level Revolucionaria
- VEREDA_ Hierarchical Temporal Sub-sampling Mechanism

## Seeds e metricas
Seeds: `11, 22, 33`.

| Variante | BPB | bytes/s | UTF-8 | memoria recorrente |
| --- | --- | --- | --- | --- |
| patch_size_1 | 5.075 | 218.1 | 1 | - |
| patch_size_2 | 5.173 | 422.7 | 1 | - |
| patch_size_4 | 5.121 | 814.1 | 1 | - |
| patch_size_8 | 4.938 | 1406 | 1 | - |

![Patching Compressao](/Users/victorprudencio/veredassm2/research/experiments/article-ablations/figures/patching-compression.png)

## Criterio de falsificacao
Rejeitar ganho de compressao se BPB/UTF-8 piorarem alem do controle local.

## Resultado
inconclusivo

## O que aprendemos
A rodada local curta produz sinais comparativos, mas nao valida a tese como resultado final.
