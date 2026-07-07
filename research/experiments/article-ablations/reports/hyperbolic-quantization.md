# Quantizacao Hiperbolica

## Pergunta
Adaptive gamma quantization e estavel e util em treino curto?

## Hipotese
Quantizacao adaptativa mantem gradientes finitos sem piorar BPB/latencia.

## Baseline
`quant_on`

## Variavel alterada
`quant_on`: Quantizacao adaptativa ligada.; `quant_off`: `use_adaptive_gamma_quant=false`.

## Artigos-alvo
- Estabilizacao Geometrica de SSMs Ternarios
- Hyperbolic Quantization for VEREDA Architecture

## Seeds e metricas
Seeds: `11, 22, 33`.

| Variante | BPB | bytes/s | UTF-8 | memoria recorrente |
| --- | --- | --- | --- | --- |
| quant_on | 5.121 | 835.2 | 1 | - |
| quant_off | 5.122 | 817 | 1 | - |

![Quantizacao Hiperbolica](/Users/victorprudencio/veredassm2/research/experiments/article-ablations/figures/hyperbolic-quantization.png)

## Criterio de falsificacao
Rejeitar se `quant_off` melhorar BPB/latencia sem custo de estabilidade.

## Resultado
inconclusivo

## O que aprendemos
A rodada local curta produz sinais comparativos, mas nao valida a tese como resultado final.
