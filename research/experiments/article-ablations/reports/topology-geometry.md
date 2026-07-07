# Topologia Geometria

## Pergunta
Geometria leve estabiliza o candidato final em treino curto?

## Hipotese
Curvatura, topologia e dendrito Lorentz reduzem tearing sem custo proibitivo.

## Baseline
`geometry_full`

## Variavel alterada
`geometry_full`: Baseline geometrico: curvatura 0.1, topologia 0.05, Lorentz/dendrito ligado.; `topology_off`: `topology_loss_weight=0.0`.; `curvature_off`: `curvature_loss_weight=0.0`.; `no_lorentz_dendrite`: `use_lorentz_dendrite=false`.

## Artigos-alvo
- Estabilizacao Geometrica de SSMs Ternarios
- VEREDA-v2_ Arquitetura Byte-Level Revolucionaria
- VEREDA_ Hyperbolic SSM Derivation

## Seeds e metricas
Seeds: `11, 22, 33`.

| Variante | BPB | bytes/s | UTF-8 | memoria recorrente |
| --- | --- | --- | --- | --- |
| geometry_full | 5.121 | 846.1 | 1 | - |
| topology_off | 5.121 | 847 | 1 | - |
| curvature_off | 5.132 | 837.3 | 1 | - |
| no_lorentz_dendrite | 5.122 | 1016 | 1 | - |

![Topologia Geometria](/Users/victorprudencio/veredassm2/research/experiments/article-ablations/figures/topology-geometry.png)

## Criterio de falsificacao
Rejeitar se variantes sem esses termos tiverem BPB/estado melhores sem falhas.

## Resultado
inconclusivo

## O que aprendemos
A rodada local curta produz sinais comparativos, mas nao valida a tese como resultado final.
