# Portugues Explicito

## Pergunta
Regularizacao sintatica explicita ajuda concordancia PT-BR?

## Hipotese
`syntax_loss_weight=0.2` melhora concordancia sem piorar BPB alem do ruido local.

## Baseline
`syntax_on`

## Variavel alterada
`syntax_on`: Syntax loss ligada com peso 0.2.; `syntax_off`: `syntax_loss_weight=0.0`.

## Artigos-alvo
- Evolucao do VEREDA para Ensino Explicito de Portugues
- VEREDA_ Alfabetizacao Radical em Portugues

## Seeds e metricas
Seeds: `11, 22, 33`.

| Variante | BPB | bytes/s | UTF-8 | memoria recorrente |
| --- | --- | --- | --- | --- |
| syntax_on | 5.828 | 844.4 | 1 | - |
| syntax_off | 5.784 | 847.3 | 1 | - |

![Portugues Explicito](/Users/victorprudencio/veredassm2/research/experiments/article-ablations/figures/portuguese-explicit.png)

## Criterio de falsificacao
Rejeitar se `syntax_off` igualar ou superar concordancia e BPB.

## Resultado
inconclusivo

## O que aprendemos
A rodada local curta produz sinais comparativos, mas nao valida a tese como resultado final.
