# Resultado: validacao mecanica inicial

Data: 2026-06-04

Este resultado usa apenas um passo de treino nos modelos de laboratorio de 2M
parametros. Ele valida software, nao qualidade conversacional.

## Positivos

- RWKV-7 e Dual-State completaram forward, backward e checkpoint no MPS.
- Persistencia de sessao foi numericamente equivalente nas duas arquiteturas.
- Dual-State: 2.038.688 parametros; RWKV-7: 2.065.312 parametros.
- Configs de 30M: 33.293.232 e 33.605.040 parametros; pacote INT8 estimado
  abaixo de 64 MB.
- Memoria episodica lembrou, corrigiu e esqueceu o registro no teste de 10
  turnos.
- Exportacao INT8 do laboratorio Dual-State: aproximadamente 2,02 MB.

## Negativos

- A geracao depois de um passo ainda e texto degenerado, como esperado.
- Medida sobre bytes brutos, a validade UTF-8 foi aproximadamente `81,3%` na
  RWKV-7 e `83,3%` no Dual-State depois de um passo.
- O Dual-State nao recuperou `Z-731` do estado recorrente depois de 10 turnos.
- Na execucao completa de uma seed, um passo e horizonte 10, o ganho de memoria
  do Dual-State sobre RWKV-7 foi `0,0` ponto percentual e
  `contribution_pass=false`.
- A contribuicao VEREDA permanece nao demonstrada ate a comparacao completa
  com tres seeds e memoria em 10, 100 e 1000 turnos.

## Problema corrigido

O backend MPS abortava em views com stride zero criadas por `expand()`. Os
buffers usados pelo decoder local e pela injecao de memoria agora sao
materializados como contiguos.

Embeddings de memorias com comprimentos diferentes eram calculados sobre o
padding do lote. Agora cada embedding usa somente seus bytes reais, com teste
de regressao.

Memorias persistentes agora registram o hash dos pesos que geraram seu
embedding. A busca semantica ignora embeddings de outro checkpoint, ainda
permite recuperacao lexical e filtra resultados sem relevancia minima. Bancos
SQLite antigos sao migrados automaticamente. Quando uma busca lexical encontra
uma memoria de outro checkpoint, o chat recalcula um embedding temporario no
modelo atual; o nucleo recusa embeddings estrangeiros recebidos diretamente.

O identificador de compatibilidade deixou de usar o arquivo inteiro do
checkpoint, que muda com RNG e otimizador, e agora deriva somente dos pesos.

Checkpoints retomaveis agora preservam separadamente os cursores dos dados de
treino e validacao, com leitura retrocompativel do formato anterior.

A taxa de UTF-8 agora e medida sobre os bytes brutos gerados. Antes, bytes
invalidos eram descartados durante a conversao para texto e podiam produzir
uma taxa artificial de `1,0`.

## Artefatos

- `research/experiments/smoke/rwkv-report.json`
- `research/experiments/smoke/dual-state-report.json`
- `research/experiments/smoke/dual-state-memory.json`
- `research/experiments/smoke/full-protocol-seed-11.json`
