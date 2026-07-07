# Sintese critica das pesquisas originais

Data: 2026-06-04

## Convergencias fortes

- Bytes exigem compressao temporal; patches fixos de quatro bytes sao o
  primeiro controle simples e comparavel.
- Estado recorrente serializavel resolve continuidade aproximada com custo
  constante, mas nao substitui memoria episodica exata.
- Correcao, esquecimento e versionamento devem ser operacoes explicitas.
- O nucleo precisa aprender fluencia por backpropagation antes de qualquer
  promessa de ensino puramente local.
- Toda inovacao deve enfrentar baseline, seeds, BPB, memoria e latencia.

## Ideias promissoras, ainda nao demonstradas

- Patching dinamico por entropia pode economizar relogios recorrentes.
- Losses de morfologia, concordancia e denoising podem ajudar um modelo pequeno.
- Adaptacao local via RLS/Kalman pode ser util em modulos pequenos e isolados.
- Geometria hiperbolica pode ajudar representacoes explicitamente hierarquicas.
- Quantizacao periferica pode reduzir o pacote final sem destruir o estado.

## Alertas

- Varios textos confundem plausibilidade matematica local com prova de que uma
  arquitetura de linguagem completa funcionara.
- Lorentz, Lie/CBH, Ricci, R-STE, Mamba-3, Hopfield e Kalman aparecem juntos;
  isso torna impossivel atribuir ganhos e multiplica riscos numericos.
- Afirmar que a geometria "ensina" gramatica ou que filtros substituem
  pre-treinamento e forte demais sem evidencia.
- Ternarizar cedo pode eliminar capacidade antes de a fluencia existir.
- Memoria externa sem treino do adaptador vira apenas RAG com outro nome.

## Decisao v0.1

Implementar apenas o conjunto minimo falsificavel: patches fixos, baseline
RWKV-7-like, Dual-State, adaptador gated treinavel, memoria episodica
controlavel, checkpoints retomaveis e avaliacoes pareadas. As demais ideias
permanecem no catalogo para uma ablação por vez.
