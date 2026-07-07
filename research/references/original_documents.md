# Catalogo dos documentos originais

Data da sintese: 2026-06-04

Os documentos originais permanecem em `~/Downloads`. Este catalogo registra o
aprendizado extraido deles sem tratar linguagem confiante como evidencia.

| Documento | Contribuicao util | Status na v0.1 |
| --- | --- | --- |
| Analise Critica do Sistema VEREDA | Separar fluencia, memoria auditavel e controle; medir riscos | Incorporado |
| Pesquisa Arquitetura Nucleo VEREDA | RWKV-7, patches fixos, estado persistente e testes sinteticos | Incorporado |
| Projeto VEREDA: Nucleo SSM Cognitivo | Gating, novidade, correcao e testes em 100/300/1000 turnos | Parcialmente incorporado |
| Evolucao para Ensino Explicito de Portugues | Curriculo PT-BR, UD e avaliacao morfossintatica | Planejado |
| Alfabetizacao Radical em Portugues | Denoising, unlikelihood e separacao de precisao | Catalogado |
| Hierarchical Temporal Sub-sampling | Relogios hierarquicos e patching dinamico | Ablacao futura |
| Arquitetura Byte-Level / VEREDA-v2 | Lie/CBH, limites dinamicos e quantizacao adaptativa | Ablacoes futuras |
| Hyperbolic SSM Derivation | Retracao e operador associativo proposto | Requer prova e benchmark independentes |
| Estabilizacao Geometrica / Hyperbolic Quantization | Lorentz, R-STE, Ricci e quantizacao | Alto risco; adiado |
| Neuronio Hiperbolico-Dendritico | Plasticidade curta e predictive coding local | Pesquisa de longo prazo |
| Roadmap Engenharia Software | Modularidade, TDD, laboratorio 2M e exportacao | Incorporado seletivamente |

## Regra de leitura

Afirmações de superioridade, estabilidade garantida ou "prova" arquitetural
nao sao aceitas sem implementacao reproduzivel, baseline pareada e medicao.
Fontes secundarias e referencias futuras devem ser verificadas antes de
fundamentar uma decisao.

## Catalogo matematico

As teorias matematicas extraidas dos documentos foram separadas por status em
`research/references/math_theory_catalog.md`: incorporadas, validadas apenas
localmente, ou ainda sem evidencia suficiente para entrar no treino oficial.

O documento principal de composicao agora e
`research/references/unified_vereda_theory.md`, que junta artigo por artigo em
uma teoria unica do VEREDA-Final e lista as obrigacoes de prova antes de cada
modulo entrar no modelo final.
