# Catalogo matematico das pesquisas VEREDA

Data: 2026-06-07

Este catalogo religa os documentos originais, a implementacao atual e as
evidencias reproduziveis. A regra usada aqui e conservadora: uma tese
matematica so pode entrar no treino oficial quando houver uma prova local
verificavel ou uma ablacao empirica pareada que mostre ganho contra baseline.

Documento principal: `research/references/unified_vereda_theory.md`.

## Documentos relidos

- `Analise Critica do Sistema VEREDA.docx`
- `Pesquisa Arquitetura Nucleo VEREDA.docx`
- `Projeto VEREDA_ Nucleo SSM Cognitivo.docx`
- `Evolucao do VEREDA para Ensino Explicito de Portugues.docx`
- `VEREDA_ Alfabetizacao Radical em Portugues.docx`
- `VEREDA_ Hierarchical Temporal Sub-sampling Mechanism.docx`
- `Arquitetura de Linguagem Byte-Level Inovadora.docx`
- `VEREDA-v2_ Arquitetura Byte-Level Revolucionaria.docx`
- `VEREDA_ Hyperbolic SSM Derivation.docx`
- `Estabilizacao Geometrica de SSMs Ternarios.docx`
- `Hyperbolic Quantization for VEREDA Architecture.docx`
- `Neuronio Hiperbolico-Dendritico e Treinamento Preditivo.docx`
- `Roadmap Engenharia Software VEREDA.docx`

## Teorias aceitas na v0.1

| Teoria | Nucleo matematico | Evidencia atual | Status para treino |
| --- | --- | --- | --- |
| Patching fixo de bytes | Reduz passos recorrentes de `N` para `ceil(N/p)`; com `p=4`, o nucleo ve 25% dos passos byte-a-byte. | `FixedBytePatcher`; configs usam `patch_size: 4`; grafico `research/experiments/math-validation/patching_cost_curve.png`. | Incorporado. Manter como controle antes de patching dinamico. |
| Baseline RWKV-7-like | Recorrencia matricial com estado explicito, sem KV-cache. | `vereda/models/rwkv7.py`; comparacao multi-seed em `research/experiments/lab-comparison/summary.json`. | Incorporado como baseline, nao como contribuicao VEREDA. |
| Dual-State fast/slow | Estado rapido vetorial + memoria lenta associativa por erro de recuperacao. | `vereda/models/dual_state.py`; comparacao multi-seed executada. | Implementado, mas a hipotese foi refutada no laboratorio atual: 0 pp de ganho de memoria e BPB pior. |
| Injecao gated de memoria episodica | Contexto externo normalizado passa por gate treinavel antes de afetar patches. | `PatchLanguageModel._inject_memories`; testes de gradiente e compatibilidade por hash. | Incorporado mecanicamente; qualidade ainda depende de treino. |
| Persistencia de sessao | Estado serializado e vinculado a arquitetura, patch size e hash dos pesos. | Testes de roundtrip e comparacao numerica. | Incorporado; requisito minimo para chat local. |

## Teorias localmente validadas, mas ainda sem ganho de linguagem

| Teoria | Prova/teste atual | Resultado | Lacuna antes de treino oficial |
| --- | --- | --- | --- |
| Identidade do hiperboloide de Lorentz | `LearnableCurvatureLorentz` deve satisfazer `<x,x>_L=-1/K`. | Passou; erro maximo `1.04e-14`. | Precisa ablacao de qualidade: Euclidiano vs Lorentz, mesmo budget. |
| Camada dendritica hiperbolica | Saida de `DendriticNeuronLayer` deve ficar em `<y,y>_L=-1`. | Passou; erro maximo `4.44e-16`. | Prova somente geometria local, nao melhora de linguagem. |
| Operador CBH step-2 | `combine_states` deve ser associativo para scan paralelo. | Passou; erro maximo `9.54e-07`. | A implementacao usa bracket vetorial elementwise; isso nao prova sozinho uma imersao fiel em `SO(1,d)`. |
| CIF paralelo vs sequencial | Forward paralelo de `VeredaV2Block` deve bater com passos sequenciais. | Passou para comprimentos 2, 4, 8 e 16; erro maximo `1.33e-15`. | Falta mostrar que fronteiras aprendidas melhoram BPB/latencia em dados reais. |
| STE ternario mascarado | Gradiente so passa dentro de `|W| <= beta` com escala `tanh(tau W)`. | Passou; erro `0.0`. | Nao e prova Riemanniana completa; falta projecao tangente e teste de estabilidade em treino longo. |
| VSA/HDC quase-ortogonal | Hipervetores aleatorios normalizados devem ser quase ortogonais. | Passou; similaridade maxima `0.123`, media absoluta `0.0249`. | Falta tarefa de recuperacao simbolica integrada ao modelo gerador. |
| VEREDA-Final state bounds | CIF, STP e covariancia Kalman ficam em faixas finitas apos rollout local. | Passou; `cif` em `[0.189,0.375]`, `kalman_cov` em `[0.0405,0.0405]`, STP em `[0,1]`. | Falta treino longo e curva de estabilidade durante 5000 steps. |
| VEREDA-Final backward + VSA textual | Gradiente deve chegar em `p_u/p_w/p_c`, slow write, memoria densa e VSA. | Passou; todos os gradientes-alvo foram nao nulos. | Falta mostrar que VSA melhora recall factual contra memoria densa pura. |
| VEREDA-Final checkpoint equivalence | Checkpointing de ativacao nao pode mudar forward/backward. | Passou; erro de hidden e gradiente `0.0`. | E prova de engenharia, nao prova de qualidade. |
| Topology/flow loss | Sequencia com tearing persistente deve ser mais penalizada que sequencia suave. | Passou; smooth `0.0`, torn `0.0383`. | Falta ablation de coeficiente e custo no treino oficial. |

Evidencia reproduzivel: `python3 research/experiments/math-validation/validate_math.py`.

## Teorias ainda nao validadas

| Teoria | Onde aparece | Estado atual | Prova minima exigida |
| --- | --- | --- | --- |
| Patching dinamico por entropia | BLT, hierarchical sub-sampling, VEREDA-v2 | Nao implementado; v0.1 usa patch fixo. | Comparar patch fixo vs dinamico com mesmo compute, BPB, UTF-8 e latencia. |
| RLS/Kalman online | Projeto SSM Cognitivo, Alfabetizacao, Arquitetura Byte-Level | Implementado no `vereda_final` como ganho de escrita e covariancia diagonal por camada/head; prova local de bounds passou. | Teste de assimilacao/correcao online com baseline sem Kalman. |
| Lyapunov/spectral constraints | Hyperbolic SSM Derivation, SSM Cognitivo, Estabilizacao | Parcialmente retorico; nao ha matriz `A` com restricao espectral explicita no codigo atual. | Medir norma/raio espectral ou expoente local em sequencias 10/100/1000 turnos. |
| Ricci flow/topological loss | Estabilizacao Geometrica, Hyperbolic Quantization | Topology/flow loss leve implementada; Ricci flow completo ainda nao. | Ablation do coeficiente `0.05`, custo, gradiente e efeito em BPB/UTF-8. |
| Mamba-3 MIMO/exponential-trapezoidal | Alfabetizacao, Roadmap, Estabilizacao | Nao implementado. | Implementacao isolada ou decisao de escopo; comparar contra RWKV-7-like. |
| Hyperbolic quantization/power companding | VEREDA-v2 e Hyperbolic Quantization | Existe `AdaptiveGammaQuant`, sem benchmark independente. | Medir erro de reconstrucao, BPB, estabilidade e custo contra INT8/ternario simples. |
| Modern Hopfield/Lorentz memory | Hyperbolic Quantization e textos de memoria | Nao implementado como memoria neural. | Retrieval exato e correcao em 100/300/1000 turnos. |
| Losses de morfologia/concordancia | Ensino Explicito, Alfabetizacao | Curriculos existem; loss dedicada ainda nao esta integrada como criterio oficial. | Avaliar concordancia, flexao e denoising em holdout PT-BR. |
| 200M ternario completo | Estabilizacao e Roadmap | Ha export e artefatos edge, mas nao ha treino oficial validado. | Escalar somente apos v0.1 passar em 2M/7M/30M com baseline. |

## Decisao matematica atual

A versao mais recente esta mecanicamente rica, mas ainda nao esta validada como
modelo final. O caminho seguro para o treino oficial e:

1. Congelar v0.1 com patch fixo, baseline RWKV-7-like, Dual-State e memoria
   episodica gated.
2. Tratar VEREDA-v2 como laboratorio matematico, nao como candidato principal.
3. Promover uma teoria de cada vez somente se passar por prova local,
   ablacao pareada e grafico em `research/experiments/math-validation/`.
4. Recusar frases de "prova arquitetural completa" quando a evidencia cobre
   apenas invariantes locais.
