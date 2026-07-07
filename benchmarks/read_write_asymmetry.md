# Benchmark: assimetria leitura/escrita em decoder congelado (PT-BR)

Consolidado em 2026-07-07 a partir das células oficiais de M1–M4 (mesmo
substrato: Qwen2.5-0.5B-Instruct CONGELADO; mesma régua: juiz duro S1,
sentinelas S3/S4/S5, splits held; n≥1024 em toda linha citada). Este documento
NÃO introduz números novos — ele fixa a definição operacional dos três eixos e
aponta a célula oficial de onde cada número vem.

## A tese empírica (o que o V2 mediu que a literatura não tem nesta forma)

Num mesmo decoder congelado, com a mesma régua e os mesmos fatos:

1. **LER dos estados internos é fácil** — cabeças pequenas extraem com
   precisão alta o que o substrato representa;
2. **ESCREVER de volta pelo canal de input é difícil** — a mesma carga útil
   que a leitura domina não vira resposta quando injetada como vetor no input;
3. **o ENDEREÇAMENTO aprendido não generaliza de graça** — a chave que
   resolve atributos treinados falha em atributo nunca visto, e esse é um
   eixo separado dos outros dois.

A literatura tem cada peça isolada (arXiv:2603.22329 para 2; arXiv:2112.08348
para o mecanismo de 2; probing clássico para 1). O que este benchmark tem de
específico: os três eixos no MESMO substrato, mesma régua, mesmos fatos,
PT-BR, com ablação causal em cada linha — e, no eixo B, a decomposição
categoria vs binding documentada por amostra.

## Eixo A — leitura (probing/retrieval sobre estados congelados)

| medida | número | célula oficial |
|---|---:|---|
| Retrieval top-1 held @ k=8 (cabeça M1, ~0.1M params) | **0.989** | `research/results/m1_retrieval_report.md` |
| Retrieval top-1 held @ k=100 | **0.919** | idem |
| Extração entidade+valor de texto cru (M3, ~1.28M params) | **0.925** | `m3/resultados/RELATORIO_M3.md` |
| — teto byte-level do V1 na mesma tarefa | ~0.59 | `herdados/licoes_v1/` |
| Fim-a-fim texto→M→resposta @ k=8 (M3) | **0.899** | idem |
| Ablação (pesos aleatórios): extração | 0.001 | idem |

Veredito A: os estados do decoder congelado carregam o fato de forma
linear-extraível por cabeças de <1.3M params; a vantagem sobre aprender o
substrato do zero é +0.28 absoluto (0.925 vs 0.59).

## Eixo B — escrita pelo canal de input (o negativo limpo do M2)

Mesmos itens, mesma chave (retrieval 0.94), mesma carga útil (estados de
token do fato). A ÚNICA variável é a rota de entrada no Θ.

| rota | EM k=8 | EM k=32 | EM k=100 | célula |
|---|---:|---:|---:|---|
| Fato como TEXTO no prompt (T) | **0.940** | 0.931 | 0.938 | `m2/resultados/RELATORIO_M2.md` |
| — capacidade: T @ k=200 | 0.922 | — | — | idem |
| Soft prefix treinado, CE (S1) | 0.040 | 0.025 | 0.041 | idem |
| Soft prefix + recon (S2) | 0.021 | 0.022 | 0.023 | idem |
| Ablação: projetor ALEATÓRIO (SR) | 0.019 | 0.013 | 0.018 | idem |

S1 ≈ S2 ≈ SR: o projetor treinado é estatisticamente indistinguível de pesos
aleatórios. O modo de falha é qualitativo e documentado por amostra
(`NEGATIVE_FINDINGS.md`, 2026-07-07): a resposta vem SEMPRE do pool certo do
atributo (cor↔cor, senha com mesma forma) e quase sempre com o valor errado —
**categoria sem binding**. Análise completa: `research/m2_failure_analysis.md`.

## Eixo C — generalização de endereçamento (o negativo novo do M4)

A chave M1 foi treinada nos 6 atributos da régua; o override do M4 pergunta
por um atributo NUNCA visto ("a capital de X") — risco declarado no
`m4/DESIGN.md` §4.B antes de rodar.

| medida | número | fonte |
|---|---:|---|
| Retrieval em atributo held-out | **0.724** | `m4/resultados/M4_override_N_k8_n1024.json` |
| — em atributos treinados (referência) | 0.93–0.99 | M1/M2 |
| EM contrafactual global | 0.710 (portão ❌ <0.90) | `m4/resultados/RELATORIO_M4.md` |
| **EM condicionado a retrieval ok** (n=741) | **0.978** | decomposição por item do JSON acima (campos `retr_ok`/`correto`) |
| **eco do prior condicionado a retrieval ok** | **0.005** | idem |
| EM / eco quando o retrieval FALHA (n=283) | 0.007 / 0.198 | idem |
| Ablação SR (chave aleatória) | EM 0.267 — desabou | `M4_override_SR_k8_n1024.json` |

Veredito C: **o prior do Qwen não é teimoso — quando endereçado, o
contrafactual ensinado vence o prior em 97.8% com eco de 0.5%.** Todo o
déficit do portão 4.B mora na generalização da chave para atributo fora do
treino. "Override" e "endereçamento" são eixos separados, e só o segundo
falhou. Registro: `NEGATIVE_FINDINGS.md` (2026-07-07).

## A barra que o benchmark impõe a qualquer mecanismo novo de escrita

Qualquer alegação de "canal de escrita não textual" sobre este substrato deve
ser medida nestas condições (as mesmas de tudo acima) e superar:

| exigência | número | por quê |
|---|---|---|
| EM held, k=8, n≥1024, fato NUNCA no prompt | ≥ 0.80 | portão F1 do knife (`m5/DESIGN.md`) |
| Margem sobre o melhor baseline não textual (kNN-LM/logit) | ≥ +0.05 | senão o mecanismo não paga o próprio custo |
| Ablação da memória (zerada, chaves embaralhadas, pesos aleatórios) | EM ≤ 0.10 | CARTA §3 — sem colapso, não é memória |
| Teto de referência textual (braço T) | 0.940 | distância honesta do teto |
| Piso (Θ sozinho, baseline A) | 0.015 | distância honesta do piso |

### Primeira tentativa contra a barra: o knife M5a (morreu) — 2026-07-07

Célula `m5/resultados/RELATORIO_M5_KNIFE.md` (n=1024, held, mesmos itens):

| braço | rota | EM | endereç. implícito |
|---|---|---:|---:|
| T (texto, reuso) | teto | 0.940 | — |
| K (kNN-LM, λ=0.5 τ=0.05) | saída | 0.215 | — |
| S (soft prefix, reuso) | input | 0.040 | — |
| **I (delta, camada 12)** | **meio** | **0.002** | **0.114 ≈ acaso (1/8)** |
| IZ / IS / IR (ablações) | — | 0.018 / 0.004 / 0.011 | ~0.11 |

Veredito: MORTE (F1 0.002 ≪ 0.80). A rota interna falhou uma etapa ANTES do
M2: o endereçamento implícito ficou em acaso — a query aprendida (W_q sobre a
camada 12) nunca aterrissou na chave M1 certa. 61.6% das respostas do braço I
caem na categoria certa do gold (assinatura "categoria sem binding"), mas essa
categoria vem do PRIOR do Qwen respondendo à pergunta, não da memória (IZ ≥ I:
a injeção é ruído). Detalhe e próxima hipótese: `NEGATIVE_FINDINGS.md`
(2026-07-07). **Achado colateral medido:** o piso não textual desta tarefa
(exact-match com valores literais, incl. senhas aleatórias) é genuinamente
baixo — kNN-LM, o "plano B trivial" da literatura, só chega a 0.215.

Consequência para o eixo B: a assimetria leitura/escrita se ESTREITA. Não é só
"o input é o canal errado" (M2); é "sem endereçamento explícito resolvido, a
rota interna também não escreve". As duas rotas não textuais testadas (input S,
meio I) morreram; a única escrita que funciona neste substrato continua sendo
o fato como TEXTO (T, 0.940) — e o produto (M4) usa exatamente isso.

## Reprodução

```
python3 -m m2.run  --device cuda --dtype float16            # eixo B (células T/S/SR)
python3 -m m3.run  --device cuda --dtype float16            # eixo A (extração/fim-a-fim)
python3 -m m4.run  --device cuda --dtype float16            # eixo C (override/decomposição)
python3 -m m5.knife_test --do run --device cuda --dtype float16   # a barra, braços K/I
```

A decomposição do eixo C sai dos campos por item (`retr_ok`, `correto`,
`ecoou_prior`) de `m4/resultados/M4_override_N_k8_n1024.json`.
