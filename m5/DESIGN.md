# M5 v0 — Teste de faca: existe canal NÃO TEXTUAL interno para um fato escrito em inferência?

Desenho congelado ANTES de rodar (P1). Data: 2026-07-07.
Literatura (P3): `research/m2_literatura.md`, `research/m2_failure_analysis.md`
e o relatório de deep research (2026-07-07). Pilares verificados na fonte:

- Retrofit de memória em decoder-only CONGELADO: arXiv:2603.22329 — em baixa
  capacidade (1×), só vencem mecanismos com prior de BINDING explícito
  (Hebbiano / slot-write / cross-attn paralela); prefix, KV-extension e ramo
  aditivo sem estrutura associativa falham (<0.4%). Em 10× todos convergem:
  o gargalo é arquitetural, não fundamental.
- Regra delta / fast weights: arXiv:2102.11174 (delta corrige o acúmulo
  Hebbiano); Gated DeltaNet arXiv:2412.06464; Gated DeltaNet-2
  arXiv:2605.22791 (separar apagar/escrever); OSDN arXiv:2605.13473
  (estabilidade com garantias).
- Baseline de saída: kNN-LM arXiv:1911.00172 (interpolação de logits).
- Por que o M2 falhou como falhou: Prompt Waywardness arXiv:2112.08348;
  compressão preserva tarefa, não conteúdo literal, arXiv:2412.17483.

## A pergunta única (e o que o knife NÃO alega)

> **M5a**: um decoder congelado pequeno consegue usar um fato novo, escrito em
> tempo de inferência numa memória associativa delta EXTERNA e lido por uma
> rota latente de camada intermediária, SEM o fato como texto no prompt — com
> a dependência provada por ablação?

O knife responde SÓ isso. Não alega internalização em pesos (M5b/ROME-style é
horizonte, fora deste teste), não alega produto (M4 é o produto), não alega
capacidade em k grande (fase 3, condicional). A memória continua sendo estado
externo materializável de um ledger — auditável e reconstruível por definição
(mesma propriedade arquitetural declarada no M4).

## Os quatro braços (mesmos itens do M0/M2: seed 1000+k, split held)

| braço | rota | fato no prompt? | treino novo? | origem |
|---|---|---|---|---|
| T | texto no prompt (controle/teto) | SIM | não | célula oficial M2 (reuso) |
| S | input, soft prefix (M2 v1) | não | não | célula oficial M2 (reuso) |
| K | saída, kNN-LM/interp. de logits | não | não (λ,τ por grid em TRAIN) | novo |
| I | MEIO: memória delta lida na camada 12 | não | ~0.6M params | novo (a hipótese) |

T e S NÃO são re-rodados: as células `m2/resultados/M2_{T,S1}_k8_n1024.json`
usam exatamente os mesmos itens (seed 1008, held) — reusá-las é mais honesto
do que re-rodar (zero risco de drift). K e I rodam frescos nos mesmos itens.

Simetria que o knife fecha: input (S, falhou no M2) × saída (K, baseline sem
risco) × meio (I, a aposta) × texto (T, teto). Uma tabela, quatro rotas.

## Braço I — escrita delta, leitura intermediária

Escrita (por forward, sem gradiente em inferência; ordem dos fatos = ordem do
item, posição já embaralhada S4):

    k_t = chave M1 do texto do fato (CONGELADA, 128-d, L2-norm — endereçamento provado a 0.94)
    v_t = ValueEncoder(estados de token do fato)          (aprendido, 896→256)
    S ← S + (v_t − S·k_t)·k_tᵀ                            (delta pura, η=1; S ∈ R^{256×128}, zero-init)

Leitura + injeção (aprendidas; hook após o bloco 12 de 24 do Qwen — 50% da
profundidade, TRAVADO antes de rodar; sem layer-shopping pós-hoc):

    q = L2norm(W_q h)            por posição               (896→128)
    m = S·q                                                 (leitura associativa)
    h̃ = h + σ(w_g·[h; LN(m)]) ⊙ W_m·LN(m)

Inicialização travada: `W_m` = zeros (passo 0 preserva o backbone: h̃ ≡ h);
viés do gate = −2 (abre devagar); LN(m) contra mismatch de escala. São
heurísticas da literatura, não teoremas — declarado como tal.

Por que esta forma e não outra: o braço fica na família VENCEDORA de
arXiv:2603.22329 (estado associativo com binding chave→valor explícito,
leitura re-consultada a CADA token gerado) e usa a regra delta porque ela
substitui em vez de acumular (é o que torna a edição um segundo write). O que
o M2 provou que NÃO funciona — vetor projetado no input sem estrutura — não é
reintroduzido aqui.

Sem retrieval top-1 no braço I: os k fatos são TODOS escritos em S; o
endereçamento é implícito na leitura (q deve aterrissar perto da chave certa).
É o teste de binding de verdade, não um pipeline com seleção externa.

## Braço K — kNN-LM sobre o episódio (piso não textual)

Datastore por item: pares (estado final normalizado, próximo token) dos k
fatos (forward do Θ cru sobre o texto de cada fato). Geração gulosa passo a
passo: p = (1−λ)·p_LM + λ·p_kNN, p_kNN = softmax(sim/τ) sobre os top-8
vizinhos por cosseno. ZERO parâmetros treinados. λ ∈ {0.3, 0.5, 0.7} e
τ ∈ {0.05, 0.2, 1.0} escolhidos por grid em n=96 itens de TREINO (seed 600)
ANTES de qualquer célula held — o grid fica em `m5/resultados/K_grid.json`.

## Treino do braço I (travado)

- Dados: episódios da PRÓPRIA régua, `build_items(k=8, n=6000, seed=500,
  split="train")` — held jamais visto.
- Módulos treinados: ValueEncoder + W_q + W_m + gate (~0.6M params). Qwen e
  cabeça M1 CONGELADOS.
- Perda: CE na resposta-ouro (teacher forcing), prompt e fato mascarados.
- 6000 passos, batch 12, AdamW lr 3e-4, seed 0. S2 teve 3000 passos no M2;
  o braço I tem canal + binding para aprender — 6000 é o orçamento, e a morte
  NÃO pode ser adiada com "mais passos" além dele.
- Termômetro a cada 500 passos (S5: não é manchete): EM em 128 itens held
  (seed 501) + EM com memória ZERADA (mini-IZ) + endereçamento implícito
  (top-1 de q̄ contra as chaves do item). Melhor checkpoint por EM →
  `modelos/vereda2_m5_knife.pt`.
- s/passo medido no 1º checkpoint, ETA à vista (P6).

Diagnóstico embutido (para atribuir falha SEM culpar o componente errado):
endereçamento implícito alto + EM baixo ⇒ gargalo na voz/capacidade do valor
(256-d por fato é compressão; risco declarado); endereçamento baixo ⇒ W_q não
mapeia camada 12 → espaço da chave M1 (gargalo de leitura). Os dois têm
próximos passos DIFERENTES — o knife só precisa dizer qual morreu.

## Portões (números travados AGORA)

| # | portão | número |
|---|---|---|
| F1 | canal: EM do braço I, k=8, held, n=1024, SEM fato no prompt | ≥ 0.80 |
| F2 | margem: EM(I) − max(EM(K), EM(S)) | ≥ +0.05 |
| F3 | ablação (CARTA §3): IZ memória zerada; IS chaves embaralhadas na escrita; IR módulos aleatórios | cada uma EM ≤ 0.10 |
| F4 | fase 2 — edição: re-write delta do alvo → EM novo ≥ 0.80, eco antigo ≤ 0.20; esquecimento: rebuild de S sem o evento → eco ≤ 0.20 | — |

Sentinelas: S1 juiz da régua; S3/S4 por construção do dataset (reportar EM por
terço de posição); S5 n=1024; S6 amostras auditáveis no JSON. Baseline A
(Θ sozinho) já congelado: 0.015.

## Critério de MORTE (honesto, sem apelação)

Morre a hipótese M5a NESTA FORMA se, na rodada travada (seed 0; se F1 cair em
0.70–0.80, repetir com seeds 1 e 2 e reportar dispersão — só isso):

- F1 < 0.80, ou
- F2 < +0.05 (o canal interno não paga o próprio custo sobre kNN-LM), ou
- F3 falha (o efeito não é da memória), ou
- o ganho só aparece com o fato textual reintroduzido em qualquer ponto.

Proibido: trocar camada/escala/gate depois de ver o resultado (vira tuning
narrativo); "só mais N passos"; repetir só o braço que "quase foi". Falhou →
`NEGATIVE_FINDINGS.md` no dia, com o diagnóstico (voz vs endereçamento) — e o
V2 segue valioso com M0–M4 + benchmark (o knife negativo É publicável como
evidência de assimetria; ver `benchmarks/read_write_asymmetry.md`).

## Fases e execução

```
python3 -m m5.knife_test --smoke                      # mecânica (CPU ok), números não contam
python3 -m m5.knife_test --do grid  --device cuda --dtype float16   # λ,τ do K (train)
python3 -m m5.knife_test --do train --device cuda                   # braço I (float32)
python3 -m m5.knife_test --do run   --device cuda --dtype float16   # células K, I, IZ, IS, IR (held, n=1024)
python3 -m m5.knife_test --do edit  --device cuda --dtype float16   # fase 2 (n=512), só se F1–F3 ✅
python3 -m m5.knife_test --do report                                # RELATORIO_M5_KNIFE.md (importa T/S do M2)
```

Fase 3 (condicional a F1–F3 ✅): células I em k ∈ {32, 100} — curva de
capacidade. Fora do veredito do knife; entra no desenho do M5 v1.

Um JSON por célula em `m5/resultados/` (retomável, house style). Commit com
config (P8). Tempo medido, não estimado (P6).
