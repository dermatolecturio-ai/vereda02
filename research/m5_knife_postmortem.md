# Post-mortem mecanístico do knife M5a — por que morreu e o que isso decide

Data: 2026-07-07, mesmo dia do veredito (`m5/resultados/RELATORIO_M5_KNIFE.md`,
morte por F1=0.002). Este documento NÃO reabre o veredito — ele explica o
mecanismo da falha com medições novas sobre os artefatos oficiais (checkpoint
`modelos/vereda2_m5_knife.pt`, células n=1024) e reduz o espaço das próximas
hipóteses. Diagnósticos, não alegações: nada aqui passa por portão.

## 1. O que o leitor treinado aprendeu de fato (anatomia do checkpoint)

| medida | valor | leitura |
|---|---:|---|
| Respostas idênticas I vs IZ (memória viva vs zerada) | **0/1024** | a injeção muda TODAS as respostas — o leitor não aprendeu a se calar |
| ‖W_m‖ (zero-init) | 9.21 | cresceu muito: injeta sinal FORTE |
| cos médio entre queries q de perguntas distintas | **0.993** | **a query COLAPSOU numa direção única** — lê a mesma mistura da memória para toda pergunta |
| gate: ‖w[h]‖ vs ‖w[m]‖ | 0.936 vs 0.086 | o gate decide por posição/contexto e IGNORA o conteúdo da memória |
| viés do gate (init −2.0) | −1.96 | nunca precisou abrir de verdade |

Mecanismo da morte, em uma frase: **sem sinal de alinhamento direto, a CE
através de 12 blocos congelados encontrou o ótimo local degenerado — query
constante → leitura constante → W_m injeta um viés forte e não-informativo,
que atrapalha em vez de ajudar (EM I=0.002 < IZ=0.018 < baseline A=0.015).**
É por isso que F3 "desabou" vacuamente: não havia informação para ablacionar.

## 2. A pergunta decisiva: a informação de endereçamento EXISTE na camada 12?

Sonda ridge supervisionada (diagnóstico; fit em 256 perguntas de TREINO com
alvo = chave M1 da pergunta; avaliação top-1 contra as chaves dos 8 fatos em
128 itens held oficiais, seed 1008; melhor λ de 4 reportado):

| query usada | endereçamento top-1 (acaso 0.125) |
|---|---:|
| Oráculo: cabeça M1 na pergunta (atenção-pooled, camada final) | **0.961** |
| Sonda ridge: último token, camada 24 (crua / chat) | 0.703 / 0.641 |
| Sonda ridge: último token, camada 12 (crua / chat) | **0.539** / 0.453 |
| W_q TREINADO no knife (último token, camada 12, chat) | 0.109 |

Três conclusões, em ordem de importância:

1. **A falha foi o SINAL DE TREINO, não ausência total de informação**: uma
   sonda supervisionada tira o endereçamento de 0.11 para 0.54 na mesma
   entrada que o W_q treinado usou. Uma perda contrastiva q↔chave teria
   aprendido ALGO. Mas —
2. **0.54 é o teto aproximado do último-token na camada 12 — e 0.54 < 0.80.**
   A rota (a) na forma simples (alinhar q do último token por perda
   contrastiva) NASCE MORTA para o portão F1: mesmo com voz perfeita, EM ≤
   endereçamento. Até na camada FINAL o último token só dá 0.70. O que
   entrega 0.96 é a ATENÇÃO-POOLING da cabeça M1 sobre todos os tokens —
   endereçamento de qualidade exige pooling, não um estado pontual.
3. **A rota (b) sai fortalecida por eliminação**: o endereçamento que
   funciona (0.94–0.96) já existe, fora do backbone, e é a cabeça M1 que o
   M2/M4 já usam. A única incógnita genuinamente NÃO testada é a VOZ
   interna: com o slot CERTO selecionado explicitamente, uma leitura em
   camada intermediária transmite o VALOR à geração? O knife M5a não testou
   isso — o endereçamento quebrado contaminou tudo rio abaixo.

## 3. Braços K e I por atributo (contexto do piso não textual)

| atributo | EM I | EM IZ | EM K |
|---|---:|---:|---:|
| a cidade natal | 0.006 | 0.050 | 0.422 |
| o nome da professora | 0.006 | 0.006 | 0.352 |
| a cor do carro | 0.000 | 0.045 | 0.230 |
| a comida favorita | 0.000 | 0.000 | 0.137 |
| o apelido do gato | 0.000 | 0.000 | 0.125 |
| a senha do wifi | 0.000 | 0.000 | **0.006** |

kNN-LM por tipo de valor: palavra-única 0.247, multi-palavra 0.254,
**senha 0.007**. O piso não textual (0.215 agregado) é sustentado por valores
de vocabulário fechado e desaba no literal — reproduzir uma string arbitrária
de 5–10 tokens exige acertar TODOS os passos com a memória dominando cada um.
O braço T faz 0.94 nisso. A régua com senhas procedurais (S3) é exatamente o
que separa "canal que carrega categoria" de "canal que carrega binding".

## 4. O que isso decide (e o que continua sendo decisão do dono)

- **Morta com causa conhecida**: M5a como desenhado (query aprendida por CE,
  último token, camada 12). Não reabrir com "mais passos" — o colapso da
  query é estrutural, não subtreino.
- **Morta por teto medido**: rota (a) simples (contrastiva no último token da
  camada 12) — teto ~0.54 < F1.
- **Viva e mais estreita**: rota (b) — endereçamento explícito pela cabeça M1
  (0.96) + voz interna lendo o valor do slot selecionado. Um knife futuro
  testaria SÓ a voz (uma variável), com os mesmos portões F1–F3. Custo: menor
  que o M5a (sem aprender endereçamento).
- **Viva e sem código novo**: parar a linha de escrita não textual aqui e
  fechar o pacote M0–M4 + benchmark (com DUAS rotas não textuais mortas e o
  piso kNN medido, o negativo ficou mais forte, não mais fraco).

Qualquer uma das duas últimas exige DESIGN novo e "vai" explícito (regra do
projeto). Este post-mortem só garante que a escolha será feita com o mecanismo
nomeado, não com um "não funcionou" genérico.
