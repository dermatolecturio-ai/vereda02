# M2 — Voz: o fato recuperado vira resposta fluente

Desenho congelado ANTES de treinar (P1). Literatura em
`research/m2_literatura.md` (P3). Data: 2026-07-07.

## Portão (números travados AGORA, derivados do M1 oficial)

EM fim-a-fim (juiz da régua, geração greedy) ≥ retrieval do M1 − 0.05,
nos MESMOS itens do M0 (seed 1000+k, split held, n=1024):

| k | retrieval M1 | portão M2 (EM) | referência B | referência C |
|---:|---:|---:|---:|---:|
| 8 | 0.989 | **≥ 0.939** | 0.781 | 0.460 |
| 32 | 0.964 | **≥ 0.914** | 0.581 | 0.371 |
| 100 | 0.919 | **≥ 0.869** | 0.463 | 0.345 |

Mais: fluência preservada (S6 — respostas em PT-BR natural, sem degeneração;
amostras auditáveis no JSON), ablação 3 partes (CARTA §3), n≥1024 (S5).

## Arquitetura da memória no M2 (decisão central: chave ≠ carga útil)

- **Chave** (endereçamento): cabeça M1 congelada (`modelos/vereda2_m1_head.pt`).
  Treinada por InfoNCE fato↔pergunta, codifica (nome, atributo). NÃO carrega o
  valor — e não precisa.
- **Carga útil** (conteúdo): estados de token do fato (40×896, do Qwen
  congelado) — os mesmos estados que a chave já usa. Nada novo a armazenar.
- Caminho: pergunta → chave M1 → top-1 sobre os k fatos → carga útil do fato
  vencedor → injetor → Θ gera a resposta. 100% aprendido, sem parser.

## A/B (P2/P5 — uma variável por vez)

| braço | injetor | treino | o que testa |
|---|---|---|---|
| T | fato recuperado como TEXTO no prompt (formato do baseline C) | nenhum | teto do Θ condicionado ao fato certo; controle |
| S-v1 | projetor → m=8 soft tokens (inputs_embeds) | CE na resposta-ouro | injeção por representação, forma pura |
| S-v2 | idem | CE resposta + CE reconstrução do fato (λ=1) | se a perda AE resolve a fidelidade literal (senhas), como prevê arXiv:2412.17483 |

Projetor: perceiver-style — m=8 latentes aprendidos, 1 bloco de cross-attention
sobre os estados de token do fato + FFN, projeção para d=896, escala calibrada
pela RMS da matriz de embeddings do Qwen. ~1M parâmetros treináveis. Qwen e
cabeça M1 CONGELADOS.

## Ablação (3 partes, CARTA §3)

1. Projetor com pesos aleatórios (mesma arquitetura) → EM deve desabar.
2. Θ sozinho: baseline A já congelado (0.015) → referência de piso.
3. Caminho 100% aprendido: sem parser/regex/banco em nenhum ponto (por
   construção; auditável no código).

Adicional diagnóstico: EM condicional ao retrieval correto (separa erro de
endereçamento de erro de voz).

## Custo (a tese do dispositivo leve)

Reportar tokens de prompt (S conta os m soft tokens) e s/item. Alvo: em k=100,
custo ~O(pergunta) + m tokens, contra 1865 tokens do B.

## Critério de morte (roadmap M2)

Se T passar mas S-v1 e S-v2 falharem: a voz por representação não fecha neste
substrato — registrar em NEGATIVE_FINDINGS.md, decidir se M2 fecha com T
(CARTA não proíbe texto recuperado aprendidamente; proíbe parser/BM25) ou se
reavaliamos decoder próprio. Se NEM T passar: o gargalo é o Θ 0.5B condicionado
— diagnóstico antes de qualquer escala.

## Protocolo de execução

1. Smoke local (CPU, n pequeno) valida mecânica; números NÃO contam (S5).
2. Colab T4: treino S-v1 e S-v2 (P6: medir s/passo no 1º checkpoint).
3. Run oficial n=1024, k={8,32,100}, braços T, S-v1, S-v2 + ablação.
4. `m2/resultados/RELATORIO_M2.md` com portão avaliado; commit por experimento
   (P8); negativo no dia (P7).
