# M4 — Memória viva completa: os 5 portões na barra cheia

Desenho congelado ANTES de rodar (P1). Herda os 5 portões de
`herdados/licoes_v1/VEREDA_M_SPEC.md` §4 e os números do ROADMAP.md.
Data: 2026-07-07.

Dependências: M1 (chave, fechado), M2 (voz via injetor T, fechado). NÃO
depende da extração do M3 (o ensino aqui entra como fato canônico; texto cru
é papel do M3/M6). A iteração de escala do M3 roda em paralelo no Colab.

## O que o M4 alega (e o que não alega)

A memória do V2 nesta fase é EXTERNA e explícita: lista de (texto do fato,
chave aprendida M1). Escrita/edição/esquecimento são operações sobre essa
lista; leitura é 100% aprendida (chave M1 → injetor T → Θ). Por construção,
Θ congelado + memória externa dão DE GRAÇA duas coisas que no V1 exigiam
cuidado (esquecer restaura o prior; persistência é serialização) — o
relatório declara isso como propriedade arquitetural, não como descoberta.
O que NÃO é de graça e precisa de número: capacidade em k grande, override
do prior, integridade de vizinhos sob edição, e o caminho pós-reinício
funcionando de ponta a ponta.

## Portões (números travados)

| # | portão | número |
|---|---|---|
| 4.A | Capacidade k=200: retrieval top-1 held | ≥0.90 |
| 4.A' | Capacidade k=200: EM fim-a-fim (injetor T) | ≥0.85 |
| 4.B | Override do prior: fato contrafactual ensinado VENCE o que Θ diria | EM contrafactual ≥0.90; eco do prior ≤0.05 |
| 4.C | Edição: valor novo responde | EM novo ≥0.90; eco do valor antigo ≤0.05 |
| 4.C' | Integridade de vizinhos após edição (roadmap) | ≥0.95 |
| 4.D | Persistência: EM pós-reinício de processo | ≥ EM mesmo-processo − 0.02 |
| 4.E | Ablação (CARTA §3): chave aleatória no override | desaba |

Tudo com n≥1024 (exceto onde declarado), posição embaralhada (S4), juiz da
régua (S1), split held (S3).

## 4.B — o experimento genuinamente novo: override do prior

O V1 (E3) provou override num núcleo próprio com portão aprendido. No V2, o
prior mora no Qwen pré-treinado: fatos de mundo real que Θ responde sozinho
(capitais de países/estados, em PT-BR). Protocolo:

1. **Prior baseline** (referência): Θ cru responde "Qual é a capital de X?"
   → mede-se em quais entidades o prior existe de fato.
2. **Override**: ensina-se o contrafactual ("Anote: a capital da França é
   Sobral.") na memória com k−1=7 distratores da régua, posição embaralhada.
   Pergunta → chave M1 → injetor T → resposta. Passa se a resposta segue o
   CONTRAFACTUAL, não o prior. Reporta-se também o EM só nas entidades onde
   o prior baseline acertou (o caso difícil de verdade).
3. **Esquecer restaura**: com Θ congelado e memória externa, remover o fato
   devolve exatamente o comportamento do passo 1 — por construção;
   declarado no relatório, com sanidade mínima medida na fase de
   persistência.

Risco conhecido e assumido: o atributo "a capital" NUNCA foi visto pela
chave M1 (treinada nos 6 atributos da régua) — retrieval aqui é
generalização de atributo held-out. `retr_ok` é reportado separado para
atribuir falhas com honestidade (endereçamento vs voz vs teimosia do prior).

## 4.C — edição com vizinhos

k=32 fatos; edita-se o VALOR do fato-alvo (texto e chave recomputados);
pergunta-alvo deve dar o valor NOVO (não ecoar o antigo); uma pergunta de
vizinho aleatório não-editado deve continuar certa (integridade ≥0.95).

## 4.D — persistência entre processos

Fase `escrever` (processo 1): constrói memórias de n itens k=8, salva
`.vereda` (torch.save de textos+chaves). Fase `ler` (processo 2, fresh):
carrega, responde, EM. Compara com o EM mesmo-processo da fase escrever.

## Capacidade (4.A) — reuso honesto

`python -m m2.run --arms T --k-list 200 --n 1024` — o runner do M2 já mede
exatamente isso (retrieval M1 + injetor T) em qualquer k. Sem código novo.
Morte (roadmap): colapso de capacidade em k grande → diagnóstico de
endereçamento (lição E1: conteúdo, não posição) antes de qualquer escala.

## Execução

1. Smoke local CPU (números não contam, S5).
2. Colab T4: `m2.run k=200` + `m4.run` (override, edição) + `m4.persist`
   (2 processos) + relatório.
3. `m4/resultados/RELATORIO_M4.md`; commit por experimento (P8); negativo
   no dia (P7).
