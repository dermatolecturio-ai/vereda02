# P3 — Literatura para o M3 (extração aprendida de texto cru)

Data: 2026-07-07. A receita central do M3 já é HERDADA e provada pelo V1
(`herdados/licoes_v1/04_aprendizados.md` §9-10, `03_o_que_nao_funcionou.md`).
Esta nota é o complemento externo, focado no ponto novo do V2: extrair sobre
representações de um LM CAUSAL pré-treinado (Qwen), não sobre um núcleo
byte-level treinado do zero.

## A receita herdada (não é P3 externo — é P3 interno, já pago)

1. **Ponteiro (início+fim), não tagging por posição.** V1 testou os dois:
   tagging por byte derrubou a entidade (0.56, fronteira ambígua); ponteiro
   ganhou. Decisão fechada: M3 usa ponteiros.
2. **Complementaridade diferenciável**: "valor = o span que NÃO é a
   entidade", via perda de coverage + mutual-exclusivity. Sem isso, o modelo
   decora o marcador léxico (atalho) e não generaliza.
3. **Marker-dropout**: remover o atalho (a palavra-conectora, ex. "é", "mora
   em") em parte do treino, forçando sinal estrutural.
4. **Contexto bidirecional > causal** para extração: o valor às vezes aparece
   ANTES da entidade na frase; um encoder causal puro erra esse caso
   (0.60→0.75 ao virar bidirecional).
5. **Early-stop é obrigatório**: held-out sobe a 0.828 e despenca a 0.562 por
   overfitting do ponteiro de fim.
6. **Teto honesto do V1** (byte-level, tamanho variável, pós-correção):
   **~0.59** — é o número que o M3 precisa bater CLARAMENTE (critério de
   morte do ROADMAP) para justificar usar Θ em vez de um núcleo do zero.

## O que é novo no V2: causalidade do substrato

Qwen2.5 é decoder-only (causal): a representação do token na posição i só viu
tokens ≤i. Isso é o MESMO problema que o V1 relata na lição #10 (causal
quebra no caso "valor-antes-da-entidade"), mas a origem é diferente (aqui é
arquitetura do substrato pré-treinado, não do nosso próprio encoder).

- Um levantamento (arXiv:2605.14449, detecção de alucinação via hidden
  states) usa exatamente o fato de que "a atenção causal garante que o TOKEN
  FINAL agrega informação de todos os tokens anteriores" — ou seja, o último
  token de uma frase Qwen já é bidirecional-por-agregação em relação ao resto
  da frase, mesmo sem um encoder bidirecional dedicado.
- Um estudo sobre representações de entidades em LLMs autorregressivos
  (arXiv:2510.09421) mostra que o ÚLTIMO TOKEN de uma menção NÃO captura bem
  todos os tokens da menção — a média sobre os tokens da menção decodifica
  melhor. Ou seja: usar o último token da FRASE como sinal global é bom;
  usar o último token de um SPAN como representante do span inteiro é ruim.
- SpanBERT (boundary representations) reforça: span = par (início, fim),
  não um único token — consistente com a decisão herdada do V1 (ponteiro).

## Decisão de mecanismo para o M3

Em vez de reescrever Qwen como bidirecional (caro, quebraria o congelamento),
adiciono um **refinador bidirecional pequeno e treinável** (2 camadas de
self-attention SEM máscara causal, dim reduzida) por cima dos estados de
token do Qwen (congelado). Esse refinador implementa a lição #10 (contexto
bidirecional) sem tocar em Θ. Os ponteiros (início/fim × entidade/valor) leem
do refinador, não direto do Qwen — reconciliando "Qwen já entende PT-BR" com
"extração precisa ver a frase inteira nos dois sentidos".

## Fontes
- Representações de entidade em LLMs autorregressivos: https://arxiv.org/pdf/2510.09421
- Agregação causal no token final (hidden states): https://arxiv.org/pdf/2605.14449
- SpanBERT (span boundary representations): via IJCAI 2024 span-based NER survey, https://www.ijcai.org/proceedings/2024/0708.pdf
- Memória treinável em LLM decoder-only congelado (contexto do M2, revisitado): https://arxiv.org/html/2603.22329
