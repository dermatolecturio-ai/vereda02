# M3 — Ensino de texto cru (o muro E4 do V1, agora com vantagem)

Desenho congelado ANTES de treinar (P1). Receita herdada do V1 em
`herdados/licoes_v1/04_aprendizados.md` §9-10 (P3 interno, já paga).
Complemento externo em `research/m3_literatura.md` (P3 externo). Data:
2026-07-07.

## Portão (números do ROADMAP.md, travados)

1. **Extração**: EM ≥0.90 em fraseados held-out com valores de tamanho
   variável (span exato: entidade E valor corretos, normalizados).
2. **Fim-a-fim**: texto cru → extração → M (cabeça M1) → resposta (injetor T,
   M2) ≥0.85 @ k=8.
3. n≥1024, ablação 3 partes (CARTA §3), S3/S4 ativos.

## Critério de morte (o mais importante deste marco)

O V1 (núcleo byte-level treinado do zero) chegou a um teto honesto de
**~0.59** em extração de valor de tamanho variável, mesmo depois de aplicar
a correção completa (complementaridade + marker-dropout + bidirecional +
early-stop). A aposta do V2 é que USAR REPRESENTAÇÕES DE UM Θ QUE JÁ ENTENDE
PT-BR (Qwen, pré-treinado) supera isso claramente. Se o M3 não bater bem
acima de 0.59 (a barra prática: ≥0.90 do próprio portão já é a prova), a
vantagem do substrato pré-treinado para extração era ilusória — parar,
registrar, deep research antes de insistir em mais dado/mais passos.

## O problema novo do V2: Qwen é causal, o V1 já sabia que causal falha

Lição herdada #10: encoder causal quebra quando o valor aparece ANTES da
entidade na frase (0.60→0.75 ao virar bidirecional). Qwen é decoder-only
(causal) — mesmo risco, origem diferente (arquitetura do pré-treino, não do
nosso encoder).

**Decisão**: um refinador bidirecional PEQUENO e TREINÁVEL (2 camadas de
self-attention SEM máscara causal, d=256) por cima dos estados de token do
Qwen (congelados). Os ponteiros leem do refinador. Isso implementa a lição
#10 sem tocar em Θ (`research/m3_literatura.md` — o token final de Qwen já
agrega a frase inteira por atenção causal; o refinador propaga essa
informação de volta para os tokens anteriores, o que a causalidade pura do
Qwen impede).

## Arquitetura

```
frase PT-BR → Qwen (congelado, hidden states por token, L×896)
            → Refinador bidirecional (2 camadas, treinável, d=256)
            → 4 ponteiros (início/fim × entidade/valor), softmax sobre L
```

Extração da entidade E do valor da MESMA frase, num único forward. Nenhum
parser/regex decide o span — só argmax sobre logits aprendidos.

## Perdas (reconstrução fiel da receita do V1; fórmula exata não sobreviveu
no código herdado, então esta é a operacionalização usada aqui — registrado
com honestidade)

- **CE**: cross-entropy padrão nos 4 ponteiros (início_e, fim_e, início_v,
  fim_v) contra os índices de token corretos.
- **Mutual-exclusivity**: para cada posição, `m_ent(pos) ≈ F_início_e(pos) ×
  F_fim_e(pos)` (probabilidade de que um span amostrado das marginais início/
  fim cubra `pos`; aproximação padrão independente). Idem `m_val`. Perda =
  `mean(m_ent × m_val)` — penaliza os dois spans se sobrepondo.
- **Coverage**: BCE entre `clip(m_ent + m_val, 0, 1)` e a máscara-união
  verdadeira (1 onde a posição pertence à entidade OU ao valor, 0 fora) —
  operacionaliza "valor = o que NÃO é a entidade, mas juntos cobrem o fato".
- **Marker-dropout**: com prob. p=0.3 por exemplo de treino, a frase-conector
  (a palavra/frase que liga entidade↔valor, ex. "é", "mora em", ":") é
  substituída por um preenchedor neutro sorteado (não correlacionado com
  nenhuma relação) — remove o atalho léxico fixo.

## Dataset (S3/S4 desde o dia 1, diversidade real de fraseado)

Ampliar de 3 para ~20 fraseados por atributo, cobrindo:
- ordem entidade-primeiro E valor-primeiro (ex.: "Recife é a cidade natal de
  Ana." vs "Ana nasceu em Recife.");
- conectores variados por template (catalogados, para o marker-dropout
  saber o que trocar);
- split train/held de TEMPLATES (não só de valores) — held-out precisa
  incluir fraseados nunca vistos, não só nomes/valores novos, para provar
  generalização estrutural (lição #5: diversidade por dimensão).

## Ablação (3 partes, CARTA §3)

1. Refinador + ponteiros com pesos aleatórios → EM deve desabar a ~acaso.
2. Θ sozinho (baseline A, já congelado em M0: 0.015) → referência de piso.
3. Caminho 100% aprendido: sem parser/regex decidindo span; ver
   `m3/extractor.py`.

## Fim-a-fim (texto→M→resposta, portão 2)

Reusa o injetor T do M2 (decisão já fechada): span de valor extraído vira o
"fato" injetado como texto no prompt, junto com a pergunta — mesmo caminho
medido no M2 (EM 0.94 @ k=8 quando o fato está certo). O gargalo novo é
puramente a EXTRAÇÃO; se extração ≥0.90 e o injetor T já preserva ~0.99 de
EM condicional (medido no M2), o fim-a-fim deve ficar perto de 0.90×0.99 ≈
0.89 — compatível com o portão ≥0.85.

## Protocolo de execução

1. Smoke local (CPU, n pequeno) valida mecânica; números não contam (S5).
2. Colab T4: treino do extrator (P6: medir s/passo no 1º checkpoint).
3. Run oficial n≥1024, extração held-out + fim-a-fim @ k=8, ablação.
4. `m3/resultados/RELATORIO_M3.md`; commit por experimento (P8); negativo no
   dia se algo falhar (P7).
