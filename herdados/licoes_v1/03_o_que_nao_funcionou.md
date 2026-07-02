# O que NÃO funcionou (falsificações e erros)

Este é, talvez, o documento mais valioso: ele impede repetir becos sem saída.
Cada item foi **medido**, não suposto.

## Ideias originais que não se sustentaram como diferencial

| hipótese | veredito medido | lição |
|---|---|---|
| Geometria de Lorentz dá capacidade exponencial | inerte (ablação: não vence o baseline euclidiano) | geometria só vale se a curvatura *medir* melhor algo; aqui não mediu |
| Scan associativo CBH nilpotente | kernel existe, nunca foi ligado ao forward; sem ganho | não conta como resultado o que não está no caminho de execução |
| Neurônio dendrítico + CIF + STP | valor negativo (pioram vs baseline) | complexidade bio-inspirada sem ablação é dívida, não ativo |
| Quantização ternária como diferencial | neutra (preserva, não melhora) | compressão ≠ capacidade |
| "VEREDA bate o GRU em state-tracking" | não reproduziu (0.19 vs 0.36) | exigir config reprodutível antes de afirmar |

## Armadilhas de medição (autoengano) — as mais perigosas

1. **Sonda fraca = papagaio com nota alta.** Uma sonda de cognição 2-way deu
   `0.703`; no juiz duro (gerar a resposta, vocab novo) caiu para `0.008`.
   → **Lição:** para cognição, só vale exact-match gerando a resposta, com
   vocabulário held-out.

2. **BPB baixo em corpus repetitivo ≠ fluência.** Um currículo de templates deu
   BPB ~0.08 (falso); em PT real, esse mesmo modelo deu 8.64 (péssimo).
   → **Lição:** medir fluência só em PT natural, nunca em dados sintéticos
   repetitivos.

3. **A cognição do híbrido é o motor semântico, NÃO o modelo.** O scorecard do
   híbrido mostra cognição 1.0 — mas trocando os pesos do modelo por
   **aleatórios**, a cognição continua 1.0 (todos os itens vão para o parser
   simbólico; o modelo nem é chamado). → **Lição (a regra da carta):** nunca
   chamar de cognição neural o que é executado pelo núcleo simbólico. Sempre rodar
   a **ablação de pesos aleatórios**: se o número não cair, não é o modelo.

## Abordagens de treino que falharam

- **Currículo-só** para fluência: ruim em PT real (8.64 BPB). Fluência precisa de
  dado natural.
- **Multitask ingênuo** (natural + cognição diluída): boa fluência (2.01), mas a
  cognição-dura colapsou a 0. Diluição + cópia-de-valor fraca.
- **Cópia de valor inédito** (lugares novos): o neural generaliza a *chave*
  (nome), mas não gera *valores* nunca vistos (0.106 mesmo com 140 lugares
  diversos). É operação de indução/cópia difícil nessa escala.
- **Fine-tunes de "reparo" (Stage 3):** melhoraram probes locais mas derrubaram a
  cognição principal → não promovidos (regra: não promover se cair).

## Refutações de 2026-06-14 (memória ensinável)

| hipótese | veredito medido | lição |
|---|---|---|
| Tagging por byte é melhor p/ extração | derruba a entidade (0.56) — fronteira ambígua | ponteiro (início+comprimento) é melhor p/ a entidade |
| Encoder causal basta p/ extrair valor | valor 0.60; bidir sobe p/ 0.75 | precisa ver a frase inteira ("valor-antes-da-entidade") |
| **Valor de tamanho fixo mede o problema real** | com tamanho variável cai 0.97→**0.589** + overfitting | era **muleta**: o modelo só acertava o início; o fim vinha de graça |
| Mais passos = melhor (valor variável) | held-out sobe a 0.828 (passo 750) e **despenca** a 0.562 | overfitting do ponteiro de fim; **early-stop importa** |

> Detalhe completo: `relatorios/2026-06-14_memoria_ensinavel_e_extracao.md` e o
> catálogo `relatorios/CATALOGO_DE_TEORIAS.md` (bloco F).

## Mais negativos do núcleo (medidos)

- **Multiescala MEGABYTE** (patch 4-byte): BPB **pior** (2.61 vs 2.25) e **~15× mais
  lento** em PT real — gargalo no decoder local. Não adotar sem decoder vetorizado.
- **1 cabeça** no delta-net: falha (0.29). **8 cabeças**: piora (0.58, dk pequeno).
  Usar 2–4.
- **Lorentz / ternário / dendrítico:** inertes ou negativos como capacidade (ver
  catálogo, bloco A).
