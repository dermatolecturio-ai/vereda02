# CARTA DO VEREDA 2

> Escrita ANTES de qualquer código de modelo (2026-07-02). Este documento é lei.
> Nenhum resultado, componente ou manchete vale se violar o que está aqui.

## 1. O norte (a estrela-guia, em uma frase)

> Um sistema **fluente E cognitivo** cuja cognição mora em **parâmetros/estado
> APRENDIDOS** — provado por ablação — que **aprende fatos em tempo de execução**
> a partir de texto PT-BR real, persiste, edita e esquece. Pequeno, local, na borda.

Este é o problema em aberto que o VEREDA 1 nomeou e não resolveu
(`herdados/licoes_v1/`). O V2 existe para resolvê-lo — pela rota corrigida.

## 2. A rota corrigida (a lição dos 90%)

O V1 gastou ~90% do tempo construindo do zero o substrato (fluência/entendimento)
— a única parte que a escala resolve de graça — e ~10% no diferencial real
(memória viva aprendida). O V2 inverte:

- **Θ (substrato) = Qwen2.5-0.5B-Instruct, CONGELADO.** Fluência e entendimento
  são commodity: compram-se prontos. Não se treina, não se compete com escala.
- **M (memória viva) + cabeças aprendidas = 100% do nosso esforço.** Escrita por
  forward, leitura aprendida, persistente e serializável (`.vereda`).
- **O cume da pesquisa = INTERNALIZAÇÃO** (marco M5): mover a cognição dos
  módulos externos para dentro dos pesos (adapters/destilação/fast-weights) —
  o passo que o multitask ingênuo do V1 não conseguiu. É pesquisa de fronteira;
  tem portão falsificável e critério de morte como tudo aqui.

## 3. REGRA OBRIGATÓRIA (herdada do V1, protocolo afiado para o V2)

**O VEREDA é um MODELO, nunca um motor semântico.** Capacidade mora em
parâmetros/estado aprendidos; parser/regex/gramática manual/SQLite/BM25 são
proibidos como mecanismo de qualquer capacidade alegada.

Protocolo de ablação do V2 (3 partes, todas obrigatórias em toda alegação):

1. **Ablação dos NOSSOS parâmetros:** aleatorizar tudo que NÓS treinamos
   (cabeças, M, adapters) → a capacidade alegada DESABA (≈ acaso/baseline).
2. **Baseline Θ-sozinho:** o Qwen cru (sem nossos módulos), nas mesmas perguntas,
   fica MUITO abaixo. Senão, a capacidade era do substrato, não nossa.
3. **Caminho da resposta 100% aprendido:** nenhum parser/regex/banco em nenhum
   ponto entre a pergunta e a resposta avaliada. Se um `if` resolve, não conta.

## 4. Os princípios fundadores (cada um mata um erro medido do V1)

| # | Princípio | Erro do V1 que ele mata |
|---|---|---|
| P1 | **A régua nasce antes do modelo.** Marco 0 = régua + sentinelas + baselines. Nada se acredita antes dela existir. | Régua construída depois da crença → 6 ilusões em série |
| P2 | **Nenhum componente entra sem vencer A/B** contra a versão sem ele, na régua oficial. | Pilha Lorentz/CBH/dendrítico por narrativa |
| P3 | **Diagnóstico → literatura → mecanismo → código**, nesta ordem. Reconhecimento bibliográfico ANTES de cada aposta cara. | Meses em geometria; a causa (binding) era conhecível por teoria |
| P4 | **Barra realista desde o dia 1:** vocab aberto, texto real, tamanho variável, posição embaralhada. Muleta descoberta depois = resultado anulado. | Muleta do tamanho fixo; atalho posicional |
| P5 | **Uma variável por vez.** Experimento que muda duas coisas não atribui causa. | — (lição já paga) |
| P6 | **Tempo se mede, não se estima.** s/passo no 1º checkpoint, ETA com a conta à vista, right-size antes de lançar. | "2–4 min" que eram 25 |
| P7 | **Resultado negativo é resultado.** Vai para `NEGATIVE_FINDINGS.md` no mesmo dia, com número. | — (o ativo mais valioso do V1) |
| P8 | **Git desde o dia 1.** Um commit por experimento, config no commit. | V1 sem versionamento |

## 5. Regra de promoção (o que pode virar "oficial")

Nenhum checkpoint/resultado vira oficial sem, simultaneamente:
(a) passar no juiz duro (geração exata, vocab/forma held-out, distribuição realista);
(b) passar no protocolo de ablação (§3, as 3 partes);
(c) passar nos 6 sentinelas (`REGUA.md`);
(d) manter o que já funcionava (sem regressão nos marcos anteriores);
(e) avaliação ≥1024 amostras para qualquer número de manchete.

Sem os cinco: é papagaio, motor semântico, muleta ou ruído — e nomeamos como tal.

## 6. Frase-lei (herdada, intacta)

> Se a ablação não derruba a capacidade, **não é VEREDA**.
> Tudo que o VEREDA sabe, ele aprendeu — e a prova está nos pesos/estado aprendido.
