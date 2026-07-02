# ⛔ REGRA OBRIGATÓRIA DO VEREDA

> **O VEREDA tem que ser um MODELO DE IA, nunca um motor semântico.**
> A capacidade (fluência, binding, fatos, raciocínio) precisa morar em
> **parâmetros aprendidos** — pesos ou um novo formato de memória aprendida —
> e **não** em regras escritas à mão, parser, regex ou banco de dados.
> Aceita-se inventar um **novo formato além do `.pt`**, desde que seja **aprendido
> e diferenciável**. Aceita-se que seja mais difícil e demore anos.

## Definição operacional (sem ambiguidade)

- **MODELO (permitido):** a resposta vem de parâmetros/estado **aprendidos** por
  treino/gradiente. Inclui pesos densos, fast weights, memória associativa
  chave→valor aprendida, Neural Knowledge Bank, NTM/DNC-like, modulações MAC —
  qualquer memória cujo **escrever/ler é aprendido**.
- **MOTOR SEMÂNTICO (proibido como mecanismo de capacidade):** a resposta vem de
  parser/roteador escrito à mão, regex, gramática manual, SQLite/BM25/grafo
  consultado por regras. Capacidade em código humano, não em parâmetros aprendidos.

## O TESTE que decide (obrigatório em toda alegação)

**Ablação de pesos aleatórios.** Troque os parâmetros aprendidos por aleatórios e
rode a MESMA avaliação:
- Se o resultado **DESABA** (≈ acaso) → a capacidade está no modelo. ✅ **válido.**
- Se o resultado **NÃO cai** → quem faz é o motor semântico. ❌ **proibido,
  rejeitar, não chamar de cognição.**

Complemento: nenhum parser/regex/banco pode produzir a resposta avaliada. Se um
`if`/regex resolve, não conta.

## O que isto PROÍBE neste repo (honesto)

- `vereda_memory.py` (roteador talâmico + hipocampo simbólico) e o caminho
  simbólico de `vereda_delta_cognitive.py` **NÃO são o VEREDA**. A cognição 1.000
  do `vereda_hybrid_report.py` é **motor semântico** (provado: pesos aleatórios →
  ainda 1.000). Pode existir só como **utilitário de dev / professor para gerar
  dados de treino**, **nunca** como a capacidade entregue/alegada.

## O que isto EXIGE (a revisão do blueprint)

A deep research recomendou "fatos → memória externa auditável (SQLite/BM25/grafo)".
**Pela regra, isso é rejeitado como mecanismo de cognição.** Em vez disso:

- **Fatos/valores → memória NEURAL aprendida** (fast weights / memória associativa
  chave→valor / Neural Knowledge Bank / linhagem NTM-DNC / cápsulas MAC com reader
  aprendido). Escrever e ler são **operações aprendidas**, e a ablação de pesos
  aleatórios derruba o recall.
- **"Ensinar / mandar PDF" → o MODELO escreve na própria memória aprendida** via
  operações aprendidas — **não** um script que faz parse do PDF para um banco.
- **Novo formato além do `.pt`** = um modelo cujo estado serializável inclui essa
  **memória aprendida e atualizável** (não pesos densos congelados). Permitido e
  até desejável.

## A semente que já temos (prova de que é possível)

O estado do delta-rule **já é uma memória associativa chave→valor aprendida**:
quando o `cog_proc` aprendeu binding a 0.95 (pesos aleatórios → 0.00), os fatos
(entidade→lugar) estavam **nos parâmetros aprendidos** — isso é modelo, não parser.
O caminho do VEREDA é **escalar/persistir essa memória aprendida**, não trocá-la
por um banco simbólico.

## Frase-lei

> Se a ablação de pesos aleatórios não derruba a capacidade, **não é VEREDA**.
> Tudo que o VEREDA sabe, ele aprendeu — e a prova está nos pesos/estado aprendido.
