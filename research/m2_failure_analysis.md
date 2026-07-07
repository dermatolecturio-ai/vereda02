# Análise formal do negativo do M2 — "categoria sem binding"

Data: 2026-07-07. Consolida o registro do dia em `NEGATIVE_FINDINGS.md` num
negativo FORMAL: o que exatamente falhou, o que a literatura já explicava, o
que segue em aberto e qual experimento separa as hipóteses restantes. Números:
`m2/resultados/RELATORIO_M2.md` (células oficiais, n=1024, held).

## 1. O que foi testado e o que aconteceu

Alegação testada: um projetor treinável (~1.09M params, perceiver-style,
m=8 soft tokens em `inputs_embeds`) transforma a carga útil do fato (estados
de token do Qwen congelado) em resposta gerada correta, chegando perto do teto
do fato-como-texto.

| braço | EM k=8 | k=32 | k=100 |
|---|---:|---:|---:|
| T (texto no prompt — teto) | 0.940 | 0.931 | 0.938 |
| S1 (CE) | 0.040 | 0.025 | 0.041 |
| S2 (CE + reconstrução λ=1) | 0.021 | 0.022 | 0.023 |
| SR (projetor ALEATÓRIO) | 0.019 | 0.013 | 0.018 |

Três fatos tornam este negativo limpo, e não um "não deu certo":

1. **O endereçamento estava resolvido** (retrieval 0.94 nos mesmos itens) — a
   falha é 100% na transmissão do valor, não em achar o fato.
2. **S1 ≈ S2 ≈ SR**: o projetor treinado não se distingue de pesos aleatórios.
   A perda de reconstrução (a correção canônica da literatura de compressão)
   não moveu o número.
3. **O modo de falha é qualitativo e sistemático**, não ruído: com retr_ok, a
   resposta vem quase sempre do POOL CERTO do atributo (cor↔cor; senha com o
   mesmo comprimento/alfabeto; cidade↔cidade) e quase sempre com o VALOR
   ERRADO (`gold='jfx2r'` → `'zqj7y'`; `gold='Oeiras'` → `'Natal'`).

Nomeamos o padrão: **categoria sem binding** — o canal contínuo de input
carrega o subespaço do atributo, mas não a associação exata entidade→valor.

## 2. O que a literatura já explicava (e nós medimos no nosso substrato)

- **Prompt Waywardness (arXiv:2112.08348):** prompts contínuos podem resolver
  a tarefa sem que sua interpretação discreta seja fiel — desempenho de tarefa
  e conteúdo literal são dissociáveis. É a assinatura exata do nosso S1/S2.
- **Compressão de prompt (arXiv:2412.17483):** gist/ICAE preservam tarefa e
  falham em reconstrução literal; a perda de autoencoding é a correção
  canônica — e no nosso caso ela NÃO bastou (S2 ≈ SR), o que é informação
  nova sobre o regime de fato-exato (senhas procedurais, juiz duro).
- **Retrofit em decoder-only CONGELADO (arXiv:2603.22329):** em baixa
  capacidade (1×), prefix e KV-extension falham (<0.4%) e só vencem mecanismos
  com prior de binding explícito (Hebbiano, slot-write, cross-attn paralela);
  em 10× todos convergem. Ou seja: nosso resultado é o ESPERADO para a rota
  input em baixa capacidade — o que valida a régua, e aponta a rota interna.
- **Lição do V1** (`herdados/licoes_v1/VEREDA_M_SPEC.md` §3): o binding que
  funcionou lá veio de um núcleo Gated Delta com (k,v) explícitos e portão de
  override — nunca de "vetor no input + esperar a self-attention decodificar".

Sob esse rótulo exato ("categoria sem binding") o padrão não tem nome
canônico na literatura — a combinação waywardness + compressão descreve o
mecanismo, mas o caso controlado (mesmo substrato para ler e escrever, juiz
duro de valor literal, ablação com pesos aleatórios, PT-BR) é contribuição
nossa. É isso que o benchmark consolida
(`benchmarks/read_write_asymmetry.md`, eixo B).

## 3. Hipóteses separáveis que restam (e como o knife as separa)

| hipótese | previsão no knife (`m5/DESIGN.md`) |
|---|---|
| H-canal: o INPUT é a rota errada; uma rota interna com prior de binding funciona com a MESMA carga útil | braço I ≥ 0.80 com braço S parado em ~0.04 |
| H-binding: falta estrutura chave→valor explícita em qualquer rota | braço I falha COM endereçamento implícito alto e IS (chaves embaralhadas) igual a I — não há binding em lugar nenhum |
| H-capacidade: 256-d por fato não carrega valor literal (compressão) | braço I falha com endereçamento implícito ALTO e erros do tipo "categoria certa, valor errado" (mesma assinatura do M2) |
| H-treino: 3000–6000 passos não bastam para nenhuma rota | previsão desfavorecida por S2 ≈ SR já em 3000 e pela convergência só em 10× de arXiv:2603.22329; o knife NÃO estende passos (morte sem apelação) |

O experimento separador é exatamente o desenho do knife: mesma carga útil,
mesma chave, mesmos itens — muda SÓ a rota (input S vs meio I vs saída K vs
texto T). Uma variável por vez (P5), com o diagnóstico embutido
(endereçamento implícito) para atribuir a falha ao componente certo.

## 4. Status

- Negativo registrado no dia: `NEGATIVE_FINDINGS.md` (2026-07-07).
- Consequência de desenho: o M4 usou o injetor T (texto) como voz — produto
  funciona; a rota soft está morta NESTA FORMA.
- Próximo teste da hipótese de canal: `m5/knife_test.py` (portões e critério
  de morte em `m5/DESIGN.md`).
