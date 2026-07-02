# Aprendizados transferíveis (o método honesto)

Estes princípios valem para qualquer projeto de IA — são o que sobra de mais
durável dos 6 meses.

## 1. Construa a régua antes de acreditar no resultado
Defina o benchmark e as métricas **antes** de otimizar. Sem juiz fixo, qualquer
demo vira hype. A régua deste projeto: fluência (BPB em PT real), gramática
(pares mínimos), cognição (exact-match, vocab held-out), eficiência.

## 2. Para detectar papagaio, exija duas coisas
- **Geração** da resposta (não escolha entre 2 opções).
- **Vocabulário held-out** (entidades/lugares nunca vistos no treino).
Se cai aqui mas vai bem na sonda fácil → é papagaio.

## 3. A ablação de pesos aleatórios é o detector de motor-semântico
Se trocar os pesos treinados por aleatórios **não derruba** o número, então quem
faz não é o modelo — é um parser/heurística. Rode isso sempre que afirmar
"o modelo aprendeu X".

## 4. Densidade de dado > volume
Dado denso e bem desenhado ensina mais por byte. Mas cuidado: denso-repetitivo
falseia o BPB; denso-diverso (combinatório, único) é o que ensina regra de
verdade. Fluência de mundo aberto ainda exige dado natural.

## 5. Generalização é diversidade-dependente, por dimensão
O modelo generaliza na dimensão em que viu diversidade. 90 nomes → memoriza
(0%); 650 nomes → aprende a operação (0.95). Quer que generalize valores? Dê
diversidade de valores. Quer estrutura? Dê estrutura variada.

## 6. Separe, e nomeie com honestidade, "modelo" de "motor"
Um híbrido neuro-simbólico é arquitetura legítima — desde que você diga qual
parte faz o quê. Fluência (modelo) e cognição exata (motor simbólico) são coisas
diferentes; não venda uma como a outra.

## 7. Resultado negativo é resultado
Falsificar a própria ideia central, com número, economiza meses. Most projects
never do this. Foi o que mais protegeu o VEREDA de virar ilusão permanente.

## 8. Escolha do regime > tamanho do modelo
Num laptop, não compita em escala. O valor está no regime que a restrição cria:
pequeno, local, PT-BR, ensinável, auditável, na borda. Lá o laptop é vantagem.

## 9. Estrutura > atalho: force o modelo a usar o sinal certo
O modelo pega o **atalho** mais fácil (Geirhos et al. 2020, *shortcut learning*). A
extração de valor dependia do marcador (atalho lexical) e não generalizava. A cura
foi tornar o sinal **estrutural** mais fácil que o atalho: **complementaridade
diferenciável** ("valor = o span que NÃO é a entidade", via coverage +
mutual-exclusivity) + **marker-dropout** (remove o atalho em parte do treino). Valor
held-out 0.52 → 0.97. Princípio: se o modelo decora um atalho, **penalize o atalho**
ou **dê o sinal estrutural de graça**, não só "treine mais".

## 10. Contexto bidirecional vale a pena para extração
Encoder **causal** não enxerga o resto da frase — quebra quando o alvo aparece antes
de sua âncora. Tornar o encoder **bidirecional** (forward + reverse) resolveu o caso
"valor-antes-da-entidade" (0.60→0.75) sem custo de capacidade. Para tarefas de
extração/compreensão, bidirecional ≫ causal.

## 11. Teste a SUPOSIÇÃO mais arriscada antes de construir em cima
Antes de levar a extração ao PT-BR natural, mudei **uma variável** (tamanho fixo →
variável) e descobri que o número estava **inflado por muleta**. Custou ~15 min e
evitou construir um andar inteiro sobre base falsa. Mude **uma variável por vez**
(senão não sabe a causa) e ataque primeiro o que, se for falso, derruba tudo.

## 12. Meça o tempo; não estime no olho
Estimar runtime "no olho" e errar por 10× quebra confiança como inflar um resultado.
Meça s/passo no 1º checkpoint, dê ETA com a conta à vista, e **right-size** o
experimento antes de lançar (1 treino, não 2; avaliação barata durante, grande no fim).
