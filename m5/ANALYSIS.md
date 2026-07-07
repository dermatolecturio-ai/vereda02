# M5 knife — o que conta como sucesso, morte ou inconclusivo

Escrito ANTES de rodar (P1), junto com o DESIGN.md. Este arquivo existe para
que a leitura do resultado não seja negociada depois que o número aparecer.

## Sucesso (todas, simultaneamente)

- F1: EM(I) ≥ 0.80 (k=8, held, n=1024, fato NUNCA no prompt).
- F2: EM(I) ≥ max(EM(K), EM(S)) + 0.05.
- F3: IZ (memória zerada), IS (chaves embaralhadas) e IR (módulos aleatórios)
  todas com EM ≤ 0.10.
- Sentinelas: EM por terço de posição sem gradiente forte (S4); amostras
  fluentes (S6); n=1024 (S5).
- F4 (fase 2): edição EM novo ≥ 0.80 com eco antigo ≤ 0.20; esquecimento por
  rebuild com eco ≤ 0.20.

Se sucesso: o canal interno existe. Próximos passos NA ORDEM: fase 3
(capacidade k ∈ {32,100}); repetição com seeds 1 e 2 (dispersão à vista);
ablação de camada (injetar na 22 deve ser PIOR — checagem causal do ponto de
injeção, só faz sentido depois do sucesso); e só então desenho do M5 v1
(consolidação em pesos / M5b).

## Morte (qualquer uma)

- F1 < 0.80 na rodada travada (exceção única: 0.70 ≤ EM < 0.80 → repetir com
  seeds 1 e 2 e reportar média±dispersão; se a média não cruzar 0.80, morreu).
- F2 < +0.05 — o canal interno não paga o próprio custo sobre kNN-LM.
- F3 falha — o efeito não é da memória; é do leitor perturbando o Θ.
- Qualquer dependência de fato textual reintroduzido.

Morte NÃO é: "quase passou", "com mais passos passa", "noutra camada passa".
Morreu → `NEGATIVE_FINDINGS.md` no dia, com o diagnóstico da tabela abaixo, e
o projeto segue com M0–M4 + benchmark (que já são o produto e o paper
negativo, respectivamente).

## Diagnóstico da falha (preencher ANTES do veredito final)

| endereçamento implícito (célula I) | EM | leitura |
|---|---|---|
| alto (≥0.85) | baixo | gargalo na VOZ/capacidade do valor (256-d comprime demais OU o leitor não converte m em geração) — mesma família do M2, agora na rota interna; variante multi-slot de valor é HIPÓTESE NOVA, knife novo |
| baixo (<0.60) | baixo | gargalo de LEITURA: W_q não mapeia camada 12 → espaço da chave M1; hipótese nova = query de outra camada ou treinada com alvo de alinhamento |
| alto | alto mas F2 falha | o canal existe mas não supera kNN-LM — registrar como "canal caro"; a rota de produto continua T/K |
| IZ NÃO desaba | qualquer | o leitor virou adapter genérico (aprendeu a tarefa, não a memória) — ablação IS decide: se IS também não desaba, é motor de tarefa, não memória |

## Autoengano — checklist pré-relatório (marcar um a um)

1. O fato aparece em QUALQUER ponto do prompt dos braços K/I? (tem de ser não)
2. Os itens são os mesmos do M0/M2 (seed 1008, held)? Entidades/templates
   held-out de TODO treino (M1, M2, knife)?
3. λ/τ do K foram escolhidos SÓ em train (K_grid.json commitado antes)?
4. O checkpoint do I foi escolhido pelo termômetro (128 itens, seed 501) e
   NÃO pelas células oficiais?
5. Camada, escala, gate: iguais ao DESIGN.md? (nenhum ajuste pós-resultado)
6. EM por terço de posição reportado? (S4 — atalho posicional)
7. As três ablações rodaram do MESMO checkpoint da célula I?
8. O relatório inclui s/item e tokens (custo honesto vs braço T)?
9. Alguma célula rodou mais de uma vez? Se sim, TODAS as rodadas reportadas?
10. O baseline A (0.015) e o teto T (0.940) estão na mesma tabela do I?

## Inconclusivo (e o que fazer)

- Célula I entre K e K+0.05 com ablações desabando: canal real mas margem
  insuficiente → registrar como negativo de MARGEM (não de existência);
  decisão de produto: K.
- Termômetro oscilando > ±0.10 entre checkpoints consecutivos no fim do
  treino: instabilidade de treino → reportar curva inteira no JSON, não o
  melhor ponto isolado (S5).
- Smoke ≠ oficial: números de --smoke NUNCA entram em relatório.
