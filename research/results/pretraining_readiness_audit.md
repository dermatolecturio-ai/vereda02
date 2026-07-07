# Auditoria de prontidao pre-treino

Data: 2026-06-07

## Veredito

A versao atual nao deve ser tratada como pronta para o treinamento oficial.
Ela esta bem melhor organizada do que o prototipo inicial e possui invariantes
matematicos locais validos, mas ainda nao demonstra a contribuicao principal
contra baseline nem a completude matematica prometida nos documentos.

## Evidencia principal

- A bancada `research/experiments/math-validation/validate_math.py` passou nos
  invariantes locais de Lorentz, CBH, CIF, STE ternario, camada dendritica e
  VSA.
- A comparacao multi-seed `Dual-State vs RWKV-7` falhou nos criterios de
  contribuicao: BPB piorou em todas as seeds e o ganho de memoria recorrente
  foi `0.0` ponto percentual.
- O Dual-State foi mais rapido nas tres seeds, com razao media de latencia
  `0.783`, mas isso nao compensa a falha de BPB e memoria.
- O run `runs/vereda-7m-mac` e o artefato mais recente encontrado, com
  `checkpoint-00000500.pt` atualizado em 2026-06-07. Aos 500 passos, o BPB de
  validacao foi `3.657`, e as amostras ainda sao ruidosas.
- O log `runs/lab-2m-vereda-v2/metrics.jsonl` contem 22 resets de passo nao
  monotonicos, indicando reexecucoes anexadas no mesmo arquivo. Ele precisa ser
  separado antes de servir como evidencia limpa.

## Criterios avaliados

| Criterio | Evidencia | Status |
| --- | --- | --- |
| Software treina e salva checkpoint | Testes existentes e runs em `runs/` | Parcialmente pronto |
| Invariantes matematicos locais | `research/experiments/math-validation/summary.json` | Passou |
| Contribuicao Dual-State | `research/experiments/lab-comparison/summary.json` | Falhou |
| Memoria recorrente longa | 0% em 10/100/1000 turnos na comparacao lab | Falhou |
| Qualidade de linguagem | BPB ainda alto; geracao ruidosa | Nao pronto |
| Logs reprodutiveis limpos | v2 tem resets anexados | Nao pronto |
| Teorias v2 completas | Catalogo matematico mostra varias lacunas | Nao pronto |

## Decisao para o modelo final

O treino oficial deve usar a linha v0.1 conservadora apenas depois que uma nova
rodada provar melhora contra RWKV-7-like. A linha VEREDA-v2 deve permanecer em
ablacao matematica, porque combina Lorentz, CBH, CIF, ternario, dendritos,
STP, quantizacao adaptativa e reset ortogonal ao mesmo tempo. Com tudo ligado
junto, nao da para atribuir ganho ou falha.

## Proximas provas necessarias

1. Reexecutar `run-comparison` com logs limpos e salvar cada tentativa em pasta
   nova.
2. Rodar ablacoes isoladas: `dual_state`, `vereda_v2_sem_lorentz`,
   `vereda_v2_sem_ternario`, `vereda_v2_sem_cif`, `vereda_v2_sem_dendrito`.
3. Adicionar avaliacao PT-BR: concordancia nominal/verbal, denoising com
   acentos, UTF-8 bruto e memoria com correcao de premissa.
4. Exigir que qualquer candidato passe: BPB ate 3% pior que baseline, memoria
   +15 pontos percentuais, latencia ate 1.5x, UTF-8 >= 98%, repeticao <= 20%.
5. So entao escalar para 30M e depois para o treino oficial.
