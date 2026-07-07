# Experimento 0001: Dual-State vs RWKV-7

## Pergunta

O estado rapido vetorial combinado com memoria associativa lenta melhora a
retencao recorrente sem degradar linguagem ou latencia?

## Hipotese e falsificacao

- Hipotese: Dual-State ganha pelo menos 15 pontos percentuais em memoria.
- Rejeitar se: BPB piorar mais de 3%, latencia exceder 1,5x, ou ganho de
  memoria ficar abaixo de 15 pontos percentuais.

## Controle

- Baseline: `configs/lab_2m_rwkv.yaml`
- Tratamento: `configs/lab_2m_dual_state.yaml`
- Seeds: 11, 22, 33
- Dados: mesma fonte e mesma sequencia por seed
- Patches: quatro bytes em ambos

## Execucao

```bash
python3 -m vereda.cli run-comparison \
  --baseline-config configs/lab_2m_rwkv.yaml \
  --dual-state-config configs/lab_2m_dual_state.yaml \
  --seeds 11,22,33
```

Status: comparacao completa executada em `research/experiments/lab-comparison/`.

## Resultado

O Dual-State nao passou nos criterios de contribuicao. A latencia foi melhor
que a baseline nas tres seeds, mas o BPB piorou mais que o limite de 3% e o
ganho de memoria recorrente foi `0.0` ponto percentual.

Ver tambem:

- `research/experiments/lab-comparison/summary.json`
- `research/experiments/math-validation/contribution_criteria.png`
- `research/results/pretraining_readiness_audit.md`
