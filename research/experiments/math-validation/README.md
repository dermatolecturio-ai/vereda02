# Validacao matematica local

Esta pasta contem uma bancada pequena para transformar as teses matematicas
dos documentos VEREDA em evidencia reproduzivel.

## Como rodar

```bash
python3 research/experiments/math-validation/validate_math.py
```

## Artefatos gerados

- `summary.json`: resultado numerico consolidado.
- `math_invariants.png`: erros dos invariantes locais em escala log.
- `contribution_criteria.png`: BPB, latencia e memoria da comparacao multi-seed.
- `training_bpb_latest_segments.png`: segmentos recentes dos logs de treino.
- `patching_cost_curve.png`: custo recorrente relativo por tamanho de patch.

## Interpretacao atual

Os invariantes locais passam, mas `pretraining_ready=false`, porque a
contribuicao experimental Dual-State ainda falha contra a baseline RWKV-7-like.
Isso significa que a matematica implementada tem sanidade local, nao que a
arquitetura completa esteja validada para treino oficial.
