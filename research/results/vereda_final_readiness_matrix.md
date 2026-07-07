# VEREDA-Final Readiness Matrix

Date: 2026-06-07

## Status curto

O `vereda_final` 30M esta implementado e localmente validado, mas ainda nao esta aprovado para treino oficial de 5000 steps nesta maquina.

Motivo: as provas matematicas locais passam, mas a contribuicao experimental ainda falha e o probe 30M nao completou 1 optimizer step em tempo pratico.

## Evidencia verde

| Item | Evidencia | Status |
| --- | --- | --- |
| Arquitetura registrada | `build_model` aceita `architecture: vereda_final` | OK |
| Parametros 30M | `37,957,428` parametros | OK |
| Config oficial | `configs/vereda_final_30m_5k.yaml` | OK |
| Manifesto final | `data/mixture_vereda_final.yaml` sem fontes ausentes/vazias | OK |
| Holdout externo | `data/pt_teste_5k.jsonl` usado via eval override, nao no treino | OK |
| Testes unitarios | `python3 -m pytest`: `41 passed` | OK |
| Provas locais | `local_math_invariants_pass: true` | OK |
| Forward/backward tiny | smoke tiny e manifesto passaram | OK |
| Eval override | `smoke-holdout-report.json` gerado | OK |
| Memory eval path | `smoke-memory-report.json` gerado | OK |
| Throughput benchmark | `throughput-report.json` + `throughput-estimate.png` gerados | OK |

## Provas matematicas novas do candidato final

| Prova | Resultado |
| --- | --- |
| `vereda_final_state_bounds` | CIF em `[0.189, 0.375]`, Kalman em `[0.0405, 0.0405]`, STP em `[0,1]` |
| `vereda_final_backward_and_vsa_grad` | gradiente nao nulo em `p_u`, `p_w`, `p_c`, `slow_write`, memoria densa e VSA textual |
| `vereda_final_checkpoint_equivalence` | hidden error `0.0`, grad error `0.0` |
| `topology_loss_tearing_order` | smooth `0.0`, torn `0.0383` |

## Evidencia vermelha

| Item | Resultado | Consequencia |
| --- | --- | --- |
| `lab_contribution_pass` | `false` | Ainda nao provou ganho contra baseline. |
| `pretraining_ready` | `false` | Nao declarar modelo final cientificamente validado. |
| Probe 30M inicial | OOM no MPS | Memoria precisava ser corrigida. |
| Probe 30M com checkpointing | 1 step nao terminou em `583.99s` | Cabia em memoria, mas lento demais. |
| Probe 30M com TBPTT streaming | 1 step nao terminou em `474.31s` | Memoria melhorou; tempo segue impeditivo. |
| Benchmark 30M chunk 128 | estimativa `411.94s/step`, `572.13h/5000 steps` | Melhor chunk medido, mas ainda impraticavel localmente. |
| Recurrent memory smoke | accuracy `0.0` em checkpoint de 2 steps | Caminho roda, mas modelo nao aprendeu ainda. |

## Melhorias obrigatorias antes dos 5000 steps

1. Reduzir overhead do loop `position x layer` do `VeredaFinalBackbone`.
2. Vetorizar ou fundir partes de `VeredaFinalBlock` que hoje rodam patch-a-patch em Python.
3. Usar `configs/vereda_final_30m_5k_chunk128.yaml` como variante local mais rapida para probes; manter `configs/vereda_final_30m_5k.yaml` como plano oficial original.
4. Criar ablation final:
   - sem Kalman
   - sem VSA textual
   - sem topology loss
   - sem Lorentz/dendrito
   - completo
5. Gerar criterio de promocao:
   - BPB nao pior que baseline por mais de `3%`
   - recall recorrente melhor que baseline
   - UTF-8 valid rate melhor ou igual
   - samples a cada 50 steps exatamente registradas

## Decisao

Treino oficial de 5000 steps nao deve ser iniciado nesta maquina ainda.

O proximo trabalho correto e otimizar throughput e rodar uma bateria curta comparativa antes de gastar o corpus final.
