# VEREDA 2

**Um sistema fluente e cognitivo cuja cognição mora em parâmetros aprendidos —
que aprende fatos em runtime a partir de texto PT-BR real, persiste, edita e
esquece. Pequeno, local, na borda. Fundado sobre as lições medidas do VEREDA 1.**

## Status

**Fundado em 2026-07-02. Marco atual: M0 (a régua antes do modelo).**
Nenhum resultado ainda — e, pela CARTA, nenhum número será citado antes de a
régua existir.

## Os documentos que são lei (ler nesta ordem)

1. [`CARTA.md`](CARTA.md) — o norte, a rota corrigida, a REGRA OBRIGATÓRIA
   com o protocolo de ablação em 3 partes, e os 8 princípios fundadores.
2. [`REGUA.md`](REGUA.md) — a spec da régua oficial: juiz duro, 6 sentinelas
   (um por ilusão documentada do V1), baselines obrigatórios.
3. [`ROADMAP.md`](ROADMAP.md) — marcos M0–M6 com portões falsificáveis e
   critérios de morte definidos ANTES.
4. [`NEGATIVE_FINDINGS.md`](NEGATIVE_FINDINGS.md) — o diário de falsificações.

## Arquitetura (decidida por número, não por estética)

```
Θ  = Qwen2.5-0.5B-Instruct, CONGELADO      ← fluência/entendimento: commodity
M  = memória viva aprendida                ← escrita por forward, leitura
     (chaves/valores sobre as reps de Θ)     aprendida, persistente, editável
.vereda = {manifest, ref a Θ, M, cabeças}  ← o artefato que se ensina e persiste
M5 = internalização nos pesos              ← o cume da pesquisa (fronteira)
```

## Linhagem

O VEREDA 1 (protótipo, falsificações e metodologia) vive em
`~/pasta sem título 2/` como arquivo histórico. Os ativos PROVADOS foram
importados para `herdados/` (código: cabeça de memória sobre Qwen, formato
`.vereda`, benchmark justo, estágios E1–E3) e `modelos/vereda_m_qwen_head.pt`
(retrieval held-out 0.965 @ k=8 — número do V1, a REVALIDAR sob a régua nova
no M1 antes de qualquer citação como resultado do V2).
As lições estão em `herdados/licoes_v1/`.

## Máquina de referência

Apple M1, 8 GB RAM, CPU/MPS, Python 3.9, torch 2.8, transformers 4.57.
Qwen2.5-0.5B-Instruct já em cache local.
