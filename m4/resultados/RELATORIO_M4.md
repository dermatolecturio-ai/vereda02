# Relatório M4 — memória viva: override, edição, persistência

Portões (DESIGN.md): override EM ≥0.90 e eco do prior ≤0.05; edição EM ≥0.90, eco antigo ≤0.05, vizinhos ≥0.95; persistência EM_ler ≥ EM_escrever − 0.02; capacidade k=200 no RELATORIO_M2 (células T k=200).

- Prior baseline (Θ cru): 0.557 das capitais respondidas certas sem memória (n=1024).
- Override: EM contrafactual 0.710 (portão ❌), eco do prior 0.059 (❌), retrieval 0.724.
  - No subconjunto DIFÍCIL (prior existia, n=570): EM contrafactual 0.693.
- Ablação SR (chave aleatória): EM 0.267, retrieval 0.263 — desabou ✅.
- Edição: EM valor novo 0.944 (✅), eco do antigo 0.003 (✅), integridade vizinhos 0.977 (✅).
- Persistência: EM escrever 0.934 → EM ler (processo novo) 0.934 — ✅.

Esquecer-restaura-o-prior é propriedade arquitetural nesta fase (Θ congelado + memória externa): remover o fato devolve o comportamento do prior baseline por construção (DESIGN.md §4.B).

