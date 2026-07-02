# ROADMAP DO VEREDA 2 — marcos com portões falsificáveis

> Sem estimativas de tempo (P6: tempo se mede). Cada marco tem PORTÃO (números
> definidos ANTES) e CRITÉRIO DE MORTE (o que nos faria parar/reformular).
> Nada avança com o marco anterior quebrado.

## M0 — Fundação: a régua existe antes do modelo
- Implementar `regua/` conforme `REGUA.md`: juiz duro + 6 sentinelas + protocolo
  de ablação, um comando.
- Medir e CONGELAR os baselines A (Θ cru), B (in-context) e C (RAG ingênuo de
  referência) em k ∈ {2, 8, 32, 100}.
- **Portão:** régua roda end-to-end; baselines documentados com n≥1024.
- **Morte:** não há. M0 não pode falhar, só atrasar.

## M1 — Âncora: reproduzir o melhor do V1 sob a régua nova
- Cabeça de memória aprendida sobre Θ congelado (herdada de
  `herdados/vereda_m_qwen.py`, retreinada com S3/S4 ativos desde o início).
- **Portão:** retrieval top-1 held-out ≥0.95 @ k=8 e ≥0.90 @ k=32; vence o
  baseline B em k=100; ablação (3 partes) passa.
- **Morte:** se com S3/S4 ativos o 0.95 do V1 não reproduzir, o número do V1
  tinha muleta → registrar, diagnosticar, e o portão real é o novo número.

## M2 — Voz: o fato recuperado vira resposta fluente
- Injetar o valor recuperado de M na GERAÇÃO de Θ (não só medir retrieval).
  Candidatos (decidir por A/B, P2): soft-prompt aprendido / cross-attention
  leve / prefixo aprendido. Literatura antes (P3).
- **Portão:** EM end-to-end da resposta gerada ≥ (retrieval do M1 − 0.05);
  fluência preservada (S6); ablação passa.
- **Morte:** se toda forma de injeção derrubar fluência ou EM, reavaliar se a
  voz precisa ser do Θ (talvez decoder próprio pequeno).

## M3 — Ensino de texto cru (o muro E4 do V1, agora com vantagem)
- Extração aprendida de (entidade→valor) a partir de frases PT-BR reais, por
  ponteiros sobre as representações do Θ (que já "entende" a frase — a aposta
  central da rota corrigida). Transferir a receita provada do V1:
  complementaridade diferenciável + marker-dropout. Spans reais, tamanho
  variável (S3), early-stop.
- **Portão:** extração EM ≥0.90 em fraseados held-out com valores variáveis;
  end-to-end texto→M→resposta ≥0.85 @ k=8.
- **Morte:** se a extração sobre reps do Θ não superar CLARAMENTE o teto
  byte-level do V1 (~0.59 variável), a vantagem do substrato era ilusória para
  extração → deep research antes de insistir.

## M4 — Memória viva completa: os 5 portões na barra cheia
- Override do prior (fato ensinado vence o que Θ "diria"), editar, esquecer,
  capacidade k=200+, persistência entre processos via `.vereda`.
- **Portão:** os 5 portões da spec herdada (`herdados/licoes_v1/VEREDA_M_SPEC.md`)
  com S1–S6 ativos; edição/esquecimento com integridade de vizinhos ≥0.95.
- **Morte:** colapso de capacidade em k grande → diagnóstico de endereçamento
  (lição E1: conteúdo, não posição) antes de qualquer aumento de escala.

## M5 — INTERNALIZAÇÃO (o cume da pesquisa; a estrela-guia)
- Mover a cognição dos módulos externos para DENTRO de parâmetros do modelo:
  candidatos (P3 primeiro — literatura: fast weights, Titans, test-time
  learning, destilação professor→aluno usando M como professor; LoRA reversível).
- **Portão:** capacidade de fato ensinado morando em pesos/estado interno,
  passando juiz duro + ablação + fluência mantida — o que o multitask do V1
  não conseguiu (colapso a 0).
- **Morte (honesta):** este é o marco de FRONTEIRA; pode falhar. Time-box por
  tentativa, cada tentativa com hipótese única (P5) e registro do negativo (P7).
  O V2 continua valioso sem M5 (M0–M4 já são o produto medido).

## M6 — O sonho, medido
- Documento PT-BR real (PDF/artigo) → o sistema lê, extrai, escreve em M →
  perguntas depois, em outra sessão → respostas fluentes corretas.
- **Portão:** EM ≥0.80 sobre fatos de documentos held-out reais; persistência
  entre sessões; ablação passa; custo < baseline B.
- Este marco transforma a frase do sonho original do V1 em um número.

## Ordem de ataque e disciplina
M0 → M1 → M2 → M3 → M4 → (M5 pesquisa em paralelo time-boxed) → M6.
Cada experimento: 1 variável (P5), tempo medido (P6), commit com config (P8),
negativo registrado no dia (P7).
