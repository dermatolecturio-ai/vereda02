# P3 — Literatura para o M2 (injeção de conhecimento em Θ congelado)

Data: 2026-07-07. Feita ANTES de qualquer código do M2 (P3: diagnóstico →
literatura → mecanismo → código).

## Pergunta de pesquisa
Como fazer um fato recuperado pela memória M (aprendida no M1) virar resposta
gerada FLUENTE pelo Θ congelado (Qwen2.5-0.5B), sem parser, com custo
compatível com dispositivo leve?

## O que a literatura diz

### 1. xRAG (Cheng et al., NeurIPS 2024, arXiv:2405.13792)
- Projeta o embedding do documento recuperado em UM token no espaço de
  embeddings de um LLM congelado; só a "ponte de modalidade" (projetor) treina.
- Retriever e LLM permanecem congelados; embeddings de documento são
  reutilizáveis offline.
- Resultado: ~10% de melhora média em 6 tarefas knowledge-intensive, com
  compressão de ~175 tokens → 1 token.
- **Lição para o M2:** a topologia soft-token + projetor treinável sobre LM
  congelado é viável e barata. É o nosso injetor S.

### 2. Gist tokens (Mu et al., 2023) e ICAE (Ge et al., 2023, arXiv:2307.06945)
- Compressão de prompt em soft tokens funciona para desempenho de tarefa
  (gist: até 26x; ICAE: 4x), MAS estudos de probing (arXiv:2412.17483)
  mostram que gist tokens **falham em reconstrução literal** do conteúdo
  original.
- Correção conhecida: **perda auxiliar de autoencoding** (reconstruir o texto
  original a partir dos soft tokens) força a retenção do conteúdo literal.
- **Lição para o M2:** nosso juiz exige o valor EXATO (inclusive senhas
  aleatórias de 4-10 chars). Injetor S sem perda de reconstrução tende a
  falhar exatamente aí. A/B obrigatório: S sem recon vs S com recon (P2/P5).
- **Consequência de desenho:** m>1 soft tokens (usamos m=8) e perda auxiliar
  de reconstrução do fato como variável de A/B.

### 3. RETRO (Borgeaud et al., 2022, arXiv:2112.04426) e memória retrofit em
   decoder-only (arXiv:2601.15324, arXiv:2603.22329)
- RETRO injeta vizinhos recuperados via cross-attention em chunks — mas exige
  encoder e cross-attention nativos (treinados juntos).
- Trabalhos de retrofit em decoder-only congelado apontam que NÃO há ponto
  natural de injeção por cross-attention: adapters precisam aprender a fazer a
  memória parecer "contexto anterior legítimo" para a self-attention.
- **Lição para o M2:** para Θ decoder-only congelado, prefixo de soft tokens
  (inputs_embeds) é o caminho de menor atrito; cross-attention leve exigiria
  cirurgia no forward do Qwen e treino mais caro. Fica como candidato B só se
  o soft-prompt falhar.

## Decisão de mecanismo (fundamentada)
1. **Injetor T (controle/teto):** fato recuperado (pela cabeça M1) colado como
   TEXTO no prompt. Sem treino novo. Mede o teto do substrato condicionado ao
   fato certo (M0 sugere teto alto: acc ≈ retrieval no baseline C).
2. **Injetor S (a aposta VEREDA):** projetor perceiver-style sobre os ESTADOS
   DE TOKEN do fato (não só mean-pool — precisamos dos caracteres da senha,
   não só do "assunto") → m=8 soft tokens no espaço de embeddings do Qwen.
   Treina só o projetor (~1M params), Qwen e cabeça M1 congelados.
   - S-v1: perda = CE na resposta-ouro (teacher forcing).
   - S-v2: + perda de autoencoding do fato (λ=1.0) — motivada por
     arXiv:2412.17483.
3. A memória do M2 tem CHAVE (cabeça M1, endereçamento provado) e CARGA ÚTIL
   (estados de token do fato). A chave do M1 foi treinada por InfoNCE
   fato↔pergunta e portanto codifica (nome, atributo), não o valor — a carga
   útil precisa ser separada da chave. Esta separação chave/valor é decisão
   central do M2.

## Fontes
- xRAG: https://arxiv.org/abs/2405.13792 (NeurIPS 2024)
- Gist tokens: https://arxiv.org/abs/2304.08467
- ICAE: https://arxiv.org/pdf/2307.06945
- Estudo de compressão gist (reconstrução falha sem AE loss):
  https://arxiv.org/html/2412.17483v1
- RETRO: https://arxiv.org/abs/2112.04426
- Survey de prompt compression (NAACL 2025):
  https://arxiv.org/pdf/2410.12388
