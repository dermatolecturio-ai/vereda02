# RÉGUA DO VEREDA 2 — especificação (escrita antes do código)

> Esta é a spec que o Marco 0 implementa em `regua/`. A régua é UMA, oficial,
> roda com um comando, e toda alegação passa por ela. Alterar a régua exige
> justificativa escrita aqui (a régua não se ajusta ao resultado; o resultado
> se submete à régua).

## 1. Eixos medidos

| eixo | métrica | como |
|---|---|---|
| **Cognição/memória** | exact-match GERANDO a resposta | juiz duro: entidades, valores e fraseados held-out; distribuição realista |
| **Fluência preservada** | Θ+módulos não pode piorar a geração do Θ cru | perplexidade/BPB em PT natural held-out + inspeção de amostras |
| **Custo** | tokens de contexto + latência por resposta | comparado aos baselines |
| **Persistência** | recall após save → kill do processo → reload | via `.vereda` |

## 2. Os 6 SENTINELAS (cada ilusão do V1 vira teste permanente)

Todo resultado de manchete roda os seis. Falhou um → não é manchete.

| sentinela | o que pega | teste |
|---|---|---|
| S1 anti-papagaio | sonda fraca inflando cognição | só vale GERAÇÃO exata com vocab held-out (nunca escolha 2-way) |
| S2 anti-motor-semântico | parser fazendo o trabalho | protocolo de ablação da CARTA §3 (3 partes) |
| S3 anti-muleta | regularidade artificial no dado | valores de tamanho/forma VARIÁVEL; fraseados múltiplos; nada de comprimento fixo |
| S4 anti-atalho-posicional | "responda o 1º valor" | posição do fato-alvo EMBARALHADA no treino e na avaliação; reportar acc por posição |
| S5 anti-ruído | checkpoint sortudo | manchete só com ≥1024 amostras; checkpoints pequenos são termômetro |
| S6 anti-template | BPB baixo em dado repetitivo | fluência medida SÓ em PT-BR natural held-out |

## 3. Baselines obrigatórios (medidos no M0, congelados)

- **A — Θ cru, sem fatos:** piso (deve errar tudo que depende de fato ensinado).
- **B — Θ cru com TODOS os fatos no contexto:** o baseline FORTE honesto
  (in-context). O VEREDA 2 só tem razão de existir se vencer B em acerto e/ou
  custo conforme k cresce (no V1: 0.60 vs 0.20 @ k=100, 16× menos tokens).
- **C — RAG ingênuo (embedding + top-1 no contexto):** PROIBIDO como mecanismo,
  OBRIGATÓRIO como comparação externa. Se um RAG de 20 linhas empata com nossa
  memória aprendida, o diferencial não existe e precisamos saber.

## 4. Condições de avaliação (distribuição realista, sempre)

- Entidades: nomes reais E procedurais, held-out do treino das cabeças.
- Valores: vocabulário ABERTO, tamanho variável, com acentos/multibyte.
- Fraseados: múltiplos templates de fato e de pergunta, held-out de forma.
- Competição: k ∈ {2, 8, 32, 100, 200+} fatos coexistindo; query embaralhada.
- Edição/esquecimento: reportar sempre os três — recall do novo, ausência do
  apagado, integridade dos vizinhos.

## 5. Relato

Cada experimento gera uma linha em `NEGATIVE_FINDINGS.md` (se negativo) ou no
relatório do marco (se positivo), com: config, seed, n da avaliação, número,
sentinelas passados, e tempo MEDIDO (s/passo × passos).
