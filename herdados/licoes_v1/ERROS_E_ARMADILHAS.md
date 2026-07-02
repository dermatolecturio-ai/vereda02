# Erros e armadilhas — o que deu errado e como não repetir

> Dois tipos de erro: **(A) autoengano** (acreditar em número que não vale) e
> **(B) engenharia** (ferramenta/ambiente). Os do tipo A custam meses; os do tipo B
> custam horas. Ambos estão aqui para não voltarem.

---

## A. Armadilhas de AUTOENGANO (as caras)

### A1. Motor semântico disfarçado de modelo
- **O que aconteceu:** a cognição do híbrido marcava **1.000**. Parecia o modelo.
- **Como pegamos:** ablação de pesos aleatórios → **continuou 1.000**. Os itens iam
  para o parser simbólico; o modelo neural nem era chamado.
- **Por quê engana:** um número alto não diz *quem* o produziu. Parser + modelo no
  mesmo pipeline confundem o crédito.
- **Correção / regra:** **REGRA OBRIGATÓRIA** — sempre rodar ablação de pesos
  aleatórios. Se o número **não cai**, não é o modelo. Nunca chamar de "cognição
  neural" o que um parser faz.

### A2. Papagaio com nota alta (sonda fraca)
- **O que aconteceu:** sonda 2-way dava cognição **0.703**.
- **Como pegamos:** juiz duro (gerar a resposta, vocab held-out) → **0.008**.
- **Por quê engana:** escolher entre 2 opções é gameável por fluência; não prova
  binding.
- **Correção:** cognição só vale **gerando** a resposta, com **vocabulário held-out**.

### A3. BPB baixo em corpus repetitivo ≠ fluência
- **O que aconteceu:** currículo de templates dava BPB **~0.08/0.53** (lindo).
- **Como pegamos:** o **mesmo** modelo em PT real deu **8.64** (péssimo).
- **Por quê engana:** dado denso-**repetitivo** tem entropia baixa → BPB cai sem
  haver fluência de mundo aberto.
- **Correção:** medir fluência **só** em PT natural held-out.

### A4. A muleta do tamanho fixo (2026-06-14)
- **O que aconteceu:** extração de valor marcava **0.97** held-out.
- **Como pegamos:** com **tamanho variável** (probe de realidade) caiu p/ **0.589**
  e ainda fez **overfitting** (pico 0.828 → despenca).
- **Por quê engana:** com comprimento fixo, o modelo só precisa acertar o **início**;
  o fim vem de graça. O número media meio-problema.
- **Correção:** não citar números de tamanho-fixo como manchete; testar **realidade**
  (valor de tamanho/forma variável) antes de comemorar.

### A5. Estimar tempo no olho (2026-06-14)
- **O que aconteceu:** afirmei "2–4 min" para um run que levou **~25 min**.
- **Por quê engana:** estimativa apresentada como fato é desonestidade, mesmo sem
  intenção — quebra confiança igual a inflar um resultado.
- **Correção:** **medir** s/passo no 1º checkpoint, calcular ETA com a conta à vista.
  (Memória `honest-timing-measure-not-guess`.)

### A6. Celebrar o checkpoint ruidoso
- **O que aconteceu:** relatei EM_resp **0.859** de uma avaliação de 256 amostras.
- **Como pegamos:** a avaliação final de 1024 amostras deu **0.713**.
- **Correção:** manchete só com avaliação grande; checkpoints pequenos são
  termômetro, não veredito.

### A7. Promover fine-tune que regrediu (Stage 3)
- **O que aconteceu:** reparos melhoravam probes locais.
- **Como pegamos:** derrubavam a cognição principal → **não promovidos**.
- **Correção:** regra de promoção — nada vira "oficial" sem passar no juiz principal
  **e** manter o que já funcionava.

---

## B. Erros de ENGENHARIA / ambiente (macOS, zsh, torch)

| Erro | Sintoma | Causa | Correção |
|---|---|---|---|
| `\| tail` em comando em background | saída "travada" até o fim | tail bufferiza até o processo terminar | escrever sem buffer em arquivo; `cat` cru; `flush=True` |
| zsh `for id in $IDS` | não itera | zsh não faz word-split de variável por padrão | lista explícita no loop |
| `timeout` ausente | comando falha | `timeout` não existe no macOS base | usar `islice`/guardas de loop no Python |
| f-string com aspas aninhadas em `python -c` | SyntaxError | aspas duplas dentro de aspas duplas inline | criar script `.py` em vez de `-c` |
| `gen_fact` duplicado | risco de NameError | função antiga referenciando pools removidos | remover a duplicata |
| chunkwise float32 | erro 0.035 (teste falhava) | sistema triangular mal-condicionado com chaves não-normalizadas | **L2-normalizar** as chaves (o modelo real já faz) → 3e-7 |
| streaming HF (wikipedia) | travava | streaming instável | download direto de parquet (Madras1/corpus-ptbr-v2) |
| `python` não encontrado | command not found | no macroambiente é `python3` | usar `python3` |

---

## C. Padrão dos erros de autoengano (a meta-lição)

Todos os erros do tipo A têm a mesma forma:

> **um número subiu, e eu quis acreditar antes de perguntar "quem o produziu e sob
> qual distribuição?"**

A defesa é sempre a mesma tríade:
1. **Geração + vocab/forma held-out** (mata papagaio);
2. **Ablação de pesos aleatórios** (mata motor semântico);
3. **Distribuição realista** (mata muleta: tamanho fixo, corpus repetitivo, sonda fácil).

Se um resultado passa pelos três, é real. Se falha em um, é ilusão — e nomeamos como tal, sem dó.

---

## D. Erros que **não** cometemos (por terem sido evitados a tempo)

- Não escalamos para 200M params atrás de um nome bonito (falsificamos antes).
- Não construímos o PT-BR natural em cima da muleta de tamanho fixo (o probe pegou).
- Não entramos em loop de retreino cego no gargalo do valor — escalamos para deep
  research e achamos a alavanca (complementaridade) com fundamento.
