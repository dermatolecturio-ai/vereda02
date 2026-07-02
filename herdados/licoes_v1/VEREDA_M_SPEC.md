# VEREDA-M — Spec (memória viva nos pesos/estado)

> Documento de REGRAS, escrito ANTES de codar (anti-autoengano aplicado ao próprio
> projeto novo). Subordinado a `REGRA_OBRIGATORIA.md`. Versão 0.1 — 2026-06-15.

## 1. O sonho, em uma frase

Um modelo **pequeno e local** que **aprende um FATO em tempo de execução** — escrevendo
no **próprio estado aprendível**, sem retreino e sem banco de dados — e isso **persiste,
pode ser editado/esquecido, e vence o que estava nos pesos**. Tudo provável pela régua.

## 2. O que é o "novo formato" (correção honesta)

O gargalo **não é o `.pt`** (que guarda qualquer tensor). É a **estrutura de uso**: um
forward congelado não tem canal para **escrever / persistir / sobrepor** um fato. Logo o
"novo formato" = **um novo TIPO de modelo cujo artefato salvo carrega uma MEMÓRIA viva**:

```
artefato VEREDA-M  =  { Θ : núcleo CONGELADO (fluência + operações)        # pesos mortos
                        M : MEMÓRIA viva (slots aprendíveis, persistente) } # estado vivo
```

`M` é escrita pelo próprio modelo via **operações aprendidas em forward** (estilo
delta-rule / NTM), **não** por gradiente nem por parser. Salvamos `{Θ, M}` juntos.

## 3. Linhagem (honestidade sobre ineditismo)

Memory Networks, NTM/DNC, fast-weights, "Neural Knowledge Bank". A classe de mecanismo
**não é inédita**. O que pode ser nosso: a demonstração **pequena, local, PT-BR, com
persistência real e provada por ablação** — e a metodologia que não se deixa enganar.
Semente: o estado do delta-rule **já é** memória associativa chave→valor aprendida
(`vereda_teach` leu valores a 0.999; cog_proc binda a 1.0, e cai a 0 sob pesos
aleatórios). A aposta = **persistir/crescer ISSO**, não trocar por um banco.

## 4. Os 5 PORTÕES (a barra, travada antes)

Um fato ensinado em runtime tem que passar em TODOS:

1. **Escrita sem retreino:** dou o fato em texto → entra em `M` por forward (sem
   gradiente, sem parser/regex/SQLite).
2. **Leitura/recall:** pergunto depois → acerta.
3. **Vence o prior:** se o fato **contradiz** o que Θ "diria", a resposta segue `M`
   (não recita o prior).
4. **Persistência + esquecimento:** salvo `M`, **reinicio o processo**, recarrego →
   ainda sabe; **descarto `M`** → esquece; ≥2 fatos coexistem (sem sobrescrita catastrófica).
5. **É modelo, não script (ablação):** zerar/aleatorizar os pesos do leitor **ou**
   apagar `M` → a capacidade cai a ~0. Se um parser produz a resposta, está proibido.

Régua de promoção: nada vira "oficial" sem os 5, com avaliação grande (≥1024) e
held-out de vocabulário/forma.

## 5. Arquitetura mínima (VEREDA-M v0)

- **Θ congelado:** `VeredaDeltaNet` (núcleo já existente). Não treina mais após o pré.
- **M:** slots `[N, d_k + d_v]`, persistente e serializável; começa vazia.
- **Cabeça de escrita (aprendida):** lê o fato → produz `(k, v)` → escreve em `M`
  (endereçamento: slot livre / menos-usado / aprendido). Forward puro.
- **Cabeça de leitura (aprendida):** na pergunta, `q` atende `M` (softmax k·q) → recupera
  `v` → **portão de override** mistura `v` sobre a saída de Θ.
- **Serialização:** `save({Θ, M})`; reload reconstrói o estado vivo.

## 6. Plano por estágios (devagar, com portão falsificável)

| Estágio | Entrega | Portão |
|---|---|---|
| **E0** | Anatomia: *medir* que Θ congelado não aprende fato em runtime; testar se o estado delta **persiste** | nº |
| **E1** | Θ+M mínimo: ensina 1 (k,v) → recall | 1, 2, 5 |
| **E2** | Persistência: salva M, reinicia, recarrega; 2+ fatos coexistem | 4 |
| **E3** | Override: fato que contradiz Θ → M vence | 3 |
| **E4** | A partir de TEXTO cru PT-BR (extração no forward) | 1–5 ponta a ponta |

Cada estágio: **ablação sempre**, barra definida antes, "não medido" onde não mediu.

## 7. Invariantes inegociáveis

- Capacidade mora em `{Θ, M}` aprendidos — **nunca** em parser/regex/DB.
- Toda manchete passa pela tríade: held-out + ablação + distribuição realista.
- Resultado negativo é resultado. Estimativa nunca é fato (mede-se o tempo).
