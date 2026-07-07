# Teoria unificada VEREDA-Final

Data: 2026-06-07

Objetivo: juntar todas as ideias dos artigos em uma teoria unica, complexa,
mas falsificavel. O modelo final nao deve ser uma pilha de ideias bonitas; ele
deve ser uma composicao de teoremas, invariantes, ablations e resultados.

## Regra central

Uma teoria entra no modelo final somente se satisfizer pelo menos uma destas
condicoes:

1. E um requisito estrutural ja provado por construcao, como patching fixo,
   persistencia de estado ou identidade do hiperboloide.
2. Passa em prova numerica local reproduzivel em
   `research/experiments/math-validation/`.
3. Ganha ablation pareada contra baseline, com mesmo budget, seed e dados.

Se nao satisfizer uma dessas condicoes, fica como modulo candidato, nao como
parte do treino oficial.

## Tudo que os artigos dizem

| Artigo | Teses extraidas | Destino na teoria unificada |
| --- | --- | --- |
| `Analise Critica do Sistema VEREDA` | Separar conhecimento simbolico, estado recorrente, memoria externa, alucinacao/abstencao, ternario, necessidade de treino estatistico inicial. | Base de governanca: nao rejeitar pretraining; memoria externa deve modular estado, nao virar RAG passivo. |
| `Pesquisa Arquitetura Nucleo VEREDA` | RWKV-7 como nucleo inicial, xLSTM/Gated DeltaNet como alternativas, Mamba/Mamba-2 como referencia, patching 4 bytes, state tuning, testes de memoria e morfologia. | RWKV-7-like vira baseline obrigatoria; patching fixo vira controle inicial; xLSTM/GDN ficam como candidatos futuros. |
| `Projeto VEREDA_ Nucleo SSM Cognitivo` | Gates `alpha/beta/gamma`, surpresa, confianca, RLS/Kalman, Lyapunov, VSA, dicionario, Carolina, benchmarks 100/300/1000 turnos, ablations. | Gating conservativo + VSA/RLS entram como teoria de memoria/adaptacao; no `vereda_final`, Kalman/RLS-style modula writes da slow memory. |
| `Evolucao para Ensino Explicito de Portugues` | Curriculo PT-BR, morfologia, concordancia, denoising, avaliacao explicita. | Fica como bateria de dados/avaliacao; nao define a matematica central. |
| `VEREDA_ Alfabetizacao Radical em Portugues` | Separacao entre nucleo fixo, memoria externa, filtros bayesianos, Mamba-3, Lorentz, R-STE, patching dinamico. | Extrai a tese de "ensinar por modulos", mas Mamba-3 e patching dinamico exigem prova isolada. |
| `VEREDA_ Hierarchical Temporal Sub-sampling Mechanism` | Lorentz, CBH step-2, CIF, limites dinamicos, reset ortogonal, hierarquia temporal, STP. | Forma a coluna vertebral do VEREDA-v2 matematico. Varios invariantes ja passam localmente. |
| `Arquitetura de Linguagem Byte-Level Inovadora` | Byte-level sem tokenizer, memoria hiperbolica externa, BLT, RLS/Kalman, energia livre, Ricci flow. | Mantem byte-level e memoria externa; topology/flow loss leve foi implementada, mas Ricci/energia livre completos ficam fora ate haver ablation. |
| `VEREDA-v2_ Arquitetura Byte-Level Revolucionaria` | Curvatura Lorentz aprendida, CBH, CIF, power companding, falhas empiricas do lite, boundaries dinamicos. | Define o laboratorio VEREDA-v2; nao e ainda candidato oficial. |
| `VEREDA_ Hyperbolic SSM Derivation` | Retracao `rsqrt`, erro cubico vs exp map, paradoxo de associatividade, Lie/CBH, Lyapunov espectral, kernel paralelo. | As provas de retracao e CBH sao aceitas sob hipoteses locais; kernel e Lyapunov espectral precisam implementacao. |
| `Estabilizacao Geometrica de SSMs Ternarios` | Ruido STE em Lorentz, campo de Jacobi, R-STE, sanctuary BF16/FP32, Ricci/topological loss, Slender-Mamba, HMamba, HELM. | R-STE local entra como teste; sanctuary de precisao vira regra de engenharia; Ricci fica bloqueado. |
| `Hyperbolic Quantization for VEREDA Architecture` | Quantizacao hiperbolica, gamma adaptativo, ternario, curva/erro de reconstrucao. | `AdaptiveGammaQuant` passa sanidade local; ainda precisa ablation contra INT8/ternario simples. |
| `Neuronio Hiperbolico-Dendritico e Treinamento Preditivo` | Ramos dendriticos, fusao somatica em Lorentz, STP, predictive coding local. | Dendrito e STP passam invariantes; predictive coding local ainda nao esta provado. |
| `Roadmap Engenharia Software VEREDA` | Traduzir teoria em modulos PyTorch, testes unitarios, MPS/CPU, configs 2M/30M, export. | Define a sequencia de implementacao e os gates de engenharia. |

## Teoria unica

O VEREDA-Final deve ser modelado como um sistema recorrente byte-level com seis
camadas matematicas acopladas, mas testaveis separadamente:

1. **Camada B: bytes e patches.**
   O fluxo bruto de bytes `b_1...b_N` e comprimido em patches `p_i`. O controle
   inicial e patch fixo de tamanho `P=4`; patching dinamico por entropia so
   entra depois de provar ganho.

2. **Camada E: encoder local.**
   Cada patch vira uma representacao local `z_i` que preserva ordem interna de
   bytes e alimenta o nucleo global. Esta camada pode ser densa ou ternaria,
   mas nao deve quantizar o estado recorrente critico antes da estabilidade.

3. **Camada S: estado recorrente duplo.**
   O estado possui:
   - estado rapido `f_i`, para continuidade local;
   - estado lento associativo `M_i`, atualizado por regra delta;
   - estado de sessao serializavel, hashado por checkpoint.

   Forma ideal:

   ```text
   f_i = r_i * f_{i-1} + w_i * tanh(W_f z_i)
   M_i = lambda_i M_{i-1} + eta_i (v_i - M_{i-1} k_i) k_i^T
   read_i = M_i q_i
   ```

4. **Camada G: geometria e algebra.**
   Representacoes hierarquicas podem ser projetadas no hiperboloide de Lorentz:

   ```text
   x_L = [sqrt(1/K + ||x_s||^2), x_s]
   <x_L, x_L>_L = -1/K
   ```

   Sequencias internas podem ser compostas por algebra nilpotente step-2:

   ```text
   X * Y = X + Y + 1/2 [X,Y]
   [n,[n,n]] = 0
   ```

   Isso permite scan paralelo se a restricao nilpotente for mantida.

5. **Camada M: memoria externa e adaptacao local.**
   Memorias explicitas entram como hipervetores VSA/HDC e embeddings vinculados
   ao hash do modelo. Adaptadores pequenos podem ser atualizados por RLS/Kalman,
   mas o core principal nao deve ser alterado online sem prova de estabilidade.

6. **Camada Q: quantizacao e sanctuary de precisao.**
   Periferia pode usar ternario/INT8. Estado recorrente, acumuladores CBH,
   curvatura, covariancia RLS e normas criticas ficam em BF16/FP32. O STE deve
   ser mascarado e finito; R-STE completo precisa projecao tangente real antes
   de ser chamado de Riemanniano.

7. **Camada D: decoder local.**
   O decoder reconstrui bytes do proximo patch. A qualidade do modelo final e
   medida em BPB, recall de memoria, latencia, repeticao e UTF-8 bruto.

## Provas ja aceitas localmente

Estas provas estao registradas em
`research/experiments/math-validation/summary.json`.

| Prova | Resultado | Significado |
| --- | --- | --- |
| Projecao Lorentz | erro `1.04e-14` | A construcao `x_0=sqrt(1/K+||x_s||^2)` preserva o hiperboloide. |
| Retracao Lorentz | slope `3.0001`, erro max `1.37e-06` | A retracao `rsqrt` tem erro cubico vs exp map em passos pequenos e preserva a restricao. |
| Dendrito Lorentz | erro `4.44e-16` | A fusao dendritica volta ao hiperboloide. |
| Delta memory ideal | erro `1.07e-06` | Se `q=k` normalizado e gate=1, a regra delta escreve valor com recall exato. |
| CBH step-2 | erro `9.54e-07` | O operador implementado e associativo dentro da tolerancia. |
| Reset ortogonal | erro `4.38e-13` | O componente projetado fica ortogonal ao vetor base. |
| CIF paralelo | erro `1.33e-15` | Forward paralelo e sequencial coincidem nos estados finais testados. |
| STP bounded | intervalo `[0,1]` | Fadiga/facilitacao permanecem limitadas. |
| STE ternario | erro `0.0` | Gradiente mascarado bate com a regra implementada. |
| Gamma quant | gradiente finito | Quantizador adaptativo nao explode no teste local. |
| RLS | erro relativo `2.12e-05`, autovalor min `0.0056` | Covariancia permanece SPD e estimativa converge em sistema linear sintetico. |
| VSA | max sim `0.123` | Hipervetores aleatorios ficam quase ortogonais. |
| VEREDA-Final bounds | erro `0.0` | CIF, STP e covariancia Kalman/RLS-style permanecem em faixas finitas em rollout local. |
| VEREDA-Final gradiente | erro `0.0` | Backward chega em projecoes ternarias, slow write, memoria densa e adaptador VSA textual. |
| Checkpoint equivalence | erro `0.0` | Checkpointing de ativacao nao altera hidden nem gradiente no teste local. |
| Topology tearing | smooth `0.0`, torn `0.0383` | A loss topologica penaliza salto persistente entre patches. |

## Provas em forma matematica

### P1. Patching reduz custo recorrente

Para sequencia de `N` bytes e patch fixo `P`, o nucleo global executa
`ceil(N/P)` passos. Logo o custo recorrente relativo e `ceil(N/P)/N`. Para
`P=4` e `N` grande, o custo tende a `1/4`. Esta prova justifica patch fixo como
controle, nao prova que patch dinamico e melhor.

### P2. Identidade do hiperboloide

Defina `x_L=[x_0,x_s]`, `x_0=sqrt(1/K+||x_s||^2)` e
`<x,y>_L=-x_0 y_0 + x_s dot y_s`. Entao:

```text
<x_L,x_L>_L = -(1/K + ||x_s||^2) + ||x_s||^2 = -1/K
```

Portanto a projecao espacial preserva a variedade por construcao.

### P3. Retracao `rsqrt`

Se `x` esta no hiperboloide e `v` esta no espaco tangente, entao
`<x,x>_L=-1/K` e `<x,v>_L=0`. Para `R_x(v)=gamma(x+v)`, imponha
`<R,R>_L=-1/K`:

```text
gamma^2(-1/K + ||v||_L^2) = -1/K
gamma = 1 / sqrt(1 - K ||v||_L^2)
```

Comparando a serie de Taylor contra o mapa exponencial, os termos de ordem
0, 1 e 2 cancelam; o primeiro erro tangencial aparece em ordem 3. A bancada
mede slope log-log `3.0001`.

### P4. CBH associativo sob nilpotencia step-2

Se a algebra satisfaz `[n,[n,n]]=0`, a serie CBH termina:

```text
X * Y = X + Y + 1/2[X,Y]
```

Como esse produto e a coordenada logaritmica de uma multiplicacao de grupo
nilpotente, ele e associativo. Expandindo `(X*Y)*Z` e `X*(Y*Z)`, os termos
aninhados que causariam diferenca sao todos zero pela hipotese step-2.

### P5. Regra delta escreve recall exato no caso ideal

Com `M' = M + (v - M k) k^T` e `||k||=1`:

```text
M' k = M k + (v - M k) k^T k
     = M k + (v - M k)
     = v
```

Logo a memoria lenta pode substituir uma associacao exata quando a chave de
leitura e a chave de escrita coincidem e o gate permite escrita total.

### P6. Reset ortogonal

Para vetor base `a`, a projecao
`c_perp = c - ((c dot a)/(a dot a))a` satisfaz:

```text
c_perp dot a = c dot a - ((c dot a)/(a dot a))(a dot a) = 0
```

Isso prova o reset ortogonal usado no VEREDA-v2.

### P7. STP bounded

As variaveis STP sao atualizadas e depois clampadas em `[0,1]`. Portanto, por
inducao, se `f_0,x_0 in [0,1]`, entao `f_t,x_t in [0,1]` para todo `t`. A
prova e simples, mas garante que essa subdinamica nao explode.

### P8. RLS preserva covariancia positiva no caso ideal

Com `P_0` SPD, `lambda>0` e denominador
`lambda + z^T P z > 0`, a atualizacao RLS padrao e equivalente a uma atualizacao
de inversa de matriz de Gram regularizada. Logo `P_t` permanece SPD em aritmetica
exata. A bancada numerica confirma autovalor minimo positivo.

### P9. Lyapunov para gate conservativo

Se o estado for:

```text
h_t = alpha_t h_{t-1} + beta_t z_t + gamma_t r_t
alpha_t,beta_t,gamma_t >= 0
alpha_t + beta_t + gamma_t <= 1
||z_t|| <= C, ||r_t|| <= C, ||h_0|| <= C
```

entao, pela desigualdade triangular:

```text
||h_t|| <= alpha_t C + beta_t C + gamma_t C <= C
```

Logo a bola de raio `C` e invariante. O modelo final deve impor essa forma se
quiser alegar estabilidade Lyapunov em inferencia.

### P10. VSA quase ortogonal

Para hipervetores aleatorios normalizados em dimensao `D`, o produto interno
tem media 0 e desvio padrao aproximado `1/sqrt(D)`. Em `D=1024`, o desvio
esperado e `0.03125`; a bancada mediu media absoluta `0.0249`.

## O que ainda nao esta provado

| Modulo | Problema | Obrigacao antes de entrar |
| --- | --- | --- |
| Patching dinamico por entropia | Pode melhorar custo, mas pode quebrar atribuicao e vazamento temporal. | Provar equivalencia causal e medir BPB/latencia contra patch fixo. |
| Ricci/topological loss | Topology/flow loss leve existe e passa sanidade local; Ricci flow completo nao. | Rodar ablation de coeficiente, custo e efeito em BPB/UTF-8. |
| R-STE completo | O teste atual prova STE mascarado, nao projecao Riemanniana completa. | Implementar projecao tangente e medir estabilidade em treino longo. |
| Mamba-3 MIMO | E arquitetura externa recente, nao implementada aqui. | Implementar modulo isolado ou manter como referencia. |
| Hyperbolic Hopfield memory | Nao existe modulo de memoria Hopfield no runtime atual. | Teste de recuperacao/correcao 100/300/1000 turnos. |
| Predictive coding local | Ainda e ideia de treino, nao mecanismo implementado. | Criar loss local e mostrar gradiente finito + ganho. |
| Full ternary core | Quantizar tudo pode destruir estado. | Sanctuary de precisao primeiro; depois ablation camada a camada. |

## Resultado experimental atual

Mesmo com as provas locais passando, o candidato atual nao esta pronto:

- `lab_contribution_pass=false`
- BPB Dual-State/RWKV por seed: `1.115`, `1.087`, `1.048`; limite era `1.03`.
- ganho de memoria recorrente: `0.0`; minimo era `0.15`.
- latencia passou, mas nao basta.
- `runs/vereda-7m-mac` ainda tem BPB de validacao `3.657` em 500 passos.

Conclusao: a matematica local esta ficando solida; a arquitetura completa ainda
nao provou ser melhor.

## Adaptacao para o modelo final

Ordem correta de construcao:

1. **Base final minima:** byte patches fixos, RWKV-7-like baseline,
   Dual-State, memoria episodica gated, sessao hashada.
2. **Adicionar delta memory corrigida:** forcar testes onde a regra delta
   consiga recall exato no caso ideal e ganho real em 10/100/1000 turnos.
3. **Adicionar geometria Lorentz:** primeiro so no adaptador ou em uma camada,
   nao no core inteiro; exigir ablation Euclidiano vs Lorentz.
4. **Adicionar CBH/CIF:** provar que scan paralelo bate com sequencial em
   sequencias maiores e que limites dinamicos melhoram custo ou BPB.
5. **Adicionar quantizacao:** manter sanctuary para estado; quantizar periferia;
   medir erro e estabilidade.
6. **Adicionar RLS/VSA:** no candidato `vereda_final`, VSA textual entra por
   adaptador treinavel e Kalman/RLS-style modula a escrita da slow memory; ainda
   falta baseline sem VSA/Kalman.
7. **Adicionar modulos especulativos:** Ricci, Hopfield hiperbolico, predictive
   coding e Mamba-3 entram apenas se cada um vencer seu baseline isolado.

## Forma do modelo final complexo

```text
bytes
  -> patcher fixo/dinamico causal
  -> encoder local byte-level
  -> recurrent core baseline-compatible
       fast state f_t
       slow delta memory M_t
       Lie-CBH accumulator g_t
       Lorentz/dendritic adapter x_L
  -> gated external memory
       VSA textual / dense embedding / Kalman-gated slow write
  -> precision sanctuary
       FP32/BF16 state, curvature, covariance
       INT8/ternary periphery
  -> local decoder
  -> byte output
```

Esta e a teoria unica atual: um modelo byte-level recorrente, com memoria lenta
associativa, geometria hiperbolica, algebra Lie/CBH para estado, memoria externa
VSA/RLS-style para fatos, quantizacao protegida por sanctuary de precisao e
validacao por provas locais + ablations.

O candidato `vereda_final` ja compoe todos esses modulos. O proximo passo real
antes do treino oficial nao e adicionar mais teoria; e provar que a composicao
vence ou pelo menos nao degrada os baselines em custo, BPB, memoria e UTF-8.
