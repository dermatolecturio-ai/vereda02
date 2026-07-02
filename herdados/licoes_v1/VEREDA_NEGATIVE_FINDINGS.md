# VEREDA - Achados negativos que guiam o próximo passo

Este arquivo existe para impedir que o projeto esqueça o que já foi testado.

## Não funcionou como diferencial principal

- Geometria de Lorentz, CBH, CIF e pilha dendrítica não explicaram o salto de
  cognição observado depois.
- O salto real veio do núcleo Gated Delta/DeltaNet, que dá binding endereçável.
- Fine-tune neural direto em corpus de reparo melhorou métricas locais, mas
  derrubou a cognição principal quando foi agressivo.

## Resultado empírico recente

- `vereda_stage2_foundation.pt` continua sendo o melhor modelo neural oficial:
  `nat_BPB=2.181`, `gram=0.750`, `cog=0.703`.
- `vereda_stage3_repair.pt` melhorou probes residuais, mas caiu para
  `cog=0.531`; não promover.
- `vereda_stage3_entity_repair.pt` melhorou residual (`entity=0.750`,
  `contrad=0.750`, `agree=1.000`), mas caiu para `cog=0.625`; útil como
  experimento, ainda não substitui o Stage 2.

## Decisão atual

O caminho mais realista para inteligência útil agora é híbrido:

1. manter o DeltaNet Stage 2 como córtex fluente;
2. usar hipocampo simbólico auditável para memória, fatos, overwrite, relações e
   conflito;
3. registrar tudo como andaime, não como cognição neural consolidada;
4. gerar dados de professor a partir desse andaime para futura destilação neural.

## Regra de promoção

Um novo checkpoint neural só substitui `vereda_stage2_foundation.pt` se melhorar
as tarefas residuais sem derrubar a métrica principal de cognição abaixo de
`0.70`.
