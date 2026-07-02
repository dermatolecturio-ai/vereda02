"""
VEREDA (regime novo) — teste decisivo e BARATO da virada: as representações do Qwen
(congelado) já dão o ENDEREÇAMENTO que nos travou em 0.52 do zero?

Codifica fatos PT reais + uma pergunta sobre um deles; recupera por similaridade das
embeddings do Qwen (SEM treino). Se top-1 for alto com distratores e vocab ABERTO,
então a memória do VEREDA-M sobre o Qwen vai funcionar com texto real — o muro de
generalização cai porque o entendimento vem pronto.
"""
import random, time, torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
print(f"Qwen base carregado | device={DEV}\n")

NAMES = ["Ana", "Bruno", "Carla", "Diego", "Elena", "Felipe", "Gabriela", "Hugo",
         "Inês", "João", "Letícia", "Marcos", "Natália", "Otávio", "Paula", "Rafael",
         "Sofia", "Tiago", "Vera", "William", "Beatriz", "Caio", "Daniela", "Eduardo"]
OBJS = ["carro", "casa", "livro", "gato", "bicicleta", "relógio", "mochila", "caneta",
        "jaqueta", "celular", "sofá", "violão"]
VALS = ["azul", "vermelho", "dourado", "enorme", "antigo", "quebrado", "novo", "barato",
        "italiano", "minúsculo", "pesado", "turquesa", "vintage", "reluzente",
        "de madeira", "importado", "veloz", "silencioso", "elegante", "rústico"]


@torch.no_grad()
def embed(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32).to(DEV)
    h = model(**enc).last_hidden_state                      # [B,T,d]
    m = enc.attention_mask.unsqueeze(-1).float()
    v = (h * m).sum(1) / m.sum(1).clamp(min=1)              # mean-pool mascarado
    return torch.nn.functional.normalize(v, dim=-1)


def trial(rng, k):
    names = rng.sample(NAMES, k); objs = [rng.choice(OBJS) for _ in range(k)]
    vals = rng.sample(VALS, k)
    facts = [f"{('O' if True else 'A')} {objs[i]} de {names[i]} é {vals[i]}." for i in range(k)]
    qi = rng.randrange(k)
    question = f"Qual é a característica do {objs[qi]} de {names[qi]}?"
    embs = embed(facts); q = embed([question])
    sims = (embs @ q.T).squeeze(-1)
    return int(sims.argmax().item() == qi)


rng = random.Random(0)
print("RECUPERAÇÃO por embedding do Qwen (sem treino), vocab ABERTO, top-1:")
for k in (2, 4, 8, 16):
    t = time.time()
    acc = sum(trial(rng, k) for _ in range(100)) / 100
    print(f"  k={k:>2} fatos: {acc:.3f}   (acaso {1/k:.3f})   [{time.time()-t:.1f}s/100]")

print("\nLeitura: se top-1 >> acaso (ex.: k=8 acima de ~0.8), o endereçamento fato↔pergunta")
print("por ENTIDADE já vem do pré-treino -> a memória do VEREDA-M funciona com texto real")
print("e vocab aberto, sem o muro de 0.52 que tivemos do zero. Próximo: memória aprendida")
print("+ persistência (.vereda) por cima disso, com leitor treinado e ablação.")
