"""
VEREDA-M sobre Qwen — cabeça de memória APRENDIDA sobre o núcleo congelado.

O Qwen cru já endereça fato↔pergunta acima do acaso (k=8: 0.63), mas sujo (mean-pool
mistura entidade+valor+template). Aqui treino uma CABEÇA leve (atenção sobre os
estados do Qwen + projeção), por CONTRASTE (InfoNCE): a pergunta deve casar com o
SEU fato vs todos os outros -> a cabeça aprende a focar a ENTIDADE. Qwen 100% CONGELADO.

Régua: retrieval top-1 (nomes NOVOS, vocab ABERTO) cabeça-aprendida vs Qwen-cru;
ablação (cabeça aleatória) tem que cair. Reps do Qwen são CACHEADAS (1x) p/ treino rápido.
"""
import os, random, time, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from build_cognition_tasks import make_name_pool

DEV = "cpu"
DIN = 896                       # hidden do Qwen2.5-0.5B
MAXLEN = 24
CACHE = "/tmp/vereda_qwen_cache.pt"

OBJS = [("carro", "O"), ("livro", "O"), ("gato", "O"), ("relógio", "O"), ("sofá", "O"),
        ("celular", "O"), ("violão", "O"), ("casaco", "O"),
        ("casa", "A"), ("bicicleta", "A"), ("mochila", "A"), ("caneta", "A"),
        ("jaqueta", "A"), ("planta", "A"), ("camisa", "A"), ("mesa", "A")]
VALS = ["azul", "vermelho", "dourado", "enorme", "antigo", "quebrado", "novo", "barato",
        "italiano", "minúsculo", "pesado", "turquesa", "vintage", "reluzente", "veloz",
        "importado", "silencioso", "elegante", "rústico", "macio", "brilhante", "raro",
        "escuro", "claro", "redondo", "quadrado", "leve", "moderno", "frágil", "robusto"]


def gen_pairs(rng, names, n):
    out = []
    for _ in range(n):
        nm = rng.choice(names); o, A = rng.choice(OBJS); v = rng.choice(VALS)
        a = A.lower()
        fact = rng.choice([f"{A} {o} de {nm} é {v}.",
                           f"{nm} tem {('um' if A=='O' else 'uma')} {o} {v}.",
                           f"{A} {o} de {nm} parece {v}."])
        ques = rng.choice([f"Como é {a} {o} de {nm}?", f"Qual é {a} {o} de {nm}?"])
        out.append((fact, ques))
    return out


@torch.no_grad()
def encode_batch(model, tok, texts):
    states = torch.zeros(len(texts), MAXLEN, DIN)
    masks = torch.zeros(len(texts), MAXLEN, dtype=torch.bool)
    for i in range(0, len(texts), 32):
        chunk = texts[i:i + 32]
        enc = tok(chunk, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=MAXLEN).to(DEV)
        h = model(**enc).last_hidden_state                       # [b,MAXLEN,DIN]
        states[i:i + 32] = h.float()
        masks[i:i + 32] = enc.attention_mask.bool()
    return states, masks


def build_cache():
    if os.path.exists(CACHE):
        return torch.load(CACHE)
    print("cacheando reps do Qwen (1x)...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to(DEV).eval()
    pool = make_name_pool(800); tr_names, he_names = pool[:680], pool[680:]
    rng = random.Random(0)
    tr = gen_pairs(rng, tr_names, 1200)
    he = gen_pairs(random.Random(1), he_names, 400)
    t0 = time.time()
    trf, trfm = encode_batch(model, tok, [f for f, _ in tr])
    trq, trqm = encode_batch(model, tok, [q for _, q in tr])
    hef, hefm = encode_batch(model, tok, [f for f, _ in he])
    heq, heqm = encode_batch(model, tok, [q for _, q in he])
    data = dict(trf=trf.half(), trfm=trfm, trq=trq.half(), trqm=trqm,
                hef=hef.half(), hefm=hefm, heq=heq.half(), heqm=heqm)
    torch.save(data, CACHE)
    print(f"cache pronto em {time.time()-t0:.0f}s ({len(tr)} train + {len(he)} held pares)")
    return data


class Head(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.qk = nn.Parameter(torch.randn(DIN) * DIN ** -0.5)
        self.proj = nn.Linear(DIN, d, bias=False)

    def forward(self, states, mask):
        sc = (states @ self.qk).masked_fill(~mask, -1e4)
        a = torch.softmax(sc, -1)
        pooled = (a.unsqueeze(-1) * states).sum(1)
        return F.normalize(self.proj(pooled), dim=-1)


def retrieval_acc(keyf, keyq, rng, k, n=2000):
    N = keyf.size(0); ok = 0
    for _ in range(n):
        idx = rng.sample(range(N), k)
        qi = rng.randrange(k)
        q = keyq[idx[qi]]
        sims = keyf[idx] @ q
        ok += int(sims.argmax().item() == qi)
    return ok / n


def raw_keys(states, mask):                       # baseline Qwen cru (mean-pool)
    m = mask.unsqueeze(-1).float()
    return F.normalize((states.float() * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


def main():
    d = build_cache()
    trf, trfm = d["trf"].float(), d["trfm"]; trq, trqm = d["trq"].float(), d["trqm"]
    hef, hefm = d["hef"].float(), d["hefm"]; heq, heqm = d["heq"].float(), d["heqm"]

    # baseline cru (held)
    rkf, rkq = raw_keys(hef, hefm), raw_keys(heq, heqm)
    erng = random.Random(99)
    print("\nQwen CRU (mean-pool, sem treino) — held:")
    for k in (8, 16): print(f"  k={k}: {retrieval_acc(rkf, rkq, erng, k):.3f}")

    torch.manual_seed(0)
    head = Head().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    N = trf.size(0); rng = random.Random(0)
    print("\ntreino da cabeça (InfoNCE, Qwen congelado):", flush=True)
    t0 = time.time()
    for step in range(1, 1501):
        idx = torch.tensor(rng.sample(range(N), 128))
        kf = head(trf[idx], trfm[idx]); kq = head(trq[idx], trqm[idx])
        logits = kq @ kf.T / 0.07
        loss = F.cross_entropy(logits, torch.arange(len(idx)))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 300 == 0:
            with torch.no_grad():
                hkf = head(hef, hefm); hkq = head(heq, heqm)
                a8 = retrieval_acc(hkf, hkq, erng, 8)
            print(f"  step {step}/1500 loss={loss.item():.3f} held k=8={a8:.3f} "
                  f"[{(time.time()-t0)/step:.2f}s/step]")

    with torch.no_grad():
        hkf = head(hef, hefm); hkq = head(heq, heqm)
        torch.manual_seed(7); rndhead = Head().to(DEV)
        akf = rndhead(hef, hefm); akq = rndhead(heq, heqm)
    print("\n=== RESULTADO (held: nomes NOVOS, vocab ABERTO, fraseado variado) ===")
    print(f"{'k':>4}{'Qwen cru':>11}{'cabeça aprendida':>18}{'ablação(aleat.)':>17}")
    for k in (8, 16, 32):
        print(f"{k:>4}{retrieval_acc(rkf, rkq, erng, k):>11.3f}"
              f"{retrieval_acc(hkf, hkq, erng, k):>18.3f}"
              f"{retrieval_acc(akf, akq, erng, k):>17.3f}")
    torch.save(head.state_dict(), "modelos/vereda_m_qwen_head.pt")
    print("\nLeitura: se a cabeça aprendida >> Qwen cru e >> ablação, o endereçamento")
    print("fato↔pergunta foi RESOLVIDO sobre texto real/vocab aberto (sem o muro 0.52),")
    print("e é APRENDIDO (ablação derruba). Base p/ memória persistente .vereda + override.")


if __name__ == "__main__":
    main()
