"""
VEREDA-M — E3: VENCER O PRIOR (portão 3). O momento "ensinei e ele sabe, contra o
que os pesos achavam".

Construção: um modelo com CONHECIMENTO embutido nos pesos (nomes conhecidos -> valor
CANÔNICO, memorizado no treino = o "prior") + memória M com um PORTÃO APRENDIDO. Em
runtime escrevo em M um fato que CONTRADIZ o canônico; o portão tem que preferir M.
Apagar M -> volta ao prior (esquecer restaura a crença).

Régua: (1) prior intacto (M sem o nome) -> dá o canônico; (2) override (M com valor
contraditório) -> dá o de M; (3) após apagar M -> volta ao canônico; (4) ablação
aleatória -> acaso. Tudo aprendido (portão inclusive); nada de if-else escrito à mão
decidindo a resposta.
"""

import math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from vereda_deltanet import DeltaNetBlock, RMSNorm
from build_cognition_tasks import make_name_pool

DEV = torch.device("cpu")
L = 12
VALUES = ["azul", "verde", "preto", "branco", "roxo", "cinza", "vermelho", "amarelo",
          "rosa", "bege", "vinho", "creme", "dourado", "prata", "marrom", "lilas",
          "novo", "velho", "leve", "forte", "fraco", "caro", "raro", "redondo",
          "liso", "quente", "frio", "seco", "limpo", "claro", "doce", "macio"]
NV = len(VALUES)
MS = 4                                        # slots de M por exemplo (1 alvo + distratores)

KNOWN = make_name_pool(200)                   # nomes "conhecidos" (prior nos pesos)
_cr = random.Random(123)
CANON = {n: _cr.randrange(NV) for n in KNOWN} # valor canônico memorizado = o prior


def enc(s):
    b = list(s.encode("utf-8"))[:L]; return b + [0] * (L - len(b)), len(b)


class MemPrior(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(256, d)
        self.block = DeltaNetBlock(d, num_heads=heads, conv_size=4)
        self.norm = RMSNorm(d)
        self.key = nn.Linear(d, d, bias=False)
        self.val_emb = nn.Embedding(NV, d)
        self.cls_prior = nn.Linear(d, NV)        # conhecimento embutido
        self.cls_mem = nn.Linear(d, NV)          # leitura da memória
        self.gate_w = nn.Parameter(torch.tensor(1.0))
        self.gate_b = nn.Parameter(torch.tensor(0.0))
        self.scale = 1.0 / math.sqrt(d)

    def encode(self, B_NL, len_N):
        h = self.norm(self.block(self.emb(B_NL)))
        return self.key(h[torch.arange(h.size(0), device=h.device), (len_N - 1).clamp(min=0)])

    def forward(self, qn, ql, mn, ml, mv):
        B = qn.size(0)
        q = self.encode(qn, ql)                                   # [B,d]
        MK = self.encode(mn.reshape(B * MS, L), ml.reshape(B * MS)).reshape(B, MS, self.d)
        MV = self.val_emb(mv)                                     # [B,MS,d]
        sims = torch.einsum("bd,bmd->bm", q, MK) * self.scale
        attn = torch.softmax(sims, -1)
        vread = torch.einsum("bm,bmd->bd", attn, MV)
        mem_logits = self.cls_mem(vread)
        prior_logits = self.cls_prior(q)
        match = sims.max(-1, keepdim=True).values                 # confiança de hit
        g = torch.sigmoid(self.gate_w * match + self.gate_b)      # portão aprendido
        return g * mem_logits + (1 - g) * prior_logits, g.mean().item()


def batch(rng, B, override_p=0.5):
    qn = torch.zeros(B, L, dtype=torch.long); ql = torch.zeros(B, dtype=torch.long)
    mn = torch.zeros(B, MS, L, dtype=torch.long); ml = torch.zeros(B, MS, dtype=torch.long)
    mv = torch.zeros(B, MS, dtype=torch.long); tg = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        name = rng.choice(KNOWN); canon = CANON[name]
        others = rng.sample([n for n in KNOWN if n != name], MS)
        qb, qln = enc(name); qn[b] = torch.tensor(qb); ql[b] = qln
        if rng.random() < override_p:                            # OVERRIDE: M tem o nome
            contra = (canon + rng.randint(1, NV - 1)) % NV
            slots = [(name, contra)] + [(others[i], rng.randrange(NV)) for i in range(MS - 1)]
            rng.shuffle(slots); tg[b] = contra
        else:                                                    # PRIOR: M sem o nome
            slots = [(others[i], rng.randrange(NV)) for i in range(MS)]
            tg[b] = canon
        for i, (nm, vv) in enumerate(slots):
            bb, ll = enc(nm); mn[b, i] = torch.tensor(bb); ml[b, i] = ll; mv[b, i] = vv
    return (qn.to(DEV), ql.to(DEV), mn.to(DEV), ml.to(DEV), mv.to(DEV), tg.to(DEV))


@torch.no_grad()
def eval_split(model, rng, B, mode, n=1536):
    model.eval(); ok = tot = 0
    for _ in range(n // B):
        qn, ql, mn, ml, mv, tg = batch(rng, B, override_p=(1.0 if mode == "override" else 0.0))
        logits, _ = model(qn, ql, mn, ml, mv)
        ok += (logits.argmax(-1) == tg).sum().item(); tot += B
    return ok / tot


def main():
    torch.manual_seed(0)
    model = MemPrior().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    rng = random.Random(0); erng = random.Random(999)
    print("VEREDA-M E3 — vencer o prior (conhecimento embutido + memória + portão)")
    print(f"params={sum(p.numel() for p in model.parameters()):,} | "
          f"{len(KNOWN)} nomes conhecidos | acaso={1/NV:.3f}\n")
    import time; t0 = time.time()
    model.train()
    for step in range(1, 2001):
        qn, ql, mn, ml, mv, tg = batch(rng, 64)
        logits, _ = model(qn, ql, mn, ml, mv)
        loss = F.cross_entropy(logits, tg)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 400 == 0:
            pr = eval_split(model, erng, 64, "prior", 512)
            ov = eval_split(model, erng, 64, "override", 512)
            print(f"  step {step:04d}/2000 loss={loss.item():.4f} prior={pr:.3f} "
                  f"override={ov:.3f} [{(time.time()-t0)/step:.2f}s/step]")
        model.train()

    model.eval()
    pr = eval_split(model, erng, 64, "prior", 1536)
    ov = eval_split(model, erng, 64, "override", 1536)
    torch.manual_seed(7); rnd = MemPrior().to(DEV).eval()
    ab = eval_split(rnd, erng, 64, "override", 768)
    print("\n=== avaliação ===")
    print(f"  (1) prior intacto (M sem o nome) -> canônico:   {pr:.3f}")
    print(f"  (2) OVERRIDE (M contradiz)       -> valor de M: {ov:.3f}")
    print(f"  (4) ablação (pesos aleatórios):                 {ab:.3f}  (acaso {1/NV:.3f})")

    # demo narrativa + (3) esquecer restaura o prior
    print("\n=== demo: ensinar contra o prior, depois esquecer ===")
    name = KNOWN[0]; canon = VALUES[CANON[name]]
    contra = VALUES[(CANON[name] + 5) % NV]
    def ask(slots):
        mn = torch.zeros(1, MS, L, dtype=torch.long); ml = torch.zeros(1, MS, dtype=torch.long)
        mv = torch.zeros(1, MS, dtype=torch.long)
        for i in range(MS):
            nm, vv = slots[i] if i < len(slots) else (KNOWN[1 + i], 0)
            bb, ll = enc(nm); mn[0, i] = torch.tensor(bb); ml[0, i] = ll; mv[0, i] = VALUES.index(vv) if isinstance(vv, str) else vv
        qb, qln = enc(name)
        lg, g = model(torch.tensor([qb]), torch.tensor([qln]), mn, ml, mv)
        return VALUES[int(lg.argmax(-1))], g
    r1, g1 = ask([(KNOWN[5], canon), (KNOWN[6], 0), (KNOWN[7], 0), (KNOWN[8], 0)])  # M sem o nome
    r2, g2 = ask([(name, contra), (KNOWN[6], 0), (KNOWN[7], 0), (KNOWN[8], 0)])     # ensina contra
    r3, g3 = ask([(KNOWN[5], canon), (KNOWN[6], 0), (KNOWN[7], 0), (KNOWN[8], 0)])  # apaga -> sem o nome
    print(f"  pergunta sobre '{name}' (prior nos pesos = '{canon}')")
    print(f"   sem ensinar     -> '{r1}'  (portão {g1:.2f})   [esperado prior '{canon}']")
    print(f"   ensino '{contra}' -> '{r2}'  (portão {g2:.2f})   [esperado override '{contra}']")
    print(f"   esqueço (apago) -> '{r3}'  (portão {g3:.2f})   [volta ao prior '{canon}']")

    print("\n=== VEREDITO E3 ===")
    okdemo = (r1 == canon and r2 == contra and r3 == canon)
    if pr > 0.9 and ov > 0.9 and ab < 0.08 and okdemo:
        print("✅ E3 PASSA: o fato novo no ESTADO vence o prior dos PESOS, e esquecer "
              "restaura o prior — tudo via portão APRENDIDO (ablação derruba). Portão 3 ok.")
    else:
        print(f"➜ E3: prior={pr:.2f} override={ov:.2f} ablação={ab:.2f} demo={okdemo}")
    torch.save(model.state_dict(), "modelos/vereda_m_e3.pt")


if __name__ == "__main__":
    main()
