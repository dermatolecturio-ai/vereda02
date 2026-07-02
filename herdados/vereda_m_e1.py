"""
VEREDA-M — E1: memória M EXPLÍCITA, endereçada por CONTEÚDO (não por posição).

E0 mostrou: o estado do delta-rule persiste 1 fato (PERSIST 1.0), mas com 2 fatos
colapsa por viés posicional (0.975/0.075). E1 constrói a solução: uma memória de
slots (chave,valor) com leitura por ATENÇÃO DE CONTEÚDO — e prova que resolve a
capacidade E a posição, num experimento que também confirma a correção honesta de E0.

Isolamento (disciplina): E1 mede o MECANISMO de memória (endereçar/armazenar/recuperar),
não a extração de texto cru (isso é E4). Cada fato entra como campos (nome, valor);
mas as REPRESENTAÇÕES e a recuperação são 100% APRENDIDAS -> ablação de pesos
aleatórios tem que cair ao acaso (1/|valores|). Sem parser/regex/DB.

Régua (tríade): (a) held-out = NOMES NOVOS (chaves nunca vistas) -> testa generalização
do endereçamento; (b) ablação aleatória -> ~acaso; (c) distribuição realista =
capacidade variável (k=2/4/8) e POSIÇÃO EMBARALHADA (query em qualquer slot). Bônus:
persistência (salva M em disco, recarrega, responde).
"""

import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from vereda_deltanet import DeltaNetBlock, RMSNorm
from build_cognition_tasks import make_name_pool

DEV = torch.device("cpu")
L = 12                                   # bytes por nome (pad)
VALUES = [                               # vocab FECHADO de valores (E1 isola memória)
    "azul", "verde", "preto", "branco", "roxo", "cinza", "vermelho", "amarelo",
    "rosa", "bege", "vinho", "creme", "dourado", "prata", "marrom", "lilas",
    "parque", "praia", "campo", "rio", "museu", "feira", "escola", "praca",
    "porto", "templo", "cinema", "clube", "ilha", "torre", "ponte", "horta",
    "novo", "velho", "leve", "forte", "fraco", "caro", "raro", "redondo",
    "liso", "quente", "frio", "seco", "limpo", "claro", "doce", "macio",
    "alto", "baixo",
]
VID = {v: i for i, v in enumerate(VALUES)}
NV = len(VALUES)


def enc_bytes(s):
    b = list(s.encode("utf-8"))[:L]
    return b + [0] * (L - len(b)), len(b)


class MemNet(nn.Module):
    """Encoder de string -> chave; memória (K,V) por sessão; leitura por atenção."""

    def __init__(self, d=64, heads=4):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(256, d)
        self.block = DeltaNetBlock(d, num_heads=heads, conv_size=4)
        self.norm = RMSNorm(d)
        self.key = nn.Linear(d, d, bias=False)          # string -> chave
        self.val_emb = nn.Embedding(NV, d)              # valor armazenado (aprendido)
        self.cls = nn.Linear(d, NV)                     # valor recuperado -> id
        self.scale = 1.0 / math.sqrt(d)

    def encode(self, bytes_NL, len_N):
        x = self.emb(bytes_NL)
        h = self.norm(self.block(x))
        last = h[torch.arange(h.size(0), device=h.device), (len_N - 1).clamp(min=0)]
        return self.key(last)                            # [N, d]

    def forward(self, fnames, flen, fvals, qname, qlen):
        B, k, _ = fnames.shape
        K = self.encode(fnames.reshape(B * k, L), flen.reshape(B * k)).reshape(B, k, self.d)
        V = self.val_emb(fvals)                          # [B,k,d]
        q = self.encode(qname, qlen)                     # [B,d]
        attn = torch.softmax(torch.einsum("bd,bkd->bk", q, K) * self.scale, -1)
        vread = torch.einsum("bk,bkd->bd", attn, V)      # [B,d]
        return self.cls(vread), attn

    # --- API de memória persistível (o "M" do VEREDA-M) ---
    @torch.no_grad()
    def write(self, names):
        """Escreve fatos numa memória M serializável: lista de (nome->chave, valor)."""
        bs = torch.tensor([enc_bytes(n)[0] for n, _ in names], device=DEV)
        ls = torch.tensor([enc_bytes(n)[1] for n, _ in names], device=DEV)
        Kk = self.encode(bs, ls)                          # [k,d]
        Vv = self.val_emb(torch.tensor([VID[v] for _, v in names], device=DEV))
        return {"K": Kk, "V": Vv}                         # <- isto salva/recarrega

    @torch.no_grad()
    def read(self, M, qname):
        b, l = enc_bytes(qname)
        q = self.encode(torch.tensor([b], device=DEV), torch.tensor([l], device=DEV))
        attn = torch.softmax((q @ M["K"].T) * self.scale, -1)
        vread = attn @ M["V"]
        return VALUES[int(self.cls(vread).argmax(-1).item())]


def make_batch(rng, B, names_pool, kmin, kmax, qpos=None):
    k = rng.randint(kmin, kmax)
    fnames = torch.zeros(B, k, L, dtype=torch.long)
    flen = torch.zeros(B, k, dtype=torch.long)
    fvals = torch.zeros(B, k, dtype=torch.long)
    qname = torch.zeros(B, L, dtype=torch.long)
    qlen = torch.zeros(B, dtype=torch.long)
    target = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        nms = rng.sample(names_pool, k)
        vals = [rng.randrange(NV) for _ in range(k)]
        for i, nm in enumerate(nms):
            bb, ll = enc_bytes(nm)
            fnames[b, i] = torch.tensor(bb); flen[b, i] = ll; fvals[b, i] = vals[i]
        qi = rng.randrange(k) if qpos is None else min(qpos, k - 1)
        bb, ll = enc_bytes(nms[qi])
        qname[b] = torch.tensor(bb); qlen[b] = ll; target[b] = vals[qi]
    return (fnames.to(DEV), flen.to(DEV), fvals.to(DEV), qname.to(DEV),
            qlen.to(DEV), target.to(DEV))


@torch.no_grad()
def acc(model, rng, B, pool, kmin, kmax, n=2048, qpos=None):
    model.eval(); ok = 0; tot = 0
    for _ in range(n // B):
        fn, fl, fv, qn, ql, tg = make_batch(rng, B, pool, kmin, kmax, qpos)
        logits, _ = model(fn, fl, fv, qn, ql)
        ok += (logits.argmax(-1) == tg).sum().item(); tot += B
    return ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kmax", type=int, default=8)
    args = ap.parse_args()

    pool = make_name_pool(800)
    train_names, held_names = pool[:680], pool[680:]
    torch.manual_seed(0)
    model = MemNet(d=args.d).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(0); erng = random.Random(999)
    npar = sum(p.numel() for p in model.parameters())

    print("VEREDA-M E1 — memória endereçada por CONTEÚDO (resolve o colapso do E0)")
    print(f"params={npar:,} | nomes TR={len(train_names)} HE={len(held_names)} | "
          f"valores={NV} (acaso={1/NV:.3f}) | k=2..{args.kmax}, posição embaralhada\n")

    import time
    t0 = time.time(); best = 0.0
    model.train()
    for step in range(1, args.steps + 1):
        fn, fl, fv, qn, ql, tg = make_batch(rng, args.batch, train_names, 2, args.kmax)
        logits, _ = model(fn, fl, fv, qn, ql)
        loss = F.cross_entropy(logits, tg)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 300 == 0:
            h = acc(model, erng, args.batch, held_names, 2, args.kmax, 512)
            sps = (time.time() - t0) / step
            if h > best:
                best = h; torch.save(model.state_dict(), "modelos/vereda_m_e1.pt")
            print(f"  step {step:04d}/{args.steps} loss={loss.item():.4f} "
                  f"held(nomes novos)={h:.3f}  [{sps:.2f}s/step]")
        model.train()

    print("\n=== avaliação (NOMES NOVOS held-out) ===")
    model.eval()
    for k in (2, 4, 8):
        a = acc(model, erng, args.batch, held_names, k, k, 1536)
        print(f"  capacidade k={k} fatos, query EMBARALHADA: {a:.3f}")
    print("  posição da query (k=8) — tem que ser SIMÉTRICO (sem viés do E0):")
    for p in (0, 3, 7):
        a = acc(model, erng, args.batch, held_names, 8, 8, 1024, qpos=p)
        print(f"     pos {p}: {a:.3f}")
    torch.manual_seed(7)
    rnd = MemNet(d=args.d).to(DEV).eval()
    ab = acc(rnd, erng, args.batch, held_names, 4, 4, 1024)
    print(f"  ablação (pesos aleatórios): {ab:.3f}   (acaso={1/NV:.3f})")

    # PERSISTÊNCIA: escreve M, salva em disco, recarrega em modelo NOVO, lê
    print("\n=== persistência (salva M -> modelo NOVO -> lê) ===")
    facts = [("Ana", "azul"), ("Bruno", "verde"), ("Carla", "preto"), ("Diego", "rio")]
    M = model.write(facts)
    torch.save(M, "/tmp/vereda_m_M.pt")
    m2 = MemNet(d=args.d).to(DEV).eval(); m2.load_state_dict(torch.load("modelos/vereda_m_e1.pt"))
    M2 = torch.load("/tmp/vereda_m_M.pt")
    okp = 0
    for nm, gold in facts:
        r = m2.read(M2, nm)
        ok = r == gold; okp += ok
        print(f"   '{nm}' -> '{r}'  (esperado '{gold}') {'✓' if ok else '✗'}")
    os.path.exists("/tmp/vereda_m_M.pt") and os.remove("/tmp/vereda_m_M.pt")

    print("\n=== VEREDITO E1 ===")
    k4 = acc(model, erng, args.batch, held_names, 4, 4, 1536)
    sym = abs(acc(model, erng, args.batch, held_names, 8, 8, 1024, 0) -
              acc(model, erng, args.batch, held_names, 8, 8, 1024, 7))
    if k4 > 0.85 and ab < 0.08 and sym < 0.1 and okp == len(facts):
        print(f"✅ E1 PASSA: memória de conteúdo recupera por NOME (não posição), "
              f"k=4 {k4:.2f}, simétrica, ablação {ab:.2f}, persiste. Resolve o colapso do E0.")
    else:
        print(f"➜ E1 parcial: k4={k4:.2f} ablação={ab:.2f} assimetria={sym:.2f} "
              f"persist={okp}/{len(facts)} — ver onde aperta.")


if __name__ == "__main__":
    main()
