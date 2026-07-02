"""
VEREDA-M — E2: capacidade + EDITAR/ESQUECER (o que torna a memória ENSINÁVEL).

Usa o modelo E1 já treinado (sem retreino). Mede:
 (1) CAPACIDADE: recall conforme M cresce (k=4..100 fatos). Treinou em k=2..8 ->
     testa se o endereçamento por conteúdo escala além do treino.
 (2) EDITAR: sobrescrever o valor de um fato (corrigir) -> leitura passa a dar o NOVO.
 (3) ESQUECER: apagar o slot de um fato -> deixa de ser recuperado; os outros ficam.
Operações de memória atuam em M (o estado); a LEITURA segue o mecanismo aprendido.
Régua: nomes NOVOS (held), exact-match; comparado ao acaso 1/50=0.020.
"""

import random
import torch
from vereda_m_e1 import MemNet, VALUES, VID, enc_bytes
from build_cognition_tasks import make_name_pool

DEV = torch.device("cpu")
model = MemNet(d=64).to(DEV).eval()
model.load_state_dict(torch.load("modelos/vereda_m_e1.pt"))
held = make_name_pool(800)[680:]            # 120 nomes NOVOS
rng = random.Random(7)


@torch.no_grad()
def keyvec(name):
    b, l = enc_bytes(name)
    return model.encode(torch.tensor([b], device=DEV), torch.tensor([l], device=DEV))[0]


@torch.no_grad()
def build_M(facts):
    K = torch.stack([keyvec(n) for n, _ in facts])
    V = model.val_emb(torch.tensor([VID[v] for _, v in facts], device=DEV))
    return {"K": K, "V": V}


@torch.no_grad()
def read(M, name):
    q = keyvec(name)
    attn = torch.softmax((M["K"] @ q) * model.scale, -1)
    vread = attn @ M["V"]
    return VALUES[int(model.cls(vread).argmax(-1).item())]


@torch.no_grad()
def slot_of(M, name):                        # endereçamento por conteúdo (aprendido)
    q = keyvec(name)
    return int(((M["K"] @ q)).argmax().item())


def edit(M, name, new_val):                  # CORRIGIR um fato
    i = slot_of(M, name)
    M["V"][i] = model.val_emb(torch.tensor(VID[new_val], device=DEV))


def forget(M, name):                         # ESQUECER um fato (remove o slot)
    i = slot_of(M, name)
    keep = [j for j in range(M["K"].size(0)) if j != i]
    M["K"] = M["K"][keep]; M["V"] = M["V"][keep]


# (1) CAPACIDADE
print("VEREDA-M E2 — capacidade + editar/esquecer (modelo E1, sem retreino)")
print(f"acaso = {1/len(VALUES):.3f}\n(1) CAPACIDADE (recall com M crescendo, nomes novos):")
for k in (4, 8, 16, 32, 64, 100):
    ok = tot = 0
    for _ in range(40):
        nms = rng.sample(held, k)
        facts = [(n, VALUES[rng.randrange(len(VALUES))]) for n in nms]
        M = build_M(facts)
        for n, gold in random.sample(facts, min(8, k)):
            ok += (read(M, n) == gold); tot += 1
    print(f"   k={k:>3} fatos: recall {ok/tot:.3f}")

# (2)+(3) EDITAR e ESQUECER, cenário ensinável
print("\n(2)+(3) ENSINÁVEL: ensina 12, edita 3, esquece 3, relê tudo:")
nms = rng.sample(held, 12)
facts = [(n, VALUES[rng.randrange(len(VALUES))]) for n in nms]
truth = dict(facts)
M = build_M(facts)
base = sum(read(M, n) == v for n, v in facts)
print(f"   recall inicial: {base}/12")

editar = nms[:3]
for n in editar:                              # corrige p/ um valor novo diferente
    nv = VALUES[(VID[truth[n]] + 7) % len(VALUES)]
    edit(M, n, nv); truth[n] = nv
ed_ok = sum(read(M, n) == truth[n] for n in editar)
print(f"   após EDITAR 3: leem o novo valor? {ed_ok}/3")

esquecer = nms[3:6]
for n in esquecer:
    forget(M, n)
restantes = nms[6:]                           # 6 intactos
gone = sum(read(M, n) != truth[n] for n in esquecer)   # esquecido = não dá mais o valor
keep_ok = sum(read(M, n) == truth[n] for n in restantes)
print(f"   após ESQUECER 3: esquecidos somem? {gone}/3 | intactos preservados? {keep_ok}/6")

print("\n=== VEREDITO E2 ===")
cap64 = None
ok = tot = 0
for _ in range(40):
    nms = rng.sample(held, 64); facts = [(n, VALUES[rng.randrange(len(VALUES))]) for n in nms]
    M = build_M(facts)
    for n, gold in random.sample(facts, 8):
        ok += (read(M, n) == gold); tot += 1
cap64 = ok / tot
if cap64 > 0.8 and ed_ok == 3 and gone == 3 and keep_ok == 6:
    print(f"✅ E2 PASSA: escala a {64} fatos ({cap64:.2f}), e é ENSINÁVEL (edita e esquece "
          f"sem quebrar o resto). Memória = primitivo completo.")
else:
    print(f"➜ E2: cap@64={cap64:.2f}, editar={ed_ok}/3, esquecer={gone}/3, intactos={keep_ok}/6.")
