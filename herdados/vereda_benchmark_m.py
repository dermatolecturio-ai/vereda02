"""
VEREDA-M — BENCHMARK JUSTO: Qwen cru vs Qwen+memória, SEM armar pra nós ganharmos.

Mesmas perguntas, mesmos fatos, MESMO critério de acerto pra todos. Damos ao cru a
melhor chance (TODOS os fatos no contexto = baseline forte). Variamos o nº de fatos.

  A) cru SEM fatos      -> piso (não pode saber fato de runtime)
  B) cru + TODOS no contexto -> baseline FORTE (LLM é ótimo in-context)
  C) VEREDA (recupera 1 fato da memória aprendida) -> responde

Métrica: o VALOR-ouro (normalizado) aparece na resposta (substring). Aplicada IGUAL
a A, B, C. Reporta acerto + tokens de prompt (custo). Imprime exemplos crus p/ auditar.
"""
import random, unicodedata, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vereda_m_qwen import Head, MAXLEN, DIN

DEV = "cpu"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
head = Head().to(DEV).eval(); head.load_state_dict(torch.load("modelos/vereda_m_qwen_head.pt", weights_only=False))

NAMES = ["Ana", "Bruno", "Carla", "Diego", "Elena", "Felipe", "Gabriela", "Hugo",
         "Inês", "João", "Letícia", "Marcos", "Natália", "Otávio", "Paula", "Rafael",
         "Sofia", "Tiago", "Vera", "Rodrigo", "Beatriz", "Caio", "Daniela", "Eduardo",
         "Fábio", "Helena", "Ivo", "Júlia", "Karina", "Lucas", "Mariana", "Nelson"]
OBJS = [("carro", "o"), ("livro", "o"), ("gato", "o"), ("relógio", "o"), ("sofá", "o"),
        ("celular", "o"), ("chapéu", "o"), ("violão", "o"), ("casaco", "o"), ("quadro", "o"),
        ("casa", "a"), ("bicicleta", "a"), ("mochila", "a"), ("caneta", "a"), ("jaqueta", "a"),
        ("planta", "a"), ("camisa", "a"), ("mesa", "a"), ("bolsa", "a"), ("xícara", "a")]
VALS = ["azul", "vermelho", "dourado", "enorme", "antigo", "novo", "barato", "italiano",
        "minúsculo", "pesado", "turquesa", "veloz", "elegante", "rústico", "macio",
        "brilhante", "escuro", "redondo", "leve", "moderno", "frágil", "verde", "roxo",
        "prateado", "quadrado", "robusto", "claro", "amarelo", "vintage", "importado"]


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def make_world(rng, n):
    facts = []; seen = set()
    while len(facts) < n:
        nm = rng.choice(NAMES); o, a = rng.choice(OBJS)
        if (nm, o) in seen: continue
        seen.add((nm, o)); facts.append((nm, o, a, rng.choice(VALS)))
    return facts


@torch.no_grad()
def gen(system, user, max_new=24):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(DEV)
    out = qwen.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return ans, ids.input_ids.shape[1]


@torch.no_grad()
def key_of(text):
    enc = tok([text], return_tensors="pt", padding="max_length", truncation=True, max_length=MAXLEN).to(DEV)
    st = qwen(**enc, output_hidden_states=True).hidden_states[-1].float()
    return head(st, enc.attention_mask.bool())[0]


SYS = "Responda em português com UMA palavra (a característica). Se não souber, diga 'não sei'."


def run(n, n_q=15, seed=0, verbose=False):
    rng = random.Random(seed)
    facts = make_world(rng, n)
    factlines = [f"{a.upper()} {o} de {nm} é {v}." for nm, o, a, v in facts]
    Kmem = torch.stack([key_of(line) for line in factlines])     # memória VEREDA
    qidx = rng.sample(range(n), min(n_q, n))
    accA = accB = accC = 0; tokB = tokC = 0
    for j, qi in enumerate(qidx):
        nm, o, a, v = facts[qi]
        question = f"Qual é a característica {('do' if a=='o' else 'da')} {o} de {nm}?"
        # A: sem fatos
        ansA, _ = gen(SYS, question)
        # B: TODOS os fatos no contexto
        ansB, tB = gen(SYS + "\nFatos:\n" + "\n".join(factlines), question)
        # C: VEREDA recupera 1 fato
        q = key_of(question); i = int((Kmem @ q).argmax())
        ansC, tC = gen(SYS + "\nFato:\n" + factlines[i], question)
        accA += norm(v) in norm(ansA); accB += norm(v) in norm(ansB); accC += norm(v) in norm(ansC)
        tokB += tB; tokC += tC
        if verbose and j < 4:
            print(f"   Q: {question}  [ouro: {v}]")
            print(f"      A(cru)='{ansA}' | B(contexto)='{ansB}' | C(VEREDA←'{facts[i][:2]}')='{ansC}'")
    nq = len(qidx)
    return dict(A=accA/nq, B=accB/nq, C=accC/nq, tB=tokB/nq, tC=tokC/nq, nq=nq)


print("BENCHMARK JUSTO — Qwen cru vs Qwen+VEREDA (mesmas perguntas/critério)\n")
print(f"{'nº fatos':>9}{'A cru':>9}{'B contexto':>12}{'C VEREDA':>10}{'tok B':>8}{'tok C':>8}")
for n in (5, 30, 100):
    r = run(n, verbose=(n == 30))
    print(f"{n:>9}{r['A']:>9.2f}{r['B']:>12.2f}{r['C']:>10.2f}{r['tB']:>8.0f}{r['tC']:>8.0f}")
    if n == 30:
        print()

print("\nLeitura honesta: B (tudo no contexto) é o baseline FORTE e justo. VEREDA (C) só")
print("vale se chegar perto de B com MUITO menos tokens (custo), e/ou superá-lo quando")
print("o nº de fatos cresce e o contexto incha. Se C < B sempre e sem economia, perdemos.")
