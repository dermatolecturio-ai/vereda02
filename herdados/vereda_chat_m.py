"""
VEREDA-M — CHAT: teste o sistema novo (memória ensinável sobre Qwen congelado).

USO:
  python3 vereda_chat_m.py            # chat interativo
  python3 vereda_chat_m.py --test     # roda um roteiro automático (verifica que funciona)

NO CHAT:
  - Escreva um FATO (sem '?')  -> ele APRENDE e guarda na memória.
      ex:  A chave de Ana é dourada.
  - Faça uma PERGUNTA (com '?') -> ele RESPONDE (usa a memória se souber; senão, o Qwen).
      ex:  Qual é a cor da chave de Ana?
  Comandos:  :memoria   :salva <arq>   :carrega <arq>   :esquece <texto>   :ajuda   :sair

COMO FUNCIONA (honesto): a COGNIÇÃO (achar o fato certo pela pergunta) é NEURAL —
embedding do Qwen congelado + a nossa cabeça aprendida; ablação derruba. O Qwen
gera a resposta fluente usando o fato recuperado. A memória é salva no nosso .vereda.
"""
import sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vereda_m_qwen import Head, MAXLEN, DIN
from vereda_format import save_vereda, load_vereda

DEV = "cpu"
THR = 0.52                                   # limiar de similaridade p/ "usar a memória"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PERSONA = ("Você é o VEREDA, um assistente em português, um experimento de memória "
           "ensinável sobre um modelo pequeno. Responda curto, honesto. Se não souber, "
           "diga que não sabe. Não invente.")
TEACH_PREFIXES = ("aprenda que ", "lembre que ", "lembre-se que ", "anote que ", "guarde que ")

print("carregando Qwen + cabeça VEREDA-M ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
head = Head().to(DEV).eval()
head.load_state_dict(torch.load("modelos/vereda_m_qwen_head.pt", weights_only=False))

MEM_K = []        # chaves (tensores) — a parte NEURAL
MEM_T = []        # texto do fato (a carga/auditoria)


@torch.no_grad()
def key_of(text):
    enc = tok([text], return_tensors="pt", padding="max_length",
              truncation=True, max_length=MAXLEN).to(DEV)
    states = qwen(**enc, output_hidden_states=True).hidden_states[-1].float()
    return head(states, enc.attention_mask.bool())[0]


@torch.no_grad()
def gen(prompt_user, system=PERSONA):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt_user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(DEV)
    out = qwen.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def teach(fact):
    MEM_K.append(key_of(fact)); MEM_T.append(fact)
    return f"aprendido ({len(MEM_T)} fato(s) na memória)."


def retrieve(question):
    if not MEM_K:
        return None, 0.0
    q = key_of(question)
    sims = torch.stack(MEM_K) @ q
    i = int(sims.argmax()); return (MEM_T[i], float(sims.max()))


def ask(question):
    fact, s = retrieve(question)
    if fact is not None and s >= THR:
        ans = gen(f"Fato que você aprendeu: {fact}\nPergunta: {question}",
                  PERSONA + " Use APENAS o fato dado para responder.")
        return f"{ans}\n   [memória · fonte: \"{fact}\" · sim {s:.2f}]"
    # sem memória relevante -> conversa com a persona (não inventa fato)
    ans = gen(question)
    tag = f"[Qwen · sem memória relevante (melhor sim {s:.2f})]" if MEM_K else "[Qwen]"
    return f"{ans}\n   {tag}"


def converse(line):                          # bate-papo: NÃO guarda como fato
    return f"{gen(line)}\n   [Qwen · conversa]"


def save(path):
    if not path.endswith(".vereda"):
        path += ".vereda"
    K = torch.stack(MEM_K) if MEM_K else torch.zeros(0, 128)
    save_vereda(path, core_state=head.state_dict(),
                core_config={"base": MODEL + "(frozen)", "head_d": 128, "din": DIN},
                M={"K": K, "V": K}, value_vocab=[],
                facts_ledger=[[t] for t in MEM_T], notes="chat VEREDA-M")
    return f"salvo em {path} ({len(MEM_T)} fatos)"


def load(path):
    if not path.endswith(".vereda"):
        path += ".vereda"
    art = load_vereda(path)
    head.load_state_dict(art["core_state"])
    MEM_K.clear(); MEM_T.clear()
    K = art["M"]["K"]
    for row, t in zip(K, art["manifest"]["facts_ledger_AUDIT_ONLY"]):
        MEM_K.append(row); MEM_T.append(t[0] if isinstance(t, list) else t)
    return f"carregado {path} ({len(MEM_T)} fatos)"


def forget(text):
    if not MEM_K:
        return "memória vazia."
    q = key_of(text); i = int((torch.stack(MEM_K) @ q).argmax())
    rem = MEM_T.pop(i); MEM_K.pop(i)
    return f"esqueci: \"{rem}\""


HELP = ("ENSINAR: ':ensina <fato>' ou 'aprenda que ...'  (só assim ele guarda)\n"
        "  PERGUNTAR: termine com '?'   |   CONVERSAR: escreva normal\n"
        "  :memoria  :salva <arq>  :carrega <arq>  :esquece <texto>  :ajuda  :sair")


def handle(line):
    line = line.strip()
    if not line:
        return ""
    if line.startswith(":"):
        cmd, *rest = line[1:].split(" ", 1); arg = rest[0] if rest else ""
        if cmd in ("sair", "q"): return None
        if cmd == "ajuda": return HELP
        if cmd == "ensina": return teach(arg) if arg else "uso: :ensina <fato>"
        if cmd == "memoria":
            return "memória:\n" + ("\n".join(f"  {i+1}. {t}" for i, t in enumerate(MEM_T)) or "  (vazia)")
        if cmd == "salva": return save(arg or "modelos/minha_memoria")
        if cmd == "carrega": return load(arg or "modelos/minha_memoria")
        if cmd == "esquece": return forget(arg)
        return f"comando? {HELP}"
    low = line.lower()
    for p in TEACH_PREFIXES:                 # ensinar só por gatilho explícito
        if low.startswith(p):
            return teach(line[len(p):].strip())
    if line.endswith("?"):
        return ask(line)
    return converse(line)                    # resto = conversa, NÃO vira fato


def main():
    if "--test" in sys.argv:
        for line in ["Olá", "Quem é você?",
                     ":ensina A chave de Ana é dourada.",
                     "aprenda que o drone de Bruno tem oito hélices",
                     "Como é a chave de Ana?", "Como é o drone de Bruno?",
                     "Quanto é 7 mais 5?", ":memoria"]:
            print(f">>> {line}\n{handle(line)}\n")
        return
    print("\nVEREDA-M pronto.\n" + HELP + "\n")
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\ntchau!"); break
        out = handle(line)
        if out is None:
            print("tchau!"); break
        if out:
            print(out)


if __name__ == "__main__":
    main()
