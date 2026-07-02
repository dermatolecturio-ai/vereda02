"""
VEREDA (regime novo) — sonda do núcleo pré-treinado: Qwen2.5-0.5B-Instruct.

Prova de por que mudar de regime: o mesmo teste em que nosso modelo-do-zero foi
PAPAGAIO ("se um carro é preto → vermelha"), agora com entendimento pré-treinado.
E mostra o buraco que a NOSSA memória vai preencher: fato dado em contexto = OK;
fato fora do contexto = ele não tem (sem persistência) -> é aí que entra o VEREDA-M.
Mede o tempo real no M1 (sem chutar).
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cpu"   # 0.5B em CPU é confiável no M1; MPS é mais rápido mas pode ter ops faltando

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
print(f"carregado {MODEL} em {time.time()-t0:.1f}s | "
      f"params={sum(p.numel() for p in model.parameters())/1e6:.0f}M | device={DEV}\n")


@torch.no_grad()
def ask(user, max_new=24):
    msgs = [{"role": "system", "content": "Você responde em português, curto e direto."},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(DEV)
    t = time.time()
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    dt = time.time() - t
    ans = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return ans, dt


print("=== 1) ENTENDIMENTO (o teste que o modelo-do-zero errou de papagaio) ===")
for q in ["Se um carro é preto, qual é a cor dele?",
          "Se um carro é azul, qual é a cor dele?",
          "A casa de Pedro é vermelha. De que cor é a casa de Pedro?"]:
    a, dt = ask(q)
    print(f"  Q: {q}\n     -> {a}   [{dt:.1f}s]")

print("\n=== 2) RACIOCÍNIO simples em PT ===")
for q in ["Maria tem 3 maçãs e ganha 2. Quantas tem agora?",
          "Qual é o oposto de 'quente'?"]:
    a, dt = ask(q)
    print(f"  Q: {q}\n     -> {a}   [{dt:.1f}s]")

print("\n=== 3) O BURACO que a memória do VEREDA-M preenche ===")
a, _ = ask("Qual é a cor da chave de Ana?")
print(f"  sem nunca ensinar -> {a}   (não tem como saber: fato não está nos pesos nem no contexto)")
a, _ = ask("A chave de Ana é dourada. Qual é a cor da chave de Ana?")
print(f"  com o fato no contexto -> {a}   (entende, mas isso SOME quando a conversa acaba)")
print("\n  => fluência+entendimento vêm DE GRAÇA do pré-treino; o que falta é")
print("     PERSISTÊNCIA/ensinabilidade no estado -> exatamente o E1-E3 + .vereda.")
