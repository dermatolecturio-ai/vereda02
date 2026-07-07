# -*- coding: utf-8 -*-
"""VEREDA 2 — chat demo local: ensine fatos em PT-BR, pergunte depois.

    python3 vereda_chat.py            # CPU, ~20s para carregar

Como usar:
  - Frase SEM "?" ao final  → ensina (extrator M3 lê entidade+valor)
  - Frase COM "?" ao final  → pergunta (chave M1 acha o fato; Θ responde)
  - /fatos                  → lista a memória
  - /esquecer N             → apaga o fato N (e só ele)
  - /sair                   → salva e sai

A memória persiste em `memoria.vereda` entre sessões (feche, reabra: lembra).

Honestidade sobre o caminho (CARTA §3): extração de entidade/valor, chave de
retrieval e geração são 100% aprendidas (M3+M1+Θ, medidas nos relatórios).
O fato guardado é a FRASE ORIGINAL do usuário (chave e injeção); a extração
M3 aparece como eco para você conferir o que foi entendido. A convenção
"termina com ? = pergunta" é interface, não capacidade alegada.
Números oficiais: extração 0.874, fim-a-fim 0.853 @ k=8.
"""
import os

import torch

from m1.head import Head, encode_batch
from m2.pipeline import M1_CKPT
from m2.run import _prompt_T
from m3.pipeline import M3_CKPT, extract_batch
from regua.qwen_io import generate_batch, load

MEMORIA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "memoria.vereda")


class Memoria(object):
    def __init__(self):
        self.fatos = []   # [{"fato": str, "original": str}]
        self.keys = torch.zeros(0, 128)

    def add(self, fato, original, key):
        self.fatos.append({"fato": fato, "original": original})
        self.keys = torch.cat([self.keys, key.unsqueeze(0)], 0)

    def esquecer(self, i):
        del self.fatos[i]
        self.keys = torch.cat([self.keys[:i], self.keys[i + 1:]], 0)

    def top1(self, qkey):
        if not self.fatos:
            return None
        return int(torch.argmax(self.keys @ qkey).item())

    def save(self, path=MEMORIA):
        torch.save({"fatos": self.fatos, "keys": self.keys}, path)

    @classmethod
    def load(cls, path=MEMORIA):
        m = cls()
        if os.path.exists(path):
            d = torch.load(path)
            m.fatos, m.keys = d["fatos"], d["keys"]
        return m


def m1_key(tok, model, head, texto):
    side = tok.padding_side
    tok.padding_side = "right"  # consistente com o treino da cabeça
    states, masks = encode_batch(model, tok, [texto], device=model.device,
                                 batch_size=1)
    tok.padding_side = side
    with torch.no_grad():
        return head(states.float().to(model.device),
                    masks.to(model.device))[0].cpu()


def main():
    print("Carregando VEREDA (Qwen 0.5B congelado + cabeças aprendidas)...",
          flush=True)
    tok, model = load("cpu", "float32")

    head = Head()
    head.load_state_dict(torch.load(M1_CKPT, map_location="cpu"))
    head.eval()

    from m3.extractor import Extractor
    extractor = Extractor()
    extractor.load_state_dict(torch.load(M3_CKPT, map_location="cpu"))
    extractor.eval()

    mem = Memoria.load()
    print("Pronto. %d fato(s) na memória (memoria.vereda)." % len(mem.fatos))
    print("Ensine com uma frase; pergunte terminando com '?'. "
          "/fatos /esquecer N /sair\n", flush=True)

    while True:
        try:
            linha = input("> ").strip()
        except EOFError:
            linha = "/sair"
        if not linha:
            continue

        if linha == "/sair":
            mem.save()
            print("memória salva em %s — até a próxima." % MEMORIA)
            return
        if linha == "/fatos":
            if not mem.fatos:
                print("(memória vazia)")
            for i, f in enumerate(mem.fatos):
                print("  [%d] %s" % (i, f["fato"]))
            continue
        if linha.startswith("/esquecer"):
            try:
                i = int(linha.split()[1])
                alvo = mem.fatos[i]["fato"]
                mem.esquecer(i)
                mem.save()
                print("esquecido: %s (%d restam)" % (alvo, len(mem.fatos)))
            except (IndexError, ValueError):
                print("uso: /esquecer N   (veja os N com /fatos)")
            continue

        if linha.endswith("?"):
            if not mem.fatos:
                print("(ainda não sei nada — me ensine um fato primeiro)")
                continue
            qkey = m1_key(tok, model, head, linha)
            i = mem.top1(qkey)
            fato = mem.fatos[i]["fato"]
            answers, _ = generate_batch(
                tok, model, [_prompt_T(fato, linha)], batch_size=1,
                max_new_tokens=32)
            print(answers[0].strip())
            print("  (memória usada: [%d] %s)" % (i, fato))
        else:
            ex = extract_batch(model, tok, extractor, [linha],
                               model.device)[0]
            key = m1_key(tok, model, head, linha)
            mem.add(linha, linha, key)
            mem.save()
            print("  [aprendido #%d] %s" % (len(mem.fatos) - 1, linha))
            print("    (entendi: entidade=%r, valor=%r)"
                  % (ex["entidade"], ex["valor"]))


if __name__ == "__main__":
    main()
