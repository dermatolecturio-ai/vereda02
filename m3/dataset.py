# -*- coding: utf-8 -*-
"""Dataset de frases PT-BR REAIS para o M3 (extração aprendida entidade+valor).

Reusa ATRIBUTOS/Pools de `regua.dataset` (mesmo vocabulário, mesmo split
train/held de nomes/valores). O que é NOVO aqui: ~18 fraseados por atributo
(contra os 3 do M0/M1), incluindo ordem valor-ANTES-da-entidade (o caso que
quebra encoders causais — lição herdada #10), split train/held também por
TEMPLATE (S3/S4: generalização estrutural, não só de vocabulário), e
marker-dropout (substitui o conector por um preenchedor neutro em parte do
treino, removendo o atalho léxico).

Cada item devolve offsets de CARACTERE da entidade e do valor na frase final
— a conversão para offsets de TOKEN (Qwen) acontece em m3/extractor.py.

Cada peça de template é ("lit", texto) ou uma das keywords
{"attr","attr_cap","nome","valor","valor_cap","marker"}. O piece "marker" já
inclui os espaços ao redor (" é ", ", " etc.) — templates NÃO colocam
literais de espaço perto dele.
"""
import random

from regua.dataset import ATRIBUTOS, PERGUNTA_TEMPLATES, Pools, _add_fato
from regua.judge import normalize

FILLERS_MARKER = [", ", " olha só, ", " ao que consta, ", " pelo visto, ",
                 " conforme dizem, "]


def _marker(default_word, dropout, rng):
    return rng.choice(FILLERS_MARKER) if dropout else " %s " % default_word


_TEMPLATES_ENTIDADE_PRIMEIRO = [
    ([("attr_cap",), ("lit", " de "), ("nome",), ("marker",), ("valor",),
      ("lit", ".")], "é"),
    ([("lit", "Anote: "), ("attr",), ("lit", " de "), ("nome",), ("marker",),
      ("valor",), ("lit", ".")], "é"),
    ([("lit", "Fiquei sabendo que "), ("attr",), ("lit", " de "), ("nome",),
      ("marker",), ("valor",), ("lit", ".")], "é"),
    ([("lit", "Segundo me disseram, "), ("attr",), ("lit", " de "),
      ("nome",), ("marker",), ("valor",), ("lit", ".")], "é"),
    ([("lit", "Confirmando: "), ("attr",), ("lit", " de "), ("nome",),
      ("marker",), ("valor",), ("lit", ", viu?")], "é"),
    ([("lit", "Pelo que eu sei, "), ("attr",), ("lit", " de "), ("nome",),
      ("marker",), ("valor",), ("lit", ".")], "é"),
    ([("attr_cap",), ("lit", " de "), ("nome",), ("lit", ", para constar,"),
      ("marker",), ("valor",), ("lit", ".")], "é"),
    ([("lit", "Registrando aqui: "), ("attr",), ("lit", " de "), ("nome",),
      ("marker",), ("valor",), ("lit", ".")], "é"),
    ([("lit", "Do que eu me lembro, "), ("attr",), ("lit", " de "),
      ("nome",), ("marker",), ("valor",), ("lit", ".")], "é"),
    ([("attr_cap",), ("lit", " de "), ("nome",), ("lit", " sempre"),
      ("marker",), ("valor",), ("lit", ".")], "foi"),
    ([("lit", "Ah, "), ("attr",), ("lit", " de "), ("nome",), ("marker",),
      ("valor",), ("lit", ", se não me engano.")], "é"),
    ([("lit", "Pode escrever: "), ("attr",), ("lit", " de "), ("nome",),
      ("marker",), ("valor",), ("lit", ".")], "é"),
]

_TEMPLATES_VALOR_PRIMEIRO = [
    ([("valor_cap",), ("marker",), ("attr",), ("lit", " de "), ("nome",),
      ("lit", ".")], "é"),
    ([("lit", "Segundo consta, "), ("valor",), ("marker",), ("attr",),
      ("lit", " de "), ("nome",), ("lit", ".")], "é"),
    ([("lit", "Anotado: "), ("valor",), ("marker",), ("attr",),
      ("lit", " de "), ("nome",), ("lit", ".")], "é"),
    ([("lit", "Dizem que "), ("valor",), ("marker",), ("attr",),
      ("lit", " de "), ("nome",), ("lit", ".")], "é"),
    ([("valor_cap",), ("lit", " — isso"), ("marker",), ("attr",),
      ("lit", " de "), ("nome",), ("lit", ".")], "é"),
    ([("lit", "Foi confirmado: "), ("valor",), ("marker",), ("attr",),
      ("lit", " de "), ("nome",), ("lit", ".")], "é"),
]

TEMPLATES = (
    [(pieces, marker, "entidade_primeiro")
     for pieces, marker in _TEMPLATES_ENTIDADE_PRIMEIRO]
    + [(pieces, marker, "valor_primeiro")
       for pieces, marker in _TEMPLATES_VALOR_PRIMEIRO])


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _split_templates(seed=7, held_frac=0.28):
    idx = list(range(len(TEMPLATES)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_held = max(2, int(len(idx) * held_frac))
    held_idx, train_idx = set(idx[:n_held]), set(idx[n_held:])
    orders_held = {TEMPLATES[i][2] for i in held_idx}
    if len(orders_held) < 2:
        for i in train_idx:
            if TEMPLATES[i][2] not in orders_held:
                held_idx.add(i)
                train_idx.discard(i)
                break
    return sorted(train_idx), sorted(held_idx)


TRAIN_TPL_IDX, HELD_TPL_IDX = _split_templates()


def render(pieces, nome, attr, valor, marker_default, marker_dropout, rng):
    out, spans = [], {}
    pos = 0

    def emit(txt):
        nonlocal pos
        out.append(txt)
        pos += len(txt)

    for kind, *lit in pieces:
        if kind == "lit":
            emit(lit[0])
        elif kind == "attr":
            emit(attr)
        elif kind == "attr_cap":
            emit(_cap(attr))
        elif kind == "nome":
            s = pos
            emit(nome)
            spans["entidade"] = (s, pos)
        elif kind == "valor":
            s = pos
            emit(valor)
            spans["valor"] = (s, pos)
        elif kind == "valor_cap":
            s = pos
            emit(_cap(valor))
            spans["valor"] = (s, pos)
        elif kind == "marker":
            emit(_marker(marker_default, marker_dropout, rng))
        else:
            raise ValueError(kind)
    return "".join(out), spans


def build_items(n, seed, split="held", marker_dropout_p=0.0):
    """split: 'train' ou 'held' (nomes/valores E templates)."""
    pools = Pools(split)
    tpl_idx = TRAIN_TPL_IDX if split == "train" else HELD_TPL_IDX
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        nome = pools.nome(rng)
        attr, pool_name = rng.choice(ATRIBUTOS)
        valor = pools.valor(rng, pool_name)
        ti = rng.choice(tpl_idx)
        pieces, marker_default, order = TEMPLATES[ti]
        dropout = rng.random() < marker_dropout_p
        frase, spans = render(pieces, nome, attr, valor, marker_default,
                              dropout, rng)
        if normalize(nome) == normalize(valor):
            continue
        items.append({
            "frase": frase,
            "nome": nome, "attr": attr, "valor": valor,
            "span_entidade_char": spans["entidade"],
            "span_valor_char": spans["valor"],
            "template_idx": ti, "order": order, "marker_dropout": dropout,
        })
    return items


def build_k_items(k, n, seed, split="held", marker_dropout_p=0.0):
    """k fatos por item, cada um em fraseado M3 (diverso, S3/S4), com os
    mesmos distratores fortes do M0/M1 (mesmo nome/outro attr; mesmo attr/
    outro nome). Usado para medir o portão fim-a-fim texto→M→resposta."""
    pools = Pools(split)
    tpl_idx = TRAIN_TPL_IDX if split == "train" else HELD_TPL_IDX
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        tuples = []
        if not _add_fato(rng, pools, tuples):
            continue
        t_nome, t_attr, _, t_valor = tuples[0]
        ok = True
        if k >= 3:
            attrs_outros = [a for a in ATRIBUTOS if a[0] != t_attr]
            ok = ok and _add_fato(rng, pools, tuples, forced_nome=t_nome,
                                  forced_attr=rng.choice(attrs_outros))
            ok = ok and _add_fato(rng, pools, tuples,
                                  forced_attr=(t_attr, tuples[0][2]))
        while ok and len(tuples) < k:
            ok = _add_fato(rng, pools, tuples)
        if not ok or len(tuples) != k:
            continue
        order = list(range(k))
        rng.shuffle(order)
        tuples = [tuples[i] for i in order]
        target_idx = order.index(0)
        frases, spans_ent, spans_val, attrs = [], [], [], []
        for (nome, attr, _, valor) in tuples:
            attrs.append(attr)
            ti = rng.choice(tpl_idx)
            pieces, marker_default, _ = TEMPLATES[ti]
            dropout = rng.random() < marker_dropout_p
            frase, spans = render(pieces, nome, attr, valor, marker_default,
                                  dropout, rng)
            frases.append(frase)
            spans_ent.append(spans["entidade"])
            spans_val.append(spans["valor"])
        pergunta = rng.choice(PERGUNTA_TEMPLATES).format(attr=t_attr,
                                                          nome=t_nome)
        items.append({
            "frases": frases, "attrs": attrs,
            "span_entidade_char": spans_ent, "span_valor_char": spans_val,
            "target_idx": target_idx, "pergunta": pergunta,
            "gold": t_valor, "nome": t_nome, "attr": t_attr,
        })
    return items


if __name__ == "__main__":
    print("templates: %d train, %d held" % (len(TRAIN_TPL_IDX),
                                            len(HELD_TPL_IDX)))
    for it in build_items(6, seed=1, split="held", marker_dropout_p=0.3):
        se, ee = it["span_entidade_char"]
        sv, ev = it["span_valor_char"]
        f = it["frase"]
        print(repr(f))
        print("  entidade=%r valor=%r ordem=%s dropout=%s" % (
            f[se:ee], f[sv:ev], it["order"], it["marker_dropout"]))
