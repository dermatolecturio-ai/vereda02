# -*- coding: utf-8 -*-
"""Datasets do M4: priores reais (capitais, PT-BR) + itens estruturados.

PRIORES: pares (entidade, capital) inequívocos que um LM pré-treinado tende a
saber — o prior que o override precisa VENCER. Casos ambíguos/polêmicos
ficaram de fora. O baseline mede em quais o prior do Qwen 0.5B existe mesmo.

Itens estruturados (tuplas expostas): edição precisa de pergunta para
QUALQUER fato do item, não só o alvo — por isso não dá para reusar
regua.dataset.build_items (que só devolve textos).
"""
import random

from regua.dataset import (ATRIBUTOS, FATO_TEMPLATES, PERGUNTA_TEMPLATES,
                           Pools, _add_fato)
from regua.judge import normalize

# (entidade, tipo, capital canônica)
PRIORES = [
    ("Brasil", "país", "Brasília"), ("França", "país", "Paris"),
    ("Portugal", "país", "Lisboa"), ("Espanha", "país", "Madri"),
    ("Itália", "país", "Roma"), ("Alemanha", "país", "Berlim"),
    ("Japão", "país", "Tóquio"), ("China", "país", "Pequim"),
    ("Rússia", "país", "Moscou"), ("Argentina", "país", "Buenos Aires"),
    ("Chile", "país", "Santiago"), ("Peru", "país", "Lima"),
    ("Colômbia", "país", "Bogotá"), ("Uruguai", "país", "Montevidéu"),
    ("Paraguai", "país", "Assunção"), ("México", "país", "Cidade do México"),
    ("Canadá", "país", "Ottawa"), ("Egito", "país", "Cairo"),
    ("Grécia", "país", "Atenas"), ("Turquia", "país", "Ancara"),
    ("Índia", "país", "Nova Délhi"), ("Austrália", "país", "Camberra"),
    ("Cuba", "país", "Havana"), ("Venezuela", "país", "Caracas"),
    ("Equador", "país", "Quito"), ("Suécia", "país", "Estocolmo"),
    ("Noruega", "país", "Oslo"), ("Dinamarca", "país", "Copenhague"),
    ("Áustria", "país", "Viena"), ("Bélgica", "país", "Bruxelas"),
    ("Polônia", "país", "Varsóvia"), ("Hungria", "país", "Budapeste"),
    ("Irlanda", "país", "Dublin"), ("Coreia do Sul", "país", "Seul"),
    ("Indonésia", "país", "Jacarta"), ("Tailândia", "país", "Bangkok"),
    ("Vietnã", "país", "Hanói"), ("Marrocos", "país", "Rabat"),
    ("Angola", "país", "Luanda"), ("Moçambique", "país", "Maputo"),
    ("Ceará", "estado", "Fortaleza"), ("Bahia", "estado", "Salvador"),
    ("Pernambuco", "estado", "Recife"), ("Piauí", "estado", "Teresina"),
    ("Maranhão", "estado", "São Luís"), ("Pará", "estado", "Belém"),
    ("Amazonas", "estado", "Manaus"), ("Paraná", "estado", "Curitiba"),
]

PERGUNTA_CAPITAL = [
    "Qual é a capital de {ent}?",
    "Me diga a capital de {ent}.",
    "Você lembra qual é a capital de {ent}?",
]
FATO_CAPITAL = [
    "Anote: a capital de {ent} é {valor}.",
    "A capital de {ent} é {valor}.",
    "Fiquei sabendo que a capital de {ent} é {valor}.",
]


def build_tuples_item(rng, pools, k):
    """k tuplas (nome, attr, pool, valor) com os invariantes da régua."""
    tuples = []
    if not _add_fato(rng, pools, tuples):
        return None
    if k >= 3:
        t_nome, t_attr = tuples[0][0], tuples[0][1]
        attrs_outros = [a for a in ATRIBUTOS if a[0] != t_attr]
        if not _add_fato(rng, pools, tuples, forced_nome=t_nome,
                         forced_attr=rng.choice(attrs_outros)):
            return None
        if not _add_fato(rng, pools, tuples,
                         forced_attr=(t_attr, tuples[0][2])):
            return None
    while len(tuples) < k:
        if not _add_fato(rng, pools, tuples):
            return None
    return tuples


def render_fato(rng, nome, attr, valor):
    tpl = rng.choice(FATO_TEMPLATES)
    return tpl.format(attr=attr, attr_cap=attr.capitalize(), nome=nome,
                      valor=valor)


def render_pergunta(rng, nome, attr):
    return rng.choice(PERGUNTA_TEMPLATES).format(attr=attr, nome=nome)


def build_override_items(n, seed, k=8, split="held"):
    """Item: fato CONTRAFACTUAL (capital errada de propósito) + k-1
    distratores da régua, posição embaralhada (S4)."""
    pools = Pools(split)
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        ent, _, canonica = rng.choice(PRIORES)
        contra = rng.choice(pools.cidades)
        cn, en, kn = normalize(contra), normalize(ent), normalize(canonica)
        if cn == kn or cn in en or en in cn:
            continue
        tuples = build_tuples_item(rng, pools, k - 1)
        if tuples is None:
            continue
        if any(kn in normalize(v) or normalize(v) in cn
               for _, _, _, v in tuples):
            continue
        fatos = [render_fato(rng, nm, at, v) for nm, at, _, v in tuples]
        fato_contra = rng.choice(FATO_CAPITAL).format(ent=ent, valor=contra)
        pos = rng.randrange(k)
        fatos.insert(pos, fato_contra)
        items.append({
            "fatos": fatos, "target_idx": pos,
            "pergunta": rng.choice(PERGUNTA_CAPITAL).format(ent=ent),
            "gold": contra, "prior": canonica, "entidade": ent,
        })
    return items


def build_edit_items(n, seed, k=32, split="held"):
    """Item: k fatos; alvo ganha valor NOVO (edição); vizinho aleatório
    não-editado ganha pergunta própria (integridade)."""
    pools = Pools(split)
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        tuples = build_tuples_item(rng, pools, k)
        if tuples is None:
            continue
        t_nome, t_attr, t_pool, t_valor = tuples[0]
        novo = pools.valor(rng, t_pool)
        vn, tn = normalize(novo), normalize(t_valor)
        if vn == tn or vn in tn or tn in vn:
            continue
        if any(vn in normalize(v) or normalize(v) in vn
               for _, _, _, v in tuples[1:]):
            continue
        if vn in normalize(t_nome) or normalize(t_nome) in vn:
            continue
        order = list(range(k))
        rng.shuffle(order)
        tuples = [tuples[i] for i in order]
        tgt = order.index(0)
        viz = rng.choice([i for i in range(k) if i != tgt])
        fatos = [render_fato(rng, nm, at, v) for nm, at, _, v in tuples]
        vz_nome, vz_attr, _, vz_valor = tuples[viz]
        items.append({
            "fatos": fatos, "target_idx": tgt,
            "fato_editado": render_fato(rng, t_nome, t_attr, novo),
            "pergunta": render_pergunta(rng, t_nome, t_attr),
            "valor_antigo": t_valor, "valor_novo": novo,
            "viz_idx": viz,
            "pergunta_viz": render_pergunta(rng, vz_nome, vz_attr),
            "gold_viz": vz_valor,
        })
    return items
