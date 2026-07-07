# -*- coding: utf-8 -*-
"""Pares (fato, pergunta, valor) para o contraste InfoNCE do M1.

Reusa os MESMOS pools/templates da régua (regua/dataset.py) — vocabulário
aberto, valores de tamanho VARIÁVEL (S3 já embutido). Treino usa split
'train'; avaliação usa 'held' (nomes/valores nunca vistos no treino da cabeça).
"""
import random

from regua.dataset import ATRIBUTOS, FATO_TEMPLATES, PERGUNTA_TEMPLATES, Pools


def gen_pairs(rng, pools, n):
    out = []
    for _ in range(n):
        nome = pools.nome(rng)
        attr, pool_name = rng.choice(ATRIBUTOS)
        valor = pools.valor(rng, pool_name)
        fato = rng.choice(FATO_TEMPLATES).format(
            attr=attr, attr_cap=attr.capitalize(), nome=nome, valor=valor)
        pergunta = rng.choice(PERGUNTA_TEMPLATES).format(attr=attr, nome=nome)
        out.append((fato, pergunta, valor))
    return out


def build(n_train, n_held, seed_train=0, seed_held=1):
    tr_pools = Pools("train")
    he_pools = Pools("held")
    tr = gen_pairs(random.Random(seed_train), tr_pools, n_train)
    he = gen_pairs(random.Random(seed_held), he_pools, n_held)
    return tr, he
