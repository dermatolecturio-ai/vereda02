# -*- coding: utf-8 -*-
"""Juiz oficial da régua (S1: só geração; aqui apenas a correção da resposta).

Critério (documentado, fixo): o valor-ouro NORMALIZADO aparece como segmento
com fronteira de palavra na resposta gerada NORMALIZADA. Normalização =
minúsculas, sem acentos, sem pontuação, espaços colapsados. A construção do
dataset garante que nenhum outro nome/valor do item contém o ouro (sem falso
positivo por eco de distrator).
"""
import re
import unicodedata


def normalize(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_correct(answer, gold):
    a = " %s " % normalize(answer)
    g = " %s " % normalize(gold)
    return g in a
