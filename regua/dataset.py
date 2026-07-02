# -*- coding: utf-8 -*-
"""Gerador do conjunto de avaliação oficial (REGUA.md §4).

Distribuição realista POR CONSTRUÇÃO:
- S3 anti-muleta: valores de vocabulário aberto, tamanho VARIÁVEL (senhas
  procedurais 4-10 chars, comidas multi-palavra, cidades com acento);
- S4 anti-atalho-posicional: o fato-alvo é embaralhado uniformemente entre os
  k fatos (shuffle), e o run reporta acurácia por terço de posição;
- múltiplos fraseados de fato e de pergunta;
- competição real: quando k≥3, o item contém um distrator com o MESMO nome
  (outro atributo) e um com o MESMO atributo (outro nome);
- pools com split train/held definidos JÁ AQUI (seed 42, 70/30): a régua
  avalia SEMPRE em 'held'; M1+ treinará cabeças só com 'train'.

Garantia anti-falso-positivo do juiz: dentro de um item, nenhum nome ou valor
contém (normalizado) qualquer outro valor.
"""
import random

from regua.judge import normalize

NOMES_REAIS = [
    "Ana", "Bento", "Carla", "Davi", "Elisa", "Fábio", "Gabriela", "Heitor",
    "Isabela", "João", "Karina", "Lucas", "Mariana", "Nicolas", "Olívia",
    "Paulo", "Quitéria", "Rafael", "Sofia", "Tiago", "Úrsula", "Vitor",
    "Wagner", "Ximena", "Yasmin", "Zeca", "Alice", "Bruno", "Cecília",
    "Diego", "Eduarda", "Felipe", "Giovana", "Hugo", "Íris", "Júlia",
    "Kléber", "Larissa", "Miguel", "Natália", "Otávio", "Priscila", "Renato",
    "Sandra", "Talita", "Ubirajara", "Valentina", "William", "Yuri", "Zilda",
    "Antônio", "Beatriz", "Caio", "Débora", "Emanuel", "Fernanda", "Gustavo",
    "Helena", "Igor", "Joana", "Kauã", "Luana", "Marcos", "Nair", "Orlando",
    "Patrícia", "Raquel", "Samuel", "Tereza", "Vinícius", "Wanda", "Yago",
    "Adriana", "Bernardo", "Clara", "Daniel", "Estela", "Francisco", "Gilda",
    "Henrique", "Ivone", "Jorge", "Lívia", "Maurício", "Norma", "Osvaldo",
    "Paula", "Rodrigo", "Simone", "Teodoro", "Vera", "Waldir", "Zuleide",
    "Aurora", "Benedito", "Celina", "Dorival", "Eunice", "Firmino", "Graça",
    "Horácio", "Inês", "Jandira", "Lauro", "Marlene", "Nestor", "Odete",
    "Plínio", "Rosângela", "Sebastião", "Terezinha", "Valdemar", "Zenaide",
]

_SILABAS = [
    "ba", "be", "bi", "bo", "bu", "ca", "co", "cu", "da", "de", "di", "do",
    "du", "fa", "fe", "fi", "fo", "ga", "go", "gu", "ja", "jo", "ju", "la",
    "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "mu", "na", "ne", "ni",
    "no", "nu", "pa", "pe", "pi", "po", "ra", "re", "ri", "ro", "ru", "sa",
    "se", "si", "so", "su", "ta", "te", "ti", "to", "tu", "va", "ve", "vi",
    "vo", "za", "ze", "zi", "zo",
]

CORES = [
    "azul", "verde", "vermelho", "amarelo", "roxo", "laranja", "cinza",
    "preto", "branco", "rosa", "marrom", "dourado", "prateado", "turquesa",
    "vinho",
]

CIDADES = [
    "São Paulo", "Teresina", "Belém", "João Pessoa", "Maceió", "Cuiabá",
    "Florianópolis", "Goiânia", "Vitória", "Natal", "São Luís",
    "Porto Alegre", "Salvador", "Recife", "Fortaleza", "Manaus", "Curitiba",
    "Belo Horizonte", "Aracaju", "Palmas", "Macapá", "Boa Vista",
    "Rio Branco", "Campo Grande", "Brasília", "Niterói", "Uberlândia",
    "Sobral", "Ilhéus", "Petrolina", "Parnaíba", "Picos", "Oeiras",
    "Caxias", "Imperatriz",
]

COMIDAS = [
    "arroz com feijão", "cuscuz", "tapioca", "açaí", "feijoada", "moqueca",
    "vatapá", "acarajé", "pão de queijo", "baião de dois", "carne de sol",
    "farofa", "brigadeiro", "canjica", "pamonha", "maniçoba", "galinhada",
    "escondidinho", "bobó de camarão", "quibebe",
]

APELIDOS = [
    "Bolinha", "Fumaça", "Pipoca", "Tigrão", "Neblina", "Faísca", "Mingau",
    "Paçoca", "Zabumba", "Trovão", "Chuvisco", "Bigode", "Serelepe",
    "Rabisco", "Estrela",
]

# (frase do atributo, nome do pool de valores)
ATRIBUTOS = [
    ("a cor do carro", "cores"),
    ("a senha do wifi", "senhas"),
    ("a cidade natal", "cidades"),
    ("a comida favorita", "comidas"),
    ("o apelido do gato", "apelidos"),
    ("o nome da professora", "professoras"),
]

FATO_TEMPLATES = [
    "{attr_cap} de {nome} é {valor}.",
    "Anote: {attr} de {nome} é {valor}.",
    "Fiquei sabendo que {attr} de {nome} é {valor}.",
]

PERGUNTA_TEMPLATES = [
    "Qual é {attr} de {nome}?",
    "Me diga {attr} de {nome}.",
    "Você lembra qual é {attr} de {nome}?",
]


def _split(pool, held_frac=0.3, seed=42):
    pool = sorted(pool)
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_held = max(3, int(len(pool) * held_frac))
    return pool[n_held:], pool[:n_held]  # (train, held)


def _nome_procedural(rng, taken_norm):
    while True:
        n = "".join(rng.choice(_SILABAS) for _ in range(rng.randint(2, 4)))
        n = n.capitalize()
        if normalize(n) not in taken_norm:
            return n


def _senha(rng):
    chars = "abcdefghjkmnpqrstuvwxyz23456789"
    return "".join(rng.choice(chars) for _ in range(rng.randint(4, 10)))


class Pools(object):
    def __init__(self, split="held"):
        idx = 0 if split == "train" else 1
        self.nomes = _split(NOMES_REAIS)[idx]
        self.cores = _split(CORES)[idx]
        self.cidades = _split(CIDADES)[idx]
        self.comidas = _split(COMIDAS)[idx]
        self.apelidos = _split(APELIDOS)[idx]
        self._nomes_reais_norm = set(normalize(n) for n in NOMES_REAIS)

    def nome(self, rng):
        if rng.random() < 0.5:
            return rng.choice(self.nomes)
        return _nome_procedural(rng, self._nomes_reais_norm)

    def valor(self, rng, pool_name):
        if pool_name == "cores":
            return rng.choice(self.cores)
        if pool_name == "cidades":
            return rng.choice(self.cidades)
        if pool_name == "comidas":
            return rng.choice(self.comidas)
        if pool_name == "apelidos":
            return rng.choice(self.apelidos)
        if pool_name == "senhas":
            return _senha(rng)
        if pool_name == "professoras":
            return rng.choice(self.nomes)
        raise ValueError(pool_name)


def _colide(cand_norm, vals_norm, nomes_norm, eh_valor):
    """Nenhum valor pode conter/estar contido em outro valor nem em um nome."""
    for v in vals_norm:
        if cand_norm in v or v in cand_norm:
            return True
    if eh_valor:
        for nm in nomes_norm:
            if cand_norm in nm or nm in cand_norm:
                return True
    return False


def _add_fato(rng, pools, tuples, forced_nome=None, forced_attr=None):
    pairs = set((t[0], t[1]) for t in tuples)
    vals_norm = [normalize(t[3]) for t in tuples]
    nomes_norm = [normalize(t[0]) for t in tuples]
    for _ in range(200):
        nome = forced_nome or pools.nome(rng)
        attr, pool_name = forced_attr or rng.choice(ATRIBUTOS)
        if (nome, attr) in pairs:
            if forced_nome and forced_attr:
                return False
            continue
        if _colide(normalize(nome), vals_norm, [], True):
            continue
        valor = pools.valor(rng, pool_name)
        if normalize(valor) == normalize(nome):
            continue
        if _colide(normalize(valor), vals_norm, nomes_norm + [normalize(nome)], True):
            continue
        tuples.append((nome, attr, pool_name, valor))
        return True
    return False


def build_items(k, n, seed, split="held"):
    pools = Pools(split)
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        tuples = []
        if not _add_fato(rng, pools, tuples):
            continue
        t_nome, t_attr, _, t_valor = tuples[0]
        ok = True
        if k >= 3:
            # competição real: mesmo nome/outro attr e mesmo attr/outro nome
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
        rng.shuffle(order)  # S4: posição do alvo uniforme
        tuples = [tuples[i] for i in order]
        target_idx = order.index(0)  # o alvo era o tuples[0] original
        fatos = []
        for (nome, attr, _, valor) in tuples:
            tpl = rng.choice(FATO_TEMPLATES)
            fatos.append(tpl.format(attr=attr, attr_cap=attr.capitalize(),
                                    nome=nome, valor=valor))
        pergunta = rng.choice(PERGUNTA_TEMPLATES).format(attr=t_attr, nome=t_nome)
        items.append({
            "fatos": fatos,
            "target_idx": target_idx,
            "pergunta": pergunta,
            "gold": t_valor,
            "nome": t_nome,
            "attr": t_attr,
        })
    return items


def stats(items):
    """Estatísticas para o relatório (documentam S3/S4 por construção)."""
    from collections import Counter
    lens = [len(it["gold"]) for it in items]
    multi = sum(1 for it in items if " " in it["gold"])
    acento = sum(1 for it in items if normalize(it["gold"]) != it["gold"].lower())
    k = len(items[0]["fatos"])
    tercos = Counter(min(2, it["target_idx"] * 3 // k) for it in items)
    return {
        "n": len(items),
        "k": k,
        "gold_len_min": min(lens), "gold_len_max": max(lens),
        "gold_len_media": sum(lens) / float(len(lens)),
        "gold_multipalavra_frac": multi / float(len(items)),
        "gold_com_acento_frac": acento / float(len(items)),
        "alvo_por_terco": {str(t): tercos.get(t, 0) for t in (0, 1, 2)},
    }


if __name__ == "__main__":
    import json
    its = build_items(k=8, n=200, seed=1008)
    print(json.dumps(stats(its), ensure_ascii=False, indent=2))
    for it in its[:3]:
        print("\n".join(it["fatos"]))
        print("Q:", it["pergunta"], "| GOLD:", it["gold"],
              "| pos:", it["target_idx"])
        print("---")
