"""
VEREDA-M — o NOSSO formato de artefato: `.vereda`.

Não é um codec numérico novo (os tensores ainda serializam via torch — fingir o
contrário seria autoengano). O que é nosso é a ESTRUTURA e a SEMÂNTICA:

    modelo.vereda  (zip)
    ├── MANIFEST.json   versão, config, AS REGRAS (5 portões), proveniência (sha),
    │                   e ledger de fatos AUDITÁVEL (metadado, NÃO funcional)
    ├── core.pt         Θ: pesos MORTOS, congelados (fluência + operações)
    └── memory.pt       M: estado VIVO, mutável (K, V) — a memória aprendida

Regras embutidas: capacidade mora em {Θ, M} aprendidos; o ledger é só auditoria
(o recall é neural sobre M, provável por ablação). load() trata M como estado mutável.
"""

import hashlib
import io
import json
import time
import zipfile
import torch

MAGIC = "VEREDA-M"
VERSION = "0.1"


def _blob(obj):
    buf = io.BytesIO(); torch.save(obj, buf); return buf.getvalue()


def _unblob(b):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def save_vereda(path, *, core_state, core_config, M, value_vocab,
                facts_ledger=None, notes=""):
    """Salva um artefato .vereda = {Θ congelado, M viva, manifesto auditável}."""
    core_b = _blob(core_state)
    mem_b = _blob({"K": M["K"].cpu(), "V": M["V"].cpu()})
    manifest = {
        "magic": MAGIC,
        "format_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules": "REGRA_OBRIGATORIA.md + VEREDA_M_SPEC.md (5 portões); "
                 "capacidade em {Θ,M} aprendidos; ledger=auditoria, não funcional",
        "core_config": core_config,
        "memory": {"n_slots": int(M["K"].shape[0]), "d": int(M["K"].shape[1])},
        "value_vocab": value_vocab,
        "facts_ledger_AUDIT_ONLY": facts_ledger or [],
        "notes": notes,
        "provenance": {"core_sha256": _sha(core_b), "memory_sha256": _sha(mem_b)},
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("core.pt", core_b)
        z.writestr("memory.pt", mem_b)
    return manifest


def load_vereda(path):
    """Carrega um .vereda -> {manifest, core_state (Θ), M (viva, mutável), value_vocab}."""
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("MANIFEST.json"))
        core = _unblob(z.read("core.pt"))
        mem = _unblob(z.read("memory.pt"))
    assert manifest.get("magic") == MAGIC, "não é um artefato VEREDA-M"
    return {"manifest": manifest, "core_state": core, "M": mem,
            "value_vocab": manifest["value_vocab"]}


# ---- demo: salva a memória do E1 como .vereda, recarrega num modelo NOVO, consulta ----
if __name__ == "__main__":
    import os
    from vereda_m_e1 import MemNet, VALUES, VID, enc_bytes

    DEV = torch.device("cpu")
    m = MemNet(d=64).to(DEV).eval()
    m.load_state_dict(torch.load("modelos/vereda_m_e1.pt", weights_only=False))

    def keyvec(name):
        b, l = enc_bytes(name)
        return m.encode(torch.tensor([b]), torch.tensor([l]))[0]

    facts = [("Belano", "azul"), ("Crelita", "verde"), ("Dorito", "preto"),
             ("Funessa", "vermelho"), ("Moraldo", "dourado")]
    K = torch.stack([keyvec(n) for n, _ in facts])
    V = m.val_emb(torch.tensor([VID[v] for _, v in facts]))

    path = "modelos/exemplo.vereda"
    man = save_vereda(path, core_state=m.state_dict(),
                      core_config={"d": 64, "layers": 2, "heads": 4, "arch": "MemNet"},
                      M={"K": K, "V": V}, value_vocab=VALUES,
                      facts_ledger=facts, notes="demo E1: 5 fatos na memória viva")

    print("=" * 64)
    print(f"SALVO: {path}  ({os.path.getsize(path)} bytes)")
    with zipfile.ZipFile(path) as z:
        print("conteúdo do .vereda:")
        for i in z.infolist():
            print(f"   {i.filename:<16} {i.file_size:>8} bytes")
    print(f"provenance: core={man['provenance']['core_sha256']} "
          f"mem={man['provenance']['memory_sha256']}")
    print(f"ledger (auditoria): {man['facts_ledger_AUDIT_ONLY']}")

    print("\nRECARREGA em modelo NOVO e consulta (recall NEURAL sobre M, sem o ledger):")
    art = load_vereda(path)
    m2 = MemNet(d=art["manifest"]["core_config"]["d"]).to(DEV).eval()
    m2.load_state_dict(art["core_state"])
    Md = art["M"]
    ok = 0
    for name, gold in facts:
        q = m2.encode(torch.tensor([enc_bytes(name)[0]]), torch.tensor([enc_bytes(name)[1]]))[0]
        attn = torch.softmax((Md["K"] @ q) * m2.scale, -1)
        pred = art["value_vocab"][int(m2.cls(attn @ Md["V"]).argmax(-1))]
        ok += pred == gold
        print(f"   '{name}' -> '{pred}'  (esperado '{gold}') {'✓' if pred == gold else '✗'}")
    print(f"\n recall do .vereda recarregado: {ok}/{len(facts)}  "
          f"(a memória viva viajou no NOSSO formato)")
