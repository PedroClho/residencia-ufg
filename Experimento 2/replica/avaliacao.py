"""Acerto por fonte: MH-QA (substring), MCC (exact match estrito) e OAB (letra da alternativa)."""
import re

from replica.autor.metricas import post_process
from replica.oab import extrair_letra

# Leitura flexível das MCC: o primeiro inteiro, depois de um "label:" opcional.
_P_LABEL = re.compile(r"(?:label\s*:\s*)?(-?\d+)", re.IGNORECASE)


def _exact_match_flexivel(saida: str, gabarito: list[str]) -> bool:
    m = _P_LABEL.search(saida or "")
    return m is not None and m.group(1) in [str(g) for g in gabarito]


def avaliar(fonte: str, saida: str, gabarito: list[str]) -> tuple[bool, dict]:
    """Devolve `(acerto, métricas)` conforme a fonte da pergunta.

    - `oab*`: acerto se a letra extraída é `gabarito[0]`; métricas `letra` e `letra_invalida`.
    - `icl*` (MCC): `post_process` do autor, acerto = `exact_match` estrito (o README do autor:
      "label: 43" conta como erro); métricas do autor mais `exact_match_flexivel`.
    - `ruler*` (MH-QA): `post_process` do autor, acerto = `substring_exact_match`.
    """
    if fonte.startswith("oab"):
        letra = extrair_letra(saida)
        return letra is not None and letra == gabarito[0], {"letra": letra, "letra_invalida": letra is None}
    if fonte.startswith("icl"):
        metricas, _ = post_process({"output": saida}, gabarito, {"sub_dataset": fonte})
        metricas = {**metricas, "exact_match_flexivel": _exact_match_flexivel(saida, gabarito)}
        return bool(metricas["exact_match"]), metricas
    if fonte.startswith("ruler"):
        metricas, _ = post_process({"output": saida}, gabarito, {"sub_dataset": fonte})
        return bool(metricas["substring_exact_match"]), dict(metricas)
    raise ValueError(f"fonte sem avaliação definida: {fonte}")
