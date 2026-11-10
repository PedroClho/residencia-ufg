"""JSON e JSONL sempre em UTF-8, sem escapar acentos; trava exclusiva e saída de console tolerante."""
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class TravaOcupada(Exception):
    """Outro processo já tem a trava."""


@contextmanager
def trava_exclusiva(caminho: Path):
    """Cria `caminho` com `O_CREAT | O_EXCL` (pid e hora de início dentro) e o apaga na saída, mesmo com exceção.

    Se o arquivo já existe, levanta `TravaOcupada` sem apagá-lo: a trava é de outro processo.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(caminho, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            dono = caminho.read_text(encoding="utf-8").strip()
        except OSError:
            dono = "?"
        raise TravaOcupada(
            f"a trava {caminho} existe ({dono}): outro rodador ou reclassificador está gastando agora. Se nenhum "
            f"estiver vivo (o processo caiu ou foi morto), apague o arquivo e rode de novo."
        ) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid {os.getpid()}, início {datetime.now().isoformat(timespec='seconds')}\n")
        yield
    finally:
        try:
            os.remove(caminho)
        except FileNotFoundError:
            pass


def saida_tolerante() -> None:
    """No stdout e no stderr, um caractere que o console não codifica (cp1252 sem `PYTHONUTF8=1`) vira escape
    `\\uXXXX` em vez de derrubar o processo no meio de uma rodada."""
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(errors="backslashreplace")


def gravar_json(caminho: Path, obj) -> None:
    """Grava `obj` em JSON UTF-8 (indent=1), criando as pastas."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def ler_json(caminho: Path):
    """Lê um JSON UTF-8."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def anexar_jsonl(caminho: Path, obj) -> None:
    """Anexa `obj` como uma linha JSON ao arquivo, criando as pastas."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ler_jsonl(caminho: Path) -> list[dict]:
    """Lê um JSONL UTF-8, pulando linhas em branco; `[]` se o arquivo não existe."""
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
