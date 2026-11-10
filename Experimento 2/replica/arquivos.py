"""JSON e JSONL sempre em UTF-8, sem escapar acentos."""
import json
from pathlib import Path


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
