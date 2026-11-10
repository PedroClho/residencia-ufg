"""Snapshot local, só de leitura, das questões da OAB e dos artigos de lei do Atlas."""
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from replica import config
from replica.arquivos import gravar_json

TABELAS = {
    "questoes_oab": "id,exam_id,ano,numero_questao,materia,enunciado,alternativas,gabarito,anulada",
    "law_articles": "id,document_id,law_sigla,law_name,eixo,article_number,content",
}


def buscar_paginado(get_json: Callable, tabela: str, colunas: str, passo: int = 1000) -> list[dict]:
    """Baixa a tabela inteira, página a página, até uma página vir incompleta."""
    linhas: list[dict] = []
    ini = 0
    while True:
        pagina = get_json(tabela, {"select": colunas, "order": "id.asc"}, f"{ini}-{ini + passo - 1}")
        linhas.extend(pagina)
        if len(pagina) < passo:
            return linhas
        ini += passo


def get_json_supabase(url: str, chave: str) -> Callable:
    """Devolve um `get_json` que faz só GET no PostgREST do Supabase."""

    def get_json(tabela: str, params: dict, intervalo: str) -> list[dict]:
        req = urllib.request.Request(
            f"{url}/rest/v1/{tabela}?{urllib.parse.urlencode(params)}",
            headers={
                "apikey": chave,
                "Authorization": f"Bearer {chave}",
                "Range-Unit": "items",
                "Range": intervalo,
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Supabase respondeu HTTP {e.code} em {tabela}") from None

    return get_json


def salvar_snapshot(pasta: Path, get_json: Callable) -> dict:
    """Grava as tabelas em `pasta/atlas/` e o manifesto em `pasta/manifesto.json`."""
    pasta = Path(pasta)
    tabelas = {}
    for tabela, colunas in TABELAS.items():
        linhas = buscar_paginado(get_json, tabela, colunas)
        caminho = pasta / "atlas" / f"{tabela}.json"
        gravar_json(caminho, linhas)
        tabelas[tabela] = {"linhas": len(linhas), "sha256": hashlib.sha256(caminho.read_bytes()).hexdigest()}
    manifesto = {
        "projeto": config.PROJETO_ATLAS,
        "data": datetime.now(timezone.utc).isoformat(),
        "tabelas": tabelas,
    }
    gravar_json(pasta / "manifesto.json", manifesto)
    return manifesto


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(config.RAIZ / ".env")
    url, chave = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not chave:
        print("Faltam SUPABASE_URL e SUPABASE_SERVICE_KEY no .env")
        return 1
    manifesto = salvar_snapshot(config.DADOS, get_json_supabase(url, chave))
    for tabela, info in manifesto["tabelas"].items():
        print(f"{tabela}: {info['linhas']}")
    print(f"manifesto: {config.DADOS / 'manifesto.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
