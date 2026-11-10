"""Perguntas e contextos; carga dos dados do autor (Hugging Face) e da OAB."""
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from replica.arquivos import ler_json
from replica.config import DADOS, HF_DATASET, HF_REVISAO


@dataclass
class Pergunta:
    id: str
    texto: str
    gabarito: list[str]
    ano: int | None = None


@dataclass
class Contexto:
    id: int
    fonte: str
    texto: str
    perguntas: list[Pergunta]


def _split(fonte: str) -> str:
    if fonte.startswith("ruler"):
        return "Accurate_Retrieval"
    if fonte.startswith("icl"):
        return "Test_Time_Learning"
    raise ValueError(f"fonte desconhecida: {fonte}")


def carregar_autor(fontes: Sequence[str], pasta: Path = DADOS / "autor",
                   baixar=hf_hub_download) -> list[Contexto]:
    """Carrega do dataset do autor, na revisão fixada, uma linha por fonte, na ordem pedida."""
    tabelas: dict[str, list[dict]] = {}
    contextos = []
    for fonte in fontes:
        split = _split(fonte)
        if split not in tabelas:
            caminho = baixar(
                repo_id=HF_DATASET,
                filename=f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=HF_REVISAO,
                local_dir=str(pasta),
            )
            tabelas[split] = pq.read_table(caminho).to_pylist()
        linha = next((l for l in tabelas[split] if l["metadata"]["source"] == fonte), None)
        if linha is None:
            raise ValueError(f"fonte não encontrada no split {split}: {fonte}")
        assert len(linha["context"]) > 2000
        perguntas = [
            Pergunta(id=f"{fonte}:{i}", texto=q, gabarito=list(linha["answers"][i]))
            for i, q in enumerate(linha["questions"])
        ]
        contextos.append(Contexto(id=len(contextos), fonte=fonte, texto=linha["context"],
                                  perguntas=perguntas))
    return contextos


def carregar_oab(nome: str, pasta: Path = DADOS / "oab") -> list[Contexto]:
    """Lê `{nome}.json`, linhas no formato do autor, e converte em contextos."""
    contextos = []
    for i, linha in enumerate(ler_json(Path(pasta) / f"{nome}.json")):
        meta = linha["metadata"]
        ids, anos = meta["qa_pair_ids"], meta.get("anos")
        perguntas = [
            Pergunta(id=str(ids[j]), texto=q, gabarito=list(linha["answers"][j]),
                     ano=int(anos[j]) if anos and anos[j] is not None else None)
            for j, q in enumerate(linha["questions"])
        ]
        contextos.append(Contexto(id=i, fonte=meta.get("source", nome),
                                  texto=linha["context"], perguntas=perguntas))
    return contextos
