"""Livro-caixa: custo por provedor e teto de gasto persistente."""
import json
import os
import tempfile
from pathlib import Path

from replica.config import PRECOS, RESERVA, TETOS

MILHAO = 1_000_000


class LimiteDeGasto(Exception):
    """A chamada passaria do teto efetivo do provedor."""


def custo_openai(modelo: str, tokens_entrada: int, tokens_cacheados: int, tokens_saida: int) -> float:
    """`tokens_entrada` inclui os cacheados."""
    p = PRECOS[modelo]
    normais = tokens_entrada - tokens_cacheados
    return (
        normais * p["entrada"] + tokens_cacheados * p["entrada_cache"] + tokens_saida * p["saida"]
    ) / MILHAO


def custo_anthropic(modelo: str, sem_cache: int, escrita_cache: int, leitura_cache: int, saida: int) -> float:
    p = PRECOS[modelo]
    longa = sem_cache + escrita_cache + leitura_cache > p["limiar_longo"]
    entrada = p["entrada_longa"] if longa else p["entrada"]
    taxa_saida = p["saida_longa"] if longa else p["saida"]
    return (
        sem_cache * entrada
        + escrita_cache * entrada * p["mult_escrita_cache"]
        + leitura_cache * entrada * p["mult_leitura_cache"]
        + saida * taxa_saida
    ) / MILHAO


def custo_embedding(modelo: str, tokens: int) -> float:
    return tokens * PRECOS[modelo]["entrada"] / MILHAO


class LivroCaixa:
    def __init__(
        self,
        caminho: Path,
        tetos: dict[str, float] = TETOS,
        reserva: dict[str, float] = RESERVA,
        usar_reserva: bool = False,
    ):
        self.caminho = Path(caminho)
        self.tetos = tetos
        self.reserva = reserva
        self.usar_reserva = usar_reserva

    def _ler(self) -> dict[str, float]:
        if not self.caminho.exists():
            return {}
        return json.loads(self.caminho.read_text(encoding="utf-8"))

    def gasto(self, provedor: str) -> float:
        return float(self._ler().get(provedor, 0.0))

    def teto_efetivo(self, provedor: str) -> float:
        return self.tetos[provedor] - (0 if self.usar_reserva else self.reserva.get(provedor, 0))

    def autorizar(self, provedor: str, custo_maximo: float) -> None:
        if self.gasto(provedor) + custo_maximo > self.teto_efetivo(provedor):
            raise LimiteDeGasto(
                f"{provedor}: gasto {self.gasto(provedor):.4f} + {custo_maximo:.4f} "
                f"passa do teto efetivo {self.teto_efetivo(provedor):.2f}"
            )

    def registrar(self, provedor: str, custo: float) -> None:
        dados = self._ler()
        dados[provedor] = float(dados.get(provedor, 0.0)) + custo
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.caminho.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.caminho)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
