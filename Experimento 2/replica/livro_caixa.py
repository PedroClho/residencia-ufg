"""Livro-caixa: custo por provedor e teto de gasto persistente."""
import json
import os
import tempfile
import time
from pathlib import Path

from replica.config import PRECOS, RESERVA, TETOS

MILHAO = 1_000_000


class LimiteDeGasto(Exception):
    """A chamada passaria do teto efetivo do provedor."""


class LimiteDoPainel(LimiteDeGasto):
    """A API recusou por falta de saldo ou pela trava de gasto do painel (cota esgotada na OpenAI, saldo ou limite
    de uso na Anthropic). Para o provedor como o teto do livro-caixa."""

    def __init__(self, provedor: str, mensagem: str):
        super().__init__(mensagem)
        self.provedor = provedor


class ErroDoLivroCaixa(Exception):
    """O livro-caixa não conseguiu gravar um gasto já feito. Fatal: a rodada inteira para, senão o gasto real passa
    do registrado sem ninguém ver."""


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


TENTATIVAS_GRAVACAO = 5
ESPERA_GRAVACAO_S = 0.2


class LivroCaixa:
    def __init__(
        self,
        caminho: Path,
        tetos: dict[str, float] = TETOS,
        reserva: dict[str, float] = RESERVA,
        usar_reserva: bool = False,
        dormir=time.sleep,
    ):
        self.caminho = Path(caminho)
        self.tetos = tetos
        self.reserva = reserva
        self.usar_reserva = usar_reserva
        self._dormir = dormir

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
        """Soma `custo` ao provedor. No Windows o `os.replace` falha com PermissionError se outro programa está
        com o arquivo aberto (antivírus, editor): tenta 5 vezes, com 0,2 s entre elas. Se não gravar, levanta
        `ErroDoLivroCaixa`, que para a rodada inteira."""
        for tentativa in range(1, TENTATIVAS_GRAVACAO + 1):
            try:
                self._gravar(provedor, custo)
                return
            except PermissionError as e:
                if tentativa == TENTATIVAS_GRAVACAO:
                    raise ErroDoLivroCaixa(self._mensagem_de_falha(provedor, custo, e)) from e
                self._dormir(ESPERA_GRAVACAO_S)
            except (OSError, ValueError) as e:
                raise ErroDoLivroCaixa(self._mensagem_de_falha(provedor, custo, e)) from e

    def _mensagem_de_falha(self, provedor: str, custo: float, e: Exception) -> str:
        return (f"não consegui gravar {self.caminho} ({e!r}): o gasto de US$ {custo:.4f} da {provedor} já foi feito "
                f"e não está no livro-caixa; some à mão antes de rodar de novo")

    def _gravar(self, provedor: str, custo: float) -> None:
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
