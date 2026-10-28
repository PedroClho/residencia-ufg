"""Único ponto de contato com a API da Anthropic.

Todo o resto do projeto (passiva, ativa, baselines, juiz) chama o modelo por aqui. Isso garante duas
coisas: a chamada é montada sempre do mesmo jeito nas quatro configurações, e cada resposta passa
pelo contador de tokens que alimenta a coluna de custo da tabela final.
"""

from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from memoria import config


@dataclass
class Contador:
    """Soma chamadas e tokens das respostas da API.

    O runner lê o contador ao fim de cada configuração e o zera para a próxima, então cada
    configuração tem o seu custo medido em separado.
    """

    chamadas: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0

    def zerar(self) -> None:
        """Volta tudo a zero, para começar a medir outra configuração."""
        self.chamadas = 0
        self.tokens_entrada = 0
        self.tokens_saida = 0

    @property
    def custo_usd(self) -> float:
        """Custo estimado em dólares, com os preços de config.py (o raciocínio do modelo conta como saída)."""
        return (
            self.tokens_entrada * config.PRECO_ENTRADA_USD_POR_MILHAO
            + self.tokens_saida * config.PRECO_SAIDA_USD_POR_MILHAO
        ) / 1e6


class LLM:
    """Cliente do modelo com contador de tokens embutido.

    Recebe um `cliente` já pronto só nos testes (um falso, para não chamar a API). Sem argumento,
    carrega o `.env` e cria o cliente real, que lê a chave de ANTHROPIC_API_KEY.
    """

    def __init__(self, cliente=None):
        if cliente is None:
            load_dotenv()
            cliente = anthropic.Anthropic()
        self.cliente = cliente
        self.contador = Contador()

    def responder(
        self,
        system: str,
        mensagens: list[dict],
        ferramentas: list[dict] | None = None,
        formato: dict | None = None,
    ):
        """Faz uma chamada ao modelo e devolve a mensagem de resposta, ou None se a API falhar.

        `ferramentas` (opcional) são as ferramentas que o modelo pode pedir; `formato` (opcional) é o
        esquema JSON que a saída deve seguir. Nenhum dos dois vai na chamada quando não é dado.
        Devolver None em vez de levantar o erro deixa quem chama decidir o que fazer: o projeto
        trata erro de API como uma resposta `[erro: api]` e segue em frente (ver `texto`).
        """
        kwargs = {
            "model": config.MODELO,
            "max_tokens": config.MAX_TOKENS,
            "system": system,
            "messages": mensagens,
        }
        if ferramentas:
            kwargs["tools"] = ferramentas
        if formato:
            kwargs["output_config"] = {"format": formato}

        try:
            msg = self.cliente.messages.create(**kwargs)
        except anthropic.APIError:
            return None

        self.contador.chamadas += 1
        self.contador.tokens_entrada += msg.usage.input_tokens
        self.contador.tokens_saida += msg.usage.output_tokens
        return msg


def texto(msg) -> str:
    """Extrai o texto de uma resposta; falha de API, recusa ou texto vazio viram `[erro: ...]`.

    Assim o turno sempre tem um texto para guardar no histórico, mesmo quando não houve resposta
    (`msg` é None), quando a API recusou o pedido ou quando o modelo não escreveu nada (por
    exemplo, parou depois de só raciocinar). Blocos que não são texto (raciocínio, pedidos de
    ferramenta) são ignorados. Texto vazio nunca é devolvido: ele entraria no histórico como
    mensagem do assistente, e a API rejeita conteúdo vazio, o que faria falhar todos os turnos
    seguintes.
    """
    if msg is None:
        return "[erro: api]"
    if msg.stop_reason == "refusal":
        return "[erro: recusa]"
    resposta = "".join(bloco.text for bloco in msg.content if bloco.type == "text")
    return resposta if resposta.strip() else "[erro: resposta vazia]"
