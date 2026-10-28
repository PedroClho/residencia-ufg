"""Exemplo 2: memória ativa, no estilo das funções de memória do MemGPT.

Aqui a memória é uma escolha do modelo, não uma política do código. O modelo recebe duas
ferramentas, `buscar_memoria` e `gravar_memoria`, e um bloco no system prompt que explica que a
janela de mensagens é curta e que o que não for gravado se perde. É ele quem decide quando ler e
quando escrever; o código só executa o que ele pede. É uma política de autocontrole por prompt,
como no MemGPT. (No exemplo 1, passiva.py, é o código quem sempre lê e sempre atualiza.)

Loop de um turno:

    mensagem do usuário
       |
       +--> monta o prompt: system base + data + bloco "Você tem memória externa..."
       |       + últimas K mensagens da sessão + mensagem nova
       |       + ferramentas [buscar_memoria, gravar_memoria]
       |
       +--> loop (no máximo MAX_ITERACOES_FERRAMENTAS rodadas):
               modelo responde
               |
               +-- stop_reason == tool_use --> o código executa cada ferramenta, devolve os
               |                               tool_result e repete
               +-- stop_reason == end_turn --> resposta final

Decisões de desenho:

- Sem resumo automático: se o modelo não gravar, perdeu. É o contraste com o exemplo 1.
- As notas só são acrescentadas, nunca editadas nem apagadas. Quando um fato muda, o modelo grava
  uma nota nova e, na busca, recebe as duas com a data de gravação; resolver a contradição pela
  data é trabalho dele. Isso expõe de propósito o risco de a memória ficar desatualizada.
- A busca é textual (FTS5 do SQLite, ver store.py), sem embeddings. Por isso a descrição da
  ferramenta avisa o modelo de que a busca casa palavras inteiras.
- O loop é manual, em vez do tool runner do SDK, porque o loop explícito é o que se consegue
  explicar linha a linha.
- Se o modelo pedir várias ferramentas de uma vez (chamadas paralelas), os resultados voltam todos
  em uma única mensagem de usuário, na mesma ordem dos pedidos.
- O limite de rodadas evita um loop sem fim. Se o modelo ainda estiver pedindo ferramenta quando o
  limite acaba, o turno vira a resposta `[erro: limite de ferramentas]`.
- A janela recente é a mesma do exemplo 1: as últimas K mensagens da sessão atual, que zeram
  quando começa uma sessão nova. Dentro da sessão a janela ajuda; entre sessões, só as notas.
- O histórico entre turnos guarda só texto: a mensagem do usuário e a resposta final do assistente.
  Os pedidos de ferramenta e seus resultados existem só dentro do turno, onde o conteúdo do
  assistente é reenviado inteiro e sem edição até o fim, como a API exige. Consequência: no turno
  seguinte o modelo não vê o resultado das buscas anteriores e precisa buscar de novo. Nisso o
  exemplo difere do MemGPT, cuja fila FIFO guarda também as chamadas de função e seus resultados.
  Um turno com erro (`[erro: ...]`) entra no histórico como resposta do assistente.

Para conversar pelo terminal: `python -m memoria.ativa`.
"""

from datetime import date

from memoria.config import CAMINHO_BANCO, K, MAX_ITERACOES_FERRAMENTAS, SYSTEM_BASE
from memoria.llm import LLM, texto
from memoria.store import Store

# As duas ferramentas que o modelo pode pedir. `strict: True` garante que a entrada do modelo siga
# o schema à risca; por isso o schema fecha com `additionalProperties: False` e lista tudo em
# `required`. As descrições dizem ao modelo quando usar cada uma: é o que faz a memória ser ativa.
FERRAMENTAS = [
    {
        "name": "buscar_memoria",
        "description": (
            "Busca nas notas que você gravou antes, inclusive em sessões anteriores. Use ANTES de "
            "responder qualquer coisa que dependa do que o usuário disse em sessões anteriores. "
            "Devolve as notas mais relevantes, cada uma com id, data de gravação e conteúdo. A busca "
            "casa palavras inteiras, sem variações de plural ou conjugação: use poucas palavras-chave "
            "e, se não achar nada, tente outras palavras."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string", "description": "Palavras-chave do que você quer lembrar."},
            },
            "required": ["consulta"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "gravar_memoria",
        "description": (
            "Grava uma nota na memória de longo prazo, que sobrevive às sessões. Use assim que o "
            "usuário disser algo durável sobre si, decisões, datas ou mudanças. Grave uma nota curta "
            "e autocontida por fato, que faça sentido lida sozinha, meses depois, sem a conversa em "
            "volta (com a data completa no lugar de \"ontem\" ou \"semana que vem\"). A nota não pode "
            "ser editada nem apagada: se um fato mudar, grave outra nota."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conteudo": {"type": "string", "description": "O fato a lembrar, em uma frase curta e autocontida."},
            },
            "required": ["conteudo"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# O bloco que acompanha SYSTEM_BASE nesta configuração e explica ao modelo a situação dele. Sem ele
# o modelo não saberia que perde as sessões anteriores e não teria motivo para usar as ferramentas.
BLOCO_MEMORIA = """\
Você tem uma memória externa, com as ferramentas buscar_memoria e gravar_memoria. Só você decide o que entra nela e quando consultá-la.
- A janela de mensagens que você enxerga é curta, e as sessões anteriores não ficam visíveis.
- O que você não gravar será perdido. Grave os fatos duráveis que o usuário contar.
- Antes de responder sobre o passado, busque na memória.
- Cada nota tem a data em que foi gravada. Se duas notas se contradizem, vale a mais recente."""


class Ativa:
    """Assistente com memória ativa: o modelo decide quando buscar e quando gravar notas.

    O banco (`store`) guarda as notas e atravessa as sessões. A instância guarda só o que é da
    sessão atual (a janela de mensagens e a data de hoje) e a contagem de ferramentas usadas.
    """

    def __init__(self, llm: LLM, store: Store):
        self.llm = llm
        self.store = store
        self.janela: list[dict] = []  # mensagens da sessão atual, texto puro, usuário e assistente
        self.data = ""  # data da sessão atual; quem usa a classe chama nova_sessao antes de responder
        # Quantas vezes o modelo usou cada ferramenta, somando todos os turnos. A avaliação lê o total.
        self.chamadas_ferramenta = {"buscar_memoria": 0, "gravar_memoria": 0}
        self.verboso = False  # True imprime cada chamada de ferramenta quando ela é executada (chat)

    def nova_sessao(self, data: str) -> None:
        """Começa uma sessão nova: guarda a data de hoje (AAAA-MM-DD) e esvazia a janela.

        O banco não é tocado: as notas atravessam a troca de sessão, e é esse o ponto da memória.
        """
        self.data = data
        self.janela = []

    def responder(self, mensagem: str) -> str:
        """Responde à mensagem do usuário, deixando o modelo usar as ferramentas. Devolve a resposta.

        Se a API falhar ou recusar, ou se o limite de rodadas de ferramenta acabar, a resposta é o
        texto `[erro: ...]` (ver `llm.texto`). Ela entra na janela como qualquer resposta, para
        manter a alternância usuário/assistente.
        """
        system = f"{SYSTEM_BASE}\n\nHoje é {self.data}.\n\n{BLOCO_MEMORIA}"
        nova = {"role": "user", "content": mensagem}
        # A lista do turno: a janela, a mensagem nova e, a cada rodada, o que o modelo pediu e o
        # que o código devolveu. Ela é descartada no fim; só o texto vai para a janela.
        mensagens = self.janela[-K:] + [nova]

        for _ in range(MAX_ITERACOES_FERRAMENTAS):
            msg = self.llm.responder(system, mensagens, FERRAMENTAS)
            if msg is None or msg.stop_reason != "tool_use":
                resposta = texto(msg)  # resposta final (ou erro de API ou recusa)
                break
            # O modelo pediu ferramentas. O conteúdo dele volta inteiro e sem edição, e todos os
            # resultados vão numa única mensagem de usuário, na mesma ordem dos pedidos.
            mensagens.append({"role": "assistant", "content": msg.content})
            pedidos = [bloco for bloco in msg.content if bloco.type == "tool_use"]
            mensagens.append({"role": "user", "content": [self._executar(bloco) for bloco in pedidos]})
        else:
            # O for terminou sem `break`: a última resposta ainda pedia ferramenta.
            resposta = "[erro: limite de ferramentas]"

        self.janela += [nova, {"role": "assistant", "content": resposta}]
        return resposta

    def _executar(self, bloco) -> dict:
        """Executa um pedido de ferramenta do modelo e devolve o `tool_result` correspondente.

        A busca devolve as notas, uma por linha, no formato `[id] (data) conteúdo`, ou uma frase
        fixa se não achou nenhuma. A gravação carimba a nota com a data da sessão atual.
        """
        self.chamadas_ferramenta[bloco.name] += 1
        if self.verboso:
            argumentos = ", ".join(f"{chave}={valor!r}" for chave, valor in bloco.input.items())
            print(f"-> {bloco.name}({argumentos})")

        if bloco.name == "buscar_memoria":
            notas = self.store.buscar_notas(bloco.input["consulta"])
            resultado = "\n".join(f"[{n['id']}] ({n['data']}) {n['conteudo']}" for n in notas)
            resultado = resultado or "Nenhuma nota encontrada."
        else:  # gravar_memoria
            id_nota = self.store.gravar_nota(self.data, bloco.input["conteudo"])
            resultado = f"Nota {id_nota} gravada."
        return {"type": "tool_result", "tool_use_id": bloco.id, "content": resultado}


if __name__ == "__main__":
    # Chat de terminal para a demonstração. As notas ficam em CAMINHO_BANCO e sobrevivem entre
    # execuções. /nova simula a troca de sessão: zera a janela, mas as notas continuam.
    store = Store(CAMINHO_BANCO)
    ativa = Ativa(LLM(), store)
    ativa.verboso = True  # mostra cada ferramenta que o modelo decide usar
    hoje = date.today().isoformat()
    ativa.nova_sessao(hoje)
    print("Chat com memória ativa. /nova começa uma sessão nova; Ctrl+D encerra (Ctrl+Z e Enter no Windows).")
    while True:
        try:
            linha = input("você> ").strip()
        except EOFError:
            break
        if not linha:  # mensagem vazia é recusada pela API e estragaria a janela
            continue
        if linha == "/nova":
            ativa.nova_sessao(hoje)
            print("(sessão nova: janela zerada, notas mantidas)")
            continue
        print(ativa.responder(linha))
    store.fechar()
