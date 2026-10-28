"""Banco SQLite compartilhado pelas duas memórias.

Um banco guarda uma conversa e tem duas tabelas:

- `resumo`: uma única linha com o resumo corrente da conversa (usada pela memória passiva).
- `notas`: as notas que o modelo decide guardar, com busca por palavra (usada pela memória ativa).

A busca usa o FTS5, o índice de texto completo que já vem dentro do SQLite. Ele casa palavras
inteiras e ordena por relevância (BM25), sem precisar de embeddings nem de serviço externo. Não
há stemming: "cachorro" não encontra "cachorros", e quem escreve a consulta (o modelo) deve saber disso.
"""

import re
import sqlite3

from memoria.config import TOP_N_NOTAS


def _consulta_fts(consulta: str) -> str:
    """Transforma texto livre numa consulta FTS5 segura: cada palavra entre aspas, unidas por OR.

    O FTS5 tem sintaxe própria (AND, OR, NOT, *, parênteses, aspas), e uma pergunta comum como
    "onde eu moro?" pode lançar erro de sintaxe. Por isso só ficam as palavras (`\\w+`, que inclui
    acentos) e cada uma vai entre aspas duplas, o que as torna texto puro, nunca operador. O OR faz
    a nota entrar se tiver qualquer uma das palavras; o BM25 põe no topo as que casam mais.
    Devolve "" quando a consulta não tem nenhuma palavra.
    """
    palavras = re.findall(r"\w+", consulta)
    return " OR ".join(f'"{palavra}"' for palavra in palavras)


class Store:
    """Resumo e notas de uma conversa, em um banco SQLite (em memória por padrão)."""

    def __init__(self, caminho: str = ":memory:"):
        """Abre o banco e cria as tabelas se ainda não existirem.

        `:memory:` serve à avaliação, que usa um banco novo por configuração; um nome de arquivo
        serve aos chats de terminal, em que a memória precisa sobreviver entre execuções.
        `isolation_level=None` deixa o SQLite em autocommit, então toda escrita já fica gravada
        no arquivo, sem precisar de `commit()` espalhado pelo código.
        """
        self.conexao = sqlite3.connect(caminho, isolation_level=None)
        # `id = 1` fixo: só existe um resumo, e gravar um novo o substitui.
        self.conexao.execute(
            "CREATE TABLE IF NOT EXISTS resumo (id INTEGER PRIMARY KEY CHECK (id = 1), texto TEXT)"
        )
        # `data` fica fora do índice (UNINDEXED): serve para mostrar a nota, não para buscá-la.
        self.conexao.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS notas USING fts5(conteudo, data UNINDEXED)"
        )

    def ler_resumo(self) -> str:
        """Devolve o resumo corrente da conversa, ou "" se ainda não há nenhum."""
        linha = self.conexao.execute("SELECT texto FROM resumo WHERE id = 1").fetchone()
        return linha[0] if linha else ""

    def gravar_resumo(self, resumo: str) -> None:
        """Substitui o resumo corrente. O banco guarda só a versão mais recente, sem histórico."""
        self.conexao.execute("INSERT OR REPLACE INTO resumo (id, texto) VALUES (1, ?)", (resumo,))

    def gravar_nota(self, data: str, conteudo: str) -> int:
        """Grava uma nota e devolve o id dela. A `data` é a da sessão em que a nota foi escrita."""
        cursor = self.conexao.execute(
            "INSERT INTO notas (data, conteudo) VALUES (?, ?)", (data, conteudo)
        )
        return cursor.lastrowid

    def buscar_notas(self, consulta: str, n: int = TOP_N_NOTAS) -> list[dict]:
        """Devolve até `n` notas que contêm alguma palavra da consulta, da mais relevante à menos.

        Cada nota é um dict com `id`, `data` e `conteudo`. Consulta sem nenhuma palavra devolve [].
        """
        match = _consulta_fts(consulta)
        if not match:
            return []
        linhas = self.conexao.execute(
            "SELECT rowid, data, conteudo FROM notas WHERE notas MATCH ? ORDER BY rank LIMIT ?",
            (match, n),
        ).fetchall()
        return [{"id": id_, "data": data, "conteudo": conteudo} for id_, data, conteudo in linhas]

    def fechar(self) -> None:
        """Fecha a conexão com o banco."""
        self.conexao.close()
