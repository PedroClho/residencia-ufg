"""Rodador: configurações em ordem de prioridade, retomada, piloto, portão do Haiku e teto por provedor.

Uso: `python -m replica.rodar (NOMES ... | --todas) --rodada DIR [--piloto N] [--usar-reserva]
[--haiku-h2 {1m,600k}]`.

Cada configuração grava uma linha JSONL por pergunta em `{rodada}/{nome}.jsonl`, logo depois de cada
resposta. Rodar o mesmo comando de novo retoma: pula as perguntas que já têm linha sem erro e refaz as
que têm erro. Um rodador por vez: `main` pega a trava `resultados/.rodador.lock` e se recusa a começar se
ela existe; antes de gastar, confere que o livro-caixa não está abaixo da soma das linhas já gravadas.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from replica.arquivos import TravaOcupada, anexar_jsonl, saida_tolerante, trava_exclusiva
from replica.autor.agente import AgentWrapper
from replica.autor.metricas import chunk_text_into_sentences
from replica.avaliacao import avaliar
from replica.clientes import ClienteEmbeddings, fabricar_cliente
from replica.config import (
    FONTE_MHQA,
    LIVRO_CAIXA,
    MCC_FONTES,
    MODELO_ANTHROPIC,
    MODELO_EMBEDDING,
    MODELO_OPENAI,
    PROVEDOR,
    RESULTADOS,
    TRAVA_RODADOR,
)
from replica.conjuntos import Contexto, Pergunta, carregar_autor, carregar_oab
from replica.livro_caixa import ErroDoLivroCaixa, LimiteDeGasto, LimiteDoPainel, LivroCaixa
from replica.templates_oab import SYSTEM_SEM_MEMORIA, TEMPLATE_SEM_MEMORIA, obter_template

LIMIAR_PORTAO = 0.90
LINHA_DE_BASE_HAIKU = "oab-sem-memoria-haiku.jsonl"
VARIANTES_HAIKU_H2 = {"1m": "oab-h2-lc-haiku", "600k": "oab-h2-lc-haiku-600k"}
CTX_MAX_OAB = 2_000_000
# Respostas pagas que contam como erro de resposta e não são refeitas na retomada. Os outros erros
# (`[erro: api]` e exceções) são refeitos.
ERROS_FINAIS = ("[erro: recusa]", "[erro: resposta vazia]")


@dataclass(frozen=True)
class Configuracao:
    nome: str
    etapa: str  # "direta" ou "oab", o prefixo do nome
    hipotese: str  # "h1", "h2" ou "linha_de_base"
    papel: str  # a coluna "papel" da tabela do plano
    agente: str  # "contexto_longo" | "rag_bm25" | "rag_embedding" | "sem_memoria"
    modelo: str
    conjunto: str
    chunk_size: int | None
    retrieve_num: int | None
    input_length_limit: int | None
    buffer_length: int | None
    generation_max_length: int
    context_max_length: int | None
    temperatura: float | None


LC, BM25, EMB, SEM = "contexto_longo", "rag_bm25", "rag_embedding", "sem_memoria"
MINI, HAIKU = MODELO_OPENAI, MODELO_ANTHROPIC
# (input_length_limit, buffer_length) do autor por agente; o Haiku no contexto longo segue a convenção
# do autor para o Claude (item 12 das Correções ao desenho).
LIMITES = {LC: (128_000, 4_000), BM25: (10_000_000, 200), EMB: (300_000, 15_000), SEM: (None, None)}
LC_HAIKU_1M = (1_000_000, 480_000)
LC_HAIKU_600K = (1_000_000, 670_000)


def _hipotese(conjunto: str) -> str:
    if conjunto == FONTE_MHQA or conjunto.startswith("oab_h1"):
        return "h1"
    if conjunto in ("mcc", "oab_h2"):
        return "h2"
    if conjunto == "oab_sem_memoria":
        return "linha_de_base"
    raise ValueError(f"conjunto desconhecido: {conjunto}")


def _cfg(nome, papel, agente, modelo, conjunto, chunk=None, k=None, saida=20, ctx_max=CTX_MAX_OAB,
         limites=None) -> Configuracao:
    limite, buffer = limites or LIMITES[agente]
    return Configuracao(
        nome=nome, etapa=nome.split("-")[0], hipotese=_hipotese(conjunto), papel=papel, agente=agente,
        modelo=modelo, conjunto=conjunto, chunk_size=chunk, retrieve_num=k, input_length_limit=limite,
        buffer_length=buffer, generation_max_length=saida, context_max_length=ctx_max,
        temperatura=0.7 if PROVEDOR[modelo] == "openai" else None,
    )


# Ordem de prioridade: se o livro-caixa cortar algo, corta do fim da lista.
CONFIGURACOES: list[Configuracao] = [
    _cfg("direta-mhqa-lc-4omini", "primária", LC, MINI, FONTE_MHQA, 4096, saida=50, ctx_max=524288),
    _cfg("direta-mhqa-bm25-4omini", "primária", BM25, MINI, FONTE_MHQA, 512, 10, saida=50, ctx_max=524288),
    _cfg("direta-mcc-lc-4omini", "primária", LC, MINI, "mcc", 4096, ctx_max=131072),
    _cfg("direta-mcc-bm25-4omini", "primária", BM25, MINI, "mcc", 4096, 10, ctx_max=131072),
    _cfg("direta-mhqa-emb-4omini", "reprodução", EMB, MINI, FONTE_MHQA, 512, 10, saida=50, ctx_max=524288),
    _cfg("oab-sem-memoria-4omini", "linha de base", SEM, MINI, "oab_sem_memoria", ctx_max=None),
    _cfg("oab-sem-memoria-haiku", "linha de base", SEM, HAIKU, "oab_sem_memoria", ctx_max=None),
    _cfg("oab-h1-lc-4omini", "primária", LC, MINI, "oab_h1", 4096),
    _cfg("oab-h1-bm25-4omini", "primária", BM25, MINI, "oab_h1", 512, 10),
    _cfg("oab-h2-lc-4omini", "primária", LC, MINI, "oab_h2", 4096),
    _cfg("oab-h2-bm25-4omini", "primária", BM25, MINI, "oab_h2", 4096, 10),
    _cfg("oab-h1-emb-4omini", "secundária", EMB, MINI, "oab_h1", 512, 10),
    _cfg("oab-h1-bm25-haiku", "secundária", BM25, HAIKU, "oab_h1", 512, 10),
    _cfg("oab-h1-emb-haiku", "secundária", EMB, HAIKU, "oab_h1", 512, 10),
    *[_cfg(f"oab-h1-bm25-4omini-k{k}", "ablação", BM25, MINI, "oab_h1", 512, k) for k in (2, 5, 20)],
    _cfg("oab-h1corpus-bm25-4omini", "secundária", BM25, MINI, "oab_h1_corpus", 512, 10),
    _cfg("oab-h2-bm25-haiku", "secundária", BM25, HAIKU, "oab_h2", 4096, 10),
    _cfg("oab-h1-lc-haiku", "secundária", LC, HAIKU, "oab_h1", 4096, limites=LC_HAIKU_1M),
    _cfg("oab-h2-lc-haiku", "secundária, com portão", LC, HAIKU, "oab_h2", 4096, limites=LC_HAIKU_1M),
    _cfg("oab-h2-lc-haiku-600k", "variante da 21", LC, HAIKU, "oab_h2", 4096, limites=LC_HAIKU_600K),
    _cfg("oab-h2-emb-4omini", "secundária", EMB, MINI, "oab_h2", 4096, 10),
    _cfg("oab-h2-emb-haiku", "secundária", EMB, HAIKU, "oab_h2", 4096, 10),
    _cfg("direta-mcc-emb-4omini", "reprodução", EMB, MINI, "mcc", 4096, 10, ctx_max=131072),
]

_NOMES_RAG = {BM25: "Simple_rag_bm25", EMB: "Embedding_rag_text_embedding_3_small"}


def agent_config(cfg: Configuracao) -> dict:
    """Configuração do agente no formato do autor (`agent_name` com os nomes do autor)."""
    if cfg.agente == LC:
        nome = f"Long_context_agent_{cfg.modelo}"
    elif cfg.agente in _NOMES_RAG:
        nome = _NOMES_RAG[cfg.agente]
    else:
        raise ValueError(f"{cfg.nome}: a configuração sem memória não usa agente")
    d = {"agent_name": nome, "model": cfg.modelo, "temperature": cfg.temperatura,
         "input_length_limit": cfg.input_length_limit, "buffer_length": cfg.buffer_length,
         "output_dir": "resultados"}  # o __init__ do autor lê a chave; nada é gravado ali
    if cfg.agente in _NOMES_RAG:
        d["retrieve_num"] = cfg.retrieve_num
    return d


def _dataset(fonte: str) -> str:
    if fonte.startswith(("ruler", "oab_h1")):
        return "Accurate_Retrieval"
    if fonte.startswith("icl") or fonte in ("oab_h2", "oab_sem_memoria"):
        return "Test_Time_Learning"
    raise ValueError(f"fonte desconhecida: {fonte}")


def dataset_config(cfg: Configuracao, fonte: str) -> dict:
    """Configuração do dataset no formato do autor, para um contexto da fonte `fonte`."""
    return {"sub_dataset": fonte, "chunk_size": cfg.chunk_size,
            "generation_max_length": cfg.generation_max_length,
            "context_max_length": cfg.context_max_length, "dataset": _dataset(fonte)}


def carregar_conjunto(cfg: Configuracao) -> list[Contexto]:
    """Contextos do conjunto da configuração: dataset do autor ou conjuntos da OAB em `dados/oab/`."""
    if cfg.conjunto == FONTE_MHQA:
        return carregar_autor([FONTE_MHQA])
    if cfg.conjunto == "mcc":
        return carregar_autor(MCC_FONTES)
    if cfg.conjunto.startswith("oab_"):
        return carregar_oab(cfg.conjunto.removeprefix("oab_"))
    raise ValueError(f"conjunto desconhecido: {cfg.conjunto}")


# --- JSONL da configuração: retomada com uma linha por pergunta ---

def _ler_linhas(caminho: Path) -> tuple[list[dict], bool]:
    """Lê o JSONL; devolve `(linhas, cortada)`.

    Uma última linha cortada (queda no meio da escrita) é ignorada com aviso. A leitura é em bytes porque
    o corte pode cair no meio de um caractere UTF-8. Linha ilegível no meio do arquivo levanta `ValueError`
    com o caminho do arquivo.
    """
    if not caminho.exists():
        return [], False
    brutas = [b for b in caminho.read_bytes().split(b"\n") if b.strip()]
    linhas = []
    for i, bruta in enumerate(brutas):
        try:
            linhas.append(json.loads(bruta.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            if i < len(brutas) - 1:
                raise ValueError(f"{caminho}: linha {i + 1} ilegível no meio do arquivo (não é queda na "
                                 f"escrita); conserte à mão antes de retomar") from e
            print(f"[aviso] {caminho.name}: última linha cortada (queda no meio da escrita), ignorada",
                  file=sys.stderr)
            return linhas, True
    return linhas, False


def _refazer(linha: dict) -> bool:
    """A linha é refeita na retomada? Só erro de API ou exceção; recusa e resposta vazia são respostas finais."""
    erro = linha.get("erro")
    return bool(erro) and erro not in ERROS_FINAIS


def _uma_por_pergunta(linhas: list[dict]) -> dict[str, dict]:
    """`pergunta_id` → linha. Uma resposta final (sem erro, recusa ou vazia) nunca é trocada por uma linha a
    refazer; no resto, vale a mais nova, na posição da primeira."""
    por_id: dict[str, dict] = {}
    for linha in linhas:
        atual = por_id.get(linha["pergunta_id"])
        if atual is None or not _refazer(linha) or _refazer(atual):
            por_id[linha["pergunta_id"]] = linha
    return por_id


def _regravar(caminho: Path, linhas: list[dict]) -> None:
    """Troca o arquivo de uma vez (temporário com fsync + `os.replace`): uma queda no meio não perde linha."""
    fd, tmp = tempfile.mkstemp(dir=caminho.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for linha in linhas:
                f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, caminho)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _arrumar(caminho: Path) -> dict[str, dict]:
    """Lê o JSONL e o deixa com uma linha por pergunta (regrava só se precisar)."""
    linhas, cortada = _ler_linhas(caminho)
    por_id = _uma_por_pergunta(linhas)
    if cortada or len(por_id) < len(linhas):
        _regravar(caminho, list(por_id.values()))
    return por_id


def _situacao(por_id: dict[str, dict]) -> tuple[int, int, int]:
    """(perguntas sem erro, perguntas com erro de API a refazer, perguntas com recusa ou resposta vazia)."""
    erros = sum(1 for linha in por_id.values() if _refazer(linha))
    recusas = sum(1 for linha in por_id.values() if linha.get("erro") in ERROS_FINAIS)
    return len(por_id) - erros - recusas, erros, recusas


def _situacao_do_arquivo(caminho: Path) -> tuple[int, int, int]:
    """`_situacao` do JSONL, só lendo (sem regravar)."""
    return _situacao(_uma_por_pergunta(_ler_linhas(caminho)[0]))


# --- Execução de uma configuração ---

class _EmbeddingsDaOpenAI:
    """Repassa `embutir` e marca o `LimiteDeGasto` com o provedor "openai": numa configuração com o Haiku de
    leitor, o teto pode vir de qualquer um dos dois."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embutir(self, textos):
        try:
            return self._embeddings.embutir(textos)
        except LimiteDeGasto as e:
            e.provedor = PROVEDOR[MODELO_EMBEDDING]
            raise


def _memorizar(cfg: Configuracao, contexto: Contexto, cliente, embeddings) -> AgentWrapper:
    """AgentWrapper novo com todos os chunks do contexto memorizados (sem chamar API)."""
    agente = AgentWrapper(agent_config(cfg), dataset_config(cfg, contexto.fonte), cliente,
                          embeddings=embeddings, obter_template=obter_template)
    for chunk in chunk_text_into_sentences(contexto.texto, chunk_size=cfg.chunk_size):
        agente.send_message(chunk, memorizing=True)
    return agente


def _perguntar(cfg: Configuracao, contexto: Contexto, pergunta: Pergunta, cliente,
               agente: AgentWrapper | None) -> tuple[str, dict]:
    """`(consulta, saída)`. Uma exceção inesperada vira saída com `erro` e a rodada segue; o teto (e o limite do
    painel) e a falha do livro-caixa propagam."""
    if agente is None:
        consulta = TEMPLATE_SEM_MEMORIA.format(question=pergunta.texto)
    else:
        consulta = obter_template(contexto.fonte, "query", agente.agent_name).format(question=pergunta.texto)
    try:
        if agente is None:
            saida = asdict(cliente.conversar(SYSTEM_SEM_MEMORIA, consulta, max_tokens=cfg.generation_max_length))
            saida["output"] = saida.pop("texto")
        else:
            saida = agente.send_message(consulta, memorizing=False, query_id=pergunta.id, context_id=contexto.id)
    except (LimiteDeGasto, ErroDoLivroCaixa):
        raise
    except Exception as e:
        print(f"[erro] {cfg.nome}, pergunta {pergunta.id}: {e!r}", file=sys.stderr)
        saida = {"output": "", "erro": f"[erro: {type(e).__name__}]"}
    return consulta, saida


def _linha(cfg: Configuracao, contexto: Contexto, pergunta: Pergunta, consulta: str, saida: dict) -> dict:
    """Linha do JSONL. Linha com erro tem `acerto` falso e `metricas` vazias (não é avaliada). Se a avaliação
    quebrar, a resposta paga é gravada mesmo assim, com `acerto` falso e `metricas={"erro_avaliacao": ...}`."""
    erro = saida.get("erro")
    texto = saida.get("output") or ""
    acerto, metricas = False, {}
    if not erro:
        try:
            acerto, metricas = avaliar(contexto.fonte, texto, pergunta.gabarito)
        except Exception as e:
            print(f"[erro] {cfg.nome}, pergunta {pergunta.id}: avaliação quebrou ({e!r}); resposta gravada",
                  file=sys.stderr)
            acerto, metricas = False, {"erro_avaliacao": repr(e)}
    return {
        "configuracao": cfg.nome, "fonte": contexto.fonte, "contexto_id": contexto.id,
        "pergunta_id": pergunta.id, "ano": pergunta.ano, "consulta": consulta, "gabarito": pergunta.gabarito,
        "saida": texto, "acerto": acerto, "metricas": metricas,
        "tokens_entrada": saida.get("tokens_entrada", 0), "tokens_cacheados": saida.get("tokens_cacheados", 0),
        "tokens_escritos_cache": saida.get("tokens_escritos_cache", 0),
        "tokens_saida": saida.get("tokens_saida", 0), "custo_usd": saida.get("custo_usd", 0.0), "erro": erro,
    }


def rodar_configuracao(cfg: Configuracao, contextos: list[Contexto], fabricar, embeddings, pasta: Path,
                       limite_perguntas: int | None = None) -> dict:
    """Roda uma configuração e grava uma linha por pergunta em `pasta/{nome}.jsonl`.

    - `fabricar(modelo, temperatura)` devolve o cliente de LLM; `embeddings` só é usado no RAG de embedding.
    - Por contexto: se todas as perguntas já têm resposta final, o contexto é pulado inteiro. Senão, um
      `AgentWrapper` novo memoriza todos os chunks e responde as perguntas pendentes em sequência (o cache
      precisa das chamadas seguidas). A linha é anexada logo depois de cada resposta.
    - Linhas com erro de API (ou exceção) são refeitas; recusa e resposta vazia são respostas finais, contam
      como erro de resposta e não são refeitas. Ao fim, o arquivo fica com uma linha por pergunta.
    - `limite_perguntas`: só as N primeiras perguntas do primeiro contexto (piloto).
    - `LimiteDeGasto` encerra com `status="parada: teto"` e `provedor_teto` com o provedor que estourou;
      `LimiteDoPainel` (cota, saldo ou trava do painel) encerra igual, com `status="parada: limite do painel"`.
    - `ErroDoLivroCaixa` propaga (para a rodada inteira), depois de compactar o arquivo.

    Devolve `{"nome", "feitas", "erros", "recusas", "custo_usd", "status", "provedor_teto"}`, com as perguntas do
    arquivo ao fim: `feitas` sem erro, `erros` de API (a refazer), `recusas` com recusa ou resposta vazia.
    `custo_usd` soma só as chamadas desta execução (sem os embeddings, que ficam só no livro-caixa).
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{cfg.nome}.jsonl"
    feitas = {pid for pid, linha in _arrumar(caminho).items() if not _refazer(linha)}
    if limite_perguntas is not None:
        contextos = [Contexto(c.id, c.fonte, c.texto, c.perguntas[:limite_perguntas]) for c in contextos[:1]]
    cliente = fabricar(cfg.modelo, cfg.temperatura)
    embeddings = _EmbeddingsDaOpenAI(embeddings) if cfg.agente == EMB else None
    custo, status, provedor_teto = 0.0, None, None
    try:
        for contexto in contextos:
            pendentes = [p for p in contexto.perguntas if p.id not in feitas]
            if not pendentes:
                continue
            print(f"{cfg.nome}: contexto {contexto.id} ({contexto.fonte}), {len(pendentes)} pergunta(s) pendente(s)",
                  file=sys.stderr, flush=True)
            agente = None if cfg.agente == SEM else _memorizar(cfg, contexto, cliente, embeddings)
            for i, pergunta in enumerate(pendentes, 1):
                consulta, saida = _perguntar(cfg, contexto, pergunta, cliente, agente)
                linha = _linha(cfg, contexto, pergunta, consulta, saida)
                anexar_jsonl(caminho, linha)
                custo += linha["custo_usd"]
                if i % 100 == 0:
                    print(f"{cfg.nome}: {i}/{len(pendentes)}, US$ {custo:.4f}", file=sys.stderr, flush=True)
    except LimiteDeGasto as e:
        painel = isinstance(e, LimiteDoPainel)
        status = "parada: limite do painel" if painel else "parada: teto"
        provedor_teto = getattr(e, "provedor", PROVEDOR[cfg.modelo])
        motivo = "pelo limite do painel do provedor" if painel else "pelo teto de gasto"
        print(f"{cfg.nome}: parada {motivo} ({e})", file=sys.stderr)
    finally:
        # Uma falha aqui (ex.: PermissionError no os.replace) não pode esconder a exceção original nem perder
        # linhas: o arquivo fica como está, com as linhas anexadas, e a próxima execução compacta.
        try:
            _arrumar(caminho)
        except Exception as e:
            print(f"[aviso] {caminho.name}: não consegui compactar ({e!r}); a próxima execução compacta",
                  file=sys.stderr)
    n_feitas, n_erros, n_recusas = _situacao_do_arquivo(caminho)
    if status is None:
        status = "com erros de API" if n_erros else "completa"
    return {"nome": cfg.nome, "feitas": n_feitas, "erros": n_erros, "recusas": n_recusas, "custo_usd": custo,
            "status": status, "provedor_teto": provedor_teto}


# --- Portão, projeção e lista de configurações ---

def portao_haiku_h2(pasta: Path, esperadas: set[str] | None = None) -> tuple[bool, float]:
    """`(libera, acurácia)` do `oab-sem-memoria-haiku` nas questões de teste da H2 (as de 2024 e 2025).

    - `esperadas`: ids das questões de teste da H2; por padrão, as de `dados/oab/h2.json`, que são as mesmas
      224 questões de 2024-2025 do conjunto sem memória.
    - Bloqueia, com `(False, nan)`, se alguma questão esperada não tem resposta final na linha de base (falta a
      linha, ou ela tem erro de API): uma linha de base parcial não decide nada. Recusa e resposta vazia são
      respostas finais e contam como erro.
    - Com tudo respondido, libera se a acurácia for menor que 0,90. Com 90% ou mais, o Haiku provavelmente já
      viu as questões, sobra pouco espaço para a memória ajudar, e o contexto longo do Haiku na H2 não roda.

    Imprime o motivo da decisão.
    """
    if esperadas is None:
        esperadas = {p.id for c in carregar_oab("h2") for p in c.perguntas}
    arquivo = Path(pasta) / LINHA_DE_BASE_HAIKU
    por_id = _uma_por_pergunta(_ler_linhas(arquivo)[0])
    respondidas = [por_id[i] for i in esperadas if i in por_id and not _refazer(por_id[i])]
    faltam = len(esperadas) - len(respondidas)
    if not esperadas or faltam:
        print(f"portão do Haiku fechado: faltam {faltam} de {len(esperadas)} questões de teste da H2 com resposta "
              f"em {arquivo} (rode a oab-sem-memoria-haiku nessa pasta até erros_api=0)", flush=True)
        return False, math.nan
    acuracia = sum(1 for l in respondidas if l["acerto"]) / len(respondidas)
    libera = acuracia < LIMIAR_PORTAO
    print(f"portão do Haiku {'aberto' if libera else 'fechado'}: o Haiku sem memória acertou {acuracia:.1%} das "
          f"{len(respondidas)} questões de teste da H2 (o portão exige menos de {LIMIAR_PORTAO:.0%})", flush=True)
    return libera, acuracia


def projecao(linhas: list[dict], n_contextos: int, n_perguntas: int) -> float:
    """Custo completo projetado: a primeira linha (carga do contexto, com a escrita no cache) vezes os
    contextos, mais a média das demais vezes as perguntas restantes. Com uma linha só, ela vale para todas."""
    if not linhas:
        raise ValueError("projeção sem linhas")
    primeira = linhas[0]["custo_usd"]
    demais = [l["custo_usd"] for l in linhas[1:]] or [primeira]
    return primeira * n_contextos + sum(demais) / len(demais) * (n_perguntas - n_contextos)


def selecionar(nomes: list[str], haiku_h2: str | None = None) -> list[Configuracao]:
    """Configurações a rodar, na ordem de prioridade. Levanta `ValueError` em nome desconhecido.

    - Sem nomes: todas, com só a variante `haiku_h2` (padrão `1m`) do contexto longo do Haiku na H2.
    - Com nomes e sem `haiku_h2`: exatamente os nomeados.
    - Com nomes e com `haiku_h2`: os nomes têm de incluir essa variante e nenhuma outra; o contrário levanta
      `ValueError`, para nunca rodar uma variante diferente da pedida (nem ignorar a opção calado).
    """
    if not nomes:
        fora = {v for k, v in VARIANTES_HAIKU_H2.items() if k != (haiku_h2 or "1m")}
        return [c for c in CONFIGURACOES if c.nome not in fora]
    desconhecidos = sorted(set(nomes) - {c.nome for c in CONFIGURACOES})
    if desconhecidos:
        raise ValueError(f"configuração desconhecida: {', '.join(desconhecidos)}")
    if haiku_h2 is not None:
        escolhida = VARIANTES_HAIKU_H2[haiku_h2]
        outras = sorted(set(nomes) & (set(VARIANTES_HAIKU_H2.values()) - {escolhida}))
        if outras:
            raise ValueError(f"--haiku-h2 {haiku_h2} conflita com {', '.join(outras)} nos nomes")
        if escolhida not in nomes:
            raise ValueError(f"--haiku-h2 {haiku_h2} sem {escolhida} nos nomes não teria efeito")
    return [c for c in CONFIGURACOES if c.nome in nomes]


def _resumo(r: dict) -> str:
    return (f"{r['nome']}: feitas={r['feitas']}, erros_api={r['erros']}, recusas_vazias={r['recusas']}, "
            f"custo=US$ {r['custo_usd']:.4f}, status={r['status']}")


def gravar_projecao(pasta: Path, cfgs: list[Configuracao], resultados: list[dict],
                    conjuntos: dict[str, list[Contexto]], piloto: int) -> Path:
    """Grava `pasta/projecao.md`: custo completo projetado por configuração e total por provedor."""
    md = [
        "# Projeção de custo do piloto",
        "",
        f"Piloto com {piloto} pergunta(s) por configuração, as primeiras do primeiro contexto. As configurações "
        "sem memória rodam completas, e nelas o valor é o custo real. A projeção é `custo da 1ª linha × "
        "contextos + média das demais × (perguntas − contextos)`, com as respostas finais (sem erro de API) do "
        "provedor do leitor; os embeddings (centavos) ficam só no livro-caixa.",
        "",
        "**A projeção supõe todos os contextos do tamanho do primeiro.** Na H1 da OAB o primeiro contexto (Ética) "
        "é menor que os do CPC, do CC e da CF, então as projeções do contexto longo na H1 saem baixas.",
        "",
        "| configuração | provedor | linhas | contextos | perguntas | custo do piloto (US$) | projeção completa (US$) | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    totais = {"openai": 0.0, "anthropic": 0.0}
    for cfg, r in zip(cfgs, resultados):
        provedor = PROVEDOR[cfg.modelo]
        try:
            todas = list(_uma_por_pergunta(_ler_linhas(Path(pasta) / f"{cfg.nome}.jsonl")[0]).values())
        except ValueError as e:
            md.append(f"| {cfg.nome} | {provedor} | ? | – | – | – | – | arquivo ilegível: {e} |")
            continue
        finais = [l for l in todas if not _refazer(l)]
        contextos = conjuntos.get(cfg.conjunto)
        if not finais or contextos is None:
            md.append(f"| {cfg.nome} | {provedor} | {len(todas)} | – | – | – | – | {r['status']} |")
            continue
        n_contextos, n_perguntas = len(contextos), sum(len(c.perguntas) for c in contextos)
        custo_piloto = sum(l["custo_usd"] for l in todas)
        valor = custo_piloto if cfg.agente == SEM else projecao(finais, n_contextos, n_perguntas)
        totais[provedor] += valor
        md.append(f"| {cfg.nome} | {provedor} | {len(todas)} | {n_contextos} | {n_perguntas} | "
                  f"{custo_piloto:.4f} | {valor:.4f} | {r['status']} |")
    md += ["", "## Total projetado por provedor", "", "| provedor | projeção (US$) |", "|---|---|"]
    md += [f"| {p} | {v:.4f} |" for p, v in totais.items()]
    caminho = Path(pasta) / "projecao.md"
    caminho.write_text("\n".join(md) + "\n", encoding="utf-8")
    return caminho


def _resultado_sem_rodar(cfg: Configuracao, pasta: Path, status: str) -> dict:
    """Resultado de uma configuração pulada ou que falhou, com a contagem do arquivo quando dá para ler."""
    try:
        feitas, erros, recusas = _situacao_do_arquivo(pasta / f"{cfg.nome}.jsonl")
    except Exception:
        feitas, erros, recusas = 0, 0, 0
    return {"nome": cfg.nome, "feitas": feitas, "erros": erros, "recusas": recusas, "custo_usd": 0.0,
            "status": status, "provedor_teto": None}


def _rodar_uma(cfg: Configuracao, pasta: Path, parados: dict[str, str], conjuntos: dict[str, list[Contexto]],
               carregar, fabricar, embeddings, piloto: int | None) -> dict:
    leitor = PROVEDOR[cfg.modelo]
    if leitor in parados:
        return _resultado_sem_rodar(cfg, pasta, f"pulada: {parados[leitor]} ({leitor})")
    if cfg.conjunto not in conjuntos:
        conjuntos[cfg.conjunto] = carregar(cfg)
    contextos = conjuntos[cfg.conjunto]
    if cfg.nome.startswith(VARIANTES_HAIKU_H2["1m"]):
        libera, _ = portao_haiku_h2(pasta, {p.id for c in contextos for p in c.perguntas})
        if not libera:
            return _resultado_sem_rodar(cfg, pasta, "pulada: portão")
    limite = None if piloto is None or cfg.agente == SEM else piloto
    r = rodar_configuracao(cfg, contextos, fabricar, embeddings, pasta, limite)
    if r["provedor_teto"]:
        parados[r["provedor_teto"]] = r["status"].removeprefix("parada: ")
    return r


def executar(cfgs: list[Configuracao], pasta: Path, carregar, fabricar, embeddings,
             piloto: int | None = None) -> list[dict]:
    """Roda as configurações na ordem dada e imprime uma linha de resumo por configuração.

    - O teto (ou o limite do painel) de um provedor pula de antemão só as configurações seguintes cujo leitor é
      daquele provedor. As de leitor Haiku com embedding da OpenAI ainda tentam: o livro-caixa decide a cada
      chamada, e um embedding sem saldo para a configuração ali.
    - O portão do Haiku é conferido logo antes de cada `oab-h2-lc-haiku*`, contra as questões dela.
    - `carregar(cfg)` é chamado uma vez por conjunto, e só para configurações que vão rodar.
    - Uma exceção inesperada numa configuração (dados ausentes, arquivo corrompido, bug) vira
      `status="falhou: ..."`, com o traceback no stderr, e a rodada segue para a próxima. `ErroDoLivroCaixa`
      não: propaga e para a rodada inteira.
    - Com `piloto`, cada configuração faz só as N primeiras perguntas do primeiro contexto (as sem memória
      rodam completas, porque o portão precisa delas) e no fim grava `projecao.md`.
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    parados: dict[str, str] = {}  # provedor -> "teto" ou "limite do painel"
    conjuntos: dict[str, list[Contexto]] = {}
    resultados = []
    for cfg in cfgs:
        try:
            r = _rodar_uma(cfg, pasta, parados, conjuntos, carregar, fabricar, embeddings, piloto)
        except ErroDoLivroCaixa:
            raise
        except Exception as e:
            traceback.print_exc()
            r = _resultado_sem_rodar(cfg, pasta, f"falhou: {type(e).__name__}: {e}")
        print(_resumo(r), flush=True)
        resultados.append(r)
    if piloto is not None:
        gravar_projecao(pasta, cfgs, resultados, conjuntos, piloto)
    return resultados


# --- Conferência do livro-caixa contra as linhas gravadas ---

TOLERANCIA_LIVRO = 1e-6
_PROVEDOR_DO_LEITOR = {c.nome: PROVEDOR[c.modelo] for c in CONFIGURACOES}


def soma_das_linhas(raiz: Path) -> dict[str, float]:
    """Soma de `custo_usd` por provedor do leitor em `raiz/**/*.jsonl`.

    - Linhas idênticas contam uma vez: são cópias entre pastas (a linha de base sem memória do piloto foi
      reaproveitada na pasta da OAB), não chamadas pagas duas vezes.
    - Linhas de configuração desconhecida ficam de fora; arquivo ilegível no meio fica de fora, com aviso.
    - Os embeddings e a reclassificação não têm linha: o livro-caixa fica acima desta soma, nunca abaixo.
    """
    somas = {"openai": 0.0, "anthropic": 0.0}
    vistas: set[str] = set()
    for caminho in sorted(Path(raiz).rglob("*.jsonl")):
        try:
            linhas, _ = _ler_linhas(caminho)
        except ValueError as e:
            print(f"[aviso] {e}; fora da conferência do livro-caixa", file=sys.stderr)
            continue
        for linha in linhas:
            provedor = _PROVEDOR_DO_LEITOR.get(linha.get("configuracao"))
            chave = json.dumps(linha, sort_keys=True, ensure_ascii=False)
            if provedor is None or chave in vistas:
                continue
            vistas.add(chave)
            somas[provedor] += float(linha.get("custo_usd") or 0.0)
    return somas


def conferir_livro_caixa(livro: LivroCaixa, raiz: Path, tolerancia: float = TOLERANCIA_LIVRO) -> list[str]:
    """Problemas (um por provedor) em que o livro-caixa está abaixo da soma das linhas de `raiz`. O livro-caixa
    fica num caminho versionado: um `git stash`, `git checkout -- .` ou `git reset --hard` o reverte calado, e o
    teto passaria a valer sobre um gasto menor que o real."""
    problemas = []
    for provedor, soma in soma_das_linhas(raiz).items():
        gasto = livro.gasto(provedor)
        if gasto < soma - tolerancia:
            problemas.append(
                f"{provedor}: o livro-caixa marca US$ {gasto:.4f}, abaixo dos US$ {soma:.4f} das linhas gravadas em "
                f"{raiz}. Ele foi revertido (git stash, checkout ou reset?); restaure-o (git log -p "
                f"{livro.caminho.name}) antes de rodar.")
    return problemas


def _criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m replica.rodar",
        description="Roda as configurações do experimento na ordem de prioridade, com retomada, teto de gasto "
                    "por provedor e portão do Haiku. Gasta dinheiro de verdade.",
    )
    parser.add_argument("nomes", nargs="*", metavar="NOMES",
                        help="configurações a rodar, sempre na ordem de prioridade; sem nomes, exige --todas")
    parser.add_argument("--todas", action="store_true",
                        help="roda todas as configurações, inclusive as da etapa direta, com só a variante do "
                             "Haiku H2 de --haiku-h2")
    parser.add_argument("--rodada", type=Path, required=True, metavar="DIR",
                        help="pasta da rodada, com um .jsonl por configuração (obrigatória)")
    parser.add_argument("--piloto", type=int, metavar="N",
                        help="só as N primeiras perguntas do primeiro contexto de cada configuração (as sem "
                             "memória rodam completas) e grava projecao.md")
    parser.add_argument("--usar-reserva", action="store_true", help="libera a reserva de US$ 4 da OpenAI")
    parser.add_argument("--haiku-h2", choices=sorted(VARIANTES_HAIKU_H2), default=None,
                        help="variante do contexto longo do Haiku na H2. Com --todas, escolhe qual entra "
                             "(padrão: 1m); com nomes, a variante tem de estar nos nomes, e a outra não")
    return parser


def main(argv=None) -> int:
    saida_tolerante()
    parser = _criar_parser()
    args = parser.parse_args(argv)
    if args.piloto is not None and args.piloto < 1:
        parser.error("--piloto precisa ser 1 ou mais")
    if bool(args.nomes) == args.todas:
        print("passe os nomes das configurações a rodar ou --todas (que roda tudo, inclusive a etapa direta), "
              "não os dois nem nenhum", file=sys.stderr)
        return 2
    try:
        cfgs = selecionar(args.nomes, args.haiku_h2)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    try:
        with trava_exclusiva(TRAVA_RODADOR):
            return _rodar_com_trava(cfgs, args)
    except TravaOcupada as e:
        print(e, file=sys.stderr)
        return 2


def _rodar_com_trava(cfgs: list[Configuracao], args) -> int:
    livro = LivroCaixa(LIVRO_CAIXA, usar_reserva=args.usar_reserva)
    problemas = conferir_livro_caixa(livro, RESULTADOS)
    if problemas:
        for p in problemas:
            print(p, file=sys.stderr)
        return 2

    def fabricar(modelo, temperatura):
        return fabricar_cliente(modelo, livro, temperatura)

    codigo = 0
    try:
        executar(cfgs, args.rodada, carregar_conjunto, fabricar, ClienteEmbeddings(livro), args.piloto)
    except KeyboardInterrupt:
        print("interrompido: rode o mesmo comando para retomar de onde parou", file=sys.stderr)
        codigo = 130
    except ErroDoLivroCaixa as e:
        print(f"[fatal] rodada parada: {e}", file=sys.stderr)
        codigo = 1
    finally:
        try:
            print(f"gasto no livro-caixa: OpenAI US$ {livro.gasto('openai'):.4f}, "
                  f"Anthropic US$ {livro.gasto('anthropic'):.4f}", flush=True)
        except Exception as e:
            print(f"gasto no livro-caixa: ilegível ({e!r})", flush=True)
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
