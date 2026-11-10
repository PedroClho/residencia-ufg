"""Rodador: configurações em ordem de prioridade, retomada, piloto, portão do Haiku e teto por provedor.

Uso: `python -m replica.rodar [NOMES ...] [--rodada DIR] [--piloto N] [--usar-reserva] [--haiku-h2 {1m,600k}]`.

Cada configuração grava uma linha JSONL por pergunta em `{rodada}/{nome}.jsonl`, logo depois de cada
resposta. Rodar o mesmo comando de novo retoma: pula as perguntas que já têm linha sem erro e refaz as
que têm erro. Nunca rode dois rodadores ao mesmo tempo: o arquivo do livro-caixa não tem trava.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from replica.arquivos import anexar_jsonl
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
)
from replica.conjuntos import Contexto, Pergunta, carregar_autor, carregar_oab
from replica.livro_caixa import LimiteDeGasto, LivroCaixa
from replica.templates_oab import SYSTEM_SEM_MEMORIA, TEMPLATE_SEM_MEMORIA, obter_template

LIMIAR_PORTAO = 0.90
LINHA_DE_BASE_HAIKU = "oab-sem-memoria-haiku.jsonl"
VARIANTES_HAIKU_H2 = {"1m": "oab-h2-lc-haiku", "600k": "oab-h2-lc-haiku-600k"}
CTX_MAX_OAB = 2_000_000


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


def provedores(cfg: Configuracao) -> set[str]:
    """Provedores que a configuração usa: o do leitor e, no RAG de embedding, o dos embeddings (OpenAI)."""
    usados = {PROVEDOR[cfg.modelo]}
    if cfg.agente == EMB:
        usados.add(PROVEDOR[MODELO_EMBEDDING])
    return usados


# --- JSONL da configuração: retomada com uma linha por pergunta ---

def _ler_linhas(caminho: Path) -> tuple[list[dict], bool]:
    """Lê o JSONL; devolve `(linhas, cortada)`.

    Uma última linha cortada (queda no meio da escrita) é ignorada com aviso. A leitura é em bytes porque
    o corte pode cair no meio de um caractere UTF-8. Linha ilegível no meio do arquivo levanta erro.
    """
    if not caminho.exists():
        return [], False
    brutas = [b for b in caminho.read_bytes().split(b"\n") if b.strip()]
    linhas = []
    for i, bruta in enumerate(brutas):
        try:
            linhas.append(json.loads(bruta.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if i < len(brutas) - 1:
                raise
            print(f"[aviso] {caminho.name}: última linha cortada (queda no meio da escrita), ignorada",
                  file=sys.stderr)
            return linhas, True
    return linhas, False


def _uma_por_pergunta(linhas: list[dict]) -> dict[str, dict]:
    """`pergunta_id` → linha. Uma linha sem erro nunca é trocada por uma com erro; no resto, vale a mais nova,
    na posição da primeira."""
    por_id: dict[str, dict] = {}
    for linha in linhas:
        atual = por_id.get(linha["pergunta_id"])
        if atual is None or not linha.get("erro") or atual.get("erro"):
            por_id[linha["pergunta_id"]] = linha
    return por_id


def _regravar(caminho: Path, linhas: list[dict]) -> None:
    """Troca o arquivo de uma vez (temporário + `os.replace`): uma queda no meio não perde nenhuma linha."""
    fd, tmp = tempfile.mkstemp(dir=caminho.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for linha in linhas:
                f.write(json.dumps(linha, ensure_ascii=False) + "\n")
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


def _situacao(por_id: dict[str, dict]) -> tuple[int, int]:
    """(perguntas com linha sem erro, perguntas com linha de erro)."""
    erros = sum(1 for linha in por_id.values() if linha.get("erro"))
    return len(por_id) - erros, erros


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
    """`(consulta, saída)`. Uma exceção inesperada vira saída com `erro` e a rodada segue; o teto propaga."""
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
    except LimiteDeGasto:
        raise
    except Exception as e:
        print(f"[erro] {cfg.nome}, pergunta {pergunta.id}: {e!r}", file=sys.stderr)
        saida = {"output": "", "erro": f"[erro: {type(e).__name__}]"}
    return consulta, saida


def _linha(cfg: Configuracao, contexto: Contexto, pergunta: Pergunta, consulta: str, saida: dict) -> dict:
    """Linha do JSONL. Linha com erro tem `acerto` falso e `metricas` vazias (não é avaliada)."""
    erro = saida.get("erro")
    texto = saida.get("output") or ""
    acerto, metricas = (False, {}) if erro else avaliar(contexto.fonte, texto, pergunta.gabarito)
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
    - Por contexto: se todas as perguntas já têm linha sem erro, o contexto é pulado inteiro. Senão, um
      `AgentWrapper` novo memoriza todos os chunks e responde as perguntas pendentes em sequência (o cache
      precisa das chamadas seguidas). A linha é anexada logo depois de cada resposta.
    - Linhas com erro são refeitas; ao fim, o arquivo fica com uma linha por pergunta.
    - `limite_perguntas`: só as N primeiras perguntas do primeiro contexto (piloto).
    - `LimiteDeGasto` encerra com `status="parada: teto"` e `provedor_teto` com o provedor que estourou.

    Devolve `{"nome", "feitas", "erros", "custo_usd", "status", "provedor_teto"}`: `feitas` e `erros` contam as
    perguntas do arquivo ao fim; `custo_usd` soma só as chamadas desta execução (sem os embeddings, que ficam
    só no livro-caixa).
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{cfg.nome}.jsonl"
    feitas = {pid for pid, linha in _arrumar(caminho).items() if not linha.get("erro")}
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
        status = "parada: teto"
        provedor_teto = getattr(e, "provedor", PROVEDOR[cfg.modelo])
        print(f"{cfg.nome}: parada pelo teto de gasto ({e})", file=sys.stderr)
    finally:
        por_id = _arrumar(caminho)
    n_feitas, n_erros = _situacao(por_id)
    if status is None:
        status = "com erros" if n_erros else "completa"
    return {"nome": cfg.nome, "feitas": n_feitas, "erros": n_erros, "custo_usd": custo, "status": status,
            "provedor_teto": provedor_teto}


# --- Portão, projeção e lista de configurações ---

def portao_haiku_h2(pasta: Path) -> tuple[bool, float]:
    """`(libera, acurácia)` do `oab-sem-memoria-haiku` nas perguntas de 2024 e 2025, as de teste da H2.

    Libera se a acurácia for menor que 0,90. Com 90% ou mais, o Haiku provavelmente já viu as questões, sobra
    pouco espaço para a memória ajudar, e o contexto longo do Haiku na H2 não roda. Linhas com erro não contam.
    Sem nenhuma linha de 2024-2025 (arquivo ausente ou incompleto), devolve `(False, nan)`: bloqueia.
    """
    linhas, _ = _ler_linhas(Path(pasta) / LINHA_DE_BASE_HAIKU)
    teste = [l for l in _uma_por_pergunta(linhas).values() if l.get("ano") in (2024, 2025) and not l.get("erro")]
    if not teste:
        return False, math.nan
    acuracia = sum(1 for l in teste if l["acerto"]) / len(teste)
    return acuracia < LIMIAR_PORTAO, acuracia


def projecao(linhas: list[dict], n_contextos: int, n_perguntas: int) -> float:
    """Custo completo projetado: a primeira linha (carga do contexto, com a escrita no cache) vezes os
    contextos, mais a média das demais vezes as perguntas restantes. Com uma linha só, ela vale para todas."""
    if not linhas:
        raise ValueError("projeção sem linhas")
    primeira = linhas[0]["custo_usd"]
    demais = [l["custo_usd"] for l in linhas[1:]] or [primeira]
    return primeira * n_contextos + sum(demais) / len(demais) * (n_perguntas - n_contextos)


def selecionar(nomes: list[str], haiku_h2: str = "1m") -> list[Configuracao]:
    """Configurações a rodar, na ordem de prioridade. Sem nomes: todas, com só a variante `haiku_h2` do
    contexto longo do Haiku na H2. Nome desconhecido levanta `ValueError`."""
    if not nomes:
        fora = {v for k, v in VARIANTES_HAIKU_H2.items() if k != haiku_h2}
        return [c for c in CONFIGURACOES if c.nome not in fora]
    desconhecidos = sorted(set(nomes) - {c.nome for c in CONFIGURACOES})
    if desconhecidos:
        raise ValueError(f"configuração desconhecida: {', '.join(desconhecidos)}")
    return [c for c in CONFIGURACOES if c.nome in nomes]


def _bloqueio(cfg: Configuracao, parados: set[str], pasta: Path) -> str | None:
    """Motivo para pular a configuração (teto de um provedor que ela usa, ou portão do Haiku), ou `None`."""
    tetos = sorted(provedores(cfg) & parados)
    if tetos:
        return f"pulada: teto ({', '.join(tetos)})"
    if cfg.nome.startswith(VARIANTES_HAIKU_H2["1m"]):
        libera, acuracia = portao_haiku_h2(pasta)
        if not libera:
            if math.isnan(acuracia):
                motivo = f"sem linhas de 2024-2025 em {Path(pasta) / LINHA_DE_BASE_HAIKU}"
            else:
                motivo = (f"o Haiku sem memória acertou {acuracia:.1%} das questões de 2024-2025 "
                          f"(o portão exige menos de {LIMIAR_PORTAO:.0%})")
            print(f"{cfg.nome}: portão do Haiku fechado: {motivo}", flush=True)
            return "pulada: portão"
    return None


def _resumo(r: dict) -> str:
    return (f"{r['nome']}: feitas={r['feitas']}, erros={r['erros']}, custo=US$ {r['custo_usd']:.4f}, "
            f"status={r['status']}")


def gravar_projecao(pasta: Path, cfgs: list[Configuracao], resultados: list[dict],
                    conjuntos: dict[str, list[Contexto]], piloto: int) -> Path:
    """Grava `pasta/projecao.md`: custo completo projetado por configuração e total por provedor."""
    md = [
        "# Projeção de custo do piloto",
        "",
        f"Piloto com {piloto} pergunta(s) por configuração, as primeiras do primeiro contexto. As configurações "
        "sem memória rodam completas, e nelas o valor é o custo real. A projeção é `custo da 1ª linha × "
        "contextos + média das demais × (perguntas − contextos)`, com as linhas sem erro do provedor do "
        "leitor; os embeddings (centavos) ficam só no livro-caixa.",
        "",
        "| configuração | provedor | linhas | contextos | perguntas | custo do piloto (US$) | projeção completa (US$) | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    totais = {"openai": 0.0, "anthropic": 0.0}
    for cfg, r in zip(cfgs, resultados):
        provedor = PROVEDOR[cfg.modelo]
        todas = list(_arrumar(Path(pasta) / f"{cfg.nome}.jsonl").values())
        sem_erro = [l for l in todas if not l.get("erro")]
        contextos = conjuntos.get(cfg.conjunto)
        if not sem_erro or contextos is None:
            md.append(f"| {cfg.nome} | {provedor} | {len(todas)} | – | – | – | – | {r['status']} |")
            continue
        n_contextos, n_perguntas = len(contextos), sum(len(c.perguntas) for c in contextos)
        custo_piloto = sum(l["custo_usd"] for l in todas)
        valor = custo_piloto if cfg.agente == SEM else projecao(sem_erro, n_contextos, n_perguntas)
        totais[provedor] += valor
        md.append(f"| {cfg.nome} | {provedor} | {len(todas)} | {n_contextos} | {n_perguntas} | "
                  f"{custo_piloto:.4f} | {valor:.4f} | {r['status']} |")
    md += ["", "## Total projetado por provedor", "", "| provedor | projeção (US$) |", "|---|---|"]
    md += [f"| {p} | {v:.4f} |" for p, v in totais.items()]
    caminho = Path(pasta) / "projecao.md"
    caminho.write_text("\n".join(md) + "\n", encoding="utf-8")
    return caminho


def executar(cfgs: list[Configuracao], pasta: Path, carregar, fabricar, embeddings,
             piloto: int | None = None) -> list[dict]:
    """Roda as configurações na ordem dada e imprime uma linha de resumo por configuração.

    - O teto de um provedor pula só as configurações seguintes que usam aquele provedor.
    - O portão do Haiku é conferido logo antes de cada `oab-h2-lc-haiku*`.
    - `carregar(cfg)` é chamado uma vez por conjunto, e só para configurações que vão rodar.
    - Com `piloto`, cada configuração faz só as N primeiras perguntas do primeiro contexto (as sem memória
      rodam completas, porque o portão precisa delas) e no fim grava `projecao.md`.
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    parados: set[str] = set()
    conjuntos: dict[str, list[Contexto]] = {}
    resultados = []
    for cfg in cfgs:
        motivo = _bloqueio(cfg, parados, pasta)
        if motivo:
            feitas, erros = _situacao(_arrumar(pasta / f"{cfg.nome}.jsonl"))
            r = {"nome": cfg.nome, "feitas": feitas, "erros": erros, "custo_usd": 0.0, "status": motivo,
                 "provedor_teto": None}
        else:
            if cfg.conjunto not in conjuntos:
                conjuntos[cfg.conjunto] = carregar(cfg)
            limite = None if piloto is None or cfg.agente == SEM else piloto
            r = rodar_configuracao(cfg, conjuntos[cfg.conjunto], fabricar, embeddings, pasta, limite)
            if r["provedor_teto"]:
                parados.add(r["provedor_teto"])
        print(_resumo(r), flush=True)
        resultados.append(r)
    if piloto is not None:
        gravar_projecao(pasta, cfgs, resultados, conjuntos, piloto)
    return resultados


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m replica.rodar",
        description="Roda as configurações do experimento na ordem de prioridade, com retomada, teto de gasto "
                    "por provedor e portão do Haiku. Gasta dinheiro de verdade.",
    )
    parser.add_argument("nomes", nargs="*", metavar="NOMES",
                        help="configurações a rodar, sempre na ordem de prioridade (padrão: todas, com só a "
                             "variante do Haiku H2 escolhida em --haiku-h2)")
    parser.add_argument("--rodada", type=Path, default=RESULTADOS / "rodada", metavar="DIR",
                        help="pasta da rodada, com um .jsonl por configuração (padrão: resultados/rodada)")
    parser.add_argument("--piloto", type=int, metavar="N",
                        help="só as N primeiras perguntas do primeiro contexto de cada configuração (as sem "
                             "memória rodam completas) e grava projecao.md")
    parser.add_argument("--usar-reserva", action="store_true", help="libera a reserva de US$ 4 da OpenAI")
    parser.add_argument("--haiku-h2", choices=sorted(VARIANTES_HAIKU_H2), default="1m",
                        help="variante do contexto longo do Haiku na H2 quando roda sem nomes (padrão: 1m)")
    args = parser.parse_args(argv)
    if args.piloto is not None and args.piloto < 1:
        parser.error("--piloto precisa ser 1 ou mais")
    try:
        cfgs = selecionar(args.nomes, args.haiku_h2)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    livro = LivroCaixa(LIVRO_CAIXA, usar_reserva=args.usar_reserva)

    def fabricar(modelo, temperatura):
        return fabricar_cliente(modelo, livro, temperatura)

    codigo = 0
    try:
        executar(cfgs, args.rodada, carregar_conjunto, fabricar, ClienteEmbeddings(livro), args.piloto)
    except KeyboardInterrupt:
        print("interrompido: rode o mesmo comando para retomar de onde parou", file=sys.stderr)
        codigo = 130
    print(f"gasto no livro-caixa: OpenAI US$ {livro.gasto('openai'):.4f}, "
          f"Anthropic US$ {livro.gasto('anthropic'):.4f}", flush=True)
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
