import hashlib
from pathlib import Path

import tiktoken

import replica.autor
from replica.autor.metricas import (
    chunk_text_into_sentences,
    drqa_exact_match_score,
    normalize_answer,
    parse_output,
    post_process,
    substring_exact_match_score,
)

SHA_TEMPLATES_AUTOR = "148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd"  # utils/templates.py @538026, finais de linha LF


def test_templates_sao_copia_literal():
    bruto = (Path(replica.autor.__file__).parent / "templates.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(bruto).hexdigest() == SHA_TEMPLATES_AUTOR


def test_metricas_do_autor():
    assert substring_exact_match_score("Yes, they were.", "yes")
    assert not drqa_exact_match_score("label: 43", "43")
    assert drqa_exact_match_score("43", "43")
    assert normalize_answer("The Chief of Protocol!") == "chief of protocol"
    assert parse_output("label: 43\noutra linha") == "label: 43"


def test_post_process_icl_e_ruler_como_no_autor():
    icl = {"sub_dataset": "icl_banking77_5900shot_balance"}
    assert post_process({"output": "43"}, ["43"], icl)[0]["exact_match"]
    assert not post_process({"output": "label: 43"}, ["43"], icl)[0]["exact_match"]
    assert post_process({"output": "It was yes."}, ["yes"], {"sub_dataset": "ruler_qa2_421K"})[0]["substring_exact_match"]


def test_chunker_respeita_o_tamanho(monkeypatch):  # sem rede: troca o sent_tokenize do nltk
    monkeypatch.setattr("nltk.sent_tokenize", lambda t: t.split(". "))
    monkeypatch.setattr("nltk.download", lambda *a, **k: True)
    pedacos = chunk_text_into_sentences(". ".join(["frase curta"] * 400), chunk_size=50)
    assert all(len(tiktoken.get_encoding("o200k_base").encode(p)) <= 50 for p in pedacos) and len(pedacos) > 1
