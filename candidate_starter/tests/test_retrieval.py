"""Testes do retriever sparse (TF-IDF + cosseno)."""
from candidate_starter.retrieval import ToolRetriever, tool_document
from common.data_loader import load_tools
from common.schemas import Tool


def test_documento_da_tool_inclui_nome_legivel_descricao_e_categoria():
    tool = Tool(name="bloquear_cartao", description="Bloqueia o cartão.", category="cartao")
    assert tool_document(tool) == "bloquear cartao Bloqueia o cartão. cartao"


def test_ranking_ordenado_por_score_e_deterministico():
    retriever = ToolRetriever().fit(load_tools())
    first = retriever.search("Perdi meu cartão, quero bloquear", k=5)
    scores = [m.score for m in first.matches]

    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 + 1e-9 for s in scores)  # cosseno de vetores não negativos
    assert [m.name for m in retriever.search("Perdi meu cartão, quero bloquear", k=5).matches] == [
        m.name for m in first.matches
    ]
