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


def test_cascata_usa_nome_ou_documento_conforme_o_limiar():
    tools = [
        Tool(name="parcelar_fatura", description="Permite dividir o valor da fatura.", category="financeiro"),
        Tool(name="consultar_saldo", description="Mostra quanto dinheiro está disponível na conta.", category="financeiro"),
    ]
    query = "quanto dinheiro tenho disponível"  # não cita nenhum nome; só a descrição do saldo casa

    so_nome = ToolRetriever(name_threshold=0.0).fit(tools).search(query, k=1)  # limiar 0: sempre o nome
    so_documento = ToolRetriever(name_threshold=1.01).fit(tools).search(query, k=1)  # >1: sempre o documento

    assert so_documento.matches[0].name == "consultar_saldo"
    assert so_nome.matches[0].score < so_documento.matches[0].score  # o nome casa pior que o documento


def test_dense_sem_encoder_disponivel_cai_para_busca_sparse(monkeypatch):
    """Sem torch/modelo, a variante SLM não quebra: o fallback é o TF-IDF."""
    import candidate_starter.retrieval as retrieval

    monkeypatch.setattr(retrieval, "load_encoder", lambda: None)
    tools = load_tools()
    query = "Quanto eu tenho disponível na conta agora?"

    dense = ToolRetriever(dense=True).fit(tools).search(query, k=2)
    sparse = ToolRetriever().fit(tools).search(query, k=2)
    assert [m.name for m in dense.matches] == [m.name for m in sparse.matches]
