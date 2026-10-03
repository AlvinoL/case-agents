"""Pilar 2 — Seleção de Tools Relevantes.

Busca sparse: cada tool vira um documento (nome + descrição + categoria) indexado com o
mesmo TF-IDF de palavras + caracteres do router (text.py). A query é vetorizada igual e as
tools são ranqueadas por similaridade de cosseno. Tool nova = linha nova no índice, sem
retreinar e sem precisar de exemplos rotulados.
"""
import time
from typing import List

import numpy as np
from sklearn.preprocessing import normalize as l2_normalize

from candidate_starter.text import build_vectorizer
from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


def tool_document(tool: Tool) -> str:
    """Texto indexado da tool: o nome em snake_case também carrega significado."""
    return f"{tool.name.replace('_', ' ')} {tool.description} {tool.category}"


class ToolRetriever(BaseToolRetriever):
    def __init__(self) -> None:
        self._tools: List[Tool] = []
        self._vectorizer = build_vectorizer()
        self._index = None  # matriz esparsa (n_tools x n_features), linhas com norma 1
        self._fitted = False

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Aprende vocabulário/IDF do catálogo e indexa cada tool como um vetor."""
        self._tools = tools
        # O FeatureUnion junta dois vetores de norma 1; renormalizar faz o produto escalar
        # ser exatamente o cosseno.
        self._index = l2_normalize(self._vectorizer.fit_transform([tool_document(t) for t in tools]))
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools por similaridade de cosseno com a query."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()
        query_vec = l2_normalize(self._vectorizer.transform([query]))
        scores = (self._index @ query_vec.T).toarray().ravel()
        top = np.argsort(-scores, kind="stable")[:k]  # stable: empate decidido pela ordem do catálogo
        matches = [ToolMatch(name=self._tools[i].name, score=float(scores[i])) for i in top]
        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(matches=matches, latency_ms=latency_ms)
