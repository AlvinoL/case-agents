"""Pilar 2 — Seleção de Tools Relevantes.

Busca sparse em cascata, com o mesmo TF-IDF de palavras + caracteres do router (text.py):
  1. Compara a query só com o NOME das tools: o nome é o rótulo curto da intenção
     (verbo + objeto, ex.: "parcelar fatura"), e o cosseno favorece o rótulo que casa por
     inteiro em vez de variações longas e específicas.
  2. Se nenhum nome casa acima do limiar, o nome não é confiável para essa query e o
     ranking usa o documento completo (nome + descrição + categoria): por similaridade
     semântica com o e5-small se `dense=True` (variante SLM), senão por TF-IDF.
Tool nova = linha nova nos índices, sem retreinar e sem exemplos rotulados.
"""
import time
from typing import List, Optional

import numpy as np
from sklearn.preprocessing import normalize as l2_normalize

from candidate_starter.embeddings import E5Encoder, load_encoder
from candidate_starter.text import build_vectorizer
from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch

# p90 da similaridade máxima nome x frase nas frases FAST_PATH do treino (o "ruído" de
# quem não pede tool). Derivado sem eval: python -m candidate_starter.validacao_retrieval
NAME_THRESHOLD = 0.53


def tool_name_text(tool: Tool) -> str:
    """Nome legível da tool: snake_case vira palavras."""
    return tool.name.replace("_", " ")


def tool_document(tool: Tool) -> str:
    """Documento completo da tool: nome legível + descrição + categoria."""
    return f"{tool_name_text(tool)} {tool.description} {tool.category}"


class _CosineIndex:
    """Índice TF-IDF com linhas de norma 1: produto escalar = similaridade de cosseno."""

    def __init__(self, texts: List[str]) -> None:
        self._vectorizer = build_vectorizer()
        # O FeatureUnion junta dois vetores de norma 1; renormalizar mantém o cosseno em [0, 1].
        self._matrix = l2_normalize(self._vectorizer.fit_transform(texts))

    def scores(self, query: str) -> np.ndarray:
        query_vec = l2_normalize(self._vectorizer.transform([query]))
        return (self._matrix @ query_vec.T).toarray().ravel()


class ToolRetriever(BaseToolRetriever):
    def __init__(self, name_threshold: float = NAME_THRESHOLD, dense: bool = False) -> None:
        self._tools: List[Tool] = []
        self._name_threshold = name_threshold
        self._dense = dense
        self._names = None
        self._documents = None
        self._encoder: Optional[E5Encoder] = None
        self._passages: Optional[np.ndarray] = None  # embeddings do documento de cada tool
        self._fitted = False

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo duas vezes: só nomes e documento completo."""
        self._tools = tools
        self._names = _CosineIndex([tool_name_text(t) for t in tools])
        self._documents = _CosineIndex([tool_document(t) for t in tools])
        if self._dense:
            self._encoder = load_encoder()  # None se indisponível: fallback para TF-IDF
            if self._encoder is not None:
                self._passages = self._encoder.encode_passages([tool_document(t) for t in tools])
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Top-k pelo nome; se nenhum nome casa acima do limiar, top-k pelo documento (e5 ou TF-IDF)."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()
        scores = self._names.scores(query)
        if scores.max() < self._name_threshold:
            if self._encoder is not None:
                scores = self._passages @ self._encoder.encode_queries([query])[0]
            else:
                scores = self._documents.scores(query)
        top = np.argsort(-scores, kind="stable")[:k]  # stable: empate decidido pela ordem do catálogo
        matches = [ToolMatch(name=self._tools[i].name, score=float(scores[i])) for i in top]
        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(matches=matches, latency_ms=latency_ms)
