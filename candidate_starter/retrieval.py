"""Pilar 2 — Seleção de Tools Relevantes.

Versão esqueleto (MVP): devolve k tools sorteadas do catálogo, ignorando a query.
Serve só para validar o fluxo ponta a ponta; será substituída por busca híbrida.
"""
import random
import time
from typing import List

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


class ToolRetriever(BaseToolRetriever):
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._tools: List[Tool] = []
        self._fitted = False

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Guarda o catálogo (não há índice nesta versão)."""
        self._tools = tools
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Sorteia k tools distintas; score 0.0 porque não há ranking."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()
        sampled = self._rng.sample(self._tools, k=min(k, len(self._tools)))
        matches = [ToolMatch(name=t.name, score=0.0) for t in sampled]
        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(matches=matches, latency_ms=latency_ms)
