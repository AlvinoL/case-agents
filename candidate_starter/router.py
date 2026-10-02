"""Pilar 1 — Router de Queries.

Versão esqueleto (MVP): escolhe a rota ao acaso entre os rótulos vistos no treino.
Serve só para validar o fluxo ponta a ponta; será substituída por um classificador real.
"""
import random
import time
from typing import List

from common.interfaces import BaseRouter
from common.schemas import RouteResult


class QueryRouter(BaseRouter):
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)  # gerador próprio: reprodutível e isolado do global
        self._labels: List[str] = []
        self._fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Guarda apenas o conjunto de rótulos possíveis (não aprende nada)."""
        self._labels = sorted(set(labels))
        self._fitted = True
        return self

    def predict(self, query: str) -> RouteResult:
        """Sorteia uma rota e mede a latência (em ms)."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()
        route = self._rng.choice(self._labels)
        latency_ms = (time.perf_counter() - start) * 1000
        return RouteResult(route=route, latency_ms=latency_ms, confidence=1 / len(self._labels))
