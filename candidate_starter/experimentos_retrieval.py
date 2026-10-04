"""Experimentos de ranking do retriever: TF-IDF, BM25, TF-IDF sublinear e e5-small.

Mede Hit@1, Hit@2, Hit@3 e MRR (ranking completo) nas 20 queries AGENT do eval, isolando o
retriever do router. Nenhuma variante aqui supera a cascata da v04; o objetivo é documentar
o que foi testado e por quê. O e5 só entra se as dependências opcionais estiverem instaladas.

Uso: python -m candidate_starter.experimentos_retrieval
"""
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize as l2_normalize

from candidate_starter.embeddings import load_encoder
from candidate_starter.retrieval import NAME_THRESHOLD, ToolRetriever, tool_document, tool_name_text
from candidate_starter.text import normalize
from common.data_loader import load_eval_dataset, load_tools

SAIDA = Path(__file__).resolve().parent.parent / "docs" / "resultados" / "experimentos_retrieval.json"
Scorer = Callable[[str], np.ndarray]


class BM25:
    """BM25 Okapi: idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * tamanho / tamanho_medio)).

    k1 satura a frequência do termo; b controla a penalidade por documento longo (0 = nenhuma).
    """

    def __init__(self, texts: List[str], k1: float = 1.2, b: float = 0.75, analyzer: str = "word",
                 ngram_range: tuple = (1, 1)) -> None:
        self._counter = CountVectorizer(preprocessor=normalize, analyzer=analyzer, ngram_range=ngram_range)
        tf = self._counter.fit_transform(texts).toarray().astype(float)
        n_docs, df = tf.shape[0], (tf > 0).sum(axis=0)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        length = tf.sum(axis=1, keepdims=True)
        self._weights = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / length.mean()))

    def scores(self, query: str) -> np.ndarray:
        return self._weights @ (self._counter.transform([query]).toarray().ravel() > 0)


def tfidf_sublinear(texts: List[str]) -> Scorer:
    """TF-IDF palavras + caracteres com log da frequência (saturação leve, como no BM25)."""
    vectorizer = FeatureUnion([
        ("palavras", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                                       sublinear_tf=True)),
    ])
    matrix = l2_normalize(vectorizer.fit_transform(texts))
    return lambda q: (matrix @ l2_normalize(vectorizer.transform([q])).T).toarray().ravel()


def cascata(name_scorer: Scorer, fallback: Scorer, threshold: float = NAME_THRESHOLD) -> Scorer:
    """Nome se algum casa acima do limiar; senão, o fallback."""
    def scorer(query: str) -> np.ndarray:
        scores = name_scorer(query)
        return scores if scores.max() >= threshold else fallback(query)
    return scorer


def avaliar(scorer: Scorer, names: List[str], queries: List[dict]) -> Dict[str, float]:
    """Posição do gabarito no ranking completo -> Hit@1/2/3 e MRR."""
    ranks = np.array([
        [names[i] for i in np.argsort(-scorer(q["query"]), kind="stable")].index(q["expected_tool"]) + 1
        for q in queries
    ])
    return {
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_2": float(np.mean(ranks <= 2)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1 / ranks)),
    }


def main() -> None:
    tools = load_tools()
    names = [t.name for t in tools]
    docs = [tool_document(t) for t in tools]
    queries = [e for e in load_eval_dataset() if e["expected_route"] == "AGENT"]
    v04 = ToolRetriever().fit(tools)  # índices TF-IDF de nomes e documentos

    experimentos: Dict[str, Scorer] = {
        "documento | TF-IDF palavras+caracteres (v03)": v04._documents.scores,
        "documento | TF-IDF sublinear": tfidf_sublinear(docs),
        "documento | BM25 palavras (k1=1.2, b=0.75)": BM25(docs).scores,
        "documento | BM25 palavras (k1=1.2, b=1.0)": BM25(docs, b=1.0).scores,
        "documento | BM25 caracteres 3-5 (b=0.75)": BM25(docs, analyzer="char_wb", ngram_range=(3, 5)).scores,
        "nome | TF-IDF palavras+caracteres": v04._names.scores,
        "nome | BM25 palavras": BM25([tool_name_text(t) for t in tools]).scores,
        "cascata nome -> TF-IDF documento (v04)": cascata(v04._names.scores, v04._documents.scores),
        "cascata nome -> BM25 documento (b=0.75)": cascata(v04._names.scores, BM25(docs).scores),
    }

    encoder = load_encoder()
    if encoder is not None:
        passages = encoder.encode_passages(docs)
        e5: Scorer = lambda q: passages @ encoder.encode_queries([q])[0]
        experimentos["documento | e5-small"] = e5
        experimentos["cascata nome -> e5-small (limiar 0,53; v05)"] = cascata(v04._names.scores, e5)
        experimentos["cascata nome -> e5-small (limiar 0,80; v05.1)"] = cascata(v04._names.scores, e5, 0.80)

    resultados = {nome: avaliar(scorer, names, queries) for nome, scorer in experimentos.items()}
    print(f"Ranking completo nas {len(queries)} queries AGENT do eval (retriever isolado do router)")
    for nome, r in resultados.items():
        print(f"  {nome:<48} Hit@1 {r['hit_at_1']:>4.0%} | Hit@2 {r['hit_at_2']:>4.0%}"
              f" | Hit@3 {r['hit_at_3']:>4.0%} | MRR {r['mrr']:.2f}")
    SAIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {SAIDA}")


if __name__ == "__main__":
    main()
