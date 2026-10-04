"""Deriva o limiar da cascata nome -> descrição do retriever, sem usar o eval nem rótulos de tool.

Critério: frases FAST_PATH do treino não pedem nenhuma tool, então a similaridade máxima
delas com os nomes das tools é o "nível de ruído". O limiar é o p90 desse ruído: uma query
que casa com o nome tão pouco quanto uma saudação casaria não confia no nome.

Uso: python -m candidate_starter.validacao_retrieval
"""
import numpy as np
from sklearn.preprocessing import normalize as l2_normalize

from candidate_starter.retrieval import tool_name_text
from candidate_starter.text import build_vectorizer
from common.data_loader import load_router_training_data, load_tools


def main() -> None:
    vectorizer = build_vectorizer()
    names = l2_normalize(vectorizer.fit_transform([tool_name_text(t) for t in load_tools()]))
    texts, labels = load_router_training_data()

    for label in ("FAST_PATH", "AGENT"):
        sims = np.array([
            (names @ l2_normalize(vectorizer.transform([t])).T).max()
            for t, lab in zip(texts, labels) if lab == label
        ])
        print(f"{label:<9} (treino, n={len(sims)}): similaridade máxima com um nome "
              f"p50 {np.percentile(sims, 50):.2f} | p90 {np.percentile(sims, 90):.2f}")
        if label == "FAST_PATH":
            threshold = np.percentile(sims, 90)
    print(f"\nLimiar da cascata (p90 do ruído FAST_PATH): {threshold:.2f}")


if __name__ == "__main__":
    main()
