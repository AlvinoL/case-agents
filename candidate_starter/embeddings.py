"""Encoder denso opcional (variante SLM): intfloat/multilingual-e5-small, licença MIT.

Dependências em requirements-slm.txt. Se não estiverem instaladas, `load_encoder()` devolve
None e o retriever cai para a busca sparse, então o case roda sem torch.
"""
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # fixada: o modelo não muda por baixo
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "embeddings"


class E5Encoder:
    """Embeddings normalizados (norma 1): produto escalar = cosseno.

    O e5 foi treinado para busca assimétrica e exige prefixos: "query: " para a pergunta
    curta do cliente e "passage: " para o texto indexado (a tool).
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # import tardio: dependência opcional

        self._model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._model.encode([f"query: {t}" for t in texts], normalize_embeddings=True)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        """Codifica o catálogo com cache em disco, chaveado por modelo, revisão e textos."""
        key = hashlib.sha256("\n".join([MODEL_ID, MODEL_REVISION, *texts]).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"passages_{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = self._model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


def load_encoder() -> Optional[E5Encoder]:
    """Carrega o e5; devolve None se as dependências opcionais ou o modelo não estiverem disponíveis."""
    try:
        return E5Encoder()
    except Exception as exc:  # ImportError, sem rede/cache do modelo, etc.
        print(f"[aviso] encoder denso indisponível ({type(exc).__name__}); usando busca sparse.")
        return None
