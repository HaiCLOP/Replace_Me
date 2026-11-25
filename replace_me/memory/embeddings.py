"""Local embeddings (fastembed, ONNX on CPU): nothing is sent anywhere to build or search your memory."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from replace_me.config import get_settings

_model = None


def embed(texts: Sequence[str], batch: int = 256) -> np.ndarray:
    """Normalised float32 vectors, one row per text."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=get_settings().EMBEDDING_MODEL)
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        for v in _model.embed(list(texts[i:i + batch])):
            v = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(v)
            out.append(v / n if n > 0 else v)
    return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)
