"""Turn text into vectors with an embedding model.

An embedding model reads a piece of text and returns a vector: a list of
numbers (384 of them for our default model) that represents its meaning.
Texts with similar meanings get similar vectors, even when they use
different words, e.g. "build data pipelines" and "develop ETL workflows".

This is the ONLY module that knows which model we use. Everything else just
calls `embed_texts()`, or, in tests, a fake function with the same shape.

We use fastembed, a library that runs small embedding models on your own
computer (no API key, no cost). The first time it runs, it downloads the
model (about 67 MB) into data/models/. After that it works offline.
"""

import os
from collections.abc import Callable
from pathlib import Path

from fastembed import TextEmbedding

# The default model. Change it with the JOBS_SEARCH_EMBEDDING_MODEL setting.
# (If you do, rebuild the index: vectors from different models don't mix.
# And per CLAUDE.md, run the eval before and after.)
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# A name for "any function that takes a list of texts and returns a list of
# vectors". `Callable[[A], B]` means "a function that takes an A and returns
# a B". rag.py uses this type so it can accept either the real embed_texts()
# or a fake one in tests.
Embedder = Callable[[list[str]], list[list[float]]]

# Loading a model takes a few seconds, so we load each one only once and
# keep it here. It's a dict of model name -> loaded model, empty at first.
_loaded_models: dict[str, TextEmbedding] = {}


def embedding_model_name() -> str:
    """The embedding model to use, from the JOBS_SEARCH_EMBEDDING_MODEL setting."""
    return os.environ.get("JOBS_SEARCH_EMBEDDING_MODEL", DEFAULT_MODEL)


def model_cache_dir() -> Path:
    """Where downloaded models are kept: data/models/ (inside JOBS_SEARCH_DATA_DIR).

    fastembed's own default is a temporary folder that Windows may clear,
    which would mean downloading the model again.
    """
    data_dir = os.environ.get("JOBS_SEARCH_DATA_DIR", "data")
    return Path(data_dir) / "models"


def get_model() -> TextEmbedding:
    """Return the embedding model, loading it the first time it's needed."""
    name = embedding_model_name()
    if name not in _loaded_models:
        _loaded_models[name] = TextEmbedding(
            model_name=name, cache_dir=str(model_cache_dir())
        )
    return _loaded_models[name]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn each text into a vector. Returns one vector per text, in the same order.

    Example:
        embed_texts(["build data pipelines", "sell software"])
        -> [[0.012, -0.044, ...], [0.031, 0.008, ...]]   (384 numbers each)
    """
    model = get_model()
    vectors = []
    # `model.embed()` gives back numpy arrays (a fast kind of number list from
    # the numpy library). `.tolist()` turns each into a plain Python list.
    for vector in model.embed(texts):
        vectors.append(vector.tolist())
    return vectors
