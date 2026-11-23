"""Tests for embeddings.py.

Per CLAUDE.md, tests never load the real embedding model (it's a 67 MB
download and slow). Instead `monkeypatch` swaps fastembed's TextEmbedding for
a tiny fake, so we can still check that our code uses it correctly.
"""

from pathlib import Path

import numpy

from jobs_search import embeddings


class FakeModel:
    """Pretends to be fastembed's TextEmbedding. Counts how often it's created."""

    created = 0  # shared by all FakeModels, so tests can read it

    def __init__(self, model_name, cache_dir):
        FakeModel.created += 1
        self.model_name = model_name
        self.cache_dir = cache_dir

    def embed(self, texts):
        # Like the real model, give back numpy arrays: here, [length, 1.0].
        for text in texts:
            yield numpy.array([float(len(text)), 1.0])


def use_fake_model(monkeypatch):
    """Swap in FakeModel and start with no models loaded, for one test."""
    FakeModel.created = 0
    monkeypatch.setattr(embeddings, "TextEmbedding", FakeModel)
    monkeypatch.setattr(embeddings, "_loaded_models", {})


def test_model_name_has_a_default(monkeypatch):
    monkeypatch.delenv("JOBS_SEARCH_EMBEDDING_MODEL", raising=False)

    assert embeddings.embedding_model_name() == "BAAI/bge-small-en-v1.5"


def test_model_name_comes_from_the_setting(monkeypatch):
    monkeypatch.setenv("JOBS_SEARCH_EMBEDDING_MODEL", "some/other-model")

    assert embeddings.embedding_model_name() == "some/other-model"


def test_models_are_kept_in_the_data_folder(monkeypatch):
    monkeypatch.setenv("JOBS_SEARCH_DATA_DIR", "somewhere")

    assert embeddings.model_cache_dir() == Path("somewhere/models")


def test_embed_texts_returns_plain_lists_in_order(monkeypatch):
    use_fake_model(monkeypatch)

    vectors = embeddings.embed_texts(["ab", "abcd"])

    assert vectors == [[2.0, 1.0], [4.0, 1.0]]
    assert type(vectors[0]) is list  # not a numpy array


def test_model_is_loaded_only_once(monkeypatch):
    use_fake_model(monkeypatch)

    embeddings.embed_texts(["one"])
    embeddings.embed_texts(["two"])

    assert FakeModel.created == 1
