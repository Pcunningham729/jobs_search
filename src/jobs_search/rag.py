"""Chunk job descriptions, embed them, and keep them in a Chroma index.

RAG stands for "retrieval-augmented generation": before asking an LLM
something, you *retrieve* the most relevant text and include it. This module
is the retrieval half. It keeps a searchable index of every job, so later
steps can ask "which jobs are about building data pipelines?".

Chroma is a database made for vectors. Give it a vector and it quickly finds
the stored vectors closest to it. It keeps its files in data/chroma/.

The index holds two "collections" (Chroma's word for a table):
- job_chunks: each job description, split into chunks of about 1,000
  characters, one vector per chunk. matching.py compares these with your
  interests.
- job_titles: one vector per job title, for title search.

Rules from CLAUDE.md:
- SQLite (store.py) is the source of truth. This index can always be rebuilt
  from it with `rebuild_index()`.
- Job vectors are calculated once, when a job is indexed, and stored here.
"""

import os
import sqlite3
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel

from jobs_search.embeddings import Embedder
from jobs_search.models import Job
from jobs_search.store import list_jobs

# Chunks are at most this many characters, about 200-250 words. The model
# reads up to 512 tokens (word pieces, roughly 350 words) and silently cuts
# off the rest, so chunks must stay under that.
# (Per CLAUDE.md: run the eval before and after changing this.)
CHUNK_MAX_CHARS = 1000

CHUNKS_COLLECTION = "job_chunks"
TITLES_COLLECTION = "job_titles"

# Tells Chroma to measure "cosine distance" between vectors. Distance is
# 1 minus similarity: 0 means the same meaning, around 1 means unrelated.
COSINE = {"hnsw": {"space": "cosine"}}


class JobIndex:
    """The open Chroma database and its two collections.

    `open_index()` creates one of these. Other functions in this module take
    it as their first argument, the way store.py functions take `conn`.
    """

    def __init__(self, client, chunks, titles):
        self.client = client  # the connection to Chroma's files
        self.chunks = chunks  # the job_chunks collection
        self.titles = titles  # the job_titles collection


class TitleMatch(BaseModel):
    """One result from a title search."""

    job_id: str
    title: str
    company: str
    # How close the title is to what you searched for: 1.0 = same meaning.
    similarity: float


def chroma_path() -> Path:
    """Where Chroma keeps its files: data/chroma/ (inside JOBS_SEARCH_DATA_DIR)."""
    data_dir = os.environ.get("JOBS_SEARCH_DATA_DIR", "data")
    return Path(data_dir) / "chroma"


def open_index(path: Path) -> JobIndex:
    """Open (or create) the Chroma index stored in the folder `path`."""
    client = chromadb.PersistentClient(
        path=str(path),
        # Chroma sends anonymous usage data to its makers by default. Turn that off.
        settings=Settings(anonymized_telemetry=False),
    )
    chunks, titles = _get_collections(client)
    return JobIndex(client, chunks, titles)


def _get_collections(client):
    """Get both collections, creating them if they don't exist yet.

    The `_` at the start of the name means "only used inside this module".
    `embedding_function=None` tells Chroma we'll always give it vectors we
    calculated ourselves, so it shouldn't load a model of its own.
    """
    chunks = client.get_or_create_collection(
        CHUNKS_COLLECTION, embedding_function=None, configuration=COSINE
    )
    titles = client.get_or_create_collection(
        TITLES_COLLECTION, embedding_function=None, configuration=COSINE
    )
    return chunks, titles


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split one very long paragraph into pieces of at most `max_chars`.

    It splits between words, never in the middle of one.
    """
    pieces = []
    current = ""
    for word in paragraph.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        elif current:
            current = current + " " + word
        else:
            current = word
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split a description into chunks of at most `max_chars` characters.

    It keeps whole paragraphs (lines) together where it can: paragraphs are
    added to a chunk until the next one wouldn't fit, then a new chunk starts.

    Example with max_chars=31:
        "Short line.\\nAnother short line.\\nA third line here."
        -> ["Short line.\\nAnother short line.", "A third line here."]
    """
    # Break the text into paragraphs, one per line, skipping blank lines.
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > max_chars:
            # Too long for one chunk on its own, so split it into pieces.
            paragraphs.extend(split_long_paragraph(line, max_chars))
        else:
            paragraphs.append(line)

    chunks = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + 1 + len(paragraph) > max_chars:
            # Adding this paragraph would be too long: finish this chunk.
            chunks.append(current)
            current = paragraph
        elif current:
            current = current + "\n" + paragraph
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def index_jobs(index: JobIndex, jobs: list[Job], embed: Embedder) -> int:
    """Embed jobs and save their vectors in the index. Returns the number of chunks.

    Use it for jobs that store.save_jobs() reported as new or changed. For a
    job that's already indexed, its old chunks are deleted first, because a
    shorter new description may have fewer chunks than before.

    `embed` is the function that turns texts into vectors: embed_texts from
    embeddings.py in the real program, or a fake one in tests.
    """
    chunk_count = 0
    for job in jobs:
        # Remove anything already stored for this job.
        index.chunks.delete(where={"job_id": job.id})

        # --- The description, in chunks ---
        chunks = chunk_text(job.description)
        if chunks:
            # Each chunk needs its own id, like "lever:acme:1#0", "lever:acme:1#1".
            chunk_ids = []
            metadatas = []
            for number in range(len(chunks)):
                chunk_ids.append(f"{job.id}#{number}")
                # Metadata: extra facts stored with each vector. `job_id` lets
                # us find (or delete) every chunk belonging to one job.
                metadatas.append({"job_id": job.id, "chunk": number})

            index.chunks.add(
                ids=chunk_ids,
                embeddings=embed(chunks),
                documents=chunks,  # the chunk text itself, kept for display
                metadatas=metadatas,
            )
            chunk_count += len(chunks)

        # --- The title ---
        # `upsert` means "insert, or update if this id already exists".
        index.titles.upsert(
            ids=[job.id],
            embeddings=embed([job.title]),
            documents=[job.title],
            metadatas=[{"job_id": job.id, "company": job.company}],
        )
    return chunk_count


def search_titles(
    index: JobIndex, query: str, embed: Embedder, limit: int = 10
) -> list[TitleMatch]:
    """Find the jobs whose titles are closest in meaning to `query`, best first.

    Example: search_titles(index, "data engineer", embed_texts) might return
    "Senior Data Engineer", "Analytics Engineer", "Data Platform Engineer"...
    """
    if index.titles.count() == 0:
        return []  # nothing indexed yet

    # Chroma can search for several queries at once, so it takes a list of
    # query vectors and returns a list of result lists. We send one query,
    # so we always read item [0] of each result list.
    results = index.titles.query(
        query_embeddings=embed([query]),
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for job_id, title, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append(
            TitleMatch(
                job_id=job_id,
                title=title,
                company=metadata["company"],
                similarity=1 - distance,  # distance 0 -> similarity 1
            )
        )
    return matches


def get_job_vectors(index: JobIndex, job_id: str) -> list[list[float]]:
    """Return the stored vectors for one job's description chunks.

    matching.py uses these to compare a job with your interests, without
    running the embedding model on the job again.
    Returns an empty list if the job isn't indexed.
    """
    result = index.chunks.get(where={"job_id": job_id}, include=["embeddings"])
    vectors = []
    for vector in result["embeddings"]:
        vectors.append(vector.tolist())  # numpy array -> plain list
    return vectors


def get_title_vector(index: JobIndex, job_id: str) -> list[float] | None:
    """Return the stored vector for one job's title, or None if it isn't indexed."""
    result = index.titles.get(ids=[job_id], include=["embeddings"])
    if len(result["ids"]) == 0:
        return None
    return result["embeddings"][0].tolist()  # numpy array -> plain list


def rebuild_index(conn: sqlite3.Connection, index: JobIndex, embed: Embedder) -> int:
    """Throw away the whole index and rebuild it from the SQLite store.

    Use this after changing the embedding model or the chunk size, or if the
    index ever gets out of step with the store. Returns the number of chunks.
    """
    # Delete both collections, then create them again, empty.
    index.client.delete_collection(CHUNKS_COLLECTION)
    index.client.delete_collection(TITLES_COLLECTION)
    index.chunks, index.titles = _get_collections(index.client)

    return index_jobs(index, list_jobs(conn), embed)
