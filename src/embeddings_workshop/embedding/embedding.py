import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from embeddings_workshop.constants.ollama import DEFAULT_EMBEDDING_MODEL
from embeddings_workshop.utils.time import timed_function

load_dotenv()


@lru_cache(maxsize=None)
def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=os.getenv("EMBEDDINGS_MODEL", DEFAULT_EMBEDDING_MODEL)
    )


@timed_function
def get_embedding(text: str) -> list[float]:
    return get_embedder().embed_query(text)


@timed_function
def get_embeddings(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:

    embedder = get_embedder()

    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed_documents(texts[i : i + batch_size]))

    return vectors
