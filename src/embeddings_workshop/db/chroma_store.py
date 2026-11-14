import hashlib
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from embeddings_workshop.constants.chroma import (CHROMA_COLLECTION,
                                                  CHROMA_DB_PATH)
from embeddings_workshop.constants.ollama import DEFAULT_EMBEDDING_MODEL

load_dotenv()

vector_store = Chroma(
    collection_name=os.getenv("CHROMA_COLLECTION", CHROMA_COLLECTION),
    persist_directory=os.getenv("CHROMA_DB_PATH", CHROMA_DB_PATH),
    embedding_function=OllamaEmbeddings(
        model=os.getenv("EMBEDDINGS_MODEL", DEFAULT_EMBEDDING_MODEL)
    ),
    collection_metadata={"hnsw:space": "cosine"},
)


def get_retriever(k: int = 5):
    return vector_store.as_retriever(search_kwargs={"k": k})


def _get_chunk_id(chunk: Document):
    raw_id = f"{chunk.metadata.get('source', '')}|{chunk.metadata.get('page', '')}|{chunk.page_content}"
    return hashlib.sha256(raw_id.encode()).hexdigest()


def save_chunks(chunks: list[Document]) -> int:

    vector_store.add_documents(chunks, ids=[_get_chunk_id(c) for c in chunks])

    return len(chunks)
