from __future__ import annotations
import os
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "rag/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4


class RegulationsRetriever:
    def __init__(self):
        self.loaded = False
        try:
            if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
                print("RAG: chroma_db not found. Call build_index() first.")
                return
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",               # Maximum Marginal Relevance for diversity
                search_kwargs={"k": TOP_K, "fetch_k": 10},
            )
            self.loaded = True
            print(f"RAG: vector store loaded ({PERSIST_DIR})")
        except Exception as e:
            print(f"RAG: failed to load vector store: {e}")

    def retrieve(self, query: str) -> tuple[str, list]:
        """
        Returns (formatted_context_str, list_of_source_names).
        """
        if not self.loaded:
            return "No regulatory context available.", []

        try:
            docs = self.retriever.invoke(query)
        except Exception as e:
            return f"Retrieval error: {e}", []

        if not docs:
            return "No regulatory guidelines found for this query.", []

        parts, sources = [], []
        for doc in docs:
            src  = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "?")
            parts.append(f"[Source: {src}, Page: {page}]\n{doc.page_content.strip()}")
            if src not in sources:
                sources.append(src)

        return "\n\n---\n\n".join(parts), sources


# Singleton — loaded once at import time
_retriever = RegulationsRetriever()


@lru_cache(maxsize=256)
def _cached_retrieve(query: str) -> tuple:
    return _retriever.retrieve(query)


def get_relevant_regulations(query: str) -> tuple[str, list]:
    """
    Public interface for retrieving regulatory context.
    Results are cached per unique query string.
    """
    return _cached_retrieve(query)
