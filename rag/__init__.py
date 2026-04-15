"""
Módulo RAG do agente Kisseki.

Exports principais:
    ingest()             — processa o cardápio.md e popula o ChromaDB
    retrieve()           — busca híbrida (BM25 + vetorial) por pratos
    get_retriever()      — retriever LangChain para uso em chains
    get_hybrid_retriever() — retriever híbrido explícito
    get_vectorstore()    — vectorstore ChromaDB
    format_docs()        — formata documentos para o contexto do LLM
    get_llm()            — factory de LLM (provider configurado via .env)
    get_embeddings()     — factory de Embeddings (provider configurado via .env)
    get_settings()       — singleton de configuração
    describe_config()    — diagnóstico da configuração ativa (sem expor chaves)
"""

from __future__ import annotations

from typing import Any


def ingest(*args: Any, **kwargs: Any):
    from rag.ingestor import ingest as _fn
    return _fn(*args, **kwargs)


def retrieve(*args: Any, **kwargs: Any):
    from rag.retriever import retrieve as _fn
    return _fn(*args, **kwargs)


def get_retriever(*args: Any, **kwargs: Any):
    from rag.retriever import get_retriever as _fn
    return _fn(*args, **kwargs)


def get_hybrid_retriever(*args: Any, **kwargs: Any):
    from rag.retriever import get_hybrid_retriever as _fn
    return _fn(*args, **kwargs)


def get_vectorstore(*args: Any, **kwargs: Any):
    from rag.retriever import get_vectorstore as _fn
    return _fn(*args, **kwargs)


def format_docs(*args: Any, **kwargs: Any):
    from rag.retriever import format_docs as _fn
    return _fn(*args, **kwargs)


def get_llm(*args: Any, **kwargs: Any):
    from rag.providers import get_llm as _fn
    return _fn(*args, **kwargs)


def get_embeddings(*args: Any, **kwargs: Any):
    from rag.providers import get_embeddings as _fn
    return _fn(*args, **kwargs)


def get_settings(*args: Any, **kwargs: Any):
    from rag.config import get_settings as _fn
    return _fn(*args, **kwargs)


def describe_config(*args: Any, **kwargs: Any):
    from rag.providers import describe_config as _fn
    return _fn(*args, **kwargs)


__all__ = [
    "ingest",
    "retrieve",
    "get_retriever",
    "get_hybrid_retriever",
    "get_vectorstore",
    "format_docs",
    "get_llm",
    "get_embeddings",
    "get_settings",
    "describe_config",
]
