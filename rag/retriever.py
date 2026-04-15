"""
Retriever do cardápio Kisseki.

Estratégia de busca híbrida (BM25 + vetorial):
- BM25 (TF-IDF léxico): captura correspondências exatas de palavras-chave.
  Resolve casos como "combo com yakisoba" que precisam de match literal.
- Vetorial com MMR: captura semântica e sinônimos ("vegetariano" → "vegana").
  MMR garante diversidade para não retornar versões duplicadas do mesmo prato.
- EnsembleRetriever combina os dois com pesos configuráveis (padrão: 40/60).

Para queries estruturadas (faixa de preço, seção), filtros de metadados podem
ser combinados com qualquer um dos dois modos.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any

# Desabilita telemetria do ChromaDB antes de qualquer import do chromadb
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.config import Settings, get_settings
from rag.providers import get_embeddings

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_JSON = os.path.join(BASE_DIR, "data", "documents.json")


# ── Preprocessador BM25 ────────────────────────────────────────────────────────

def _bm25_preprocess(text: str) -> list[str]:
    """
    Tokenizador customizado para BM25 em português.

    O tokenizador padrão do LangChain (split por whitespace) produz tokens
    como "vegetariana," (com vírgula) que não casam com a query "vegetariana".
    Este preprocessador:
    1. Normaliza unicode (ex.: "á" → "a") para matching mais tolerante.
    2. Remove pontuação e caracteres especiais.
    3. Converte para minúsculas.
    4. Elimina tokens muito curtos (stop words simples como "e", "a", "o").
    """
    # Normaliza acentos: "vegetariana" e "vegetariana" se tornam equivalentes
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove pontuação mantendo espaços
    text = re.sub(r"[^\w\s]", " ", text.lower())
    # Tokeniza e remove tokens muito curtos (< 3 chars) — são stop words comuns
    return [t for t in text.split() if len(t) >= 3]


# ── Vectorstore ────────────────────────────────────────────────────────────────

def get_vectorstore(settings: Settings | None = None) -> Chroma:
    """Retorna o ChromaDB. Levanta RuntimeError se o índice ainda não existe."""
    cfg         = settings or get_settings()
    persist_dir = os.path.abspath(cfg.chroma_persist_dir)

    if not os.path.exists(persist_dir):
        raise RuntimeError(
            f"Índice ChromaDB não encontrado em '{persist_dir}'.\n"
            "Execute primeiro:  python -m rag.ingestor"
        )
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(cfg),
        collection_name=cfg.chroma_collection,
    )


# ── BM25 ───────────────────────────────────────────────────────────────────────

def _load_bm25_retriever(k: int) -> BM25Retriever:
    """
    Carrega os documentos do JSON gerado pelo ingestor e constrói o BM25Retriever.
    O BM25 opera sobre o texto completo (preamble + Markdown), então keywords
    como "yakisoba", "vegano", "nikuman" são encontradas com precisão exata.
    """
    if not os.path.exists(DOCS_JSON):
        raise RuntimeError(
            f"Arquivo de documentos não encontrado em '{DOCS_JSON}'.\n"
            "Execute primeiro:  python -m rag.ingestor"
        )
    with open(DOCS_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw]
    retriever = BM25Retriever.from_documents(docs, preprocess_func=_bm25_preprocess)
    retriever.k = k
    return retriever


# ── Retrievers públicos ────────────────────────────────────────────────────────

def get_vector_retriever(
    k: int | None = None,
    search_type: str | None = None,
    filter: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> BaseRetriever:
    """
    Retriever puramente vetorial (MMR ou similarity).
    Use quando o contexto já é bem semântico e não há palavras-chave críticas.
    """
    cfg            = settings or get_settings()
    effective_k    = k           if k           is not None else cfg.retriever_k
    effective_type = search_type if search_type is not None else cfg.retriever_search_type

    search_kwargs: dict[str, Any] = {"k": effective_k}
    if filter:
        search_kwargs["filter"] = filter
    if effective_type == "mmr":
        search_kwargs["fetch_k"]     = effective_k * 4
        search_kwargs["lambda_mult"] = 0.7  # ligeiramente mais relevância, menos diversidade

    return get_vectorstore(cfg).as_retriever(
        search_type=effective_type,
        search_kwargs=search_kwargs,
    )


def _reciprocal_rank_fusion(
    results_list: list[list[Document]],
    weights: list[float],
    k_rrf: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) — combina múltiplas listas rankeadas em uma só.

    RRF é o mesmo algoritmo usado internamente pelo LangChain EnsembleRetriever.
    Fórmula: score(d) = Σ weight_i / (k_rrf + rank_i(d))

    Vantagens sobre média simples:
    - Não requer normalização de scores entre sistemas distintos.
    - Documentos no topo de qualquer lista ganham boost significativo.
    - Documentos presentes em múltiplas listas são fortemente rerankeados.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results, weight in zip(results_list, weights):
        for rank, doc in enumerate(results, start=1):
            # Usa o nome do prato como chave de deduplicação
            key = doc.metadata.get("prato", doc.page_content[:80])
            scores[key] = scores.get(key, 0.0) + weight / (k_rrf + rank)
            doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def get_hybrid_retriever(
    k: int | None = None,
    filter: dict[str, Any] | None = None,
    vector_weight: float = 0.45,
    bm25_weight: float = 0.55,
    settings: Settings | None = None,
):
    """
    Retriever híbrido: BM25 + vetorial combinados por Reciprocal Rank Fusion.

    Por que híbrido?
    - BM25 cobre keywords exatas: "yakisoba", "nikuman", "gyoza" → 0 falsos negativos.
    - Vetorial cobre semântica: "vegetariano" → Berinjela Vegana; "dividir" → compartilhar.
    - RRF combina as listas sem precisar normalizar scores de sistemas distintos.

    Pesos padrão (0.55 BM25 / 0.45 vetorial):
    - BM25 levemente dominante para garantir que matches de keyword exata ganhem
      mesmo quando o vetor retorna resultados irrelevantes (ex.: índice desatualizado).
    - Após re-indexar com preambles semânticos, o vetor melhora e ambos se reforçam.

    fetch_k = max(k * 6, 20):
    - Coleta candidatos suficientes do vetor para que o BM25 top-1 sempre apareça
      no pool do RRF, mesmo que o índice vetorial seja impreciso.

    Args:
        k:             Número de resultados finais a retornar.
        filter:        Filtros de metadados aplicados ao retriever vetorial.
        vector_weight: Peso do vetorial no RRF (padrão 0.45).
        bm25_weight:   Peso do BM25 no RRF (padrão 0.55).
        settings:      Instância de Settings.
    """
    cfg         = settings or get_settings()
    effective_k = k if k is not None else cfg.retriever_k

    # fetch_k generoso: garante que candidatos BM25 competem no pool do RRF
    # mesmo quando o índice vetorial ainda é o antigo (sem preambles).
    fetch_k = max(effective_k * 6, 20)

    vector_r = get_vector_retriever(
        k=fetch_k, search_type="similarity", filter=filter, settings=cfg
    )
    bm25_r = _load_bm25_retriever(k=fetch_k)

    class _HybridRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
            vec_docs  = vector_r.invoke(query)
            bm25_docs = bm25_r.invoke(query)
            fused = _reciprocal_rank_fusion(
                [vec_docs, bm25_docs],
                [vector_weight, bm25_weight],
            )
            return fused[:effective_k]

    return _HybridRetriever()


def get_retriever(
    k: int | None = None,
    search_type: str | None = None,
    filter: dict[str, Any] | None = None,
    hybrid: bool = True,
    settings: Settings | None = None,
) -> BaseRetriever:
    """
    Ponto de entrada unificado para retrievers.

    Args:
        hybrid: Se True (padrão), usa busca híbrida BM25 + vetorial.
                Se False, usa apenas busca vetorial (MMR).
    """
    if hybrid:
        return get_hybrid_retriever(k=k, filter=filter, settings=settings)
    return get_vector_retriever(
        k=k, search_type=search_type, filter=filter, settings=settings
    )


# ── Interface de alto nível ────────────────────────────────────────────────────

def retrieve(
    query: str,
    k: int | None = None,
    filter: dict[str, Any] | None = None,
    hybrid: bool = True,
    settings: Settings | None = None,
) -> list[Document]:
    """
    Busca os documentos mais relevantes. Usa busca híbrida por padrão.

    Args:
        query:    Pergunta ou texto do usuário.
        k:        Número de resultados.
        filter:   Filtros de metadados (opcional).
        hybrid:   True = BM25 + vetorial. False = só vetorial.
        settings: Instância de Settings.

    Exemplos:
        retrieve("pratos vegetarianos")
        retrieve("combo com yakisoba")
        retrieve("algo barato", filter={"secao": "Porções Individuais"})
        retrieve("opção para compartilhar", k=5)
    """
    return get_retriever(k=k, filter=filter, hybrid=hybrid, settings=settings).invoke(query)


def format_docs(docs: list[Document]) -> str:
    """
    Formata docs para o prompt do LLM.
    Retorna apenas o bloco Markdown original (após o preamble interno).
    """
    parts = []
    for doc in docs:
        meta    = doc.metadata
        header  = f"### {meta.get('prato', 'Prato')}"
        if secao := meta.get("secao"):
            header += f" ({secao})"

        # Remove o preamble interno (antes do separador "---")
        content = doc.page_content
        if "\n---\n" in content:
            content = content.split("\n---\n", 1)[1]

        parts.append(f"{header}\n{content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "opções vegetarianas",
        "algo para compartilhar com amigos",
        "prato mais barato do cardápio",
        "combo com yakisoba",
    ]

    print("=" * 60)
    print("BUSCA HÍBRIDA (BM25 + Vetorial)")
    print("=" * 60)
    for q in queries:
        print(f"\n🔍 Query: {q}")
        docs = retrieve(q, k=3)
        for doc in docs:
            prato = doc.metadata.get("prato", "N/A")
            preco = doc.metadata.get("preco_str", "N/A")
            secao = doc.metadata.get("secao", "N/A")
            dieta = doc.metadata.get("dieta", "")
            dieta_str = f" [{dieta}]" if dieta else ""
            print(f"  • {prato}{dieta_str} | {secao} | {preco}")
