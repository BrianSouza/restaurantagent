"""
Factory de provedores de LLM e Embeddings.

Suporte:
    LLM          → OpenAI | Anthropic | Google Gemini | Ollama (local)
    Embeddings   → OpenAI | Google Gemini | Ollama (local) | HuggingFace (local)

Design:
- Imports dos SDKs são feitos de forma lazy (dentro de cada branch) para que
  o projeto funcione mesmo que apenas alguns pacotes estejam instalados.
- Erros de import produzem mensagens claras com a instrução de instalação.
- As chaves de API são extraídas via `.get_secret_value()` apenas no momento
  de criação do cliente, nunca armazenadas em variáveis intermediárias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


# ── Helpers internos ───────────────────────────────────────────────────────────

def _import_error(package: str, provider: str) -> ImportError:
    return ImportError(
        f"Pacote '{package}' não encontrado.\n"
        f"Instale com:  pip install {package}\n"
        f"(necessário para usar o provedor '{provider}')"
    )


# ── Factory de LLM ─────────────────────────────────────────────────────────────

def get_llm(settings: Settings | None = None) -> "BaseChatModel":
    """
    Retorna uma instância de LLM configurada para o provedor definido em
    LLM_PROVIDER.

    Args:
        settings: Instância de Settings. Se None, usa o singleton global.

    Returns:
        Instância de BaseChatModel (LangChain) pronta para invocar.

    Raises:
        ImportError:  Se o pacote do provedor não estiver instalado.
        ValueError:   Se o provedor não for reconhecido.
    """
    cfg = settings or get_settings()
    provider = cfg.llm_provider
    model = cfg.resolved_llm_model

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise _import_error("langchain-openai", provider)
        return ChatOpenAI(
            model=model,
            api_key=cfg.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise _import_error("langchain-anthropic", provider)
        return ChatAnthropic(
            model=model,
            api_key=cfg.anthropic_api_key.get_secret_value(),  # type: ignore[union-attr]
        )

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise _import_error("langchain-google-genai", provider)
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=cfg.google_api_key.get_secret_value(),  # type: ignore[union-attr]
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise _import_error("langchain-ollama", provider)
        return ChatOllama(
            model=model,
            base_url=cfg.ollama_base_url,
        )

    raise ValueError(
        f"Provedor de LLM desconhecido: '{provider}'. "
        "Valores aceitos: openai | anthropic | google | ollama"
    )


# ── Factory de Embeddings ──────────────────────────────────────────────────────

def get_embeddings(settings: Settings | None = None) -> "Embeddings":
    """
    Retorna uma instância de Embeddings configurada para o provedor definido
    em EMBEDDING_PROVIDER.

    Nota: Anthropic não possui API de embeddings. Se llm_provider=anthropic,
    configure embedding_provider para outro provedor (openai, google, ollama
    ou huggingface).

    Args:
        settings: Instância de Settings. Se None, usa o singleton global.

    Returns:
        Instância de Embeddings (LangChain) pronta para uso.

    Raises:
        ImportError:  Se o pacote do provedor não estiver instalado.
        ValueError:   Se o provedor não for reconhecido.
    """
    cfg = settings or get_settings()
    provider = cfg.embedding_provider
    model = cfg.resolved_embedding_model

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise _import_error("langchain-openai", provider)
        return OpenAIEmbeddings(
            model=model,
            api_key=cfg.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        )

    if provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise _import_error("langchain-google-genai", provider)
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=cfg.google_api_key.get_secret_value(),  # type: ignore[union-attr]
        )

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise _import_error("langchain-ollama", provider)
        return OllamaEmbeddings(
            model=model,
            base_url=cfg.ollama_base_url,
        )

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise _import_error("langchain-huggingface", provider)
        # sentence-transformers roda 100% offline após o primeiro download
        return HuggingFaceEmbeddings(model_name=model)

    raise ValueError(
        f"Provedor de embeddings desconhecido: '{provider}'. "
        "Valores aceitos: openai | google | ollama | huggingface"
    )


# ── Diagnóstico ────────────────────────────────────────────────────────────────

def describe_config(settings: Settings | None = None) -> str:
    """
    Retorna um resumo legível da configuração ativa.
    Nunca expõe valores de chaves de API — apenas indica se estão presentes.
    """
    cfg = settings or get_settings()

    def _key_status(key) -> str:
        return "✅ definida" if key else "❌ ausente"

    lines = [
        "=== Configuração RAG ===",
        f"LLM provider   : {cfg.llm_provider}",
        f"LLM model      : {cfg.resolved_llm_model}",
        f"Embed provider : {cfg.embedding_provider}",
        f"Embed model    : {cfg.resolved_embedding_model}",
        "--- Chaves de API ---",
        f"OPENAI_API_KEY    : {_key_status(cfg.openai_api_key)}",
        f"ANTHROPIC_API_KEY : {_key_status(cfg.anthropic_api_key)}",
        f"GOOGLE_API_KEY    : {_key_status(cfg.google_api_key)}",
        "--- ChromaDB ---",
        f"Persist dir    : {cfg.chroma_persist_dir}",
        f"Collection     : {cfg.chroma_collection}",
        "--- Retrieval ---",
        f"k              : {cfg.retriever_k}",
        f"search_type    : {cfg.retriever_search_type}",
    ]
    if cfg.llm_provider == "ollama" or cfg.embedding_provider == "ollama":
        lines.append(f"Ollama URL     : {cfg.ollama_base_url}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_config())
