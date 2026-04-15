"""
Configuração centralizada do RAG — lida exclusivamente de variáveis de ambiente.

Design de segurança:
- Todas as chaves de API são armazenadas como `SecretStr`: o valor nunca
  aparece em repr(), logs ou tracebacks acidentais.
- `model_validate_env()` levanta erros claros quando um provedor está
  selecionado mas a chave correspondente está ausente.
- Nenhum valor sensível é hardcoded aqui; defaults cobrem apenas nomes de
  modelos públicos e URLs locais.
- O arquivo `.env` nunca deve ser commitado; use `.env.example` como template.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Tipos ──────────────────────────────────────────────────────────────────────

LLMProvider = Literal["openai", "anthropic", "google", "ollama"]
EmbeddingProvider = Literal["openai", "google", "ollama", "huggingface"]

# Modelos padrão por provedor (usados quando o usuário não define LLM_MODEL /
# EMBEDDING_MODEL explicitamente — fácil de sobrescrever via .env)
_LLM_DEFAULTS: dict[str, str] = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "google":    "gemini-1.5-flash",
    "ollama":    "llama3.2",
}

_EMBEDDING_DEFAULTS: dict[str, str] = {
    "openai":       "text-embedding-3-small",
    "google":       "models/text-embedding-004",
    "ollama":       "nomic-embed-text",
    "huggingface":  "sentence-transformers/all-MiniLM-L6-v2",
}


# ── Settings ───────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Todas as configurações do projeto.
    Lidas automaticamente do arquivo `.env` e de variáveis de ambiente do sistema.
    Variáveis de ambiente têm precedência sobre o `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignora vars desconhecidas sem explodir
        case_sensitive=False,    # LLM_PROVIDER = llm_provider = ok
    )

    # ── Provedor de LLM ────────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default="openai",
        description="Provedor do modelo de linguagem.",
    )
    llm_model: str = Field(
        default="",
        description=(
            "Nome do modelo LLM. Se vazio, usa o padrão do provedor selecionado."
        ),
    )

    # ── Provedor de Embeddings ─────────────────────────────────────────────────
    # Nota: Anthropic não oferece API de embeddings — use outro provedor para
    # embeddings quando llm_provider=anthropic.
    embedding_provider: EmbeddingProvider = Field(
        default="openai",
        description="Provedor de embeddings (pode ser diferente do LLM).",
    )
    embedding_model: str = Field(
        default="",
        description="Modelo de embeddings. Se vazio, usa o padrão do provedor.",
    )

    # ── Chaves de API (SecretStr — nunca aparecem em logs) ─────────────────────
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="Chave da API OpenAI. Obrigatória se llm_provider=openai ou embedding_provider=openai.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Chave da API Anthropic. Obrigatória se llm_provider=anthropic.",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        description="Chave da API Google. Obrigatória se llm_provider=google ou embedding_provider=google.",
    )

    # ── Ollama (execução local) ────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL base do servidor Ollama local.",
    )

    # ── HuggingFace (embeddings locais) ───────────────────────────────────────
    hf_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description=(
            "Modelo HuggingFace para embeddings locais. "
            "Usado quando embedding_provider=huggingface."
        ),
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Diretório local para persistir o índice vetorial ChromaDB.",
    )
    chroma_collection: str = Field(
        default="cardapio_kisseki",
        description="Nome da coleção dentro do ChromaDB.",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de documentos a retornar por consulta.",
    )
    retriever_search_type: Literal["mmr", "similarity"] = Field(
        default="mmr",
        description=(
            "Estratégia de busca: 'mmr' (diversidade) ou 'similarity' (máx. relevância)."
        ),
    )

    # ── Propriedades derivadas ────────────────────────────────────────────────

    @property
    def resolved_llm_model(self) -> str:
        """Retorna o modelo LLM efetivo (explícito ou default do provedor)."""
        return self.llm_model or _LLM_DEFAULTS[self.llm_provider]

    @property
    def resolved_embedding_model(self) -> str:
        """Retorna o modelo de embedding efetivo."""
        if self.embedding_provider == "huggingface":
            return self.hf_embedding_model
        return self.embedding_model or _EMBEDDING_DEFAULTS[self.embedding_provider]

    # ── Validação de segurança ────────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_required_keys(self) -> "Settings":
        """
        Garante que a chave de API necessária está presente para o provedor
        configurado. Falha em tempo de inicialização, não durante a primeira
        chamada à API.
        """
        missing: list[str] = []

        if self.llm_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY (necessária para LLM_PROVIDER=openai)")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY (necessária para LLM_PROVIDER=anthropic)")
        if self.llm_provider == "google" and not self.google_api_key:
            missing.append("GOOGLE_API_KEY (necessária para LLM_PROVIDER=google)")

        if self.embedding_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY (necessária para EMBEDDING_PROVIDER=openai)")
        if self.embedding_provider == "google" and not self.google_api_key:
            missing.append("GOOGLE_API_KEY (necessária para EMBEDDING_PROVIDER=google)")

        if missing:
            raise ValueError(
                "Variáveis de ambiente obrigatórias não encontradas:\n"
                + "\n".join(f"  • {m}" for m in missing)
                + "\n\nCopie .env.example para .env e preencha os valores."
            )

        return self


# ── Singleton ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna a instância singleton de Settings.
    O cache garante que o arquivo .env é lido apenas uma vez por processo.

    Use `get_settings.cache_clear()` em testes para recarregar as configurações.
    """
    return Settings()
