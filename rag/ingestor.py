"""
Ingestor do cardápio Kisseki.

Melhorias de qualidade de retrieval:
- Preamble semântico: cada chunk começa com um parágrafo em linguagem natural
  descrevendo o prato. Isso resolve casos onde a query usa sinônimos ou conceitos
  ("vegetariano", "para dividir", "barato") que não aparecem literalmente no MD.
- Tags de dieta: metadata `dieta` com valores como ["vegetariano", "vegano"]
  para permitir filtragem direta quando necessário.
- Persistência em JSON: salva os docs processados em data/documents.json para
  alimentar o BM25Retriever sem precisar re-embedar.
- Provedor de embeddings configurável via .env — sem hardcode.
"""

import json
import os
import re
import shutil

# Desabilita telemetria do ChromaDB antes de qualquer import do chromadb
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from rag.config import Settings, get_settings
from rag.providers import get_embeddings

# ── Caminhos ───────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKDOWN_PATH = os.path.join(BASE_DIR, "data", "cardapio.md")
DOCS_JSON     = os.path.join(BASE_DIR, "data", "documents.json")  # para BM25


# ── Parsing de campos estruturados ────────────────────────────────────────────

def _parse_price(text: str) -> float | None:
    match = re.search(r"R\$\s*([\d]+[,.][\d]+)", text)
    return float(match.group(1).replace(",", ".")) if match else None


def _parse_weight(text: str) -> str | None:
    match = re.search(r"(\d+\s*g)\b", text)
    return match.group(1).strip() if match else None


def _parse_field(text: str, field: str) -> str:
    """Extrai valor de um campo no formato **Campo:** valor"""
    match = re.search(rf"\*\*{field}:\*\*\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _is_summary_table(text: str) -> bool:
    return text.strip().startswith("|") or "Resumo de todos" in text


# ── Tags de dieta ─────────────────────────────────────────────────────────────

# Palavras/substrings que indicam proteína animal — tornam o prato NÃO vegetariano.
# Inclui nomes de pratos que contêm carne/frango (ex.: "karaague" = frango frito,
# "nikuman" = carne suína) para cobrir combos/kits que listam pratos pelo nome.
_PROTEINAS_ANIMAIS = [
    # ingredientes diretos
    "frango", "carne", "salmão", "peixe", "pernil", "bacon",
    "suína", "porco", "atum", "camarão",
    # nomes de pratos que contêm carne/frango (relevante para combos e kits)
    "karaague", "yakisoba", "chicken", "katsu", "teriyaki",
    "nikuman", "gyoza", "nishime", "chahan", "hot roll",
]

# Palavras/substrings que indicam sobremesa/doce — esses pratos NÃO recebem
# tags de dieta mesmo sendo tecnicamente vegetarianos/veganos.
# Motivo: uma query "opções vegetarianas" deve retornar pratos salgados, não doces.
# O agente pode responder sobre sobremesas quando perguntado diretamente.
_SOBREMESAS_KEYWORDS = [
    # tipos de massa/preparo típicos de doces
    "choux", "cream puff", "éclair",
    "confeiteiro", "chantilly", "merengue",
    # sobremesas japonesas
    "mochi", "daifuku", "dorayaki", "taiyaki", "anko", "wagashi",
    # termos genéricos de doce
    "pudim", "mousse", "sorvete", "gelato",
    "calda de açúcar", "cobertura de chocolate",
    # palavras que indicam categoria
    "sobremesa", "doce",
]


def _is_dessert(prato: str, content: str) -> bool:
    """
    Detecta se o prato é uma sobremesa/doce.

    Sobremesas não recebem tags de dieta (vegetariano/vegano) mesmo que não
    contenham proteínas animais — elas pertencem a uma categoria própria e
    não devem aparecer em queries como "opções vegetarianas".
    """
    texto = (prato + " " + content).lower()
    return any(kw in texto for kw in _SOBREMESAS_KEYWORDS)


def _extract_dietary_tags(prato: str, content: str) -> list[str]:
    """
    Detecta se o prato é vegetariano/vegano com base no nome e ingredientes.

    Sobremesas são excluídas automaticamente — mesmo sendo vegetarianas,
    não devem aparecer em resultados de busca por dieta.

    Para combos e kits, verifica os nomes dos pratos na composição —
    ex.: "Karaague" na composição do Combo Hokkaido indica proteína animal.

    Retorna lista de tags, ex.: ["vegetariano", "vegano"] ou [].
    """
    # Sobremesas não recebem tags de dieta — têm categoria própria
    if _is_dessert(prato, content):
        return []

    texto = (prato + " " + content).lower()
    tags: list[str] = []

    tem_proteina_animal = any(p in texto for p in _PROTEINAS_ANIMAIS)

    if "vegano" in texto or "vegana" in texto:
        tags.extend(["vegano", "vegetariano"])
    elif not tem_proteina_animal:
        tags.append("vegetariano")

    return tags


# Faixas de preço — usadas para enriquecer o preamble com vocabulário de preço
_PRECO_BARATO  = 12.0   # até este valor → "econômico", "barato", "mais em conta"
_PRECO_MEDIO   = 22.0   # até este valor → "preço médio"
# acima → "premium", "generoso"


# ── Preamble semântico ─────────────────────────────────────────────────────────

def _build_preamble(metadata: dict, content: str) -> str:
    """
    Gera um parágrafo em linguagem natural que precede o chunk Markdown.

    Inclui sinônimos, variações morfológicas e contexto semântico para que
    a busca — tanto BM25 quanto vetorial — encontre o chunk por múltiplos
    caminhos de linguagem:
    - "vegetarianas/os" → Berinjela Vegana, Kimpirá Gobo, Bifum
    - "barato/econômico/em conta" → Nikuman, Chahan, Bifum
    - "yakisoba" → Combo Kyoto (via composição)
    - "dividir/compartilhar/grupo" → seção "Para Compartilhar"
    """
    prato      = metadata.get("prato", "")
    secao      = metadata.get("secao", "")
    preco      = metadata.get("preco")
    peso       = metadata.get("peso", "")
    descricao  = _parse_field(content, "Descrição")
    composicao = _parse_field(content, "Composição")
    dieta_tags = _extract_dietary_tags(prato, content)

    linhas: list[str] = []

    # ── Linha 1: identidade completa ──────────────────────────────────────────
    preco_str = f"R$ {preco:.2f}".replace(".", ",") if preco else ""
    partes    = [p for p in [prato, secao, preco_str, peso] if p]
    linhas.append("Prato: " + " | ".join(partes))

    # ── Linha 2: composição ou descrição ─────────────────────────────────────
    if composicao:
        linhas.append(f"Contém: {composicao}")
        ingredientes = [i.strip().split("(")[0].strip() for i in composicao.split("+") if i.strip()]
        # IMPORTANTE: usa vocabulário correto por seção.
        # "Ingredientes do combo" só aparece em Combos — Kits usam "Itens do kit".
        # Isso evita que Kits (ex.: Kit Favoritos) casem BM25 para queries "combo com X".
        if "Combo" in secao:
            linhas.append("Ingredientes do combo: " + ", ".join(ingredientes) + ".")
        elif "Kit" in secao:
            linhas.append("Itens do kit: " + ", ".join(ingredientes) + ".")
    if descricao:
        linhas.append(f"Descrição: {descricao}")

    # ── Linha 3a: sobremesa — contexto próprio, sem vocabulário de dieta ────────
    # Sobremesas recebem vocabulário de "doce" para serem encontradas em queries
    # como "tem sobremesa?", "algum doce?", "opção de doce?".
    # NÃO recebem tags vegetariano/vegano para não aparecerem em queries de dieta.
    if _is_dessert(prato, content):
        linhas.append(
            "Tipo: sobremesa, doce, opção de doce, opção de sobremesa, "
            "para adoçar, para finalizar a refeição, docinho."
        )
    # ── Linha 3b: dieta — múltiplas formas para cobrir variações morfológicas ──
    # BM25 não faz stemming; precisamos de "vegetariano", "vegetariana",
    # "vegetarianos", "vegetarianas" para cobrir todas as queries em pt-BR.
    # Sobremesas são excluídas via _extract_dietary_tags (retorna []).
    elif dieta_tags:
        dieta_formas: list[str] = []
        if "vegano" in dieta_tags:
            dieta_formas += [
                "opção vegana", "opção vegano", "prato vegano", "prato vegana",
                "sem carne", "sem frango", "sem peixe", "sem produto animal",
                "vegetariano", "vegetariana", "vegetarianos", "vegetarianas",
            ]
        elif "vegetariano" in dieta_tags:
            dieta_formas += [
                "opção vegetariana", "opção vegetariano",
                "prato vegetariano", "prato vegetariana",
                "vegetarianos", "vegetarianas",
                "sem carne", "sem frango", "sem peixe",
            ]
        linhas.append("Dieta: " + ", ".join(dieta_formas) + ".")

    # ── Linha 4: faixa de preço em linguagem natural ──────────────────────────
    if preco is not None:
        if preco <= _PRECO_BARATO:
            linhas.append(
                f"Preço: {preco_str}. Opção econômica, barata, mais em conta, "
                "acessível, custo baixo."
            )
        elif preco <= _PRECO_MEDIO:
            linhas.append(f"Preço: {preco_str}. Preço intermediário.")
        else:
            linhas.append(f"Preço: {preco_str}.")

    # ── Linha 5: contexto de uso por seção ───────────────────────────────────
    if "Compartilhar" in secao:
        linhas.append(
            "Ideal para compartilhar com amigos, família ou grupos. "
            "Porção para dividir, para duas ou mais pessoas."
        )
    elif "Combo" in secao:
        linhas.append(
            "Opção de combo: refeição completa combinando múltiplos pratos "
            "em um único pedido. Prático e variado."
        )
    elif "Kit" in secao:
        linhas.append(
            "Kit com múltiplas refeições para grupos, festas ou para a semana. "
            "Variedade em um único pedido."
        )
    elif "Individual" in secao:
        linhas.append("Porção individual para uma pessoa.")

    return "\n".join(linhas)


# ── Ingestão principal ─────────────────────────────────────────────────────────

def ingest(force: bool = False, settings: Settings | None = None) -> Chroma:
    """
    Lê o cardápio .md, enriquece os chunks com preamble semântico e metadados,
    persiste no ChromaDB e salva os docs em JSON (para BM25 híbrido).

    Args:
        force:    Se True, apaga e recria o índice.
        settings: Instância de Settings. Se None, usa o singleton global.
    """
    cfg         = settings or get_settings()
    persist_dir = os.path.abspath(cfg.chroma_persist_dir)
    collection  = cfg.chroma_collection
    embeddings  = get_embeddings(cfg)

    if not force and os.path.exists(persist_dir):
        print("📦 ChromaDB já existe. Carregando coleção existente...")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection,
        )

    # Remove índice antigo para garantir consistência
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"🗑️  Índice antigo removido: {persist_dir}")

    print(f"📄 Lendo: {MARKDOWN_PATH}")
    with open(MARKDOWN_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # ── 1. Chunking semântico por cabeçalho Markdown ──────────────────────────
    headers_to_split_on = [
        ("#",  "titulo"),
        ("##", "secao"),
        ("###","prato"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    raw_chunks = splitter.split_text(content)
    dish_chunks = [
        c for c in raw_chunks
        if c.metadata.get("prato") and not _is_summary_table(c.page_content)
    ]

    # ── 2. Enriquecimento de metadados ─────────────────────────────────────────
    for chunk in dish_chunks:
        text  = chunk.page_content
        price = _parse_price(text)
        weight = _parse_weight(text)

        if price is not None:
            chunk.metadata["preco"]     = price
            chunk.metadata["preco_str"] = f"R$ {price:.2f}".replace(".", ",")
        if weight:
            chunk.metadata["peso"] = weight

        dieta = _extract_dietary_tags(chunk.metadata.get("prato", ""), text)
        if dieta:
            chunk.metadata["dieta"] = ", ".join(dieta)  # ChromaDB só aceita str em metadata

        # Marca sobremesas com tipo próprio para facilitar filtragem futura
        if _is_dessert(chunk.metadata.get("prato", ""), text):
            chunk.metadata["tipo"] = "sobremesa"

        chunk.metadata["source"]      = "cardapio.md"
        chunk.metadata["restaurante"] = "Kisseki"

    # ── 3. Substituição do page_content por preamble + Markdown original ───────
    # O preamble em linguagem natural melhora drasticamente o recall semântico.
    # O Markdown original é mantido após "---" para preservar os dados completos.
    enriched: list[Document] = []
    for chunk in dish_chunks:
        preamble = _build_preamble(chunk.metadata, chunk.page_content)
        new_content = f"{preamble}\n---\n{chunk.page_content}"
        enriched.append(Document(page_content=new_content, metadata=chunk.metadata))

    print(f"✅ {len(enriched)} pratos enriquecidos com preamble semântico.")

    # ── 4. Persiste em JSON para uso no BM25Retriever (busca híbrida) ──────────
    docs_json_data = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in enriched
    ]
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(docs_json_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Docs salvos em JSON: {DOCS_JSON}")

    # ── 5. Cria e persiste o índice vetorial ────────────────────────────────────
    print(f"   Embedding: {cfg.embedding_provider} / {cfg.resolved_embedding_model}")
    vectorstore = Chroma.from_documents(
        documents=enriched,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection,
    )
    print(f"💾 Índice vetorial salvo em: {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    ingest(force=True)
