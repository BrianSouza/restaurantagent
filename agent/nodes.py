"""
Nós do grafo LangGraph do agente Kisseki.

Cada nó recebe AgentState e retorna um dict com os campos a atualizar.

Nós públicos:
    node_classificador       — classifica intenção (haiku, max_tokens=20)
    node_rag                 — recupera chunks do cardápio via busca híbrida
    node_resposta_cardapio   — responde dúvidas de cardápio (sonnet, max_tokens=600)
    node_resposta_saudacao   — responde saudações (sonnet, max_tokens=300)
    node_fora_do_escopo      — mensagem hardcoded, sem LLM
    node_gerenciador_pedido  — conduz o fluxo de coleta de pedido (sonnet, max_tokens=400)

Auxiliares:
    _extrair_itens(texto)    — extrai lista de itens de um texto livre (haiku)
    rotear_por_intencao      — função de roteamento para conditional_edges
"""

from __future__ import annotations

import ast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage

from agent.prompts import SYSTEM_CLASSIFICADOR, SYSTEM_PEDIDO, SYSTEM_PRINCIPAL
from agent.state import AgentState, PedidoState
from rag.config import get_settings
from rag.retriever import retrieve

# ── Constantes ─────────────────────────────────────────────────────────────────

_INTENCOES_VALIDAS = {"cardapio", "pedido", "saudacao", "fora_do_escopo"}

_PALAVRAS_CONFIRMACAO = {"sim", "confirmo", "pode", "ok", "isso", "certo", "s"}


# ── Utilitários internos ───────────────────────────────────────────────────────

def _get_anthropic_key() -> str:
    """
    Lê ANTHROPIC_API_KEY do .env via Settings.

    Levanta ValueError com mensagem clara se a chave não estiver configurada.
    O agente sempre usa modelos Anthropic independentemente do LLM_PROVIDER do RAG.
    """
    settings = get_settings()
    if settings.anthropic_api_key:
        return settings.anthropic_api_key.get_secret_value()
    raise ValueError(
        "ANTHROPIC_API_KEY não encontrada.\n"
        "O agente Kisseki requer a chave da API Anthropic.\n"
        "Adicione ANTHROPIC_API_KEY=sk-ant-... ao seu arquivo .env"
    )


def _ultima_mensagem_humana(state: AgentState) -> str:
    """Retorna o conteúdo da última mensagem não-AI no histórico."""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, AIMessage):
            content = msg.content if hasattr(msg, "content") else str(msg)
            return content
    return ""


def _llm_haiku(api_key: str, max_tokens: int = 200) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        api_key=api_key,
    )


def _llm_sonnet(api_key: str, max_tokens: int = 600) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        api_key=api_key,
    )


# ── Nós do grafo ───────────────────────────────────────────────────────────────

def node_classificador(state: AgentState) -> dict:
    """
    Classifica a intenção da última mensagem do usuário.

    Atalho: se há pedido em andamento (etapa != "finalizado"), retorna
    {"intencao": "pedido"} imediatamente, sem chamar o LLM.
    """
    pedido = state.get("pedido")
    if pedido and pedido.get("etapa") != "finalizado":
        return {"intencao": "pedido"}

    api_key = _get_anthropic_key()
    llm = _llm_haiku(api_key, max_tokens=20)

    ultima = _ultima_mensagem_humana(state)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_CLASSIFICADOR),
        {"role": "user", "content": ultima},
    ])

    intencao = response.content.strip().lower().rstrip(".").strip()
    if intencao not in _INTENCOES_VALIDAS:
        intencao = "cardapio"

    return {"intencao": intencao}


def node_rag(state: AgentState) -> dict:
    """
    Recupera os 4 chunks mais relevantes do cardápio via busca híbrida (BM25 + vetorial).
    """
    ultima = _ultima_mensagem_humana(state)
    docs = retrieve(ultima, k=4)
    contexto = "\n\n".join(doc.page_content for doc in docs)
    return {"contexto_rag": contexto}


def node_resposta_cardapio(state: AgentState) -> dict:
    """
    Responde perguntas sobre o cardápio usando o contexto RAG recuperado.
    O LLM recebe instrução explícita para usar APENAS as informações fornecidas.
    """
    api_key = _get_anthropic_key()
    llm = _llm_sonnet(api_key, max_tokens=600)

    contexto = state.get("contexto_rag") or ""
    system = (
        f"{SYSTEM_PRINCIPAL}\n\n"
        "─── Informações do cardápio ───\n"
        "Use APENAS as informações abaixo para responder. "
        "Nunca invente pratos, preços ou ingredientes que não estejam aqui.\n\n"
        f"{contexto}"
    )

    messages = [SystemMessage(content=system)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


def node_resposta_saudacao(state: AgentState) -> dict:
    """
    Responde saudações, agradecimentos e despedidas com a persona da Yuki.
    Não usa contexto RAG — o LLM usa apenas a persona e o histórico.
    """
    api_key = _get_anthropic_key()
    llm = _llm_sonnet(api_key, max_tokens=300)

    messages = [SystemMessage(content=SYSTEM_PRINCIPAL)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


def node_fora_do_escopo(state: AgentState) -> dict:
    """
    Resposta hardcoded para mensagens fora do escopo da Kisseki.
    Sem chamada ao LLM — resposta instantânea e consistente.
    """
    return {
        "messages": [AIMessage(
            content=(
                "Hmm, esse assunto foge um pouco do meu alcance por aqui 😅 "
                "Sou especialista só no universo Kisseki!\n\n"
                "Mas posso te ajudar com cardápio, preços, ingredientes ou montar um pedido. "
                "O que você gostaria de saber? 🍱"
            )
        )]
    }


def _extrair_itens(texto: str) -> list[str]:
    """
    Extrai itens de pedido de um texto livre usando o LLM.

    Retorna lista no formato ["2x Yakisoba", "1x Hot Roll"].
    Retorna [] se não encontrar itens claros ou em caso de erro de parsing.
    """
    api_key = _get_anthropic_key()
    llm = _llm_haiku(api_key, max_tokens=200)

    prompt = (
        "Extraia os itens de pedido do texto abaixo e retorne APENAS uma lista Python.\n"
        "Formato esperado: [\"2x Yakisoba\", \"1x Hot Roll\"]\n"
        "Inclua quantidade e nome. Se não houver itens de pedido claros, retorne [].\n"
        "Retorne SOMENTE a lista Python, sem explicações.\n\n"
        f"Texto: {texto}"
    )

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        # Isola apenas a lista Python da resposta
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        return ast.literal_eval(content[start:end])
    except Exception:
        return []


def node_gerenciador_pedido(state: AgentState) -> dict:
    """
    Gerencia o fluxo de coleta de pedido em 5 etapas:
    itens → nome → endereco → confirmacao → finalizado.

    Inicializa o pedido na primeira vez. Avança de etapa conforme os dados são
    coletados e delega ao LLM a geração da resposta para cada etapa.
    """
    api_key = _get_anthropic_key()
    llm = _llm_sonnet(api_key, max_tokens=400)

    ultima = _ultima_mensagem_humana(state)

    # Inicializa ou copia o pedido existente
    pedido_atual = state.get("pedido")
    pedido: PedidoState = dict(pedido_atual) if pedido_atual else {
        "itens": [],
        "nome": None,
        "endereco": None,
        "etapa": "itens",
    }

    etapa = pedido.get("etapa", "itens")

    # ── Lógica de avanço de etapa ──────────────────────────────────────────────
    if etapa == "itens":
        itens = _extrair_itens(ultima)
        if itens:
            pedido["itens"] = itens
            pedido["etapa"] = "nome"

    elif etapa == "nome":
        nome = ultima.strip()
        if len(nome) > 2:
            pedido["nome"] = nome
            pedido["etapa"] = "endereco"

    elif etapa == "endereco":
        endereco = ultima.strip()
        if len(endereco) > 5:
            pedido["endereco"] = endereco
            pedido["etapa"] = "confirmacao"

    elif etapa == "confirmacao":
        palavras_msg = set(ultima.lower().split())
        if palavras_msg & _PALAVRAS_CONFIRMACAO:
            pedido["etapa"] = "finalizado"

    # ── Monta system prompt com estado atual ───────────────────────────────────
    system = SYSTEM_PEDIDO.format(
        etapa=pedido.get("etapa", "itens"),
        itens=", ".join(pedido.get("itens") or []) or "nenhum ainda",
        nome=pedido.get("nome") or "não informado",
        endereco=pedido.get("endereco") or "não informado",
    )

    messages = [SystemMessage(content=system)] + list(state["messages"])
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "pedido": pedido,
    }


# ── Função de roteamento ───────────────────────────────────────────────────────

def rotear_por_intencao(state: AgentState) -> str:
    """Retorna a intenção classificada para o conditional_edges do LangGraph."""
    return state["intencao"]
