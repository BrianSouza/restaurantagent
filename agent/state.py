"""
Definição dos TypedDicts que compõem o estado do agente Kisseki.

PedidoState  — dados coletados durante o fluxo de pedido.
AgentState   — estado completo do grafo LangGraph.
"""

from __future__ import annotations

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class PedidoState(TypedDict):
    """
    Dados do pedido em construção.

    Ciclo de vida da etapa:
        "itens" → "nome" → "endereco" → "confirmacao" → "finalizado"
    """

    itens: list[str]          # ex.: ["2x Yakisoba", "1x Hot Roll"]
    nome: Optional[str]       # nome completo do cliente
    endereco: Optional[str]   # endereço de entrega
    etapa: str                # etapa atual da coleta


class AgentState(TypedDict):
    """
    Estado global do grafo.

    - messages:     histórico completo da conversa; append automático via add_messages.
    - intencao:     intenção classificada ("cardapio" | "pedido" | "saudacao" | "fora_do_escopo").
    - contexto_rag: chunks do cardápio recuperados pelo retriever (preenchido apenas na rota cardápio).
    - pedido:       estado do pedido em andamento; None quando não há pedido ativo.
    """

    messages: Annotated[list, add_messages]
    intencao: str
    contexto_rag: str
    pedido: Optional[PedidoState]
