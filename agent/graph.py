"""
Grafo LangGraph do agente Kisseki.

Topologia:
    [START]
      └─► classificador ─┬─► rag ──► resposta_cardapio ──► [END]
                         ├─► gerenciador_pedido ──────────► [END]
                         ├─► resposta_saudacao ───────────► [END]
                         └─► fora_do_escopo ─────────────► [END]

Persistência: SqliteSaver com "checkpoints.db" mantém histórico por thread_id.
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent.nodes import (
    node_classificador,
    node_fora_do_escopo,
    node_gerenciador_pedido,
    node_rag,
    node_resposta_cardapio,
    node_resposta_saudacao,
    rotear_por_intencao,
)
from agent.state import AgentState

# ── Persistência ───────────────────────────────────────────────────────────────
# check_same_thread=False permite que o mesmo objeto Connection seja usado em
# múltiplos threads, necessário quando o servidor recebe requests concorrentes.
_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(_conn)


# ── Construção do grafo ────────────────────────────────────────────────────────

def criar_graph():
    """
    Constrói e compila o StateGraph do agente Kisseki.

    Retorna o grafo compilado com checkpointer SqliteSaver pronto para uso.
    """
    builder = StateGraph(AgentState)

    # Registro de nós
    builder.add_node("classificador",      node_classificador)
    builder.add_node("rag",                node_rag)
    builder.add_node("resposta_cardapio",  node_resposta_cardapio)
    builder.add_node("resposta_saudacao",  node_resposta_saudacao)
    builder.add_node("fora_do_escopo",     node_fora_do_escopo)
    builder.add_node("gerenciador_pedido", node_gerenciador_pedido)

    # Entry point
    builder.set_entry_point("classificador")

    # Roteamento condicional a partir do classificador
    builder.add_conditional_edges(
        "classificador",
        rotear_por_intencao,
        {
            "cardapio":       "rag",
            "pedido":         "gerenciador_pedido",
            "saudacao":       "resposta_saudacao",
            "fora_do_escopo": "fora_do_escopo",
        },
    )

    # RAG alimenta sempre a resposta de cardápio
    builder.add_edge("rag", "resposta_cardapio")

    # Todos os nós de resposta terminam o turno
    builder.add_edge("resposta_cardapio",  END)
    builder.add_edge("resposta_saudacao",  END)
    builder.add_edge("fora_do_escopo",     END)
    builder.add_edge("gerenciador_pedido", END)

    return builder.compile(checkpointer=memory)


# ── Instância global ───────────────────────────────────────────────────────────
graph = criar_graph()
