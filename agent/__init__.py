"""
Agent Core do Kisseki — implementado com LangGraph.

Exports principais:
    graph        — instância compilada do StateGraph (com SqliteSaver)
    conversar()  — utilitário de alto nível para invocar o agente
"""

from agent.graph import graph

__all__ = ["graph"]
