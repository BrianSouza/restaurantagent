"""
Ponto de entrada e script de testes do agente Kisseki.

Uso rápido:
    python main.py

A função conversar() pode ser importada por qualquer integração (WhatsApp, API REST, etc.):
    from main import conversar
    resposta = conversar("thread-123", "Qual o preço do Yakisoba?")
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.graph import graph


def conversar(thread_id: str, mensagem: str) -> str:
    """
    Envia uma mensagem ao agente e retorna a resposta como string.

    Args:
        thread_id: Identificador da conversa (mantém histórico por sessão).
        mensagem:  Texto da mensagem do usuário.

    Returns:
        Conteúdo da última mensagem gerada pelo agente.
    """
    result = graph.invoke(
        {"messages": [HumanMessage(content=mensagem)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


# ── Testes ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "=" * 60

    # ── Teste 1: cardápio ──────────────────────────────────────────────────────
    print(sep)
    print("TESTE 1 — Consulta de cardápio")
    print(sep)
    pergunta = "Oi! Qual o preço do Yakisoba?"
    print(f"Usuário: {pergunta}")
    resposta = conversar("t1", pergunta)
    print(f"Yuki:    {resposta}")
    print()

    # ── Teste 2: fluxo de pedido completo (5 turnos) ───────────────────────────
    print(sep)
    print("TESTE 2 — Fluxo de pedido completo (5 turnos)")
    print(sep)
    turnos = [
        "Quero fazer um pedido",
        "2 Yakisobas e 1 Hot Roll",
        "Maria Souza",
        "Rua das Flores, 123, apto 42",
        "Sim, pode confirmar!",
    ]
    for i, msg in enumerate(turnos, 1):
        print(f"[{i}] Usuário: {msg}")
        r = conversar("t2", msg)
        print(f"     Yuki:    {r}")
        print()

    # ── Teste 3: fora do escopo ────────────────────────────────────────────────
    print(sep)
    print("TESTE 3 — Fora do escopo")
    print(sep)
    pergunta = "Você sabe a previsão do tempo pra amanhã?"
    print(f"Usuário: {pergunta}")
    resposta = conversar("t3", pergunta)
    print(f"Yuki:    {resposta}")
    print()

    # ── Teste 4: memória entre turnos ─────────────────────────────────────────
    print(sep)
    print("TESTE 4 — Memória entre turnos")
    print(sep)
    p1 = "Qual o combo mais barato?"
    print(f"[1] Usuário: {p1}")
    r1 = conversar("t4", p1)
    print(f"     Yuki:    {r1}")
    print()

    p2 = "E o mais caro?"
    print(f"[2] Usuário: {p2}")
    r2 = conversar("t4", p2)
    print(f"     Yuki:    {r2}")
    print()
