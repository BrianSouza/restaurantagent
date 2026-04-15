# Kisseki - Agente Conversacional para Restaurante Japonês

Agente de IA conversacional para um restaurante japonês caseiro. O agente responde dúvidas sobre o cardápio, processa pedidos em múltiplas etapas e mantém o histórico de conversa por sessão de usuário.

## Funcionalidades

- **Consultas ao cardápio** - Busca híbrida (semântica + lexical) para responder perguntas sobre pratos, preços e ingredientes
- **Pedidos conversacionais** - Fluxo guiado em 5 etapas: itens → nome → endereço → confirmação → conclusão
- **Respostas contextuais** - Personalidade definida (Yuki, atendente do Kisseki)
- **Fora do escopo** - Rejeita educadamente perguntas não relacionadas ao restaurante
- **Memória por sessão** - Estado persistente por `thread_id` via SQLite

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Orquestração do agente | LangGraph + LangChain |
| LLM (classificação) | Claude Haiku (Anthropic) |
| LLM (respostas) | Claude Sonnet (Anthropic) |
| Banco vetorial | ChromaDB |
| Busca lexical | BM25 (rank-bm25) |
| Busca semântica | EnsembleRetriever (BM25 + vetores) |
| Persistência de estado | SQLite (SqliteSaver) |
| Configuração | Pydantic v2 + variáveis de ambiente |

Provedores de LLM e embeddings alternativos suportados: OpenAI, Google Gemini, Ollama (local), HuggingFace.

## Estrutura do Projeto

```
restaurantagent/
├── main.py                 # Ponto de entrada e API pública (conversar)
├── requirements.txt        # Dependências Python
├── .env.example            # Template de configuração
│
├── agent/
│   ├── graph.py            # Topologia do grafo LangGraph
│   ├── state.py            # Definição dos estados (AgentState, PedidoState)
│   ├── nodes.py            # Nós de processamento e roteamento
│   └── prompts.py          # Prompts do sistema (persona, classificação, pedidos)
│
├── rag/
│   ├── config.py           # Configurações via variáveis de ambiente
│   ├── providers.py        # Fábricas de LLM e embeddings
│   ├── retriever.py        # Recuperação híbrida (BM25 + vetorial)
│   └── ingestor.py         # Pipeline de ingestão do cardápio
│
└── data/
    ├── cardapio.md         # Cardápio em Markdown (fonte de dados)
    └── documents.json      # Documentos processados (gerado automaticamente)
```

## Instalação

**Pré-requisitos:** Python 3.7+

```bash
# 1. Clone ou navegue até o diretório do projeto
cd restaurantagent

# 2. Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

## Configuração

```bash
# Copie o arquivo de exemplo e edite com suas chaves de API
cp .env.example .env
```

Edite o `.env` de acordo com o provedor desejado:

**Opção 1 — Anthropic (padrão):**
```env
LLM_PROVIDER=anthropic
EMBEDDING_PROVIDER=openai
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

**Opção 2 — OpenAI:**
```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

**Opção 3 — 100% local com Ollama:**
```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

Consulte o `.env.example` para ver todas as opções disponíveis (modelos, ChromaDB, retriever, etc.).

## Inicialização (primeira execução)

Antes de usar o agente, é necessário construir o banco vetorial a partir do cardápio:

```bash
python -m rag.ingestor
```

Isso cria o diretório `chroma_db/` com os embeddings e o arquivo `data/documents.json`.

## Execução

**Testes interativos:**
```bash
python main.py
```

Executa 4 cenários de teste:
1. Consulta de preço ("Qual o preço do Yakisoba?")
2. Fluxo completo de pedido (5 turnos)
3. Pergunta fora do escopo (clima)
4. Memória entre turnos (combo mais barato/caro)

**Uso via código:**
```python
from main import conversar

# thread_id identifica a sessão do usuário
resposta = conversar("usuario-123", "Qual o preço do Yakisoba?")
print(resposta)

# A conversa é persistida — mensagens anteriores são lembradas
resposta = conversar("usuario-123", "E o Karaague?")
print(resposta)
```

## Arquitetura do Agente

```
Mensagem do usuário
      │
      ▼
 classificador  ──────────────────────────────┐
      │                                        │
      ├─ cardapio ──► rag ──► resposta_cardapio│
      ├─ pedido   ──► gerenciador_pedido       │
      ├─ saudacao ──► resposta_saudacao        │
      └─ fora_do_escopo ──► resposta fixa      │
                                               │
                                         Resposta final
```

- **Claude Haiku** — classificação de intenção (rápido e eficiente)
- **Claude Sonnet** — geração de respostas (qualidade e raciocínio)
- **EnsembleRetriever** — combina BM25 (palavras-chave exatas) + vetorial (semântica)
