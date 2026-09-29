# Multi-Agent AI Systems — Lab Work (TP1 → TP3)

Hands-on labs from a **Multi-Agent Systems / Generative AI** course, going from prompt engineering
to Retrieval-Augmented Generation (RAG) and finally tool-using, agentic RAG with LangChain and LangGraph.

| Lab | Topic | Main tools |
|---|---|---|
| [TP1](TP1/) | Prompt engineering for multi-agent systems | OpenAI, Ollama, Groq, tiktoken, LangChain |
| [TP2](TP2/) | Retrieval-Augmented Generation over PDFs | ChromaDB, OpenAI embeddings, Streamlit |
| [TP3](TP3/) | Agentic RAG & tool-calling agents | LangChain agents, LangGraph, Tavily |

---

## Setup

Each lab is an independent [uv](https://docs.astral.sh/uv/) project (Python ≥ 3.13).

```bash
# 1. API keys — one .env at the repository root is picked up by every lab
cp .env.example .env        # then fill in OPENAI_API_KEY (and GROQ / TAVILY keys if needed)

# 2. Install a lab's dependencies
cd TP2
uv sync                     # or: pip install -e .
```

| Variable | Used by | Description |
|---|---|---|
| `OPENAI_API_KEY` | all | OpenAI API key (**required**) |
| `OPENAI_MODEL` | TP2, TP3 scripts | Chat model (default `gpt-4o`) |
| `OPENAI_EMBEDDING_MODEL` | TP2 app | Embedding model (default `text-embedding-3-small`) |
| `GROQ_API_KEY` | TP1 | Groq-hosted models |
| `TAVILY_API_KEY` | TP3 | Web-search tool |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` / `RAG_TOP_K` | TP2 app | Chunking & retrieval settings |

TP1 also uses a local [Ollama](https://ollama.com/) server (`ollama pull llama3.2`).

---

## TP1 — Prompt Engineering for Multi-Agent Systems

Notebook: `TP1 SMA- Prompt Engineerin For Multi Agent Systems.ipynb`

- Background: AI & distributed AI, reinforcement learning, prompt engineering
- Tokenization with **tiktoken**
- Prompting **OpenAI**, local **Ollama** and **Groq** models — via SDK and raw HTTP (Postman)
- Multimodal prompting: image generation and image description
- Use case — **aspect-based sentiment analysis**: define objectives & metrics, assemble data,
  design the prompt, evaluate prompts

Scratch notebooks: `test.ipynb`, `tp2.ipynb`.

## TP2 — Retrieval-Augmented Generation

Notebook: `RAGV2.ipynb` — applied to the **OCP 2023 annual financial report** (`pdfs/`)

- RAG building blocks: indexing and retrieval workflow
- Choosing an embedding model, chunking strategies
- Vector store with **ChromaDB**, similarity search
- Prompt design and a RAG answer function
- Evaluation: **groundedness** checking with an LLM judge

Streamlit app — upload any PDFs and chat with them:

```bash
cd TP2
streamlit run rag.py
```

## TP3 — Agentic RAG

Notebook: `sma.ipynb` — LangChain agents, dynamic model selection middleware
(basic vs. advanced model), multi-agent patterns.

Script: `Agentic_Rag.py` — an agent that combines

- a **retriever tool** over a small vector store of employee profiles,
- a mock **HR lookup** tool,
- a simulated **send-email** tool.

```bash
cd TP3
python Agentic_Rag.py       # runs a sample question
langgraph dev               # opens the agent in LangGraph Studio (graph "my_agent_rag")
```

---

## Repository structure

```
TP_multiai/
├── .env.example
├── TP1/   prompt-engineering notebooks
├── TP2/   RAGV2.ipynb, rag.py (Streamlit), pdfs/
└── TP3/   sma.ipynb, Agentic_Rag.py, langgraph.json
```
