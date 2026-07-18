# ⚡ Apoorva's Digital Twin

An AI agent built for the **"Build Your Digital Twin with Tools and RAG"** challenge —
a conversational digital twin that answers questions about my background using
Retrieval-Augmented Generation (RAG), and helps with everyday tasks using tool calling.

**Live demo:** *(add your Gradio share link or Hugging Face Space link here once deployed)*

## What it does

- Answers questions about my education, work experience, and projects — grounded in
  real facts via RAG, not hallucinated
- Speaks in first person, as a natural conversational agent
- Goes beyond Q&A with three utility tools:
  - 🧮 **Calculator** — arithmetic and percentage calculations
  - 📅 **Deadline tracker** — days remaining until a target date
  - 🔎 **Web search** — live information beyond the knowledge base

## Tech stack

| Component | Choice |
|---|---|
| Agent framework | LangChain + LangGraph (`create_react_agent`) |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | Chroma |
| UI | Gradio (custom-themed) |

## Architecture

```
User query
   │
   ▼
ReAct Agent (LangGraph) ── decides which tool(s) to call
   │
   ├── get_background_info  →  Chroma retriever  →  knowledge base (resume + bio)
   ├── calculator            →  arithmetic evaluation
   ├── days_until_deadline    →  date math
   └── web_search             →  DuckDuckGo search
   │
   ▼
Response (first-person, grounded in retrieved facts)
```

## Running it locally

1. **Clone the repo**
   ```
   git clone https://github.com/apoorva14-unique/apoorva-digital-twin.git
   cd apoorva-digital-twin
   ```

2. **Create a virtual environment and install dependencies**
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   pip install -r requirements.txt
   ```

3. **Set your API key**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   Get a free key at https://console.groq.com/keys

4. **Run it**
   ```
   python apoorva_digital_twin.py
   ```
   This prints a local URL and a temporary public `.gradio.live` link.

## Notes

- The knowledge base is embedded directly in `apoorva_digital_twin.py` as plain text
  (see `KNOWLEDGE_TEXT`), so the whole project runs from a single file — no external
  resume PDF or database needed to get started.
- Embeddings run locally (no API key required for that part), so only one API key
  (Groq) is needed to run the whole project.