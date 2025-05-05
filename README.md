# 🤖 Slack Research Analyst Bot

A cutting-edge Slack-integrated research assistant powered by **LangGraph**, **LangChain**, **FAISS**, and **Gemini**.  
It retrieves, summarizes, and reasons over academic research (via ArXiv) and web data — with persistent memory, personalization, and contextual awareness.

---

## 🚀 Features

- **Multi-turn conversation** with contextual understanding
- **Intent detection**: understands user queries in context and rewrites follow-ups
- **Academic paper search** via ArXiv API (to be extended, semantic scholar integration is underway)
- **Web fallback search** using DuckDuckGo (in case papers aren't available)
- **RAG (Retrieval-Augmented Generation)** over academic papers
- **Implementation suggestions** (code ideas, experiments, scaffolds)
- **Persistent memory** using `ConversationSummaryBufferMemory`
- **User preferences** tracking (interests, focus, etc.)
- **Dynamic contextual profiles**: tracks intent patterns, keywords, and more
- **Slack integration** with real-time feedback and formatting
- **Low-cost and high-speed** with `gemini-1.5-flash` and lightweight embeddings

# Instructions to run
Create a new venv, and install the project dependencies using 'poetry'.
Run 'main.py' with uvicorn.
Create a reverse proxy using ngrok, exposing the port to the web.
Configure the Slack app's event subscription to the created URL, type away in slack!
