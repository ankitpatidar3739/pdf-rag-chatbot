# 📄 DocuChat — RAG-Powered PDF Chatbot

A conversational AI app that lets you **chat with any PDF** using Retrieval-Augmented Generation (RAG). Upload your documents, ask questions in natural language, and get accurate answers with page-level source citations.

Built with **LangChain**, **Groq (LLaMA 3)**, **FAISS**, and **Streamlit**.

---

## 🎯 Why I Built This

I wanted to go beyond basic ML models and understand how production LLM apps actually work. This project forced me to figure out:
- Why you can't just dump a whole PDF into a prompt (context limits)
- How vector embeddings let you search by *meaning*, not just keywords
- How conversation history breaks naive RAG systems — and how to fix it

The result is something I'd actually use: paste in a research paper and ask it questions instead of reading the whole thing.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌──────────────────────────────┐
│  Condense Question Chain     │  ← rewrites follow-up Qs into standalone Qs
│  (Groq LLaMA 3 + history)   │    so retrieval works across multi-turn chat
└──────────────┬───────────────┘
               │ standalone question
               ▼
┌──────────────────────────────┐
│  FAISS Vector Store          │  ← cosine similarity search
│  (all-MiniLM-L6-v2 embeds)  │    returns top-k most relevant chunks
└──────────────┬───────────────┘
               │ relevant chunks
               ▼
┌──────────────────────────────┐
│  QA Chain                    │  ← custom prompt, grounded in context
│  (Groq LLaMA 3)              │    returns answer + source documents
└──────────────────────────────┘
```

**Key design decisions:**
- `RecursiveCharacterTextSplitter` with overlap so context isn't cut mid-sentence
- Separate "condense" LLM call handles multi-turn conversation properly
- `source_documents=True` on the chain so every answer cites specific pages
- HuggingFace `all-MiniLM-L6-v2` for embeddings — fast, local, no API cost

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/pdf-chatbot-rag.git
cd pdf-chatbot-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
```bash
cp .env.example .env
# Open .env and add your Groq API key
# Get one free at: https://console.groq.com
```

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| LLM | Groq (LLaMA 3.1 8B) | Free tier, fast inference |
| Embeddings | `all-MiniLM-L6-v2` | Local, no API calls needed |
| Vector DB | FAISS | Lightweight, no server required |
| RAG Framework | LangChain | `ConversationalRetrievalChain` handles history |
| PDF Parsing | PyPDF | Simple, handles most PDFs |
| Frontend | Streamlit | Fast to iterate, clean enough |

---

## ✨ Features

- **Multi-PDF upload** — process multiple documents at once; they're all indexed together
- **Conversational memory** — follow-up questions work ("tell me more about that")
- **Source citations** — every answer links back to specific pages + text snippets
- **Configurable** — adjust chunk size, number of retrieved sources, and model from the UI
- **Clean UI** — custom CSS, no default Streamlit look

---

## 🛠️ How the RAG Pipeline Works

**Step 1: Ingestion**

Each PDF page is extracted as text and split into overlapping chunks (~600 chars by default). Each chunk is embedded into a 384-dimensional vector and stored in FAISS.

**Step 2: Retrieval**

When you ask a question, the question is embedded with the same model and compared against all chunk vectors using cosine similarity. The top-k most similar chunks are fetched.

**Step 3: Generation**

The retrieved chunks are injected into a prompt template alongside the question. The LLM reads the context and generates an answer grounded in your documents.

**Step 4: Conversation handling**

Multi-turn chat is tricky — "Who wrote that?" doesn't make sense without knowing what "that" is. A separate LLM call condenses each follow-up question + chat history into a standalone question before retrieval. This keeps the retriever working correctly across turns.

---

## ⚠️ Known Limitations

- Scanned PDFs (images) won't work — need OCR first (add `pytesseract` for that)
- Very large PDFs (200+ pages) will be slow to process on first load
- FAISS index is in-memory — it resets on page refresh (add persistence with `faiss.write_index` for production)
- Context window limits on 8B models mean very long answers can get cut off

---

## 🔮 Possible Extensions

- [ ] Persistent vector store (save/load FAISS index to disk)
- [ ] OCR support for scanned documents
- [ ] Switch to Chroma or Pinecone for scalable vector storage
- [ ] Add LangSmith tracing for debugging retrieval quality
- [ ] Docker deployment

---

## 📁 Project Structure

```
pdf-chatbot-rag/
├── app.py                  # Streamlit frontend + session state management
├── src/
│   ├── pdf_processor.py    # PDF loading, chunking, FAISS index building
│   ├── rag_chain.py        # LangChain RAG pipeline (prompts + chain setup)
│   └── utils.py            # Source formatting, helpers
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🤝 Contributing

Open to PRs — especially for OCR support or better chunking strategies.

---

## 📄 License

MIT
