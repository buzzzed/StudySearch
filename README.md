# StudySearch v2

An AI-powered study assistant that lets students upload lecture slides, PDFs, and notes — then ask questions and get cited answers drawn directly from their material.

## Features

- **Semantic search** — n-gram hashing vectors with cosine similarity (swap for Voyage/OpenAI embeddings for production)
- **Multi-turn chat memory** — full conversation history sent to Claude on every turn; ask follow-ups naturally
- **Persistent sessions** — sessions stored in SQLite, survive page refresh via localStorage
- **Multi-format ingestion** — PPTX (slide-by-slide), DOCX, TXT, MD
- **Cited answers** — every answer shows which slides were used as sources
- **Conversation history** — all past chats saved and browsable in the sidebar
- **Auto study questions** — AI generates suggested questions from uploaded content

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vanilla HTML/CSS/JS (zero dependencies) |
| Backend | FastAPI + Python 3.11 |
| Database | SQLite (persistent, zero config) |
| AI | Claude claude-sonnet-4-20250514 (Anthropic) |
| Search | Cosine similarity over hashed n-gram vectors |
| Deployment | Docker Compose |

## Quick Start

### Option A: Docker (recommended)

```bash
git clone <your-repo>
cd studysearch
export ANTHROPIC_API_KEY=sk-ant-...
docker-compose up
```

Open `http://localhost:3000`

### Option B: Manual

**Backend:**
```bash
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
# Open index.html directly, or serve with:
python -m http.server 3000
```

Open `http://localhost:3000`

## Architecture

```
browser
  └── index.html
        ├── Session management (localStorage)
        ├── File parsing (JSZip for PPTX)
        └── REST API calls
              │
              ▼
         FastAPI (port 8000)
              ├── POST /api/sessions           → create session
              ├── GET  /api/sessions/:id       → get session + docs
              ├── POST /api/sessions/:id/documents  → upload + embed
              ├── POST /api/sessions/:id/chat  → RAG + chat memory
              ├── GET  /api/sessions/:id/conversations
              └── GET  /api/sessions/:id/suggestions
                    │
                    ▼
               SQLite (studysearch.db)
                    ├── sessions
                    ├── documents
                    ├── chunks (with embeddings as JSON)
                    ├── conversations
                    └── messages
```

## RAG Pipeline

1. **Upload**: file parsed → split into chunks → each chunk embedded → stored in SQLite
2. **Query**: question embedded → cosine similarity against all chunks → top-K retrieved
3. **Generate**: top-K chunks + full conversation history → Claude → cited answer
4. **Persist**: user message + assistant response saved to conversation

## Upgrading to Production Embeddings

In `backend/main.py`, replace `tfidf_embed()` with:

```python
# Option A: Voyage AI (best for RAG)
import voyageai
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
async def embed_texts(texts):
    result = vo.embed(texts, model="voyage-3-large", input_type="document")
    return result.embeddings

# Option B: OpenAI
from openai import AsyncOpenAI
client = AsyncOpenAI()
async def embed_texts(texts):
    resp = await client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]
```

Also increase vector dimension from 256 to 1024/1536 in the DB schema.

## Deploying to Production

**Railway (easiest):**
```bash
railway login
railway init
railway up
```

**Render:**
- Connect GitHub repo
- Set `ANTHROPIC_API_KEY` environment variable
- Deploy backend as Web Service, frontend as Static Site

## What Makes This Portfolio-Worthy

- Full RAG pipeline with semantic retrieval (not just keyword search)
- Multi-turn conversational memory with proper message history management
- Persistent backend with real database (not just localStorage)
- Clean REST API design with proper error handling
- Docker deployment ready
- Extensible architecture — swap embedding model, add vector DB, add auth

## Next Steps (to go further)

- Replace SQLite with PostgreSQL + pgvector for scalable vector search
- Add Voyage AI or Cohere embeddings for better semantic accuracy
- Add PDF.js for proper PDF text extraction
- Add slide image rendering (PPTX → PNG thumbnails)
- Add user auth (Supabase or Clerk)
- Add study guide generation endpoint
- Add highlight/citation of exact text used in answer
