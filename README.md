# NCERT Class 10 Science Doubt-Solver with Smart Caching

A chatbot that answers doubts from the NCERT Class 10 Science textbook, cites the chapter it used, and declines anything the book doesn't cover. It handles follow-ups such as *"what about its laws?"*. A semantic cache reuses past answers **only when they answer the same question**. Cache hits make no LLM call and return in tens of milliseconds.

**How it works:** [docs/APPROACH.md](docs/APPROACH.md) is a one-page explainer with a flowchart, the cache guards and evaluation numbers.

| Part | Tech |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b` to answer, `openai/gpt-oss-20b` to condense follow-ups) via an OpenAI-compatible API |
| Orchestration | LangGraph (`backend/app/chat/graph.py`) + LangChain |
| Vector store | FAISS, for both textbook retrieval and cache lookup |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally with fastembed (ONNX) |
| API | FastAPI (`POST /session`, `POST /chat`) |
| UI | Streamlit (`frontend/streamlit_app.py`) |

## Run locally

Requires Python 3.11+ and a free Groq API key from <https://console.groq.com/keys>.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r frontend/requirements.txt
cp .env.example .env                 # then put your GROQ_API_KEY in .env
```

The FAISS index of the textbook is already committed in `data/textbook_index/`. To rebuild it from the NCERT PDFs (takes about 10 minutes):

```bash
python scripts/download_textbook.py
python scripts/ingest.py
```

Start the backend and the UI in two terminals:

```bash
uvicorn backend.app.main:app --port 8000
streamlit run frontend/streamlit_app.py      # uses BACKEND_URL, default http://localhost:8000
```

To try the API directly:

```bash
curl -X POST localhost:8000/session
# {"session_id": "..."}
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"session_id": "...", "message": "What is refraction?"}'
# {"reply": "...", "citations": ["Light – Reflection and Refraction"], "cache_hit": false, "latency_ms": 1432}
```

Add `?debug=1` to `/chat` to see the decision trace: which cached questions were considered and which guard rejected each one. `GET /cache/stats` shows the cache size and hit counts.

## Tests and evaluation

```bash
pytest                          # guards, turn classifier, API with a fake LLM (no key needed)
python scripts/eval_cache.py    # similarity-only vs guarded cache on labelled question pairs
```

The API tests check that a cache hit makes **zero LLM calls** and returns in under 500 ms. They also check each row of the assignment's caching table: paraphrases, concave/convex, R = 20 vs 30 cm, follow-ups and "explain it more simply".

## Deploy

**Backend on Render** (free, Docker):
1. Push this repo to GitHub.
2. On Render, create a **New → Blueprint** from the repo. It picks up `render.yaml`.
3. Set the `GROQ_API_KEY` environment variable.
4. Note the service URL, for example `https://ncert-science-backend.onrender.com`.

**Frontend on Streamlit Community Cloud:**
1. Choose **New app**, select the repo, and set the main file to `frontend/streamlit_app.py`.
2. Under *Advanced settings → Secrets*, add `BACKEND_URL = "https://<your-render-url>"`.

Render's free disk is wiped on restart. To make sure the first visitors still get cache hits, run `python scripts/build_seed_cache.py` once. It needs a key and writes `data/seed_cache.jsonl`, which the backend loads into an empty cache at startup. Commit that file.

The free Render instance sleeps when idle. The first request after a sleep can take up to a minute, and the UI shows a "waking up" spinner while it waits.

## Layout

```
backend/app/
  main.py            FastAPI app
  chat/graph.py      LangGraph pipeline: classify → lookup → condense → retrieve → generate → store
  chat/turns.py      rule-based turn classifier (standalone / follow-up / transform / smalltalk)
  chat/llm.py        all LLM calls (swappable for tests)
  cache/store.py     SQLite + FAISS semantic cache, plus derived ("explain simpler") answers
  cache/guards.py    reuse safety checks
  cache/text.py      question analysis: terms, numbers, intent, negation
  cache/lexicon.py   stopwords, synonyms, contrast pairs
  textbook.py        textbook retrieval
scripts/             download, ingest, seed cache, cache evaluation
frontend/            Streamlit app
tests/               pytest suite
```
