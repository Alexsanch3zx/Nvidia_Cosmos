# How to Implement Supabase pgvector for Video Summary Search

This guide walks you through adding a vector database so you can store video summaries and run similarity search (e.g. “find videos like this” or “videos about X”). Do each step in order.

**Visual overview:** Open `FLOW_DIAGRAM.html` in a browser to see flowcharts of the current pipeline and the flow after summaries are stored as embeddings (save path + search path).

---

## Overview

- **What you’re adding:** PostgreSQL with the pgvector extension, plus an embedding step so summary text becomes vectors. You store video metadata + summary text + embedding; search is “query text → same embedding model → vector similarity in the DB.”
- **What stays the same:** Your existing flow (upload → frames → Cosmos → summary) stays as-is. You add a “save to DB” step after the summary and a “search” path that queries the DB by similarity.

---

## Step 1: Enable pgvector in Supabase

Open your Supabase project, then go to the **SQL Editor**, and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Step 2: Ensure the `video_summaries` table exists

Create (or verify) a table that holds one row per video: metadata, summary text, and one vector per summary. Example:

```sql
CREATE TABLE video_summaries (
  id            BIGSERIAL PRIMARY KEY,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  filename      TEXT,
  duration_sec  NUMERIC(10,2),
  summary_style TEXT,
  summary_text  TEXT NOT NULL,
  embedding     vector(384)  -- 384 for all-MiniLM-L6-v2; change if you use another model
);

-- Optional: index for faster similarity search once you have many rows
-- CREATE INDEX ON video_summaries USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

The `vector(384)` dimension must match the embedding size of the model you use (e.g. 384 for `all-MiniLM-L6-v2`, 768 for many others). Create the index later when you have enough rows (e.g. hundreds); leave it commented at first if you prefer.

---

## Step 3: Choose an embedding model

You need a **text** embedding model (not Cosmos). Use the same model for:

- Indexing: `summary_text` → vector → store in `embedding`
- Search: `query_text` → vector → similarity search in the DB

Options:

- **Local (no API key):** e.g. `sentence-transformers` with `all-MiniLM-L6-v2` (384 dimensions). Install with pip; runs on CPU.
- **Hosted:** e.g. OpenAI `text-embedding-3-small`, Cohere, etc. You’ll need an API key and the same dimension in your table (e.g. 1536 for OpenAI).

Pick one and note the **output dimension**; your `embedding` column must be `vector(<dimension>)`.

---

## Step 4: Add Python dependencies

In your project (e.g. in `requirements.txt` or a separate one for this feature), add:

- A Postgres driver: `psycopg2-binary` or `psycopg[binary]`.
- pgvector for Python: `pgvector` (so you can read/write `vector` type and use distance in SQL).
- Embeddings: for sentence-transformers use `sentence-transformers`; for OpenAI use `openai`, etc.

Install with:

```bash
pip install psycopg2-binary pgvector sentence-transformers
```

(Adjust if you use a different driver or embedding provider.)

---

## Step 5: Store the Supabase connection string (and keep it out of Git)

Use a `.env` file in the project root for the Supabase connection string. Example:

```
SUPABASE_DB_URL=postgresql://postgres:your_password@db.YOUR_PROJECT.supabase.co:5432/postgres?sslmode=require
```

The project’s `.gitignore` already includes `.env`, so this file will **not** be pushed to the repo. Never commit real passwords or URLs.

Load `.env` in your app with `python-dotenv` (e.g. `load_dotenv()` at startup) or by reading the env var from the environment where you run the app.

---

## Step 6: Implement “save summary to DB”

In your own code, after you generate the summary (in the same place you currently have the summary string):

1. Connect to Postgres using `SUPABASE_DB_URL` (or whatever your app loads via `db/connection.py`).
2. Turn the summary string into a vector with your chosen embedding model (same model and dimension as in the table).
3. Insert one row into `video_summaries`: filename (or path), duration, summary_style, summary_text, embedding. You can get duration from your video processing step; store whatever metadata you want.
4. Close the connection (or use a connection pool).

Use the `pgvector` package so the `embedding` value is sent as the correct type. Example pattern in SQL: `INSERT INTO video_summaries (..., embedding) VALUES (..., %s::vector)` and pass the list of floats.

---

## Step 7: Implement “similarity search”

When the user submits a search query:

1. Connect to Postgres.
2. Turn the query string into a vector with the **same** embedding model and same dimension.
3. Run a single SQL query that:
   - Orders rows by distance between `embedding` and the query vector (e.g. `ORDER BY embedding <=> %s` for cosine distance in pgvector).
   - Limits to top N (e.g. 10).
4. Return those rows (e.g. id, filename, summary_text, duration, and optionally the distance). Render them in your UI (e.g. list or cards with summary snippets).

Again, use the pgvector client so you can pass the query vector and read results correctly.

---

## Step 8: Wire it into your app

- **After “Generate Summary”:** Optionally or always call your “save summary to DB” logic with the current video’s metadata and the generated summary. You can add a checkbox “Save to library” if you want it optional.
- **Search UI:** Add a text input and a “Search” button (e.g. in the sidebar or a separate page). On submit, call your similarity-search function and display the returned videos/summaries.

Handle errors (e.g. DB unreachable, missing `SUPABASE_DB_URL`) so the app doesn’t crash when the DB isn’t configured.

---

## Step 9: Optional – add an index for speed

Once you have a lot of rows (e.g. hundreds or more), create an index on the vector column so similarity search stays fast. Example for cosine distance and 384-dimensional vectors:

```sql
CREATE INDEX ON video_summaries
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

`lists` is a tuning parameter; start with something like `sqrt(row_count)` or 100 and adjust if needed. You can also try HNSW instead of IVFFlat depending on your pgvector version.

---

## Checklist

- [ ] Postgres with pgvector running (Docker or local).
- [ ] Database created; `CREATE EXTENSION vector` and `video_summaries` table (with correct `vector(N)` dimension).
- [ ] Embedding model chosen; Python deps installed (Postgres driver, pgvector, embedding library).
- [ ] `.env` with `SUPABASE_DB_URL`; `.env` in `.gitignore` (already done) so it’s not pushed.
- [ ] Code: embed summary → insert into `video_summaries` after generating summary.
- [ ] Code: embed query → similarity query → return top N; show in UI.
- [ ] Optional: vector index for larger datasets.

If you follow these steps in order, you’ll have video summaries stored in Postgres and similarity search working without writing code into the repo for you; this file is the guide to do it yourself.
