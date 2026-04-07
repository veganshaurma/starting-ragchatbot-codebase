/# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

The server must be started from the `backend/` directory (relative paths in `app.py` depend on this):

```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

Or use the convenience script from the repo root:

```bash
./run.sh
```

App runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Setup

```bash
uv sync                        # Install dependencies
cp .env.example .env           # Then add ANTHROPIC_API_KEY to .env
```

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

## Architecture

Full-stack RAG chatbot: FastAPI backend + vanilla JS/HTML/CSS frontend + ChromaDB vector store + Anthropic Claude.

**Request flow:**
1. Frontend (`frontend/`) sends POST `/api/query` with `{query, session_id}`
2. `app.py` → `RAGSystem.query()` builds a prompt and calls `AIGenerator`
3. `AIGenerator` calls Claude with the `search_course_content` tool available
4. Claude decides whether to call the tool; if so, `ToolManager` routes to `CourseSearchTool`
5. `CourseSearchTool` queries `VectorStore` (ChromaDB) and returns formatted snippets
6. Claude generates a final answer; `SessionManager` stores the exchange in memory

**Backend modules (`backend/`):**

| File | Role |
|---|---|
| `app.py` | FastAPI app; mounts `../frontend` as static files; 2 API endpoints (`/api/query`, `/api/courses`) |
| `rag_system.py` | Top-level orchestrator wiring all components together |
| `ai_generator.py` | Anthropic API wrapper; handles the tool-use loop (initial call → tool execution → final call) |
| `vector_store.py` | ChromaDB wrapper; two collections: `course_catalog` (titles/metadata) and `course_content` (chunked text) |
| `document_processor.py` | Parses `.txt` course files into `Course`/`Lesson`/`CourseChunk` objects; sentence-aware chunker |
| `search_tools.py` | `Tool` ABC + `CourseSearchTool` + `ToolManager`; defines the Claude tool schema for `search_course_content` |
| `session_manager.py` | In-memory conversation history (lost on restart); keyed by `session_N` IDs |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |
| `config.py` | `Config` dataclass loaded from `.env`; single `config` singleton used app-wide |

**Key config values** (all in `config.py`):
- `ANTHROPIC_MODEL`: `claude-sonnet-4-20250514`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2` (via sentence-transformers)
- `CHUNK_SIZE`: 800 chars, `CHUNK_OVERLAP`: 100 chars
- `MAX_HISTORY`: 2 exchanges kept per session
- `CHROMA_PATH`: `./chroma_db` (relative to `backend/`)

## Course Document Format

Documents in `docs/` must follow this structure for `DocumentProcessor` to parse them correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <lesson title>
...
```

- Course `title` is used as the unique ID in ChromaDB — duplicate titles are skipped on reload
- On startup, `app.py` auto-loads all `.txt`/`.pdf`/`.docx` files from `../docs` (skipping already-indexed courses)
- ChromaDB data persists on disk at `backend/chroma_db/`; delete this folder to force a full re-index
