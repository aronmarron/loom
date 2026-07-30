# 🧶 Loom

A private, local-first AI workbench built with Streamlit. Loom lets you chat with
either cloud models (via OpenRouter) or fully-local models (via Ollama), and adds
document Q&A, image generation, image analysis, and web-assisted writing on top.

## Features

- **Cloud or local chat** — switch between OpenRouter models and local Ollama models with one toggle. A badge always shows whether data is leaving your machine.
- **PDF question-answering (RAG)** — load a PDF, and Loom chunks it, embeds it locally via Ollama, and answers questions using cosine-similarity retrieval. Vectors are cached in a pickle file, so no database is required.
- **Image generation** — sends prompts to a local Stable Diffusion XL server (`sdxl_server.py`).
- **Image analysis** — describes pasted or uploaded images locally via LLaVA (nothing leaves your machine).
- **Web search** — optional DuckDuckGo results injected into answers.
- **Essay agent** — a multi-step pipeline: outline → research → draft → editorial polish.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally, with at least an embedding model:
  `ollama pull nomic-embed-text` (and any chat/vision models you want, e.g. `llava`).
- *(Optional)* An OpenRouter API key for cloud models: <https://openrouter.ai/keys>
- *(Optional, for image generation)* An NVIDIA GPU with CUDA. Image generation uses the included `sdxl_server.py`, which downloads Stable Diffusion XL (~7 GB) on first launch.

## Setup

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets — copy the template and fill in your values
copy .env.example .env         # Windows
# cp .env.example .env          # macOS / Linux
#   → open .env and paste your OpenRouter key (and adjust server addresses)
```

## Running

```bash
streamlit run loom.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

**Image generation (optional).** To use the image-generation tab, start the SDXL server in a separate terminal. It needs an NVIDIA GPU with CUDA:

```bash
python sdxl_server.py     # serves on http://localhost:5050
```

Loom reaches it at the address in `SDXL_BASE`.

## Configuration

All configuration lives in `.env` (never committed). Copy `.env.example` to get started.

| Variable             | Purpose                                      | Default                   |
|----------------------|----------------------------------------------|---------------------------|
| `OPENROUTER_API_KEY` | Key for cloud models (blank = local only)    | *(none)*                  |
| `OPENROUTER_BASE`    | OpenRouter API base URL                      | `https://openrouter.ai/api/v1` |
| `OLLAMA_BASE`        | Local Ollama server address                  | `http://localhost:11434`  |
| `EMBED_MODEL`        | Ollama embedding model for PDF RAG           | `nomic-embed-text`        |
| `SDXL_BASE`          | Local SDXL image server address              | `http://localhost:5050/`  |
| `LOOM_VECTOR_STORE`  | Path to the cached vector store              | `./loom_vectors.pkl`      |
| `MEMORY_WINDOW`      | Recent message pairs kept verbatim           | `6`                       |

## Security note

Never commit your `.env` file — it is already listed in `.gitignore`. If a key is
ever exposed, rotate it immediately at your provider.
