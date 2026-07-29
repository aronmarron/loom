import torch
torch.classes.__path__ = []  # Fix for torch inspection error

import streamlit as st
import pandas as pd
import requests, json, os, tempfile, datetime, shutil, base64
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchResults
import numpy as np
import pickle

# ─────────────────────────────────────────────
#  CONFIG  — values are read from the environment / a local .env file.
#  Copy .env.example to .env and fill in your own values. Never commit .env.
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # loads variables from a .env file in the project root, if present

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OLLAMA_BASE        = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
# Comma-separated list of Ollama hosts to spread batch work (e.g. PDF embedding)
# across — this machine plus the garage box. Falls back to just OLLAMA_BASE.
OLLAMA_HOSTS       = [h.strip().rstrip("/") for h in
                      os.getenv("OLLAMA_HOSTS", OLLAMA_BASE).split(",") if h.strip()]
VECTOR_STORE_PATH  = os.getenv("LOOM_VECTOR_STORE", "./loom_vectors.pkl")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "nomic-embed-text")

if not OPENROUTER_API_KEY:
    st.warning(
        "No OPENROUTER_API_KEY found. Cloud models will fail until you set it in "
        ".env (see .env.example). Local Ollama models still work without a key."
    )


# ─────────────────────────────────────────────
#  SIMPLE VECTOR STORE (numpy cosine similarity)
#  No ChromaDB, no version mismatches — just embeddings + pickle
# ─────────────────────────────────────────────
def get_embedding(text: str, host: str = None) -> np.ndarray:
    """Get an embedding from an Ollama host (defaults to the primary host)."""
    resp = requests.post(
        f"{host or OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    return np.array(resp.json()["embedding"], dtype=np.float32)


def build_vector_store(chunks: list) -> dict:
    """Embed all chunks, spreading the work across every reachable Ollama host.

    Embedding a large PDF is hundreds of independent calls, so we fan them out
    over a thread pool and round-robin across hosts. With two machines online
    this roughly halves the time; with one it behaves like the old serial loop.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    hosts = healthy_embed_hosts() or [OLLAMA_BASE]
    base  = (f"Embedding {len(texts)} chunks across {len(hosts)} hosts"
             if len(hosts) > 1 else f"Embedding {len(texts)} chunks")
    vectors = [None] * len(texts)
    progress = st.progress(0, text=f"{base}...")
    done = 0

    # Cap in-flight requests at the number of hosts so no single Ollama is
    # flooded — each machine works one embedding at a time.
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futures = {pool.submit(get_embedding, txt, hosts[i % len(hosts)]): i
                   for i, txt in enumerate(texts)}
        for fut in as_completed(futures):
            i = futures[fut]
            vectors[i] = fut.result()
            done += 1
            progress.progress(done / len(texts), text=f"{base}... {done}/{len(texts)}")
    progress.empty()
    return {"texts": texts, "metas": metas, "vectors": np.stack(vectors)}


def similarity_search(store: dict, query: str, k: int = 3) -> list:
    """Cosine similarity search. Returns list of (text, metadata) tuples."""
    embed_host = (healthy_embed_hosts() or [OLLAMA_BASE])[0]
    q_vec = get_embedding(query, embed_host)
    vecs  = store["vectors"]
    # Cosine similarity
    norms  = np.linalg.norm(vecs, axis=1) * np.linalg.norm(q_vec)
    norms  = np.where(norms == 0, 1e-10, norms)
    scores = np.dot(vecs, q_vec) / norms
    top_k  = np.argsort(scores)[::-1][:k]
    return [(store["texts"][i], store["metas"][i]) for i in top_k]


def save_vector_store(store: dict, pdf_name: str):
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"store": store, "pdf_name": pdf_name}, f)


def load_vector_store_from_disk() -> tuple:
    """Returns (store, pdf_name) or (None, None)."""
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["store"], data["pdf_name"]
    return None, None
MEMORY_WINDOW      = int(os.getenv("MEMORY_WINDOW", "6"))  # last N exchange pairs verbatim; older → summary
SDXL_BASE          = os.getenv("SDXL_BASE", "http://localhost:5050/")  # sdxl_server.py endpoint

search = DuckDuckGoSearchResults(output_format="json")


# ─────────────────────────────────────────────
#  MODEL FETCHERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_openrouter_models():
    try:
        resp = requests.get(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=8,
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            ids = sorted(
                m["id"] for m in models
                if m.get("architecture", {}).get("modality", "")
                in ("text->text", "text+image->text", "")
            )
            return ids if ids else ["(no models found)"]
        return [f"(API error {resp.status_code})"]
    except Exception as e:
        return [f"(request failed: {e})"]


@st.cache_data(ttl=30)
def ollama_host_catalog() -> dict:
    """Map each reachable Ollama host -> the list of models it has installed.
    Offline hosts are silently skipped so one dead machine never breaks Loom."""
    catalog = {}
    for host in OLLAMA_HOSTS:
        try:
            resp = requests.get(f"{host}/api/tags", timeout=4)
            if resp.status_code == 200:
                catalog[host] = sorted(m["name"] for m in resp.json().get("models", []))
        except Exception:
            continue
    return catalog


def get_ollama_models():
    """Union of models across all reachable hosts (for the model dropdown)."""
    catalog = ollama_host_catalog()
    if not catalog:
        return ["(Ollama not running — start with: ollama serve)"]
    names = sorted({name for models in catalog.values() for name in models})
    return names if names else ["(no local models found)"]


def online_ollama_hosts() -> list:
    """Hosts currently reachable."""
    return list(ollama_host_catalog().keys())


def healthy_embed_hosts() -> list:
    """Reachable hosts that have the embedding model installed."""
    catalog = ollama_host_catalog()
    hosts = [h for h, models in catalog.items()
             if any(EMBED_MODEL in m for m in models)]
    return hosts or list(catalog.keys())


def host_for_model(model: str) -> str:
    """Pick a reachable host that has the given model; fall back to primary."""
    for host, models in ollama_host_catalog().items():
        if model in models:
            return host
    return OLLAMA_BASE


# ─────────────────────────────────────────────
#  ESSAY DETECTION
# ─────────────────────────────────────────────
ESSAY_KEYWORDS = [
    "essay", "article", "draft", "write about", "write an",
    "write a", "compose", "piece on", "piece about", "long-form",
    "longform", "blog post", "op-ed", "editorial",
]

def is_essay_request(prompt: str) -> bool:
    return any(kw in prompt.lower() for kw in ESSAY_KEYWORDS)


# ─────────────────────────────────────────────
#  IMAGE GENERATION DETECTION + CALL
# ─────────────────────────────────────────────
IMAGE_GEN_KEYWORDS = [
    "generate", "create", "draw", "paint", "make", "render",
    "imagine", "show me", "picture of", "image of", "photo of",
    "illustration of", "sketch of",
]

def is_image_request(prompt: str) -> bool:
    p = prompt.lower()
    # Must have an image keyword AND some visual noun to avoid false positives
    visual_nouns = ["image", "picture", "photo", "illustration", "drawing",
                    "painting", "sketch", "portrait", "landscape", "art"]
    has_keyword = any(kw in p for kw in IMAGE_GEN_KEYWORDS)
    has_noun    = any(n in p for n in visual_nouns)
    return has_keyword and has_noun


# Follow-up "edit" phrasings ("put it on a skateboard", "make it blue"). These
# are only treated as image requests when an image was JUST generated (guarded
# by img_thread_active below), which keeps false positives low.
IMAGE_EDIT_KEYWORDS = [
    "add ", "put ", "change ", "replace ", "remove ", "give it", "give him",
    "give her", "give them", "make it", "make him", "make her", "make them",
    "turn it", "now ", "instead", "but with", "without", "wearing", "holding",
    "on a ", "in a ", "with a ", "recolor", "color it", "bigger", "smaller",
    "different", "another", "more ", "less ", "background", "zoom", "same but",
]

def is_image_edit(prompt: str) -> bool:
    p = prompt.lower()
    return any(kw in p for kw in IMAGE_EDIT_KEYWORDS)


# ─────────────────────────────────────────────
#  WEB-SEARCH DETECTION
# ─────────────────────────────────────────────
# Phrasings that usually need fresh / real-time info a local model can't know.
# Heuristic only — errs toward searching when a query looks time-sensitive.
WEB_SEARCH_KEYWORDS = [
    "latest", "current", "currently", "today", "tonight", "right now",
    "this week", "this month", "this year", "recent", "recently", "news",
    "headline", "price of", "stock", "weather", "forecast", "score",
    "who won", "when did", "when is", "how old is", "as of", "release date",
    "update on", "happening", "2024", "2025", "2026", "near me",
]

def needs_web_search(prompt: str) -> bool:
    p = prompt.lower()
    return any(kw in p for kw in WEB_SEARCH_KEYWORDS)


# ─────────────────────────────────────────────
#  IMAGE SEARCH (find existing images on the web, via DuckDuckGo)
# ─────────────────────────────────────────────
IMAGE_SEARCH_VERBS = ["find", "search", "look for", "look up", "google"]
_VISUAL_NOUNS = ["image", "picture", "photo", "illustration", "drawing",
                 "painting", "sketch", "portrait", "logo", "art"]

def is_image_search(prompt: str) -> bool:
    """A request to FIND existing images on the web (not generate one)."""
    p = prompt.lower()
    return (any(v in p for v in IMAGE_SEARCH_VERBS)
            and any(n in p for n in _VISUAL_NOUNS))

def image_search_query(prompt: str) -> str:
    """Turn a request into a clean image-search query. 'similar' requests reuse
    the last generated image's prompt so 'find a similar image' actually works."""
    import re
    p = prompt.lower()
    if "similar" in p and st.session_state.get("last_image_prompt"):
        return st.session_state["last_image_prompt"]
    q = prompt
    for junk in ["can you", "could you", "please", "for me", "find me", "find",
                 "search for", "search", "look for", "look up", "on google",
                 "google", "images of", "image of", "pictures of", "picture of",
                 "a similar", "similar", "some"]:
        q = re.sub(junk, "", q, flags=re.I)
    return q.strip(" ?.,") or prompt

def ddg_image_search(query: str, max_results: int = 6) -> list:
    """Return a list of image-result dicts from DuckDuckGo, or [] on failure."""
    DDGS = None
    for mod in ("duckduckgo_search", "ddgs"):
        try:
            DDGS = __import__(mod, fromlist=["DDGS"]).DDGS
            break
        except Exception:
            continue
    if DDGS is None:
        return []
    try:
        with DDGS() as d:
            return list(d.images(query, max_results=max_results))
    except Exception:
        return []


# ─────────────────────────────────────────────
#  CONVERSATION PERSISTENCE (save/load chats to disk)
# ─────────────────────────────────────────────
CONV_DIR = os.getenv("LOOM_CONV_DIR", "./loom_conversations")

def _safe_name(s: str) -> str:
    keep = "".join(c if (c.isalnum() or c in " -_") else "" for c in s).strip()
    return (keep[:40] or "chat").replace(" ", "_")

def save_conversation(messages: list) -> str:
    os.makedirs(CONV_DIR, exist_ok=True)
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "chat")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}__{_safe_name(first_user)}.json"
    with open(os.path.join(CONV_DIR, fname), "w", encoding="utf-8") as f:
        json.dump(messages, f)
    return fname

def list_conversations() -> list:
    if not os.path.isdir(CONV_DIR):
        return []
    return sorted((f for f in os.listdir(CONV_DIR) if f.endswith(".json")), reverse=True)

def load_conversation(fname: str) -> list:
    with open(os.path.join(CONV_DIR, fname), "r", encoding="utf-8") as f:
        return json.load(f)

def delete_conversation(fname: str):
    path = os.path.join(CONV_DIR, fname)
    if os.path.exists(path):
        os.remove(path)


def call_sdxl(prompt: str, width=512, height=512, steps=40,
              guidance=7.0, seed=-1, negative_prompt="") -> dict:
    """Call the local SDXL Flask server. Returns dict with 'image' (base64) and 'seed'."""
    payload = {
        "prompt":          prompt,
        "negative_prompt": negative_prompt,
        "width":           width,
        "height":          height,
        "steps":           steps,
        "guidance":        guidance,
        "seed":            seed,
    }
    resp = requests.post(f"{SDXL_BASE}/generate", json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"SDXL server error {resp.status_code}: {resp.text}")
    return resp.json()


@st.cache_data(ttl=5)
def sdxl_available() -> bool:
    """Cached health check — re-pings at most every 5 seconds."""
    try:
        r = requests.get(f"{SDXL_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────
#  LLM CALL ROUTING — OpenRouter or Ollama
# ─────────────────────────────────────────────
def call_llm(messages, model, temperature, max_tokens=2000, local_mode=False) -> str:
    """Non-streaming call. Routes to Ollama if local_mode=True."""
    if local_mode:
        return call_ollama(messages, model, temperature, max_tokens)
    return call_openrouter(messages, model, temperature, max_tokens)


def call_openrouter(messages, model, temperature, max_tokens=2000) -> str:
    payload = {
        "model": model, "messages": messages,
        "stream": False, "temperature": temperature, "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://loom-ai.local",
        "X-Title": "Loom AI",
    }
    resp = requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        json=payload, headers=headers, timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


def call_ollama(messages, model, temperature, max_tokens=2000) -> str:
    """Non-streaming call to local Ollama /api/chat."""
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(f"{host_for_model(model)}/api/chat", json=payload, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    return resp.json()["message"]["content"]


# ─────────────────────────────────────────────
#  IMAGE ANALYSIS via llava (fully local)
# ─────────────────────────────────────────────
def analyze_image_with_llava(image_bytes: bytes, prompt: str) -> str:
    """Send image + prompt to llava:latest via Ollama. Nothing leaves your machine."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": "llava:latest",
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    resp = requests.post(f"{host_for_model('llava:latest')}/api/chat", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"llava error {resp.status_code}: {resp.text}")
    return resp.json()["message"]["content"]


# ─────────────────────────────────────────────
#  CLIPBOARD IMAGE GRAB (Windows)
# ─────────────────────────────────────────────
def grab_image_from_clipboard():
    """
    Read an image from the Windows clipboard using win32clipboard.
    Returns raw PNG bytes or None if clipboard has no image.
    Requires: pip install pywin32
    """
    try:
        import win32clipboard
        from PIL import Image
        import io

        win32clipboard.OpenClipboard()
        try:
            # CF_DIB = Device Independent Bitmap (standard image clipboard format)
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
                # Pillow can read DIB from a BMP stream — prepend the BMP file header
                bmp_header = b'BM' + (len(data) + 14).to_bytes(4, 'little') + b'\x00\x00\x00\x00' + b'\x36\x00\x00\x00'
                bmp_data   = bmp_header + data
                img = Image.open(io.BytesIO(bmp_data))
                # Convert to PNG bytes
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
            else:
                return None
        finally:
            win32clipboard.CloseClipboard()
    except ImportError:
        return "missing_pywin32"
    except Exception:
        return None


# ─────────────────────────────────────────────
#  MEMORY: Rolling summary + window
# ─────────────────────────────────────────────
def build_api_history(session_messages: list, model: str, temperature: float,
                      local_mode: bool = False) -> list:
    clean = [{"role": m["role"], "content": m["content"]} for m in session_messages]
    cutoff = MEMORY_WINDOW * 2
    if len(clean) <= cutoff:
        return clean

    older  = clean[:-cutoff]
    recent = clean[-cutoff:]

    summary_prompt = (
        "Summarize the following conversation into 3-5 concise bullet points "
        "capturing key facts, decisions, and context. Be brief.\n\n"
        + "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in older)
    )
    try:
        summary = call_llm(
            [{"role": "user", "content": summary_prompt}],
            model=model, temperature=0.2, max_tokens=300, local_mode=local_mode,
        )
    except Exception:
        summary = f"[Earlier conversation ({len(older)} messages) omitted due to length.]"

    return [{"role": "assistant",
             "content": f"**[Conversation Memory]**\n{summary}"}] + recent


# ─────────────────────────────────────────────
#  ESSAY AGENT
# ─────────────────────────────────────────────
def run_essay_agent(topic, full_context, model, temperature, use_web,
                    max_tokens=1575, local_mode=False):
    with st.status("🧶 Weaving Essay...", expanded=True) as status:

        status.write("📝 Structuring arguments...")
        outline = call_llm([
            {"role": "system", "content": (
                "You are an expert essay architect. Produce a detailed outline with thesis, "
                "main sections, key points, and conclusion direction. Be concise.")},
            {"role": "user", "content": f"Create an outline for an essay on: {topic}"},
        ], model, temperature=0.3, max_tokens=min(800, max_tokens), local_mode=local_mode)

        status.write("🌐 Scouring sources...")
        web_results = ""
        if use_web:
            try:
                web_results = search.run(topic)
            except Exception:
                web_results = "Web search unavailable."

        research_parts = []
        if full_context:
            research_parts.append(full_context)
        if web_results:
            research_parts.append(f"--- WEB SEARCH RESULTS ---\n{web_results}")
        combined_context = "\n\n".join(research_parts) or "No external sources loaded."

        status.write("✍️ Synthesizing draft...")
        draft = call_llm([
            {"role": "system", "content": (
                "You are a skilled professional writer. Using the outline and research, "
                "write a full compelling essay. Follow the outline, cite sources inline.")},
            {"role": "user", "content": (
                f"Topic: {topic}\n\nOutline:\n{outline}\n\n"
                f"Research & Context:\n{combined_context}\n\nWrite the full essay now.")},
        ], model, temperature=temperature, max_tokens=max_tokens, local_mode=local_mode)

        status.write("✨ Finalizing & formatting...")
        final_essay = call_llm([
            {"role": "system", "content": (
                "You are a master editor. Polish the draft for clarity, flow, and impact. "
                "Format in clean Markdown with headings. End with a '## References' section.")},
            {"role": "user", "content": (
                f"Polish this essay:\n\n{draft}\n\n"
                f"Sources for references:\n{combined_context[:3000]}")},
        ], model, temperature=0.4, max_tokens=max_tokens, local_mode=local_mode)

        status.update(label="✅ Essay complete!", state="complete")
        return final_essay, combined_context


# ─────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(page_title="Loom AI", page_icon="🧶", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f5f4f0; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e0ddd8; }
    .main .block-container { padding-top: 2rem; }
    p, li, span, label, div, h1, h2, h3, h4, h5 { color: #1a1a1a; }
    .stMarkdown p { color: #1a1a1a; }
    [data-testid="stSidebar"] h3 {
        color: #111111; font-size: 0.85rem; font-weight: 700;
        letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem;
    }
    .stSelectbox label, .stSlider label,
    .stToggle label, .stFileUploader label { color: #333333; }
    .stChatInput textarea { color: #1a1a1a; background: #ffffff; border: 1px solid #d0cdc8; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { color: #1a1a1a; }
    .loom-title {
        font-family: 'Courier New', monospace; font-size: 1.8rem;
        font-weight: 700; color: #111111; letter-spacing: 0.08em; margin-bottom: 0;
    }
    .loom-subtitle {
        font-family: 'Courier New', monospace; font-size: 0.7rem;
        color: #888888; letter-spacing: 0.3em; text-transform: uppercase; margin-top: 2px;
    }
    .badge {
        display: inline-block; padding: 3px 9px; border-radius: 4px;
        font-size: 0.7rem; font-family: monospace; font-weight: 700; margin-right: 4px;
    }
    .badge-green  { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .badge-yellow { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .badge-red    { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-blue   { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .badge-purple { background: #f3e8ff; color: #6b21a8; border: 1px solid #d8b4fe; }
    hr { border-color: #e0ddd8; }
    [data-testid="stSidebar"] .block-container { padding-top: 0.75rem; padding-bottom: 0.5rem; }
    [data-testid="stSidebar"] h3 { margin-top: 0.25rem; margin-bottom: 0.1rem; font-size: 0.75rem; }
    [data-testid="stSidebar"] .stSelectbox { margin-bottom: 0; }
    [data-testid="stSidebar"] .stFileUploader { margin-bottom: 0; }
    [data-testid="stSidebar"] .stToggle { margin-bottom: 0; }
    [data-testid="stSidebar"] .stSlider { margin-bottom: 0; padding-bottom: 0; }
    [data-testid="stSidebar"] hr { margin: 0.35rem 0; }
    [data-testid="stSidebar"] .stButton button { padding: 0.2rem 0.5rem; font-size: 0.8rem; }
    [data-testid="stSidebar"] .element-container { margin-bottom: 0.05rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="loom-title">🧶 LOOM</p>', unsafe_allow_html=True)
    st.markdown('<p class="loom-subtitle">Local Orchestrator of Omni-Models</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOCAL MODEL SELECTION HELPERS
# ─────────────────────────────────────────────
# Substrings that flag a model as NOT a text-chat model (image / audio /
# embedding). Loom must never default the chat box to one of these.
NON_CHAT_HINTS = [
    "stablediffusion", "stable-diffusion", "sdxl", "flux", "dreamshaper",
    "embed", "nomic-embed", "bge", "whisper", "clip",
    "tts", "neutts", "bark", "musicgen", "rerank",
]
# Preferred chat-model families, best first.
CHAT_PREFERENCE = [
    "llama3.1", "llama3.2", "llama3", "llama", "qwen2.5", "qwen",
    "mistral", "mixtral", "gemma", "phi", "deepseek",
]

def is_chat_model(name: str) -> bool:
    """True unless the model name looks like an image/audio/embedding model."""
    n = name.lower()
    return not any(h in n for h in NON_CHAT_HINTS)

def default_local_index(models: list) -> int:
    """Pick a sensible default: a preferred chat family first, then any
    chat-capable model, falling back to index 0 only if nothing else fits."""
    for pref in CHAT_PREFERENCE:
        for i, m in enumerate(models):
            if pref in m.lower() and is_chat_model(m):
                return i
    for i, m in enumerate(models):
        if is_chat_model(m):
            return i
    return 0


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    # ── MODEL (always visible — most important) ───────────────────────────────
    local_mode = st.toggle("🔒 Local Mode (Ollama)", value=True,
                           help="Routes all chat to local Ollama — nothing leaves your machine")

    if local_mode:
        st.markdown('<span class="badge badge-purple">🔒 FULLY PRIVATE</span>', unsafe_allow_html=True)
        ollama_models = get_ollama_models()
        ollama_ok = bool(ollama_models) and not ollama_models[0].startswith("(")
        col_lm, col_lr = st.columns([5, 1])
        with col_lm:
            ST_MODEL = st.selectbox(
                "Local Model", ollama_models,
                index=default_local_index(ollama_models) if ollama_ok else 0,
                disabled=not ollama_ok, label_visibility="collapsed",
            )
        with col_lr:
            if st.button("↻", key="refresh_local", help="Refresh local models"):
                ollama_host_catalog.clear()
                st.rerun()
        if not ollama_ok:
            st.caption(ollama_models[0])
        elif not is_chat_model(ST_MODEL):
            st.warning(
                "⚠ This looks like an image/audio/embedding model, not a chat "
                "model — replies won't work. Pick a llama / qwen / mistral model."
            )
        else:
            n_hosts = len(online_ollama_hosts())
            host_txt = f"  ·  ⚡ {n_hosts} compute hosts" if n_hosts > 1 else ""
            st.caption(f"🖥️ {len(ollama_models)} local model(s){host_txt}")
    else:
        available_models = get_openrouter_models()
        has_models = bool(available_models) and not available_models[0].startswith("(")
        st.markdown(
            '<span class="badge badge-green">● ONLINE</span>' if has_models
            else '<span class="badge badge-red">● OFFLINE</span>',
            unsafe_allow_html=True,
        )
        col_model, col_refresh = st.columns([5, 1])
        with col_model:
            default_idx = 0
            preferred = [
                "anthropic/claude-3.5-sonnet", "openai/gpt-4o",
                "openai/gpt-4o-mini", "meta-llama/llama-3.1-8b-instruct",
            ]
            for p in preferred:
                if p in available_models:
                    default_idx = available_models.index(p)
                    break
            ST_MODEL = st.selectbox(
                "Cloud Model", available_models,
                index=default_idx, disabled=not has_models,
                label_visibility="collapsed",
            )
        with col_refresh:
            if st.button("↻", key="refresh_cloud", help="Refresh models"):
                get_openrouter_models.clear()
                st.rerun()
        if not has_models:
            st.caption(f"Error: {available_models[0]}")

    st.divider()

    # ── QUICK TOGGLES (always visible) ───────────────────────────────────────
    use_web      = st.toggle("🌐 Web Search",  value=False)
    sdxl_ok      = sdxl_available()
    image_mode   = st.toggle("🎨 Image Gen",   value=False,
                              help="Everything you type goes to SDXL as a prompt",
                              disabled=not sdxl_ok)
    if image_mode:
        st.caption("💬 Type what to generate →")
    show_context = st.toggle("🔍 Show Sources", value=False)

    st.divider()

    # ── TABBED SOURCES + SETTINGS ───────────────────────────────────────────
    tab_sources, tab_settings = st.tabs(["📂 Sources", "⚙️ Settings"])

    # ── SOURCES TAB ──────────────────────────────────────────────────────────
    with tab_sources:

        # PDF
        pdf_status = f"✅ {st.session_state.get('pdf_name','')}  ({st.session_state.get('pdf_chunks','?')} chunks)" \
                     if "vector_store" in st.session_state else "No PDF loaded"
        st.markdown(f"**📄 PDF** — {pdf_status}")
        uploaded_pdf = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
        if uploaded_pdf:
            _chunk_size    = st.session_state.get("_chunk_size", 1000)
            _chunk_overlap = st.session_state.get("_chunk_overlap", 150)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Weave", use_container_width=True):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_pdf.getvalue())
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        chunks = RecursiveCharacterTextSplitter(
                            chunk_size=_chunk_size, chunk_overlap=_chunk_overlap
                        ).split_documents(loader.load())
                        os.unlink(tmp_path)
                        # Build numpy vector store (progress bar shown inside)
                        store = build_vector_store(chunks)
                        save_vector_store(store, uploaded_pdf.name)
                        st.session_state.vector_store = store
                        st.session_state.pdf_name     = uploaded_pdf.name
                        st.session_state.pdf_chunks   = len(chunks)
                        st.success(f"✅ {len(chunks)} chunks embedded")
                    except Exception as e:
                        st.error(f"{e}")
            with c2:
                if st.button("Load DB", use_container_width=True):
                    store, pdf_name = load_vector_store_from_disk()
                    if store:
                        st.session_state.vector_store = store
                        st.session_state.pdf_name     = pdf_name
                        st.session_state.pdf_chunks   = len(store["texts"])
                        st.success(f"✅ Loaded {len(store['texts'])} chunks")
                    else:
                        st.warning("No saved vectors found. Weave a PDF first.")

        st.divider()

        # Data
        data_status = f"✅ {st.session_state.get('df_name','')}  ({st.session_state['df'].shape[0]}r × {st.session_state['df'].shape[1]}c)" \
                      if "df" in st.session_state else "No data loaded"
        st.markdown(f"**📊 Data** — {data_status}")
        uploaded_data = st.file_uploader("CSV/XLSX", type=["csv", "xlsx"], label_visibility="collapsed")
        if uploaded_data:
            try:
                df = (
                    pd.read_csv(uploaded_data)
                    if uploaded_data.name.endswith(".csv")
                    else pd.read_excel(uploaded_data)
                )
                st.session_state.df      = df
                st.session_state.df_name = uploaded_data.name
                st.caption(f"{uploaded_data.name}  ({df.shape[0]}r × {df.shape[1]}c)")
            except Exception as e:
                st.error(f"{e}")

        st.divider()

        # Image (llava)
        img_status = f"✅ {st.session_state.get('image_name','')} — ready as context" \
                     if "image_analysis" in st.session_state else "No image loaded"
        st.markdown(f"**🖼️ Image** *(llava)* — {img_status}")

        # ── Paste from clipboard ──────────────
        if st.button("📋 Paste from Clipboard", use_container_width=True,
                     help="Grab whatever image is in your clipboard (Win+Shift+S, then paste here)"):
            clip_result = grab_image_from_clipboard()
            if clip_result == "missing_pywin32":
                st.error("Install pywin32 first:  `pip install pywin32`")
            elif clip_result is None:
                st.warning("No image found in clipboard. Copy an image first.")
            else:
                # Store raw bytes in session state — treat same as a file upload
                st.session_state.clipboard_image_bytes = clip_result
                st.session_state.pop("image_analysis", None)
                st.session_state.pop("image_key", None)
                st.success("✅ Image grabbed from clipboard")
                st.rerun()

        uploaded_image = st.file_uploader(
            "Image", type=["png", "jpg", "jpeg", "webp", "gif"],
            label_visibility="collapsed", key="image_uploader",
        )
        # Resolve image source — clipboard takes priority over file upload
        _img_bytes = None
        _img_name  = None
        _img_mime  = "image/png"

        if "clipboard_image_bytes" in st.session_state:
            _img_bytes = st.session_state.clipboard_image_bytes
            _img_name  = "clipboard.png"
        elif uploaded_image:
            image_key = f"{uploaded_image.name}_{len(uploaded_image.getvalue())}"
            if st.session_state.get("image_key") != image_key:
                st.session_state.image_key = image_key
                st.session_state.pop("image_analysis", None)
            _img_bytes = uploaded_image.getvalue()
            _img_name  = uploaded_image.name
            _img_mime  = uploaded_image.type

        if _img_bytes:
            # Thumbnail always visible in sidebar
            st.image(_img_bytes, width=200, caption=_img_name)

            if "image_analysis" not in st.session_state:
                with st.spinner("llava analyzing..."):
                    try:
                        result = analyze_image_with_llava(
                            _img_bytes,
                            "Describe this image in comprehensive detail — objects, text, colors, "
                            "layout, any data or charts, and anything else notable."
                        )
                        st.session_state.image_analysis  = result
                        st.session_state.image_name      = _img_name
                        st.session_state.image_bytes_b64 = base64.b64encode(_img_bytes).decode("utf-8")
                        st.session_state.image_mime      = _img_mime
                        # Auto-post image + short analysis into chat
                        import textwrap
                        preview  = textwrap.shorten(result, width=300, placeholder="...")
                        b64      = st.session_state.image_bytes_b64
                        chat_msg = (
                            f"**[Image loaded: {_img_name}]**\n\n"
                            f"![{_img_name}](data:{_img_mime};base64,{b64})\n\n"
                            f"*llava:* {preview}"
                        )
                        st.session_state.messages.append({
                            "role": "user", "content": chat_msg, "context_used": "",
                        })
                        # Clear clipboard buffer after use
                        st.session_state.pop("clipboard_image_bytes", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"llava error: {e}")
                        st.caption("Install: `ollama pull llava`")
            else:
                st.caption("✅ Analysis ready — active as context")

    # ── SETTINGS TAB ─────────────────────────────────────────────────────────
    with tab_settings:

        st.markdown("**Chat**")
        temperature  = st.slider("Temp",       0.0,  1.0,  0.7,  0.05)
        max_tokens   = st.slider("Max Tokens", 256, 4096, 1575,   64,
                                 help="Lower if you hit 402 credit errors")
        st.caption(f"~{max_tokens} tokens per reply")

        st.divider()
        st.markdown("**PDF Chunking**")
        top_k_docs    = st.slider("Top-K Chunks",  1,   10,   3,   1)
        chunk_size    = st.slider("Chunk Size",   200, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap",  0,  400,  150,  25)
        st.caption(f"Each chunk ≈ {chunk_size // 4} tokens")
        st.session_state["_chunk_size"]    = chunk_size
        st.session_state["_chunk_overlap"] = chunk_overlap

        st.divider()
        st.markdown("**Image Gen (SDXL)**")
        sdxl_ok = sdxl_available()
        st.markdown(
            '<span class="badge badge-green">● SDXL READY</span>' if sdxl_ok
            else '<span class="badge badge-red">● SDXL OFFLINE</span>  `python sdxl_server.py`',
            unsafe_allow_html=True,
        )
        img_width    = st.slider("Width",    512, 1024,  512,  64, key="img_w")
        img_height   = st.slider("Height",   512, 1024,  512,  64, key="img_h")
        img_steps    = st.slider("Steps",     10,   60,   40,   5, key="img_steps")
        img_guidance = st.slider("Guidance",  1.0, 15.0,  7.0, 0.5, key="img_cfg")
        img_neg      = st.text_area(
            "Negative Prompt",
            value="blurry, low quality, watermark, text, ugly, deformed, "
                  "bad anatomy, oversaturated, duplicate, mutilated, "
                  "poorly drawn, bad proportions, extra limbs",
            height=80,
            key="img_neg",
            help="Tell SDXL what NOT to generate — big quality improvement",
        )
        st.session_state["_img_width"]    = img_width
        st.session_state["_img_height"]   = img_height
        st.session_state["_img_steps"]    = img_steps
        st.session_state["_img_guidance"] = img_guidance
        st.session_state["_img_neg"]      = img_neg
    st.divider()
    with st.expander("💬 Conversations", expanded=False):
        if st.session_state.get("messages"):
            if st.button("💾 Save current chat", use_container_width=True):
                saved_name = save_conversation(st.session_state.messages)
                st.success(f"Saved: {saved_name}")
        else:
            st.caption("Nothing to save yet — send a message first.")

        saved_chats = list_conversations()
        if saved_chats:
            # Show a friendly label but keep the filename as the value.
            pick = st.selectbox(
                "Saved chats",
                saved_chats,
                format_func=lambda f: f.replace(".json", "").replace("__", " · "),
                label_visibility="collapsed",
            )
            c_load, c_del = st.columns(2)
            with c_load:
                if st.button("📂 Load", use_container_width=True):
                    st.session_state.messages = load_conversation(pick)
                    st.rerun()
            with c_del:
                if st.button("🗑️ Delete", use_container_width=True):
                    delete_conversation(pick)
                    st.rerun()
        else:
            st.caption("No saved conversations yet.")

    st.divider()
    if st.button("🗑️ Reset", use_container_width=True):
        if os.path.exists(VECTOR_STORE_PATH):
            os.remove(VECTOR_STORE_PATH)
        st.session_state.clear()
        st.rerun()

# ─────────────────────────────────────────────
#  ACTIVE SOURCE STATUS BAR
# ─────────────────────────────────────────────
sources_active = []
if local_mode:
    sources_active.append('<span class="badge badge-purple">🔒 LOCAL</span>')
if "vector_store" in st.session_state:
    sources_active.append('<span class="badge badge-blue">📄 PDF</span>')
if "df" in st.session_state:
    sources_active.append('<span class="badge badge-green">📊 DATA</span>')
if use_web:
    sources_active.append('<span class="badge badge-yellow">🌐 WEB</span>')
if "image_analysis" in st.session_state:
    sources_active.append('<span class="badge badge-purple">🖼️ IMG</span>')
if sdxl_available():
    sources_active.append('<span class="badge badge-green">🎨 SDXL</span>')
if len(sources_active) == 0 or (len(sources_active) == 1 and "LOCAL" in sources_active[0]):
    sources_active.append('<span class="badge badge-red">⚠ NO SOURCES</span>')

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(" ".join(sources_active), unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHAT HISTORY
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_context and m.get("context_used"):
            with st.expander("📎 Context Used"):
                st.text(m["context_used"][:3000])

# ─────────────────────────────────────────────
#  CHAT INPUT & ORCHESTRATION
# ─────────────────────────────────────────────
_chat_placeholder = "Describe what to generate..." if image_mode else "Ask Loom anything..."
if prompt := st.chat_input(_chat_placeholder):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context_blocks = []
        context_labels = []
        now = datetime.datetime.now()
        current_time_str = now.strftime("%A, %B %d, %Y, %I:%M %p")

        def gather_context():
            blocks, labels = [], []
            if "vector_store" in st.session_state:
                try:
                    results = similarity_search(st.session_state.vector_store, prompt, k=top_k_docs)
                    pdf_ctx = "\n".join(
                        f"[Page {meta.get('page','?')}] {text}" for text, meta in results
                    )
                    blocks.append(f"--- PDF CONTEXT ---\n{pdf_ctx}")
                    labels.append(f"📄 PDF ({len(results)} chunks)")
                except Exception as e:
                    st.warning(f"PDF retrieval error: {e}")
            if "df" in st.session_state:
                df = st.session_state.df

                import re

                data_ctx = (
                    f"File: {st.session_state.get('df_name','data')}\n"
                    f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                    f"Columns: {', '.join(df.columns.tolist())}\n\n"
                )

                # ── Detect row-number lookup ("what's on row 82", "row 6658", etc.) ──
                row_lookup = re.search(r'\brow\s+(\d+)\b', prompt.lower())

                if row_lookup:
                    # Direct index lookup — row N in spreadsheet = index N-2 (0-based, skip header)
                    sheet_row = int(row_lookup.group(1))
                    df_idx    = sheet_row - 2  # spreadsheet row 2 = df index 0
                    if 0 <= df_idx < len(df):
                        row_data  = df.iloc[[df_idx]]
                        display   = row_data.copy()
                        display.insert(0, "ROW_NUMBER", sheet_row)
                        data_ctx += (
                            f"DIRECT ROW LOOKUP — spreadsheet row {sheet_row}:\n"
                            f"{display.to_string(index=False)}\n"
                        )
                        matched_rows = row_data
                    else:
                        data_ctx += (
                            f"Row {sheet_row} is out of range. "
                            f"Valid rows: 2 to {len(df) + 1}.\n"
                        )
                        matched_rows = df.iloc[0:0]  # empty

                else:
                    # ── Keyword/phrase search ─────────────────────────────────────────
                    def search_df(dataframe, terms, require_all=True):
                        def row_matches(row):
                            row_str = " ".join(str(v) for v in row).lower()
                            return all(t in row_str for t in terms) if require_all                                    else any(t in row_str for t in terms)
                        return dataframe[dataframe.apply(row_matches, axis=1)]

                    def phrase_search(dataframe, phrase):
                        def row_has_phrase(row):
                            return phrase in " ".join(str(v) for v in row).lower()
                        return dataframe[dataframe.apply(row_has_phrase, axis=1)]

                    quoted     = re.findall(r'"([^"]+)"', prompt)
                    unquoted   = re.sub(r'"[^"]+"', '', prompt)
                    # Filter out pure number tokens to avoid matching row numbers in data
                    terms      = quoted + [w.lower() for w in unquoted.split()
                                           if len(w) > 2 and not w.isdigit()]
                    noun_chunk = " ".join(terms).lower()

                    matched_rows = phrase_search(df, noun_chunk)
                    if matched_rows.empty and terms:
                        matched_rows = search_df(df, terms, require_all=True)
                    if matched_rows.empty and terms:
                        matched_rows = search_df(df, terms, require_all=False)

                    if not matched_rows.empty:
                        display = matched_rows.head(25).copy()
                        display.insert(0, "ROW_NUMBER", matched_rows.head(25).index + 2)
                        data_ctx += (
                            f"SEARCH RESULTS — {len(matched_rows)} row(s) matched:\n"
                            f"{display.to_string(index=False)}\n\n"
                            f"NOTE: ROW_NUMBER is the spreadsheet row (header = row 1).\n"
                        )
                    else:
                        data_ctx += (
                            f"No rows matched. Sample (first 20 rows):\n"
                            f"{df.head(20).to_string(index=False)}\n\n"
                            f"Statistics:\n{df.describe().to_string()}"
                        )

                blocks.append(f"--- SPREADSHEET DATA ---\n{data_ctx}")
                labels.append(f"📊 Data ({df.shape[0]}x{df.shape[1]}, {len(matched_rows)} match(es))")
            auto_web = needs_web_search(prompt)
            if use_web or auto_web:
                try:
                    web_results = search.run(prompt)
                    blocks.append(f"--- WEB SEARCH RESULTS ---\n{web_results}")
                    labels.append("🌐 Web (auto)" if auto_web and not use_web else "🌐 Web")
                except Exception:
                    st.warning("Web search timed out.")
            # Image silo — llava analysis injected as context automatically
            if "image_analysis" in st.session_state:
                img_ctx = (
                    f"File: {st.session_state.get('image_name', 'image')}\n"
                    f"{st.session_state.image_analysis}"
                )
                blocks.append(f"--- IMAGE ANALYSIS (llava) ---\n{img_ctx}")
                labels.append(f"🖼️ Image")
            return blocks, labels

        # ── IMAGE SEARCH PATH (find existing images on the web) ──────────────
        prev_img_prompt = st.session_state.get("last_image_prompt", "")
        new_image  = sdxl_ok and is_image_request(prompt)
        edit_image = (sdxl_ok and st.session_state.get("img_thread_active")
                      and bool(prev_img_prompt) and is_image_edit(prompt))
        auto_image = new_image or edit_image

        if is_image_search(prompt) and not auto_image:
            q = image_search_query(prompt)
            with st.status(f"🔎 Searching the web for images: _{q}_", expanded=True) as s:
                results = ddg_image_search(q, max_results=6)
                s.update(label=(f"✅ {len(results)} image(s) found" if results
                                else "No images found"), state="complete")
            if results:
                cols = st.columns(3)
                lines = [f"**Web image results for _{q}_:**\n"]
                for idx, r in enumerate(results):
                    thumb = r.get("thumbnail") or r.get("image")
                    link  = r.get("url", "#")
                    with cols[idx % 3]:
                        if thumb:
                            st.image(thumb, use_container_width=True)
                        st.caption(f"[{r.get('source', 'source')}]({link})")
                    lines.append(f"- [{r.get('title', 'image')}]({link})")
                full_res = "\n".join(lines)
            else:
                full_res = (f"_No web images found for '{q}'. "
                            "DuckDuckGo may be rate-limiting — try again in a moment._")
            st.session_state.messages.append({
                "role": "assistant", "content": full_res, "context_used": "",
            })

        # ── IMAGE GENERATION PATH ────────────────────────────────────────────
        # Fires when the user forces it with the toggle, when the message is
        # clearly an image request, OR when it's a follow-up tweak to an image
        # we just generated ("put it on a skateboard", "make it blue", ...).
        elif image_mode or auto_image:
            # For a follow-up edit, build on the previous prompt so tweaks stack
            # (e.g. dinosaur -> dinosaur on a skateboard -> ... in space).
            effective_prompt = (f"{prev_img_prompt}, {prompt}"
                                 if (edit_image and not new_image) else prompt)
            with st.status("🎨 Generating image...", expanded=True) as gen_status:
                gen_status.write(f"Prompt: _{effective_prompt}_")
                try:
                    result = call_sdxl(
                        prompt=effective_prompt,
                        width=st.session_state.get("_img_width", 512),
                        height=st.session_state.get("_img_height", 512),
                        steps=st.session_state.get("_img_steps", 40),
                        guidance=st.session_state.get("_img_guidance", 7.0),
                        negative_prompt=st.session_state.get("_img_neg", ""),
                    )
                    img_b64   = result["image"]
                    used_seed = result.get("seed", "?")
                    gen_status.update(label="✅ Image generated!", state="complete")

                    import base64 as _b64
                    img_bytes = _b64.b64decode(img_b64)
                    # Show at fixed width so it doesn't eat the whole screen
                    col_img, col_pad = st.columns([1, 1])
                    with col_img:
                        st.image(img_bytes, caption=f"seed: {used_seed}", width=320)

                    meta = (
                        f"**Seed:** {used_seed}  |  "
                        f"**Size:** {result.get('width','?')}x{result.get('height','?')}  |  "
                        f"**Steps:** {st.session_state.get('_img_steps', 30)}"
                    )
                    full_res = (
                        "**Generated Image**\n\n"
                        f"**Prompt:** {effective_prompt}\n\n"
                        f"{meta}\n\n"
                        f"![generated](data:image/png;base64,{img_b64})"
                    )
                    # Remember this image "thread" so the next message can be a
                    # follow-up edit that stacks on top of this prompt.
                    st.session_state["last_image_prompt"] = effective_prompt
                    st.session_state["img_thread_active"] = True
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
                    full_res = f"_Image generation error: {e}_"

            st.session_state.messages.append({
                "role": "assistant", "content": full_res, "context_used": "",
            })

        # ── ESSAY AGENT ───────────────────────────────────────────────────────
        elif is_essay_request(prompt):
            st.session_state["img_thread_active"] = False
            with st.status("🧶 Retrieving sources...", expanded=False) as src_status:
                context_blocks, context_labels = gather_context()
                src_status.update(
                    label=f"🧶 Sources: {', '.join(context_labels)}" if context_labels
                    else "Complete — No external sources",
                    state="complete",
                )
            full_context = "\n\n".join(context_blocks)
            try:
                full_res, combined_context = run_essay_agent(
                    topic=prompt, full_context=full_context, model=ST_MODEL,
                    temperature=temperature, use_web=use_web,
                    max_tokens=max_tokens, local_mode=local_mode,
                )
                st.markdown(full_res)
            except Exception as e:
                st.error(f"Essay agent error: {e}")
                full_res, combined_context = f"_Error: {e}_", ""

            if show_context and combined_context:
                with st.expander("📎 Context Used"):
                    st.text(combined_context[:3000])
            st.session_state.messages.append({
                "role": "assistant", "content": full_res, "context_used": combined_context,
            })

        # ── NORMAL CHAT ───────────────────────────────────────────────────────
        else:
            st.session_state["img_thread_active"] = False
            with st.status("🧶 Retrieving sources...", expanded=False) as status:
                context_blocks, context_labels = gather_context()
                status.update(
                    label=f"🧶 Sources: {', '.join(context_labels)}" if context_labels
                    else "Complete — No external sources",
                    state="complete",
                )

            privacy_note = "All processing is fully local and private. " if local_mode else ""
            system_instruction = (
                f"You are Loom, a high-precision AI orchestrator. "
                f"System clock: {current_time_str}. {privacy_note}\n\n"
                "Synthesize information from PDFs, spreadsheet data, and web search.\n\n"
                "Guidelines:\n"
                "1. CROSS-REFERENCE: Compare PDF and Web data; flag discrepancies.\n"
                "2. TEMPORAL AWARENESS: Note if PDF data appears outdated vs. web.\n"
                "3. CITATION: [PDF, Page N] for PDFs, [Title](URL) for web.\n"
                "4. TABLES: Use Markdown tables for structured comparisons.\n"
                "5. UNCERTAINTY: Say so if context is insufficient.\n"
                "6. CONCISENESS: Be precise and direct.\n"
            )

            full_context = "\n\n".join(context_blocks)
            history_for_api = build_api_history(
                st.session_state.messages[:-1],
                model=ST_MODEL, temperature=temperature, local_mode=local_mode,
            )
            enriched_prompt = prompt
            if full_context:
                enriched_prompt = f"RETRIEVED CONTEXT:\n{full_context}\n\n---\nUSER QUESTION: {prompt}"
            history_for_api.append({"role": "user", "content": enriched_prompt})

            res_placeholder = st.empty()
            full_res = ""

            # ── LOCAL streaming (Ollama) ──────────────────────────────────────
            if local_mode:
                try:
                    payload = {
                        "model": ST_MODEL,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            *history_for_api,
                        ],
                        "stream": True,
                        "options": {"temperature": temperature, "num_predict": max_tokens},
                    }
                    with requests.post(
                        f"{host_for_model(ST_MODEL)}/api/chat",
                        json=payload, stream=True, timeout=180,
                    ) as r:
                        if r.status_code != 200:
                            st.error(f"Ollama error {r.status_code}: {r.text}")
                            full_res = f"_Error: {r.status_code}_"
                        else:
                            for line in r.iter_lines():
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line.decode("utf-8"))
                                    token = chunk.get("message", {}).get("content", "")
                                    if token:
                                        full_res += token
                                        res_placeholder.markdown(full_res + "▌")
                                    if chunk.get("done"):
                                        break
                                except json.JSONDecodeError:
                                    continue
                            res_placeholder.markdown(full_res)
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach Ollama. Run: `ollama serve`")
                    full_res = "_Error: Ollama not running._"
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    full_res = f"_Error: {e}_"

            # ── CLOUD streaming (OpenRouter) ──────────────────────────────────
            else:
                try:
                    payload = {
                        "model": ST_MODEL,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            *history_for_api,
                        ],
                        "stream": True,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    headers = {
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://loom-ai.local",
                        "X-Title": "Loom AI",
                    }
                    with requests.post(
                        f"{OPENROUTER_BASE}/chat/completions",
                        json=payload, headers=headers, stream=True, timeout=120,
                    ) as r:
                        if r.status_code != 200:
                            st.error(f"OpenRouter returned {r.status_code}: {r.text}")
                            full_res = f"_Error: HTTP {r.status_code}_"
                        else:
                            for line in r.iter_lines():
                                if not line:
                                    continue
                                decoded = line.decode("utf-8")
                                if decoded.startswith("data: "):
                                    decoded = decoded[6:]
                                if decoded.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(decoded)
                                    choices = chunk.get("choices", [])
                                    token = choices[0].get("delta", {}).get("content", "") if choices else ""
                                    if token:
                                        full_res += token
                                        res_placeholder.markdown(full_res + "▌")
                                except json.JSONDecodeError:
                                    continue
                            res_placeholder.markdown(full_res)
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach OpenRouter. Check internet connection.")
                    full_res = "_Error: Connection failed._"
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out.")
                    full_res = "_Error: Timeout._"
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    full_res = f"_Error: {e}_"

            if show_context and full_context:
                with st.expander("📎 Context Used"):
                    st.text(full_context[:3000])

            st.session_state.messages.append({
                "role": "assistant", "content": full_res, "context_used": full_context,
            })
