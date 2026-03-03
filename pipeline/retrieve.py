import json
import os
import re

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL    = "text-embedding-3-small"   # must match embed.py

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
STORE_DIR  = os.path.join(BASE_DIR, "data", "vector_store")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
META_PATH  = os.path.join(STORE_DIR, "metadata.json")

# How many candidates to return from FAISS before reranking
DEFAULT_TOP_K    = 10
CANDIDATE_FACTOR = 2    # retrieve 2× so the reranker has room to re-order



_URL_RE = re.compile(
    r"^https?://"          # starts with http:// or https://
    r"[\w\-.~]+"           # domain
    r"(?::\d+)?"           # optional port
    r"(?:/[^\s]*)?\s*$",   # optional path
    re.IGNORECASE,
)


def is_url(text: str) -> bool:
    """Return True if the query looks like a URL."""
    return bool(_URL_RE.match(text.strip()))


def extract_text_from_url(url: str) -> str:
    
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 100:
                return text.strip()
    except Exception:
        pass

    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 100:
            return text[:4000]
    except Exception:
        pass

    return ""


def resolve_query(query: str) -> str:
    
    query = query.strip()
    if is_url(query):
        extracted = extract_text_from_url(query)
        return extracted if extracted else query
    return query


def _embed_query(text: str) -> np.ndarray:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Cannot embed query.")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    vec = np.array([response.data[0].embedding], dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec



class Retriever:
    def __init__(
        self,
        index_path: str = INDEX_PATH,
        meta_path:  str = META_PATH,
    ):
        print(f"[Retriever] Using OpenAI '{EMBED_MODEL}' for query embedding …")

        print(f"[Retriever] Loading FAISS index from {index_path} …")
        self.index = faiss.read_index(index_path)

        print(f"[Retriever] Loading metadata from {meta_path} …")
        with open(meta_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[Retriever] Ready — {self.index.ntotal} vectors indexed.")

    def search(
        self,
        query:  str,
        top_k:  int = DEFAULT_TOP_K,
        expand: bool = True,
    ) -> list[dict]:
        
        resolved = resolve_query(query)
        k_fetch  = (top_k * CANDIDATE_FACTOR) if expand else top_k
        k_fetch  = min(k_fetch, self.index.ntotal)

        # Embed query using OpenAI
        query_vec = _embed_query(resolved)

        # FAISS search
        scores, indices = self.index.search(query_vec, k_fetch)

        # Build result list
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:       # FAISS pads with -1 when fewer results exist
                continue
            item = dict(self.metadata[idx])
            item["score"] = round(float(score), 4)
            results.append(item)

        return results   # already sorted by score descending

if __name__ == "__main__":
    retriever = Retriever()

    QUERIES = [
        "cognitive ability test for a software engineer",
        "personality assessment for sales manager",
        "Java developer with OOP and data structures skills",
        "verbal ability English comprehension communication skills",
    ]

    for q in QUERIES:
        print(f"\n{'─'*60}")
        kind = "URL" if is_url(q) else "TEXT"
        print(f"[{kind}] Query: {q[:70]}")
        results = retriever.search(q, top_k=5)
        for rank, r in enumerate(results, 1):
            print(f"  #{rank}  score={r['score']:.4f}  [{r['test_type']}] {r['name']}")
