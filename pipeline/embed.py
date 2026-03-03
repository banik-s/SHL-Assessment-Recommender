import json
import os
import sys
import time

import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.ingest import load_and_prepare

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL     = "text-embedding-3-small"   # 1536 dims, best quality/cost
EMBED_DIM       = 1536
BATCH_SIZE      = 100                 # OpenAI allows up to 2048 texts per call

BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
STORE_DIR   = os.path.join(BASE_DIR, "data", "vector_store")
INDEX_PATH  = os.path.join(STORE_DIR, "index.faiss")
META_PATH   = os.path.join(STORE_DIR, "metadata.json")
MODEL_PATH  = os.path.join(STORE_DIR, "embed_model.txt")  # records which model was used

# Fields to drop before saving metadata (keep file lean)
STRIP_FIELDS = {"composite_text"}



def embed_with_openai(texts: list[str]) -> np.ndarray:
    
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set — cannot use OpenAI embeddings.")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    all_embeddings = []
    print(f"  Embedding {len(texts)} texts with '{EMBED_MODEL}' (batch_size={BATCH_SIZE}) …")
    t0 = time.time()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Batches"):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)

    elapsed = time.time() - t0
    arr = np.array(all_embeddings, dtype="float32")

    # L2-normalise so cosine similarity == inner product (required for IndexFlatIP)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0, 1, norms)

    print(f"  Done in {elapsed:.1f}s | shape: {arr.shape} | dtype: {arr.dtype}")
    return arr


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index built | dim={dim} | ntotal={index.ntotal}")
    return index


def save_metadata(assessments: list[dict]) -> None:
    """Save assessment metadata (without composite_text) as JSON."""
    meta = [
        {k: v for k, v in a.items() if k not in STRIP_FIELDS}
        for a in assessments
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved → {META_PATH}")


def main():
    print("=" * 55)
    print("SHL Assessment Embedding Pipeline (OpenAI)")
    print("=" * 55)

    if not OPENAI_API_KEY:
        print("\n[ERROR] OPENAI_API_KEY is not set in your .env file.")
        print("        Cannot build OpenAI embeddings without an API key.")
        sys.exit(1)

    # ── 1. Load & prepare ──────────────────────────────────────────────────
    print("\n[1/4] Loading assessments from ingest …")
    assessments = load_and_prepare()
    texts       = [a["composite_text"] for a in assessments]
    print(f"  {len(assessments)} assessments ready | avg text len: {sum(len(t) for t in texts)//len(texts)} chars")

    # ── 2. Embed with OpenAI ───────────────────────────────────────────────
    print(f"\n[2/4] Embedding with OpenAI '{EMBED_MODEL}' …")
    embeddings = embed_with_openai(texts)

    # ── 3. Build FAISS index & save ────────────────────────────────────────
    print("\n[3/4] Building FAISS index …")
    os.makedirs(STORE_DIR, exist_ok=True)
    index = build_faiss_index(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"  FAISS index saved → {INDEX_PATH}")

    # ── 4. Save metadata & model record ─────────────────────────────────────
    print("\n[4/4] Saving metadata …")
    save_metadata(assessments)

    # Record which embedding model was used (so retrieve.py knows)
    with open(MODEL_PATH, "w") as f:
        f.write(EMBED_MODEL)
    print(f"  Embedding model recorded → {MODEL_PATH}")

    # ── Quick sanity check ─────────────────────────────────────────────────
    print("\n── Sanity check: query 'Python programming test for developers' ──")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=["Python programming test for developers"])
    qvec = np.array([resp.data[0].embedding], dtype="float32")
    qvec /= np.linalg.norm(qvec)
    scores, ids = index.search(qvec, 5)
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
        print(f"  #{rank}  score={score:.4f}  {assessments[idx]['name']}")

    print("\n" + "=" * 55)
    print(f"Done! Vector store rebuilt with {EMBED_MODEL}.")
    print(f"Saved to: {STORE_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
