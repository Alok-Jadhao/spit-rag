"""
Tasks 2-5: read knowledge_base/ -> clean -> chunk -> embed -> FAISS index (saved under index/<tag>/).

    python build_index.py
    python build_index.py --strategy fixed                 # second chunking strategy (advanced challenge)
    python build_index.py --embed-model sentence-transformers/all-MiniLM-L6-v2
    python build_index.py --chunk-tokens 300 --overlap 40  # chunk-size study (analysis question 5)
    python build_index.py --keep-cols col1 col2            # ONLY if ma'am confirms these ERP columns are safe
"""
import argparse

import config as C
import rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=C.CHUNK_STRATEGY, choices=["sentence", "fixed"])
    ap.add_argument("--chunk-tokens", type=int, default=C.CHUNK_TOKENS)
    ap.add_argument("--overlap", type=int, default=C.CHUNK_OVERLAP)
    ap.add_argument("--embed-model", default=C.EMBED_MODEL)
    ap.add_argument("--keep-cols", nargs="*", default=[])
    a = ap.parse_args()

    emb = rag.Embedder(a.embed_model)
    store, tag = rag.build_store(emb, a.strategy, a.chunk_tokens, a.overlap, keep_cols=tuple(a.keep_cols))
    print(f"\nTag: {tag}")
    print("Chunk record format  chunk_id | source | page/section | text  ->  index/%s/chunks.csv" % tag)
    for c in store.chunks[:3]:
        print(f"  {c['chunk_id']} | {c['source']} | {c['page_or_section']} | {c['text'][:80].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
