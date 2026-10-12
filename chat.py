"""
Interactive SPIT RAG chatbot (terminal). Shows retrieved chunks + scores + sources, then the grounded answer.

    python chat.py                 # uses the default index
    python chat.py --tag <tag>     # a specific index under index/
In-chat commands:  :k 5   :mode vector|mmr|hybrid   :thr 0.55 (relevance threshold, :thr off)   :quit
"""
import argparse

import config as C
import rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=rag.make_tag(C.EMBED_MODEL, C.CHUNK_STRATEGY, C.CHUNK_TOKENS, C.CHUNK_OVERLAP))
    ap.add_argument("--k", type=int, default=C.DEFAULT_K)
    a = ap.parse_args()

    store = rag.Store.load(a.tag)
    emb = rag.Embedder(store.meta["embed_model"])
    k, mode, thr = a.k, "vector", None
    print(f"SPIT RAG chatbot | index={a.tag} | {len(store.chunks)} chunks | LLM={C.LLM_PROVIDER}:{C.LLM_MODEL}")
    print("Ask a question about SPIT (':quit' to exit).\n")

    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q in (":quit", ":q", "exit"):
            break
        if q.startswith(":k "):
            k = int(q.split()[1]); print(f"k = {k}"); continue
        if q.startswith(":mode "):
            mode = q.split()[1]; print(f"mode = {mode}"); continue
        if q.startswith(":thr "):
            v = q.split()[1]; thr = None if v == "off" else float(v); print(f"threshold = {thr}"); continue

        out = rag.rag_answer(q, store, emb, k=k, mode=mode, min_score=thr)
        print("\n--- Retrieved evidence (top-%d, mode=%s) ---" % (k, mode))
        rag.show_results(out["results"], max_chars=400)
        print("--- Answer ---")
        print(out["answer"])
        srcs = sorted({f"{r['chunk']['source']} ({r['chunk']['url'] or 'no url'})" for r in out["results"]})
        print("\nSources retrieved:", "; ".join(srcs) if srcs else "none")
        print(f"[retrieval {out['t_retrieve']:.2f}s | generation {out['t_generate']:.2f}s]\n")


if __name__ == "__main__":
    main()
