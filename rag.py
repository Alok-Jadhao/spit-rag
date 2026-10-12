"""
rag.py - the core RAG logic, written out explicitly (no one-line RAG chain):

    Task 2  extract + clean text           -> extract_pdf(), text_to_units(), erp_units(), load_corpus()
    Task 3  chunking                       -> chunk_doc(), build_chunks()
    Task 4  embeddings                     -> Embedder
    Task 5  FAISS index + id->chunk map    -> Store
    Task 6  retrieval (vector / MMR / hybrid, optional score threshold) -> retrieve()
    Task 7  RAG prompt                     -> SYSTEM_PROMPT, build_context(), build_prompt()
    Task 8  generation                     -> llm(), rag_answer()
    Exp. D  ICL baseline                   -> icl_answer()
"""
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

import config as C

REFUSAL = "I could not find this information in the SPIT knowledge base."


# =====================================================================================
# helpers
# =====================================================================================
def approx_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per whitespace word). Good enough for chunk sizing."""
    return int(len(text.split()) * 1.3) + 1


_REFUSAL_RE = re.compile(
    r"could not find this information|not (?:available|mentioned|specified|provided|stated|contained|present)"
    r"|no (?:such )?information|does not (?:contain|mention|provide|specify)|cannot (?:find|answer)"
    r"|don't have (?:enough )?information|unable to find", re.I)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer or ""))


def load_metadata() -> dict:
    """filename -> metadata row (source_name, source_url, ...) from knowledge_base/metadata.csv"""
    if not C.META_CSV.exists():
        print("WARNING: metadata.csv not found - run build_kb.py first. Sources will fall back to file names.")
        return {}
    with open(C.META_CSV, newline="", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


# =====================================================================================
# Task 2 - extract and clean text
# =====================================================================================
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“•])|\s*•\s*")


def split_sentences(text: str):
    return [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]


def extract_pdf(path: Path):
    """Return [(page_no, cleaned_text)], removing repeated headers/footers and bare page numbers."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i, txt.replace("\xa0", " ").replace("\u200b", "")))

    def key(line):  # digits are ignored ONLY in "Page 3 of 10"-style lines; other lines must match exactly
        line = line.strip()
        return re.sub(r"\d+", "#", line) if re.search(r"\bpage\b", line, re.I) else line

    repeated = set()
    if len(pages) >= 3:  # a line that appears on >=50% of pages is a header/footer
        counter = Counter()
        for _, t in pages:
            for ln in {key(l) for l in t.splitlines() if l.strip()}:
                counter[ln] += 1
        thr = max(3, int(0.5 * len(pages)))
        repeated = {ln for ln, c in counter.items() if c >= thr and len(ln) < 120}

    out = []
    for pno, t in pages:
        keep = []
        for l in t.splitlines():
            ls = l.strip()
            if not ls or _PAGE_NUM_RE.match(ls) or key(ls) in repeated:
                continue
            keep.append(re.sub(r"[ \t]+", " ", ls))
        out.append((pno, "\n".join(keep)))
    return out


def pdf_units(path: Path):
    """PDF -> list of (sentence, page_no)"""
    units = []
    for pno, txt in extract_pdf(path):
        para = re.sub(r"\s*\n\s*", " ", txt)
        units += [(s, pno) for s in split_sentences(para)]
    return units


def text_to_units(text: str, default_label="page"):
    """Cleaned web text -> list of (line_or_sentence, section_heading). '## ' lines are headings."""
    units, label = [], default_label
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            label = line[3:].strip() or label
            units.append((label, label))
            continue
        if len(line.split()) > 60:                     # long paragraph -> sentences
            units += [(s, label) for s in split_sentences(line)]
        else:
            units.append((line, label))
    return units


# ---- ERP (anonymized export) ------------------------------------------------------
SENSITIVE_TOKENS = {"roll", "rollno", "phone", "mobile", "email", "mail", "marks", "mark", "grade", "cgpa",
                    "sgpa", "gpa", "attendance", "password", "passwd", "pwd", "credential", "credentials",
                    "aadhaar", "aadhar", "dob", "birth", "pan"}
SENSITIVE_EXACT = {"name", "student_name", "studentname", "student name", "full_name", "fullname"}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)")


def is_sensitive(col) -> bool:
    c = str(col).strip().lower()
    return c in SENSITIVE_EXACT or any(t in SENSITIVE_TOKENS for t in re.split(r"[^a-z0-9]+", c))


def erp_units(path: Path, keep_cols=()):
    """CSV/JSON -> one text unit per row. Columns that look like personal data are DROPPED (privacy rule)."""
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        df = pd.json_normalize(data)
    drop = [c for c in df.columns if is_sensitive(c) and c not in keep_cols]
    if drop:
        print(f"  [privacy] {path.name}: dropped columns {drop}  (use keep_cols=[...] only if ma'am confirms they are safe)")
        df = df.drop(columns=drop)
    units = []
    for i, row in df.iterrows():
        parts = [f"{c}: {v}" for c, v in row.items() if pd.notna(v) and str(v).strip()]
        text = f"Record {i + 1}: " + "; ".join(parts) + "."
        units.append((text, i + 1))
    return units


def redact_pii(text: str) -> str:
    return _PHONE_RE.sub("[REDACTED-PHONE]", _EMAIL_RE.sub("[REDACTED-EMAIL]", text))


def load_corpus(keep_cols=()):
    """Read every file in knowledge_base/ -> list of docs with (text_unit, label) lists + source metadata."""
    meta = load_metadata()
    docs = []

    def add(path, doc_type, units):
        m = meta.get(path.name, {})
        if not units:
            print(f"  ! no text extracted from {path.name} (scanned PDF? needs OCR) - skipped")
            return
        docs.append(dict(filename=path.name, doc_type=doc_type, units=units,
                         source=m.get("source_name") or path.name, url=m.get("source_url", "")))

    for p in sorted(C.WEB_DIR.glob("*.txt")):
        add(p, "web", text_to_units(p.read_text(encoding="utf-8", errors="ignore")))
    for p in sorted(C.PDF_DIR.glob("*.pdf")):
        add(p, "pdf", pdf_units(p))
    for p in sorted(C.PDF_DIR.glob("*.txt")):
        add(p, "web", text_to_units(p.read_text(encoding="utf-8", errors="ignore")))
    # ERP dataset is not available for this experiment. Keep ingestion disabled until an approved, anonymized export is provided.
    # for p in sorted(C.ERP_DIR.glob("*")):
    #     if p.suffix.lower() in (".csv", ".json"):
    #         add(p, "erp", erp_units(p, keep_cols))
    #     elif p.suffix.lower() == ".pdf":
    #         units = [(redact_pii(s), pg) for s, pg in pdf_units(p)]
    #         add(p, "pdf", units)
    #     elif p.suffix.lower() == ".txt":
    #         add(p, "web", text_to_units(redact_pii(p.read_text(encoding="utf-8", errors="ignore"))))
    return docs


# =====================================================================================
# Task 3 - chunking  (record format: chunk_id | source | page/section | text)
# =====================================================================================
def _explode(units, size):
    """Split any single unit that is longer than `size` tokens into word windows."""
    max_words = max(20, int(size / 1.3))
    out = []
    for text, label in units:
        w = text.split()
        if len(w) <= max_words:
            out.append((text, label))
        else:
            out += [(" ".join(w[i:i + max_words]), label) for i in range(0, len(w), max_words)]
    return out


def _pack_sentences(units, size, overlap):
    """Strategy 'sentence': greedily pack whole sentences/lines up to `size` tokens; carry ~`overlap` tokens over."""
    chunks, cur, cur_tok = [], [], 0
    for u in units:
        t = approx_tokens(u[0])
        if cur and cur_tok + t > size:
            chunks.append(cur)
            keep, tok = [], 0
            for prev in reversed(cur):
                pt = approx_tokens(prev[0])
                if tok + pt > overlap:
                    break
                keep.insert(0, prev)
                tok += pt
            cur, cur_tok = keep, tok
        cur.append(u)
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


def _fixed_windows(units, size, overlap):
    """Strategy 'fixed': naive sliding window over words (ignores sentence boundaries) - the baseline to compare."""
    words, labels = [], []
    for text, label in units:
        w = text.split()
        words += w
        labels += [label] * len(w)
    W = max(20, int(size / 1.3))
    step = max(1, W - int(overlap / 1.3))
    chunks = []
    for s in range(0, len(words), step):
        seg = words[s:s + W]
        if not seg:
            break
        chunks.append([(" ".join(seg), labels[s])])
        if s + W >= len(words):
            break
    return chunks


def _fmt_label(labels, doc_type):
    first, last = labels[0], labels[-1]
    if doc_type == "pdf":
        return f"p.{first}" if first == last else f"p.{first}-{last}"
    if doc_type == "erp":
        return f"rows {first}-{last}" if first != last else f"row {first}"
    return str(first)


def chunk_doc(doc, strategy=None, size=None, overlap=None):
    strategy = strategy or C.CHUNK_STRATEGY
    size = size or C.CHUNK_TOKENS
    overlap = C.CHUNK_OVERLAP if overlap is None else overlap
    if doc["doc_type"] == "erp":
        overlap = 0                                   # independent table rows: no overlap needed
    units = _explode(doc["units"], size)
    groups = _pack_sentences(units, size, overlap) if strategy == "sentence" else _fixed_windows(units, size, overlap)
    joiner = " " if doc["doc_type"] == "pdf" else "\n"
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(doc["filename"]).stem)[:40]
    out = []
    for n, g in enumerate(groups):
        out.append(dict(
            chunk_id=f"{stem}__{n:03d}",
            source=doc["source"],
            page_or_section=_fmt_label([x[1] for x in g], doc["doc_type"]),
            text=joiner.join(x[0] for x in g),
            filename=doc["filename"],
            url=doc["url"],
            doc_type=doc["doc_type"],
        ))
    return out


def build_chunks(docs, strategy=None, size=None, overlap=None):
    chunks = []
    for d in docs:
        chunks += chunk_doc(d, strategy, size, overlap)
    return chunks


def embed_input(chunk) -> str:
    """Text actually embedded: the stored chunk text with its title/section prepended (helps retrieval)."""
    return f"{chunk['source']} | {chunk['page_or_section']}\n{chunk['text']}"


# =====================================================================================
# Task 4 - embeddings
# =====================================================================================
class Embedder:
    """Ollama embeddings or a local Sentence-Transformers model; vectors are normalized."""

    def __init__(self, name=None):
        raw_name = name or C.EMBED_MODEL
        lower_name = raw_name.lower()
        # Accept both the explicit ollama:model form and the old mistaken
        # HF-style value that users may already have exported.
        if lower_name.startswith("ollama:"):
            self.backend = "ollama"
            self.ollama_model = raw_name.split(":", 1)[1]
        elif lower_name.startswith("sentence-transformers/nomic-embed-text"):
            self.backend = "ollama"
            self.ollama_model = raw_name.rsplit("/", 1)[-1]
        elif "/" not in raw_name and (":" in raw_name or lower_name.startswith("nomic-embed-text")):
            # Ollama model tags (for example nomic-embed-text:latest) have no namespace slash.
            self.backend = "ollama"
            self.ollama_model = raw_name
        else:
            self.backend = "sentence-transformers"
            self.ollama_model = None
        self.name = f"ollama:{self.ollama_model}" if self.backend == "ollama" else raw_name
        self.model = None
        self.q_prefix = self.d_prefix = ""
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name, device="cpu")
            self.model.max_seq_length = 512
            n = self.name.lower()
            if "bge" in n:
                self.q_prefix, self.d_prefix = "Represent this sentence for searching relevant passages: ", ""
            elif "e5" in n:
                self.q_prefix, self.d_prefix = "query: ", "passage: "

    def _ollama_encode(self, texts):
        import requests
        vectors = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = requests.post(
                f"{C.OLLAMA_BASE_URL}/api/embed",
                json={"model": self.ollama_model, "input": batch},
                timeout=300,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Ollama embedding request failed for {self.ollama_model!r}. "
                    f"Check that Ollama is running and the model is installed (ollama list). "
                    f"Server response: {response.text[:500]}"
                ) from exc
            vectors.extend(response.json().get("embeddings", []))
        if len(vectors) != len(texts):
            raise RuntimeError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} inputs")
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def encode_docs(self, texts, batch_size=32):
        if self.backend == "ollama":
            return self._ollama_encode(["search_document: " + t for t in texts])
        v = self.model.encode([self.d_prefix + t for t in texts], batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return np.asarray(v, dtype="float32")

    def encode_query(self, q):
        if self.backend == "ollama":
            return self._ollama_encode(["search_query: " + q])[0]
        v = self.model.encode([self.q_prefix + q], normalize_embeddings=True)
        return np.asarray(v, dtype="float32")[0]


# =====================================================================================
# Task 5 - vector index + (position -> chunk/metadata) mapping
# =====================================================================================
def make_tag(embed_model, strategy, size, overlap):
    return f"{embed_model.split('/')[-1]}_{strategy}_{size}_{overlap}"


class Store:
    """FAISS IndexFlatIP + parallel list `chunks` : chunks[i] is the chunk (with metadata) of vector i."""

    def __init__(self, chunks, emb, meta):
        self.chunks, self.emb, self.meta = chunks, np.asarray(emb, dtype="float32"), dict(meta)
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)
        self._bm25 = None

    # -- persistence
    def save(self, tag):
        d = C.INDEX_DIR / tag
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        np.save(d / "embeddings.npy", self.emb)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        pd.DataFrame(self.chunks)[["chunk_id", "source", "page_or_section", "text", "url"]] \
            .to_csv(d / "chunks.csv", index=False)       # the 'chunk_id | source | page/section | text' record
        return d

    @classmethod
    def load(cls, tag):
        d = C.INDEX_DIR / tag
        chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        s = cls(chunks, np.load(d / "embeddings.npy"), json.loads((d / "meta.json").read_text(encoding="utf-8")))
        if (d / "index.faiss").exists():
            s.index = faiss.read_index(str(d / "index.faiss"))
        return s

    # -- derived stores (used by Experiments B and C, no re-embedding needed)
    def subset(self, keep_fn):
        idx = [i for i, c in enumerate(self.chunks) if keep_fn(c)]
        return Store([self.chunks[i] for i in idx], self.emb[idx], self.meta)

    def extend(self, new_chunks, new_emb):
        return Store(self.chunks + list(new_chunks), np.vstack([self.emb, new_emb]), self.meta)

    # -- BM25 for hybrid retrieval
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_tok(c["text"] + " " + c["source"]) for c in self.chunks])
        return self._bm25


def _tok(s):
    return re.findall(r"\w+", s.lower())


def build_store(embedder=None, strategy=None, size=None, overlap=None, keep_cols=(), save=True, verbose=True):
    strategy = strategy or C.CHUNK_STRATEGY
    size = size or C.CHUNK_TOKENS
    overlap = C.CHUNK_OVERLAP if overlap is None else overlap
    embedder = embedder or Embedder()
    docs = load_corpus(keep_cols)
    chunks = build_chunks(docs, strategy, size, overlap)
    if verbose:
        toks = [approx_tokens(c["text"]) for c in chunks]
        print(f"{len(docs)} documents -> {len(chunks)} chunks "
              f"(avg {np.mean(toks):.0f} tokens, max {max(toks)}) | strategy={strategy} size={size} overlap={overlap}")
    emb = embedder.encode_docs([embed_input(c) for c in chunks])
    store = Store(chunks, emb, dict(embed_model=embedder.name, strategy=strategy, chunk_tokens=size,
                                    overlap=overlap, n_chunks=len(chunks)))
    tag = make_tag(embedder.name, strategy, size, overlap)
    if save:
        d = store.save(tag)
        if verbose:
            print(f"index saved to {d}")
    return store, tag


# =====================================================================================
# Task 6 - retrieval
# =====================================================================================
def retrieve(store: Store, embedder: Embedder, query: str, k=3, mode="vector", min_score=None,
             mmr_lambda=0.5, fetch_k=None):
    """
    Returns a list of dicts: {rank, score (cosine), idx, chunk}.
      mode="vector": plain top-k by cosine similarity
      mode="mmr"   : Maximal Marginal Relevance (relevance vs. redundancy)
      mode="hybrid": BM25 keyword + vector search merged with Reciprocal Rank Fusion
      min_score    : drop results whose cosine < min_score (relevance threshold -> may return [])
    """
    k = min(k, len(store.chunks))
    q = embedder.encode_query(query)
    cos = store.emb @ q                                  # cosine sims to ALL chunks (embeddings are normalised)
    if mode == "vector":
        scores, ids = store.index.search(q[None, :], k)  # <- the FAISS search
        order = [int(i) for i in ids[0] if i >= 0]
    elif mode == "mmr":
        n = min(max(fetch_k or 4 * k, k), len(store.chunks))
        _, ids = store.index.search(q[None, :], n)
        cand = [int(i) for i in ids[0] if i >= 0]
        order = []
        while cand and len(order) < k:
            best, best_val = None, -1e9
            for i in cand:
                red = max((float(store.emb[i] @ store.emb[j]) for j in order), default=0.0)
                val = mmr_lambda * float(cos[i]) - (1 - mmr_lambda) * red
                if val > best_val:
                    best, best_val = i, val
            order.append(best)
            cand.remove(best)
    elif mode == "hybrid":
        n = min(max(20, 4 * k), len(store.chunks))
        vec_rank = list(np.argsort(-cos)[:n])
        bm = store.bm25().get_scores(_tok(query))
        bm_rank = list(np.argsort(-bm)[:n])
        rrf = {}
        for ranking in (vec_rank, bm_rank):
            for r, i in enumerate(ranking):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + r)
        order = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])[:k]]
    else:
        raise ValueError(mode)

    results = []
    for i in order:
        if min_score is not None and cos[i] < min_score:
            continue
        results.append(dict(rank=len(results) + 1, score=float(cos[i]), idx=i, chunk=store.chunks[i]))
    return results


def show_results(results, max_chars=500):
    """Pretty-print retrieved chunks with similarity scores (use for the required screenshots)."""
    if not results:
        print("(no chunk passed the relevance threshold)")
    for r in results:
        c = r["chunk"]
        print(f"[{r['rank']}] score={r['score']:.4f} | {c['source']} | {c['page_or_section']} | {c['chunk_id']}")
        print(f"    url: {c['url']}")
        t = c["text"].replace("\n", " ")
        print("    " + (t[:max_chars] + " ..." if len(t) > max_chars else t) + "\n")


# =====================================================================================
# Task 7 - prompt construction
# =====================================================================================
SYSTEM_PROMPT = f"""You are the SPIT Assistant, a question-answering chatbot for Sardar Patel Institute of Technology (SPIT), Mumbai.
Rules:
1. Answer ONLY from the passages in the CONTEXT block. Do not use outside knowledge, even if you think you know the answer.
2. If the context does not contain the answer, reply with exactly: "{REFUSAL}"
3. If the context answers only part of the question, give the supported part and say what is missing.
4. If passages contradict each other, say so and name the sources instead of silently choosing one.
5. Cite the passages you used with their tags, e.g. [S1], [S2].
6. Be concise. Never output personal data about individuals."""


def build_context(results) -> str:
    blocks = []
    for r in results:
        c = r["chunk"]
        blocks.append(f"[S{r['rank']}] source: {c['source']} | location: {c['page_or_section']} | url: {c['url'] or 'n/a'}\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(question, results) -> str:
    return f"CONTEXT:\n{build_context(results)}\n\nQUESTION: {question}\n\nANSWER (use only the context; cite [S#]):"


# =====================================================================================
# Task 8 - generation
# =====================================================================================
_client = {}


def llm(system: str, user: str, max_tokens=None) -> str:
    """Single place that talks to the approved LLM. Swap provider via config / env vars."""
    max_tokens = max_tokens or C.LLM_MAX_TOKENS
    last = None
    for attempt in range(3):
        try:
            if C.LLM_PROVIDER == "anthropic":
                if "a" not in _client:
                    import anthropic
                    _client["a"] = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
                r = _client["a"].messages.create(model=C.LLM_MODEL, max_tokens=max_tokens, temperature=0,
                                                 system=system, messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
            else:                                                 # OpenAI-compatible (OpenAI, Ollama, Groq, vLLM ...)
                if "o" not in _client:
                    from openai import OpenAI
                    _client["o"] = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", C.LLM_BASE_URL),
                                          api_key=os.getenv("OPENAI_API_KEY", "ollama"))
                r = _client["o"].chat.completions.create(
                    model=C.LLM_MODEL, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after 3 attempts: {last}")


def rag_answer(question, store, embedder, k=3, mode="vector", min_score=None, **ret_kw):
    t0 = time.perf_counter()
    results = retrieve(store, embedder, question, k=k, mode=mode, min_score=min_score, **ret_kw)
    t1 = time.perf_counter()
    if not results:                                   # nothing passed the threshold -> refuse without calling the LLM
        answer, prompt = REFUSAL, ""
    else:
        prompt = build_prompt(question, results)
        answer = llm(SYSTEM_PROMPT, prompt)
    t2 = time.perf_counter()
    return dict(question=question, answer=answer, results=results, context=build_context(results), prompt=prompt,
                t_retrieve=t1 - t0, t_generate=t2 - t1, t_total=t2 - t0)


# =====================================================================================
# Experiment D - In-Context Learning baseline (no retrieval)
# =====================================================================================
ICL_SYSTEM = "You are an assistant that answers questions about Sardar Patel Institute of Technology (SPIT), Mumbai."


def icl_answer(question, icl_prompt: str):
    """`icl_prompt` = the few-shot / fact-sheet prompt from your previous ICL experiment (icl_prompt.txt)."""
    t0 = time.perf_counter()
    answer = llm(ICL_SYSTEM, f"{icl_prompt}\n\nQ: {question}\nA:")
    dt = time.perf_counter() - t0
    return dict(question=question, answer=answer, results=[], context=icl_prompt, t_retrieve=0.0,
                t_generate=dt, t_total=dt)
