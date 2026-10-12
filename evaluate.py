"""
Evaluation + experiments.

    python evaluate.py --exp A                      # Top-k study: k = 1, 3, 5, 10
    python evaluate.py --exp B --remove "about/"    # retrieval failure: delete the source holding the answers
    python evaluate.py --exp C                      # irrelevant / conflicting context (distractors/ folder)
    python evaluate.py --exp D                      # RAG vs ICL
    python evaluate.py --exp ADV                    # chunking / embedding / MMR / hybrid / threshold (retrieval only)
    python evaluate.py --exp ALL --judge            # A, C, D, ADV
Add --judge to let the LLM grade correctness + faithfulness (otherwise: keyword match + refusal check).
LLM judging is a convenience - spot-check some rows by hand and fill the Observation Sheet yourself.

qa_set.csv columns
  qid, type, question, answerable (1/0), gold_answer,
  gold_source   - substring(s) of the source name / URL / filename that contain the answer ('|' = alternatives)
  gold_keywords - facts the answer must contain: ';' separates required facts, '|' separates alternative wordings
                  e.g.  "Munshi Nagar;Andheri"   or   "University of Mumbai|Mumbai University"
  exp_c         - (optional) 1 = use this question in Experiment C
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import rag
from rag import Embedder, Store

_EMB = {}


def get_embedder(name):
    if name not in _EMB:
        _EMB[name] = Embedder(name)
    return _EMB[name]


def default_tag():
    return rag.make_tag(C.EMBED_MODEL, C.CHUNK_STRATEGY, C.CHUNK_TOKENS, C.CHUNK_OVERLAP)


def load_qa(path):
    df = pd.read_csv(path).fillna("")
    df["answerable"] = df["answerable"].astype(int)
    return df


# =====================================================================================
# metrics
# =====================================================================================
def match_source(chunk, gold):
    """Does this chunk come from a gold source? (substring match on source name / url / filename)"""
    if not gold:
        return False
    hay = " ".join([chunk.get("source", ""), chunk.get("url", ""), chunk.get("filename", "")]).lower()
    return any(g.strip().lower() in hay for g in str(gold).split("|") if g.strip())


def kw_match(text, kw):
    """True/False, or None if the question has no gold_keywords."""
    if not kw:
        return None
    t = text.lower()
    return all(any(alt.strip().lower() in t for alt in fact.split("|")) for fact in kw.split(";") if fact.strip())


def retrieval_metrics(results, gold_src):
    flags = [match_source(r["chunk"], gold_src) for r in results]
    hit = int(any(flags))                                        # Hit@k
    prec = sum(flags) / len(flags) if flags else 0.0             # Precision@k
    rr = next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0)   # reciprocal rank (mean over questions = MRR)
    return hit, prec, rr


JUDGE_SYS = "You are a strict grader for a retrieval-augmented QA system. Reply with a single JSON object and nothing else."


def judge_answer(question, gold, context, answer, answerable):
    ref = gold if (answerable and gold) else "(none - the knowledge base does NOT contain the answer; the right behaviour is to refuse)"
    prompt = (f"Question: {question}\nReference answer: {ref}\n\nContext given to the system:\n{context[:6000]}\n\n"
              f"System answer:\n{answer}\n\n"
              'Return JSON: {"correct": 0 or 1, "faithful": 0 or 1}\n'
              "- correct = 1 if the system answer agrees with the reference answer (wording may differ).\n"
              "- faithful = 1 if EVERY factual claim in the system answer is supported by the context above "
              "(a refusal counts as faithful); 0 if any claim is unsupported.")
    try:
        raw = rag.llm(JUDGE_SYS, prompt, max_tokens=60)
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return int(j["correct"]), int(j["faithful"])
    except Exception:
        return None, None


def grade(row, out, judge):
    """-> (answer_correct, faithful, refused)"""
    ans = out["answer"]
    refused = rag.is_refusal(ans)
    answerable = bool(row["answerable"])
    jc = jf = None
    if judge:
        jc, jf = judge_answer(row["question"], row["gold_answer"], out["context"], ans, answerable)
    if not answerable:
        correct = int(refused)
    elif jc is not None:
        correct = jc
    else:
        kw = kw_match(ans, row.get("gold_keywords", ""))
        correct = np.nan if kw is None else int(kw and not refused)
    if refused:
        faithful = 1
    elif jf is not None:
        faithful = jf
    else:
        faithful = np.nan if answerable else 0       # unanswerable but answered -> hallucination
    return correct, faithful, int(refused)


def summarize(df):
    a = df[df["answerable"] == 1]
    return dict(
        hit_rate=a["hit"].mean() if "hit" in a and a["hit"].notna().any() else np.nan,
        precision_at_k=a["precision"].mean() if "precision" in a else np.nan,
        mrr=a["rr"].mean() if "rr" in a else np.nan,
        answer_accuracy=df["answer_correct"].mean(),
        faithfulness=df["faithful"].mean(),
        hallucination_rate=1 - df["faithful"].mean(),
        avg_time_s=df["t_total"].mean(),
        n=len(df),
    )


# =====================================================================================
# running questions
# =====================================================================================
def run_rag(qa, store, embedder, k=3, judge=False, mode="vector", min_score=None, verbose=False):
    rows = []
    for _, row in qa.iterrows():
        out = rag.rag_answer(row["question"], store, embedder, k=k, mode=mode, min_score=min_score)
        hit, prec, rr = retrieval_metrics(out["results"], row["gold_source"]) if row["answerable"] else (np.nan, np.nan, np.nan)
        correct, faithful, refused = grade(row, out, judge)
        ev = kw_match(" ".join(r["chunk"]["text"] for r in out["results"]), row.get("gold_keywords", ""))
        rows.append(dict(
            qid=row["qid"], type=row["type"], question=row["question"], answerable=row["answerable"], k=k,
            hit=hit, precision=prec, rr=rr, evidence_available=np.nan if (ev is None or not row["answerable"]) else int(ev),
            top_sources=" ; ".join(f"{r['chunk']['source']} [{r['chunk']['page_or_section']}] ({r['score']:.2f})" for r in out["results"]),
            answer=out["answer"], refused=refused, answer_correct=correct, faithful=faithful,
            t_retrieve=out["t_retrieve"], t_generate=out["t_generate"], t_total=out["t_total"]))
        if verbose:
            print(f"  {row['qid']}: hit={hit} correct={correct} faithful={faithful} refused={refused} ({out['t_total']:.1f}s)")
    return pd.DataFrame(rows)


def run_icl(qa, icl_prompt, judge=False):
    rows = []
    for _, row in qa.iterrows():
        out = rag.icl_answer(row["question"], icl_prompt)
        correct, faithful, refused = grade(row, out, judge)
        ev = kw_match(icl_prompt, row.get("gold_keywords", ""))
        rows.append(dict(qid=row["qid"], type=row["type"], question=row["question"], answerable=row["answerable"],
                         evidence_available=np.nan if (ev is None or not row["answerable"]) else int(ev),
                         answer=out["answer"], refused=refused, answer_correct=correct, faithful=faithful,
                         t_total=out["t_total"]))
    return pd.DataFrame(rows)


def save(df, name):
    p = C.RESULTS_DIR / name
    df.to_csv(p, index=False)
    print(f"  -> saved {p}")


def md(df):
    return df.to_markdown(index=False, floatfmt=".2f")


# =====================================================================================
# Experiment A - effect of top-k
# =====================================================================================
def exp_a(qa, tag=None, judge=False, ks=(1, 3, 5, 10)):
    store = Store.load(tag or default_tag())
    emb = get_embedder(store.meta["embed_model"])
    table = []
    for k in ks:
        print(f"[Exp A] k={k}")
        df = run_rag(qa, store, emb, k=k, judge=judge)
        save(df, f"expA_answers_k{k}.csv")
        s = summarize(df)
        table.append({"k": k, "Retrieval Hit Rate": s["hit_rate"], "Precision@k": s["precision_at_k"], "MRR": s["mrr"],
                      "Answer Accuracy": s["answer_accuracy"], "Faithfulness": s["faithfulness"],
                      "Avg. Time (s)": s["avg_time_s"], "Observations": ""})
    t = pd.DataFrame(table)
    save(t, "expA_topk_table.csv")
    print(md(t))
    return t


# =====================================================================================
# Experiment B - retrieval failure (remove the gold source from the corpus)
# =====================================================================================
def exp_b(qa, remove, tag=None, judge=False, k=3):
    store = Store.load(tag or default_tag())
    emb = get_embedder(store.meta["embed_model"])
    terms = [t.strip() for t in remove.split("|") if t.strip()]
    sel = qa[(qa["answerable"] == 1) & qa["gold_source"].apply(
        lambda g: any(t.lower() in g.lower() or (g.lower() and g.lower() in t.lower()) for t in terms))]
    if sel.empty:
        print("No answerable question has gold_source matching --remove; nothing to test.")
        return None
    reduced = store.subset(lambda c: not match_source(c, remove))
    print(f"[Exp B] removed {len(store.chunks) - len(reduced.chunks)} chunks matching {terms}; testing {len(sel)} questions")
    before = run_rag(sel, store, emb, k=k, judge=judge)
    after = run_rag(sel, reduced, emb, k=k, judge=judge)

    def outcome(r):
        if r["refused"]:
            return "refused (good)"
        if r["answer_correct"] == 1:
            return "answered correctly (answer exists elsewhere / model memory leak)"
        if pd.isna(r["answer_correct"]):
            return "answered - check manually"
        return "HALLUCINATED / wrong"

    after["outcome"] = after.apply(outcome, axis=1)
    cmp_ = pd.DataFrame({"qid": sel["qid"].values, "question": sel["question"].values,
                         "correct_before_removal": before["answer_correct"].values,
                         "answer_after_removal": after["answer"].values, "outcome_after_removal": after["outcome"].values,
                         "top_sources_after": after["top_sources"].values})
    save(cmp_, "expB_retrieval_failure.csv")
    print(md(cmp_[["qid", "correct_before_removal", "outcome_after_removal"]]))
    print(after["outcome"].value_counts().to_string())
    return cmp_


# =====================================================================================
# Experiment C - irrelevant / conflicting context
# =====================================================================================
def load_distractors(kind, embedder):
    chunks = []
    for p in sorted((C.DISTRACTOR_DIR / kind).glob("*.txt")):
        doc = dict(filename=p.name, doc_type="web", source=f"DISTRACTOR-{kind}: {p.stem}", url="",
                   units=rag.text_to_units(p.read_text(encoding="utf-8")))
        chunks += rag.chunk_doc(doc)
    if not chunks:
        print(f"  (no files in distractors/{kind}/)")
        return [], None
    return chunks, embedder.encode_docs([rag.embed_input(c) for c in chunks])


def exp_c(qa, tag=None, judge=False, k=3):
    store = Store.load(tag or default_tag())
    emb = get_embedder(store.meta["embed_model"])
    if "exp_c" in qa and (qa["exp_c"].astype(str) == "1").any():
        sel = qa[qa["exp_c"].astype(str) == "1"]
    else:
        sel = qa[qa["answerable"] == 1].head(10)
    conditions = {"baseline": store}
    for kind in ("irrelevant", "conflicting"):
        ch, ev = load_distractors(kind, emb)
        if ch:
            conditions[f"+{kind}"] = store.extend(ch, ev)
    res = {}
    for name, st in conditions.items():
        print(f"[Exp C] condition={name}")
        df = run_rag(sel, st, emb, k=k, judge=judge)
        df["distractor_in_topk"] = df["top_sources"].str.contains("DISTRACTOR").astype(int)
        res[name] = df
        save(df, f"expC_{name.strip('+')}.csv")
    base = res["baseline"]
    rows = []
    for name, df in res.items():
        s = summarize(df)
        eff = "-"
        if name != "baseline":
            d = df["answer_correct"].fillna(0).values - base["answer_correct"].fillna(0).values
            eff = f"improved {int((d > 0).sum())} / unchanged {int((d == 0).sum())} / harmed {int((d < 0).sum())}"
        rows.append({"Condition": name, "Hit rate": s["hit_rate"], "Answer Accuracy": s["answer_accuracy"],
                     "Faithfulness": s["faithfulness"], "Distractor in top-k": df["distractor_in_topk"].mean(),
                     "Effect vs baseline": eff})
    t = pd.DataFrame(rows)
    save(t, "expC_summary.csv")
    print(md(t))
    return t


# =====================================================================================
# Experiment D - RAG vs ICL
# =====================================================================================
def exp_d(qa, tag=None, judge=False, k=3, icl_path=None):
    store = Store.load(tag or default_tag())
    emb = get_embedder(store.meta["embed_model"])
    icl_prompt = Path(icl_path or C.ROOT / "icl_prompt.txt").read_text(encoding="utf-8")
    print("[Exp D] ICL")
    icl = run_icl(qa, icl_prompt, judge=judge)
    print("[Exp D] RAG")
    rg = run_rag(qa, store, emb, k=k, judge=judge)
    save(icl, "expD_icl_answers.csv")
    save(rg, "expD_rag_answers.csv")
    rows = []
    for name, df in (("ICL", icl), ("RAG", rg)):
        s = summarize(df) if name == "RAG" else dict(
            answer_accuracy=df["answer_correct"].mean(), faithfulness=df["faithful"].mean(),
            hallucination_rate=1 - df["faithful"].mean())
        rows.append({"System": name, "Answer Accuracy": s["answer_accuracy"], "Faithfulness": s["faithfulness"],
                     "Hallucination Rate": s["hallucination_rate"],
                     "Evidence available": df["evidence_available"].mean(), "Observation": ""})
    t = pd.DataFrame(rows)
    save(t, "expD_rag_vs_icl.csv")
    print(md(t))
    print("\nPer question type (answer accuracy):")
    comp = pd.DataFrame({"ICL": icl.groupby("type")["answer_correct"].mean(), "RAG": rg.groupby("type")["answer_correct"].mean()})
    print(comp.round(2).to_markdown())
    return t


# =====================================================================================
# Advanced challenge - chunking / embedding / MMR / hybrid / relevance threshold (retrieval only: no LLM cost)
# =====================================================================================
def retrieval_only(qa, store, emb, k=3, mode="vector"):
    a = qa[qa["answerable"] == 1]
    ms = []
    for _, row in a.iterrows():
        res = rag.retrieve(store, emb, row["question"], k=k, mode=mode)
        ms.append(retrieval_metrics(res, row["gold_source"]))
    m = np.array(ms)
    return dict(hit_at_k=m[:, 0].mean(), precision_at_k=m[:, 1].mean(), mrr=m[:, 2].mean(), chunks=len(store.chunks))


def exp_adv(qa, k=3):
    rows = []

    def get_store(model, strategy):
        tag = rag.make_tag(model, strategy, C.CHUNK_TOKENS, C.CHUNK_OVERLAP)
        e = get_embedder(model)
        if (C.INDEX_DIR / tag / "index.faiss").exists():
            return Store.load(tag), e
        st, _ = rag.build_store(e, strategy, C.CHUNK_TOKENS, C.CHUNK_OVERLAP, verbose=False)
        return st, e

    base_store, base_emb = get_store(C.EMBED_MODEL, C.CHUNK_STRATEGY)
    # 1) chunking strategies
    for strat in ("sentence", "fixed"):
        st, e = get_store(C.EMBED_MODEL, strat)
        rows.append({"Experiment": "chunking", "Config": strat, **retrieval_only(qa, st, e, k)})
    # 2) embedding models
    for model in (C.EMBED_MODEL, C.ALT_EMBED_MODEL):
        st, e = get_store(model, C.CHUNK_STRATEGY)
        rows.append({"Experiment": "embedding", "Config": model.split("/")[-1], **retrieval_only(qa, st, e, k)})
    # 3) retrieval mode
    for mode in ("vector", "mmr", "hybrid"):
        rows.append({"Experiment": "retriever", "Config": mode, **retrieval_only(qa, base_store, base_emb, k, mode)})
    t = pd.DataFrame(rows)
    save(t, "adv_comparison.csv")
    print(md(t))

    # 4) relevance threshold: top-1 cosine for answerable vs unanswerable questions
    top1 = []
    for _, row in qa.iterrows():
        r = rag.retrieve(base_store, base_emb, row["question"], k=1)
        top1.append((row["answerable"], r[0]["score"] if r else 0.0))
    top1 = pd.DataFrame(top1, columns=["answerable", "top1"])
    print("\nTop-1 similarity: answerable mean=%.3f | unanswerable mean=%.3f" % (
        top1[top1.answerable == 1].top1.mean(), top1[top1.answerable == 0].top1.mean()))
    trows = []
    for thr in np.arange(0.40, 0.76, 0.05):
        un, an = top1[top1.answerable == 0], top1[top1.answerable == 1]
        trows.append({"threshold": round(thr, 2),
                      "unanswerable correctly rejected": (un.top1 < thr).mean() if len(un) else np.nan,
                      "answerable wrongly rejected": (an.top1 < thr).mean() if len(an) else np.nan})
    tt = pd.DataFrame(trows)
    save(tt, "adv_threshold.csv")
    print(md(tt))
    return t, tt


# =====================================================================================
# Observation sheet (section 13 of the manual) from a run_rag() dataframe
# =====================================================================================
def observation_sheet(df, name="observation_sheet.csv"):
    sheet = pd.DataFrame({
        "Q.No.": range(1, len(df) + 1), "Question": df["question"],
        "Relevant Source Retrieved?": df["hit"].map({1: "Yes", 0: "No"}).fillna("n/a (unanswerable)"),
        "Top-k": df["k"], "Answer Correct?": df["answer_correct"].map({1: "Yes", 0: "No"}).fillna("check"),
        "Grounded?": df["faithful"].map({1: "Yes", 0: "No"}).fillna("check"),
        "Notes": ""})
    save(sheet, name)
    return sheet


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["A", "B", "C", "D", "ADV", "ALL"])
    ap.add_argument("--qa", default=str(C.ROOT / "qa_set.csv"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge for correctness/faithfulness")
    ap.add_argument("--remove", default="", help="Exp B: source substring(s) to delete, '|' separated")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (quick test)")
    a = ap.parse_args()

    qa = load_qa(a.qa)
    if a.limit:
        qa = qa.head(a.limit)
    if a.exp in ("A", "ALL"):
        t = exp_a(qa, a.tag, a.judge)
        df3 = pd.read_csv(C.RESULTS_DIR / "expA_answers_k3.csv")
        observation_sheet(df3)
    if a.exp == "B":
        if not a.remove:
            ap.error("--exp B needs --remove <source substring>")
        exp_b(qa, a.remove, a.tag, a.judge, a.k)
    if a.exp in ("C", "ALL"):
        exp_c(qa, a.tag, a.judge, a.k)
    if a.exp in ("D", "ALL"):
        exp_d(qa, a.tag, a.judge, a.k)
    if a.exp in ("ADV", "ALL"):
        exp_adv(qa, a.k)


if __name__ == "__main__":
    main()
