# Experiment 8 - SPIT RAG Chatbot (CE413 Generative AI)

## Files
| File | Manual task |
|---|---|
| `build_kb.py` | Task 1 - scrape spit.ac.in pages + public PDFs -> `knowledge_base/` + `metadata.csv` (source name + URL) |
| `rag.py` | Tasks 2-8 - extraction/cleaning, chunking, embeddings, FAISS, retrieval, prompt, LLM call, ICL baseline |
| `build_index.py` | Tasks 2-5 runner (chunks -> embeddings -> FAISS, saved in `index/<tag>/`) |
| `chat.py` | Terminal chatbot: shows retrieved chunks + scores + sources, then the grounded answer |
| `evaluate.py` | Metrics (Hit@k, Precision@k, MRR, accuracy, faithfulness, hallucination, latency) + Experiments A-D + advanced challenge |
| `SPIT_RAG_Experiment8.ipynb` | Notebook that walks through everything (use it for screenshots) |
| `qa_set.csv` | **Template** QA set - extend to 30 questions |
| `icl_prompt.txt` | Placeholder ICL prompt - replace with the one from your previous ICL experiment |
| `distractors/` | Irrelevant + conflicting documents for Experiment C |

## Run order (Ollama + Gemma 4 E2B)

The app uses Ollama for both generation and embeddings. No paid API key is needed; the chat client sends the placeholder key `ollama`. Ollama embeddings use `/api/embed`, with Nomic's `search_document:` and `search_query:` prefixes. If you previously set a cloud provider/key in your shell, unset those overrides first.

```bash
cd /home/alok/Desktop/spit_rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install CPU-only PyTorch first (do not use the default PyPI torch wheel):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the remaining project dependencies:
pip install -r requirements.txt

# In another terminal, start Ollama if it is not already running:
ollama serve

# In this terminal, download the model and select the local Ollama endpoint:
ollama pull gemma4:e2b
export LLM_PROVIDER=openai
export LLM_MODEL=gemma4:e2b
export OPENAI_BASE_URL=http://localhost:11434/v1
export EMBED_MODEL=ollama:nomic-embed-text:latest
export OLLAMA_BASE_URL=http://localhost:11434
unset OPENAI_API_KEY

# Scrape SPIT pages and the configured public PDFs into knowledge_base/:
python build_kb.py

# Inspect a few knowledge_base/website_pages/*.txt files to confirm extraction.
# Build the vector index, then start the terminal chatbot:
python build_index.py
python chat.py

# Run the full evaluation suite (optional --judge uses Gemma via Ollama):
python evaluate.py --exp A --judge
python evaluate.py --exp B --remove "about/" --judge
python evaluate.py --exp C --judge
python evaluate.py --exp D --judge
python evaluate.py --exp ADV
```

For a quick smoke run before the full suite, use `python evaluate.py --exp A --limit 2`. If Ollama is running on a different host or port, set `OPENAI_BASE_URL` accordingly. The generator (`gemma4:e2b`) and embedder (`nomic-embed-text:latest`) both run through Ollama on CPU. Ollama downloads each model only if it is not already installed. `SentenceTransformer` remains available for the optional alternate-embedding comparison in `--exp ADV`; PyTorch is installed from its CPU-only wheel index above.

If you already started an install that is pulling CUDA/NVIDIA packages, stop it with `Ctrl+C`. To remove the partially installed environment and start clean, run `deactivate` if the virtual environment is active, then remove only this project's `.venv` folder and recreate it with the commands above. Do not run `pip install -r requirements.txt` before the CPU-only PyTorch command.

## Things you must do yourself
1. **Check the scraped text.** The site is WordPress, but I could not run the scraper against the live site, so
   inspect `knowledge_base/website_pages/*.txt`. If a page contains menus or is empty, tune `CONTENT_SELECTORS`
   in `build_kb.py`.
2. **Finish `qa_set.csv` (30 questions).** Recommended mix: 10 direct factual, 5 multi-document, 5 comparison,
   5 reasoning, 5 unanswerable. Fill `gold_source` (a substring of the source name/URL that holds the answer)
   and `gold_keywords` (`;` = all of these facts, `|` = alternative wordings) after reading your corpus. The 9
   rows given are only a template - verify them against your scraped pages.
3. **ICL baseline.** Paste the prompt you used in the ICL experiment into `icl_prompt.txt`.
4. **ERP data.** ERP ingestion is commented out in `rag.py` because no dataset is available. The current corpus consists of the SPIT website and configured public PDFs.
5. **Judge vs. manual grading.** `--judge` uses the LLM to grade correctness/faithfulness. It is a shortcut,
   so hand-check a sample and fill the Observation Sheet (`results/observation_sheet.csv`) yourself.
   Without `--judge`, accuracy comes from `gold_keywords` and faithfulness is only known for unanswerable questions.

## Design notes (useful for the viva)
* Embeddings are L2-normalised and the index is `IndexFlatIP`, so inner product = cosine similarity.
* `bge-small-en-v1.5` has a 512-token window, so 400-600 token chunks are not truncated; `all-MiniLM-L6-v2`
  (second model in the advanced challenge) was trained on 256 tokens.
* Each chunk embeds `source | section` + text; the stored chunk keeps `chunk_id, source, page_or_section, text, url`.
* Prompt rules: answer only from context, exact refusal sentence when missing, cite `[S#]`, flag conflicts.
* Advanced challenge implemented: two chunking strategies, two embedding models, MMR, hybrid (BM25 + vector via
  RRF), relevance threshold (`--thr` in chat / `min_score` in code), source display.
