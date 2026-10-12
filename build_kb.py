"""
Task 1 - Prepare the knowledge base.

Downloads selected pages from https://www.spit.ac.in/ (cleaned to plain text) and a few public PDFs,
stores them under knowledge_base/ and records source name + source URL for EVERY file in
knowledge_base/metadata.csv  (the RAG pipeline reads this file to give source attribution).

Usage:
    python build_kb.py                    # pages + seed PDFs
    python build_kb.py --no-pdfs          # pages only (use the PDFs your ma'am gives you)
    python build_kb.py --max-extra-pdfs 20

After running, OPEN a few files in knowledge_base/website_pages/ and check they contain real
content (not menus). If a page is empty/noisy, adjust CONTENT_SELECTORS below.
Be polite to the server: the default delay between requests is 1 second.
"""
import argparse
import csv
import re
import time
from datetime import date
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup, NavigableString

import config as C

HEADERS = {"User-Agent": "Mozilla/5.0 (educational RAG lab experiment - SPIT GenAI CE413)"}
BASE = "https://www.spit.ac.in/"

# Pages taken from the SPIT website menu (About / Academics / Administration / Students / Research ...)
SEED_PAGES = [
    "", "about/", "about/principals-message/", "about/vision-mission/", "about/institutional-values/",
    "about/campus/", "about/contact/", "about/organogram/",
    "academics/graduate/", "academics/post-graduate/", "academics/ph-d-programme/",
    "academics/departments/", "academics/students-empowerment-policy/",
    "academics/faculty-empowerment-policy/", "best-practices/", "curriculum-2/", "academic-calendar/",
    "sp-it-teaching-learning-and-evaluation-tle-model/", "academic-council-minutes/", "results-analysis/",
    "resources/it-infrastructure/", "central-library/",
    "administration/governing-body/", "deans/", "administration/finance-committee/",
    "administration/academic-council/", "administration/faculty/", "administration/committees/",
    "iqac/", "iqac/constitution/", "perspective-plan/", "institutional-distinctiveness/",
    "scholarships/", "scholarships/vidya-lakshmi-portal/", "students-council/", "achievement/",
    "innovation-and-entrepreneurship/", "capacity-development-programme/",
    "faculty-research/preamble/", "faculty-research/rd-committee/", "faculty-research/policy-research-areas/",
    "faculty-research/funded-projects/",
    "admissions/", "mandatory-disclosure/", "naac/", "nirf", "nba/", "ariia/",
]
UP = BASE + "wp-content/uploads/"
SEED_PDFS = [
    ("SPIT Examination Manual", UP + "2024/11/Final_23-24_Exam_Manual_25th_Jan.pdf"),
    ("Academic Calendar ODD Semester 2026-27", UP + "2026/04/Academic-Calendar_ODD_2026-27.pdf"),
    ("First Year B.Tech Syllabus 2026-27", UP + "2026/09/First-Year-Syllabus_2026-27_Final_18-09-2026.pdf"),
    ("Code of Conduct for Students", UP + "2026/08/Code-of-conduct-for-students.pdf"),
    ("Code of Conduct for Faculty", UP + "2026/08/code-of-conduct-for-faculty.pdf"),
    ("B.Tech FRA Approved Fee Notice 2025-26", UP + "2025/10/B.Tech_._FRA-Fee-Approved-Notice-2025-26.pdf"),
    ("M.Tech FRA Approved Fee Notice 2025-26", UP + "2025/10/M.Tech_._FRA-Approved-Fee-Notice-2025-26.pdf"),
    ("MCA FRA Approved Fee Notice 2025-26", UP + "2025/10/MCA_FRA-Approved-Fee-Notice-2025-26.pdf"),
]
DEPT_EXPAND_PAGE = BASE + "academics/departments/"   # its outbound links = department websites

# --- HTML cleaning knobs (tune if a page comes out noisy/empty) ---
CONTENT_SELECTORS = ["main", "article", "#content", ".entry-content", ".post-content", "#primary",
                     ".site-content", "#main"]
NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "aside", "form", "svg", "iframe",
              "button", "select", "input"]
NOISE_CLASS_RE = re.compile(r"(menu|breadcrumb|sidebar|cookie|share|social|related|comment|pagination|navigation)", re.I)
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "tr", "br", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"]
BOILERPLATE_LINES = {"read more", "share", "home", "skip to content", "back to top", "menu"}

META_FIELDS = ["filename", "folder", "source_name", "source_url", "doc_type", "retrieved_on"]


# ------------------------------------------------------------------ metadata helpers
def read_meta():
    if not C.META_CSV.exists():
        return {}
    with open(C.META_CSV, newline="", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def write_meta(meta):
    with open(C.META_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        for r in sorted(meta.values(), key=lambda r: (r["folder"], r["filename"])):
            w.writerow({k: r.get(k, "") for k in META_FIELDS})


# ------------------------------------------------------------------ fetching / parsing
def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r
    except Exception as e:  # network errors, 404s ...
        print(f"  ! failed {url}: {e}")
        return None


def slugify(url, suffix=""):
    p = urlparse(url)
    s = (p.netloc.replace("www.", "") + p.path).strip("/")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "home"
    return s[:120] + suffix


def _safe_attr(el, key):
    try:
        v = el.get(key)
    except Exception:  # element already decomposed
        return None
    return v


def html_to_text(html, url):
    """Return (title, cleaned_text, pdf_links, container_soup)."""
    soup = BeautifulSoup(html, "lxml")

    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = urldefrag(urljoin(url, a["href"]))[0]
        if href.lower().endswith(".pdf") and urlparse(href).netloc.endswith("spit.ac.in"):
            pdf_links.append(href)

    title = (soup.title.get_text(strip=True) if soup.title else "") or url
    title = re.split(r"\s+[–—|-]\s+(?:Sardar|SPIT|S\.P\.I\.T)", title)[0].strip() or title

    # 1) choose the main content container (fallback: <body>)
    container = None
    for sel in CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el is not None and len(el.get_text(strip=True)) >= 200:
            container = el
            break
    use_body = container is None
    if use_body:
        container = soup.body or soup

    # 2) strip noise inside the container
    for tag in container.find_all(NOISE_TAGS):
        tag.decompose()
    if use_body:
        for tag in container.find_all(["header"]):
            tag.decompose()
    else:  # class-name based removal only inside a trusted container
        for tag in container.find_all(["div", "ul", "section", "span"]):
            cls = _safe_attr(tag, "class")
            ident = _safe_attr(tag, "id")
            if cls is None and ident is None:
                continue
            if NOISE_CLASS_RE.search(" ".join(cls or []) + " " + (ident or "")):
                tag.decompose()

    # 3) tables -> "cell | cell | cell" rows, headings -> "## heading"
    for table in container.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))
    for h in container.find_all(["h1", "h2", "h3", "h4"]):
        txt = h.get_text(" ", strip=True)
        h.replace_with(NavigableString(f"\n## {txt}\n" if txt else "\n"))
    for tag in container.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    # 4) normalise lines, drop boilerplate + short duplicate lines
    seen, lines = set(), []
    for ln in container.get_text("").replace("\xa0", " ").splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln or ln.lower() in BOILERPLATE_LINES:
            continue
        if len(ln) < 80 and ln in seen:
            continue
        seen.add(ln)
        lines.append(ln)
    return title, "\n".join(lines), pdf_links, container


def save_page(url, meta):
    r = fetch(url)
    if r is None:
        return [], []
    title, text, pdfs, container = html_to_text(r.text, url)
    if len(text) < 80:
        print(f"  ? very little text extracted from {url} ({len(text)} chars) - page may be JS-rendered")
    fname = slugify(url, ".txt")
    (C.WEB_DIR / fname).write_text(f"## {title}\n{text}\n", encoding="utf-8")
    meta[fname] = dict(filename=fname, folder="website_pages", source_name=title, source_url=url,
                       doc_type="web", retrieved_on=str(date.today()))
    print(f"  saved {fname}  ({len(text)} chars)")
    return pdfs, container


def download_pdf(url, name, meta):
    fname = re.sub(r"[^A-Za-z0-9._-]+", "_", urlparse(url).path.split("/")[-1]) or slugify(url, ".pdf")
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    r = fetch(url)
    if r is None:
        return
    (C.PDF_DIR / fname).write_bytes(r.content)
    meta[fname] = dict(filename=fname, folder="institutional_pdfs", source_name=name or fname,
                       source_url=url, doc_type="pdf", retrieved_on=str(date.today()))
    print(f"  saved {fname}  ({len(r.content)//1024} KB)")


def register_local_files(meta):
    """Files placed manually (instructor PDFs, ERP export) also get a metadata row. Fill source_url by hand."""
    for folder, doc_type in (("institutional_pdfs", "pdf"), ("erp_export", "erp")):
        for p in sorted((C.KB_DIR / folder).glob("*")):
            if p.is_file() and p.name not in meta:
                meta[p.name] = dict(filename=p.name, folder=folder, source_name=p.stem.replace("_", " "),
                                    source_url="", doc_type=doc_type, retrieved_on=str(date.today()))
                print(f"  registered local file {folder}/{p.name} (edit source_name/source_url in metadata.csv)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdfs", action="store_true", help="skip downloading PDFs")
    ap.add_argument("--max-extra-pdfs", type=int, default=10, help="PDFs discovered on pages (besides seed PDFs)")
    ap.add_argument("--max-dept-sites", type=int, default=10, help="department sites linked from the Departments page")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    meta = read_meta()
    queue = [urljoin(BASE, p) for p in SEED_PAGES]
    seen, discovered_pdfs = set(), []

    print(f"== Crawling {len(queue)} seed pages ==")
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        print(f"[page] {url}")
        pdfs, container = save_page(url, meta)
        discovered_pdfs += [p for p in pdfs if p not in discovered_pdfs]
        if url == DEPT_EXPAND_PAGE and container is not None:
            host_ok = lambda u: urlparse(u).netloc.endswith(".spit.ac.in") and urlparse(u).netloc != "www.spit.ac.in"
            dept = []
            for a in container.find_all("a", href=True):
                h = urljoin(url, a["href"])
                if host_ok(h) and h not in dept:
                    dept.append(h)
            print(f"  found {len(dept)} department/sub-site links; taking {args.max_dept_sites}")
            queue += dept[: args.max_dept_sites]
        time.sleep(args.delay)

    if not args.no_pdfs:
        print("== Downloading PDFs ==")
        seed_urls = {u for _, u in SEED_PDFS}
        for name, u in SEED_PDFS:
            print(f"[pdf] {u}")
            download_pdf(u, name, meta)
            time.sleep(args.delay)
        extra = [u for u in discovered_pdfs if u not in seed_urls][: args.max_extra_pdfs]
        for u in extra:
            print(f"[pdf] {u}")
            download_pdf(u, "", meta)
            time.sleep(args.delay)

    print("== Registering local files ==")
    register_local_files(meta)
    write_meta(meta)
    print(f"\nDone. {len(meta)} documents listed in {C.META_CSV}")


if __name__ == "__main__":
    main()
