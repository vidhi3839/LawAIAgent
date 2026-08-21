import os
import re
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

PDF_FOLDER = os.path.join(BASE_DIR, "data", "case_pdfs")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "past_cases"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RRF_K = 60

embedder = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

CASE_METADATA = {
    "ATT_Mobility_v_Concepcion": {
        "case_name": "AT&T Mobility LLC v. Concepcion",
        "year": "2011",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "arbitration, class action waiver, federal preemption, FAA",
        "citation": "563 U.S. 333"
    },
    "Ashcroft_v_Iqbal": {
        "case_name": "Ashcroft v. Iqbal",
        "year": "2009",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "civil procedure, pleading standards, motion to dismiss, Twombly",
        "citation": "556 U.S. 662"
    },
    "Burlington_Northern_v_White": {
        "case_name": "Burlington Northern & Santa Fe Railway Co. v. White",
        "year": "2006",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "Title VII, retaliation, employment discrimination, adverse action",
        "citation": "548 U.S. 53"
    },
    "Van_Buren_v_United_States": {
        "case_name": "Van Buren v. United States",
        "year": "2021",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "Computer Fraud and Abuse Act, CFAA, exceeding authorized access, computer crime",
        "citation": "593 U.S. 374"
    },
    "Celotex_v_Catrett": {
        "case_name": "Celotex Corp. v. Catrett",
        "year": "1986",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "summary judgment, Rule 56, burden of proof, civil procedure",
        "citation": "477 U.S. 317"
    },
    "Miranda_v_Arizona": {
        "case_name": "Miranda v. Arizona",
        "year": "1966",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "criminal procedure, Fifth Amendment, self-incrimination, custodial interrogation",
        "citation": "384 U.S. 436"
    },
    "New_York_Times_v_Sullivan": {
        "case_name": "New York Times Co. v. Sullivan",
        "year": "1964",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "First Amendment, defamation, actual malice, freedom of the press",
        "citation": "376 U.S. 254"
    },
    "Daubert_v_Merrell_Dow": {
        "case_name": "Daubert v. Merrell Dow Pharmaceuticals, Inc.",
        "year": "1993",
        "court": "Supreme Court of the United States",
        "jurisdiction": "federal",
        "legal_issues": "evidence law, expert testimony, Federal Rule of Evidence 702, scientific evidence standard",
        "citation": "509 U.S. 579"
    },
    "Palsgraf_v_Long_Island_Railroad": {
        "case_name": "Palsgraf v. Long Island Railroad Co.",
        "year": "1928",
        "court": "New York Court of Appeals",
        "jurisdiction": "state",
        "legal_issues": "negligence, proximate cause, duty of care, foreseeability, tort law",
        "citation": "248 N.Y. 339"
    },
    "Lucy_v_Zehmer": {
        "case_name": "Lucy v. Zehmer",
        "year": "1954",
        "court": "Supreme Court of Virginia",
        "jurisdiction": "state",
        "legal_issues": "contract formation, mutual assent, objective theory of contract, intent",
        "citation": "196 Va. 493"
    }
}


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        text = " ".join(text.split())
        return text
    except Exception as e:
        log.error(f"Error reading {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_case_metadata_hints(text: str) -> dict:
    header = text[:3000]
    result = {"citation": "", "court": "", "jurisdiction": "", "year": ""}

    citation_patterns = [
        r'(\d+)\s+U\.S\.\s+(\d+)',                                   # e.g. 593 U.S. 374
        r'(\d+)\s+F\.(?:2d|3d|4th)\s+(\d+)',                          # e.g. 123 F.3d 456
        r'(\d+)\s+F\.\s*Supp\.\s*(?:2d|3d)?\s+(\d+)',                 # e.g. 45 F. Supp. 2d 678
        r'(\d+)\s+S\.\s*Ct\.\s+(\d+)',                                # e.g. 141 S. Ct. 1648
    ]
    for pattern in citation_patterns:
        m = re.search(pattern, header)
        if m:
            result["citation"] = m.group(0).strip()
            break

    court_patterns = [
        (r'SUPREME COURT OF THE UNITED STATES', "Supreme Court of the United States", "federal"),
        (r'UNITED STATES COURT OF APPEALS FOR THE (\w+) CIRCUIT', None, "federal"),
        (r'UNITED STATES DISTRICT COURT (?:FOR THE )?([A-Z .]+?)(?:\n|DISTRICT)', None, "federal"),
        (r'SUPREME COURT OF ([A-Z][A-Za-z]+)', None, "state"),
        (r'COURT OF APPEALS OF ([A-Z][A-Za-z]+)', None, "state"),
        (r'([A-Z][A-Za-z]+) COURT OF APPEALS', None, "state"),
    ]
    header_upper = header.upper()
    for pattern, fixed_label, jurisdiction in court_patterns:
        m = re.search(pattern, header_upper)
        if m:
            if fixed_label:
                result["court"] = fixed_label
            elif m.groups():
                result["court"] = m.group(0).title().strip()
            result["jurisdiction"] = jurisdiction
            break

    year_match = re.search(r'\((?:19|20)\d{2}\)', header) or re.search(r'\b(19|20)\d{2}\b', header)
    if year_match:
        year_digits = re.search(r'(19|20)\d{2}', year_match.group(0))
        if year_digits:
            result["year"] = year_digits.group(0)

    return result


def ingest_single_case(
    file_path: str,
    case_name: str,
    citation: str = "",
    year: str = "",
    court: str = "",
    jurisdiction: str = "",
    legal_issues: str = "",
    uploaded_by: str = "",
) -> dict:

    if not case_name or not case_name.strip():
        return {"success": False, "error": "Case name is required."}

    text = extract_text_from_pdf(file_path)
    if not text:
        return {"success": False, "error": "Could not extract any text from this PDF -- it may be scanned/image-only or corrupted."}

    hints = extract_case_metadata_hints(text)
    resolved_citation = citation.strip() if citation and citation.strip() else hints["citation"]
    resolved_court = court.strip() if court and court.strip() else hints["court"]
    resolved_jurisdiction = jurisdiction.strip().lower() if jurisdiction and jurisdiction.strip() else hints["jurisdiction"]
    resolved_year = year.strip() if year and year.strip() else hints["year"]

    chunks = chunk_text(text)
    if not chunks:
        return {"success": False, "error": "No text chunks were produced from this PDF."}

    case_key = re.sub(r'[^a-zA-Z0-9]+', '_', case_name.strip()).strip('_')[:80]
    if not case_key:
        return {"success": False, "error": "Case name produced an empty key after sanitization -- use a name with at least one letter or number."}

    existing_ids = set(collection.get()["ids"])
    stale_ids = [cid for cid in existing_ids if cid.startswith(f"{case_key}_chunk_")]
    if stale_ids:
        collection.delete(ids=stale_ids)
        log.info(f"Cleared {len(stale_ids)} existing chunks for '{case_key}' before re-ingesting (re-upload).")

    try:
        embeddings = embedder.encode(chunks).tolist()
    except Exception as e:
        log.error(f"Embedding failed for lawyer-uploaded case '{case_name}': {e}")
        return {"success": False, "error": f"Embedding failed: {e}"}

    metadata_base = {
        "case_name": case_name.strip(),
        "citation": resolved_citation,
        "year": resolved_year,
        "court": resolved_court,
        "jurisdiction": resolved_jurisdiction,
        "legal_issues": legal_issues.strip(),
        "uploaded_by": uploaded_by.strip(),
        "source_type": "lawyer_upload",
    }

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{case_key}_chunk_{i}")
        documents.append(chunk)
        metadatas.append({**metadata_base, "chunk_index": i, "total_chunks": len(chunks)})

    try:
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    except Exception as e:
        log.error(f"ChromaDB add failed for lawyer-uploaded case '{case_name}': {e}")
        return {"success": False, "error": f"Failed to store in the case database: {e}"}

    log.info(f"Ingested lawyer-uploaded case '{case_name}' as '{case_key}' ({len(chunks)} chunks, uploaded_by={uploaded_by!r}, "
              f"resolved metadata: citation={resolved_citation!r} court={resolved_court!r} jurisdiction={resolved_jurisdiction!r} year={resolved_year!r})")
    return {
        "success": True,
        "case_key": case_key,
        "chunks_ingested": len(chunks),
        "auto_detected": {
            "citation": resolved_citation if not citation.strip() else "",
            "court": resolved_court if not court.strip() else "",
            "jurisdiction": resolved_jurisdiction if not jurisdiction.strip() else "",
            "year": resolved_year if not year.strip() else "",
        },
    }


def list_uploaded_cases() -> list:
    """Returns one entry per distinct lawyer-uploaded case currently in
    the collection (deduplicated across chunks), for a simple "cases in
    the database" listing on the upload page. Does NOT include the
    original 10 hardcoded cases unless they were also re-ingested with
    source_type='lawyer_upload' -- this only reflects uploads made
    through ingest_single_case()."""
    try:
        all_rows = collection.get(where={"source_type": "lawyer_upload"})
    except Exception as e:
        log.error(f"list_uploaded_cases failed: {e}")
        return []

    seen = {}
    metadatas = all_rows.get("metadatas", [])
    for meta in metadatas:
        key = meta.get("case_name", "")
        if key and key not in seen:
            seen[key] = {
                "case_name": meta.get("case_name", ""),
                "citation": meta.get("citation", ""),
                "year": meta.get("year", ""),
                "court": meta.get("court", ""),
                "jurisdiction": meta.get("jurisdiction", ""),
                "legal_issues": meta.get("legal_issues", ""),
                "uploaded_by": meta.get("uploaded_by", ""),
            }
    return list(seen.values())


def ingest_cases():
    log.info("Starting ingestion pipeline...")

    if not os.path.isdir(PDF_FOLDER):
        log.error(f"PDF_FOLDER does not exist: {PDF_FOLDER} — create it and add case PDFs before running ingestion.")
        return

    existing_ids = set(collection.get()["ids"])
    ingested = 0

    for filename in os.listdir(PDF_FOLDER):
        if not filename.endswith(".pdf"):
            continue

        case_key = filename.replace(".pdf", "")
        metadata_base = CASE_METADATA.get(case_key)

        if not metadata_base:
            log.error(f"No metadata found for {filename} — skipping. Add an entry to CASE_METADATA first.")
            continue

        pdf_path = os.path.join(PDF_FOLDER, filename)
        log.info(f"Processing {filename}...")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            log.error(f"Could not extract text from {filename}")
            continue

        chunks = chunk_text(text)
        log.info(f"  {len(chunks)} chunks created")

        stale_ids = [cid for cid in existing_ids if cid.startswith(f"{case_key}_chunk_")]
        if stale_ids:
            collection.delete(ids=stale_ids)
            existing_ids -= set(stale_ids)
            log.info(f"  Cleared {len(stale_ids)} existing chunks for {case_key} before re-ingesting")

        try:
            embeddings = embedder.encode(chunks).tolist()
        except Exception as e:
            log.error(f"Embedding failed for {case_key}: {e}")
            continue

        ids, documents, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{case_key}_chunk_{i}")
            documents.append(chunk)
            metadatas.append({
                **metadata_base,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": filename
            })

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        ingested += len(chunks)
        log.info(f"  Ingested {case_key} ({len(chunks)} chunks)")

    log.info(f"Ingestion complete. Total chunks added: {ingested}")
    log.info(f"Total chunks in collection: {collection.count()}")


def _deduplicate_cases(cases: list) -> list:
    """ChromaDB returns chunks, not whole documents — the same case can appear multiple times if several of its chunks each match well,
    inflating both the displayed 'Cases Retrieved' count and the averaged m_score used in compute_past_cases_confidence. Same fix already
    applied to tasks/mock_court.py's case retrieval."""
    def _rank_key(c: dict) -> float:
        
        return c.get("rrf_score", c.get("similarity_score", 0))

    best_by_case = {}
    for c in cases:
        key = c.get("citation") or c.get("case_name") or repr(c)
        if key not in best_by_case or _rank_key(c) > _rank_key(best_by_case[key]):
            best_by_case[key] = c
    return sorted(best_by_case.values(), key=_rank_key, reverse=True)


def _tokenize(text: str) -> list:
    """Simple lowercase alphanumeric tokenizer for BM25. Deliberately basic
    (no stemming/stopword removal) -- BM25's own term-frequency weighting
    already down-weights common words naturally, and exact-token matching is
    the whole point of the keyword side of hybrid search."""
    return re.findall(r"[a-z0-9]+", text.lower())


_bm25_cache = {"index": None, "ids": None, "corpus_count": None}


def _get_bm25_index():
    """Builds (or reuses a cached) BM25 index over every chunk currently in
    the ChromaDB collection. Rebuilt only when collection.count() changes
    (a case was added/re-ingested) -- ingestion happens rarely compared to
    how often search runs, so caching against chunk count is a cheap and
    correct invalidation signal, not a full rebuild-every-query cost."""
    current_count = collection.count()
    if _bm25_cache["index"] is not None and _bm25_cache["corpus_count"] == current_count:
        return _bm25_cache["index"], _bm25_cache["ids"]

    try:
        all_rows = collection.get(include=["documents"])
    except Exception as e:
        log.error(f"BM25 index build failed to read collection: {e}")
        return None, []

    ids = all_rows.get("ids", [])
    documents = all_rows.get("documents", [])
    if not documents:
        return None, []

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    index = BM25Okapi(tokenized_corpus)

    _bm25_cache["index"] = index
    _bm25_cache["ids"] = ids
    _bm25_cache["corpus_count"] = current_count
    return index, ids


def _keyword_search(query: str, raw_fetch_count: int) -> list:
    """Returns [(chunk_id, bm25_score), ...] sorted descending, top raw_fetch_count.
    Empty list (not an error) if the collection is empty or BM25 can't build --
    hybrid search should degrade to pure-vector, not fail outright."""
    index, ids = _get_bm25_index()
    if index is None:
        return []

    scores = index.get_scores(_tokenize(query))
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [(cid, score) for cid, score in ranked[:raw_fetch_count] if score > 0]


def search_past_cases(
    query: str,
    n_results: int = 5,
    jurisdiction_filter: Optional[str] = None
) -> dict:
    """Hybrid search: fuses dense (embedding/cosine) retrieval with sparse
    (BM25 keyword) retrieval via Reciprocal Rank Fusion, so exact citations
    and party names (which embeddings alone are weak on) get a fair shot
    alongside paraphrase/semantic matches (which BM25 alone is weak on).
    Falls back gracefully to pure-vector ranking if BM25 finds nothing for
    the query."""
    query_embedding = embedder.encode(query).tolist()

    where_filter = None
    if jurisdiction_filter and jurisdiction_filter.lower() not in ["any", "all", "general"]:
        where_filter = {"jurisdiction": jurisdiction_filter}

    raw_fetch_count = min(n_results * 4, 40)

    try:
        if where_filter:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=raw_fetch_count,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=raw_fetch_count,
                include=["documents", "metadatas", "distances"]
            )
    except Exception as e:
        log.error(f"ChromaDB query failed: {e}")
        return {"error": str(e), "results": []}

    dense_ids = results.get("ids", [[]])[0]
    dense_documents = results.get("documents", [[]])[0]
    dense_metadatas = results.get("metadatas", [[]])[0]
    dense_distances = results.get("distances", [[]])[0]

    if not dense_documents:
        return {"error": "No matching cases found", "results": []}

    # --- Dense (vector) ranking, by position (rank 0 = best) ---
    dense_rank = {cid: rank for rank, cid in enumerate(dense_ids)}
    similarity_by_id = {
        cid: round(1 - dist, 4) for cid, dist in zip(dense_ids, dense_distances)
    }
    doc_by_id = dict(zip(dense_ids, dense_documents))
    meta_by_id = dict(zip(dense_ids, dense_metadatas))

    # --- Sparse (keyword/BM25) ranking, by position ---
    keyword_hits = _keyword_search(query, raw_fetch_count)
    keyword_rank = {cid: rank for rank, (cid, _score) in enumerate(keyword_hits)}

    missing_ids = [cid for cid in keyword_rank if cid not in doc_by_id]
    if missing_ids:
        try:
            extra = collection.get(ids=missing_ids, include=["documents", "metadatas"])
            for cid, doc, meta in zip(extra.get("ids", []), extra.get("documents", []), extra.get("metadatas", [])):
                doc_by_id[cid] = doc
                meta_by_id[cid] = meta
        except Exception as e:
            log.error(f"Fetching keyword-only hit metadata failed: {e}")

    if where_filter:
        keyword_rank = {
            cid: rank for cid, rank in keyword_rank.items()
            if meta_by_id.get(cid, {}).get("jurisdiction") == where_filter["jurisdiction"]
        }

    # --- Reciprocal Rank Fusion ---
    all_ids = set(dense_rank) | set(keyword_rank)
    fused = []
    for cid in all_ids:
        rrf_score = 0.0
        if cid in dense_rank:
            rrf_score += 1.0 / (RRF_K + dense_rank[cid])
        if cid in keyword_rank:
            rrf_score += 1.0 / (RRF_K + keyword_rank[cid])

        meta = meta_by_id.get(cid, {})
        fused.append({
            "text": doc_by_id.get(cid, ""),
            "case_name": meta.get("case_name", ""),
            "citation": meta.get("citation", ""),
            "year": meta.get("year", ""),
            "court": meta.get("court", ""),
            "jurisdiction": meta.get("jurisdiction", ""),
            "legal_issues": meta.get("legal_issues", ""),
            "source_file": meta.get("source_file", ""),
            "similarity_score": similarity_by_id.get(cid, 0.0),
            "rrf_score": round(rrf_score, 5),
            "matched_by": "both" if cid in dense_rank and cid in keyword_rank
                          else ("vector" if cid in dense_rank else "keyword"),
        })

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)


    RRF_FLOOR = 1.0 / (RRF_K + 30) 
    fused = [r for r in fused if r["rrf_score"] > RRF_FLOOR]

    if not fused:
        return {"error": "No matching cases found", "results": []}

    deduped = _deduplicate_cases(fused)[:n_results]
    return {"results": deduped, "query": query}


def compute_past_cases_confidence(
    query: str,
    retrieved_results: list,
    llm_answer: str,
    requested_jurisdiction: Optional[str] = None
) -> dict:
    if not retrieved_results:
        return {
            "final_score": 0.0,
            "above_threshold": False,
            "threshold": 0.7,
            "task": "past_cases",
            "m_score": 0.0,
            "j_score": 0.0,
            "g_score": 0.0,
            "has_citations": False,
            "explanation": "No results retrieved"
        }


    similarities = [r["similarity_score"] for r in retrieved_results]
    weights = [1 / (i + 1) for i in range(len(similarities))]
    m_score = round(sum(s * w for s, w in zip(similarities, weights)) / sum(weights), 4)

    if not requested_jurisdiction or requested_jurisdiction.lower() in ["any", "all", "general", ""]:
        j_score = 1.0
    else:
        req_lower = requested_jurisdiction.lower()
        matched = sum(
            1 for r in retrieved_results
            if req_lower in r.get("jurisdiction", "").lower()
            or req_lower in r.get("court", "").lower()
        )
        j_score = round(matched / len(retrieved_results), 4)

    all_source_text = " ".join([r["text"] for r in retrieved_results]).lower()
    answer_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', llm_answer.lower()))
    if answer_words:
        grounded = sum(1 for w in answer_words if w in all_source_text)
        g_score = round(grounded / len(answer_words), 4)
    else:
        g_score = 1.0

    final_score = round((m_score * 0.4) + (j_score * 0.3) + (g_score * 0.3), 4)

    has_citations = any(bool(r.get("citation")) for r in retrieved_results)

    threshold = 0.70
    return {
        "final_score": final_score,
        "above_threshold": final_score >= threshold,
        "threshold": threshold,
        "task": "past_cases",
        "m_score": m_score,
        "j_score": j_score,
        "g_score": g_score,
        "has_citations": has_citations,
        "explanation": (
            f"Vector match: {m_score} × 0.4 + "
            f"Jurisdiction: {j_score} × 0.3 + "
            f"Grounding: {g_score} × 0.3 = {final_score}"
        )
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ingest_cases()