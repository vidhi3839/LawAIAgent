import os
import re
from typing import Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer, util

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)


# 12,000 characters (~3,000 tokens) for the document sample, plus prompt template overhead (~300-400 tokens) and typical completion length
# (~500-1000 tokens for structured extraction), keeps a single call to roughly 4,000-4,500 tokens — comfortably under the per-minute cap even
# with other concurrent calls (router calls, other lawyers) sharing the same account-wide budget. If this account is upgraded to a higher Groq
# tier, this can be raised — check the new TPM/TPD limits first.
MAX_CHARS_FOR_LLM = 12000

DATE_PATTERNS = [
    r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
    r'\b\d{4}\b',  
]


ALL_SECTION_HEADERS = ["TIMELINE:", "CAST OF CHARACTERS:", "CAUSES OF ACTION:", "KEY FACTS:"]

# Small, deliberately non-exhaustive stopword list for the "distinctive word" checks used in proximity matching. Not meant to be complete NLP-grade
# stopword filtering — just enough to stop obviously generic words from passing as "evidence" a claim is grounded near a match.
STOP_WORDS = {
    "which", "there", "their", "about", "state", "court", "other", "after", "before", "shall", "under", "these", "those", "would",
    "could", "should", "where", "while", "based"
}

DECLINE_PHRASES = [
    "not stated in the document", "not stated", "does not state", "does not specify", "not specified", "no information",
    "not mentioned", "does not mention", "cannot determine", "unable to determine", "not found in the document",
]

# Common words that are only capitalized because they start a sentence, not because they're names — without this, a plain regex on capitalized
# words treats "The", "This", "If" etc. as proper nouns, and since they're common words that trivially appear anywhere, they falsely inflate
# n_score to look "grounded" for sentences with no real named entities.
SENTENCE_STARTER_WORDS = {
    "the", "this", "that", "these", "those", "it", "if", "when", "where", "who", "what", "why", "how", "a", "an", "in", "on", "at", "for", "to",
    "of", "and", "but", "or", "so", "because", "since", "before", "after", "during", "while", "however", "therefore", "thus", "then", "there",
    "here", "do", "does", "did", "can", "could", "should", "would", "will", "shall", "may", "might", "must", "is", "are", "was", "were",
    "be", "been", "being", "as", "such", "given", "yes", "no"
}


def _strip_think_tags(text: str) -> str:
    """Some models wrap reasoning in <think>...</think> before the real
    answer. Strip it if present."""
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def _truncation_notice(raw_text: str, limit: int = MAX_CHARS_FOR_LLM) -> str:
    """Visible note appended to the model's output whenever the source document was too long to fully analyze — replaces the previous silent
    truncation, where content past the cutoff was dropped with zero indication to the lawyer."""
    if len(raw_text) > limit:
        return (
            f"\n\n*Note: this document is {len(raw_text):,} characters long; "
            f"only the first {limit:,} characters were analyzed. Content "
            f"beyond that point was not seen by the model.*"
        )
    return ""


def _find_all_occurrences(needle: str, haystack_lower: str) -> list:
    """Word-boundary-safe search — fixes the old plain-substring check, where a short name like 'Roe' would match inside unrelated words like
    'wardrobe'. haystack_lower must already be lowercased; needle is lowercased and escaped here."""
    if not needle:
        return []
    pattern = r'\b' + re.escape(needle.lower()) + r'\b'
    return [m.start() for m in re.finditer(pattern, haystack_lower)]


def _significant_words(text: str, min_len: int = 5) -> list:
    """Extracts 'distinctive' words from text for proximity checks — excludes short/common words that would make a false-positive proximity
    match trivially easy to hit."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOP_WORDS]


def _proximity_match(occurrence_indices: list, haystack_lower: str, keywords: list, window: int = 200) -> bool:
    """Checks whether any of `keywords` appears within `window` characters of any occurrence of the anchor term. This is what turns 'does this
    date/name exist anywhere in the document' into 'does this date/name exist near the words describing what it's supposedly attached to' —
    the actual fix for the existence-only grounding problem."""
    for idx in occurrence_indices:
        window_text = haystack_lower[max(0, idx - window): idx + window]
        if any(k in window_text for k in keywords):
            return True
    return False


def _extract_first_date(text: str) -> Optional[str]:
    """First recognizable date in text, preferring a full date (month/day/year, slash, dash) over a bare 4-digit year — and only accepting a bare
    year in a plausible 1900-2099 range, since an unfiltered \\d{4} match also catches docket numbers and dollar amounts."""
    text_lower = text.lower()
    for pattern in DATE_PATTERNS[:-1]:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0)
    match = re.search(DATE_PATTERNS[-1], text_lower)
    if match and 1900 <= int(match.group(0)) <= 2099:
        return match.group(0)
    return None


def _looks_like_decline(answer: str) -> bool:
    """True if the model explicitly said the information wasn't in the document, rather than fabricating something to check."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in DECLINE_PHRASES)


def _extract_section(extraction: str, header: str) -> str:
    """Pulls the text between `header` and whichever other known section header comes next (or the end of the extraction if none does)."""
    if header not in extraction:
        return ""
    after = extraction.split(header, 1)[1]
    for other_header in ALL_SECTION_HEADERS:
        if other_header != header and other_header in after:
            after = after.split(other_header)[0]
    return after

_embedding_model = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _split_sentences(text: str) -> list:
    """Simple sentence splitter — good enough for semantic-similarity scoring, not meant to handle every edge case of legal citation punctuation perfectly."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list:
    """Overlapping chunks so a fact split across a chunk boundary still has a decent chance of being fully captured in at least one chunk."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def compute_semantic_grounding(answer: str, raw_text: str) -> Optional[float]:
    """
    Semantic grounding via sentence embeddings — NOT a correctness or legal-reasoning checker. For each sentence in the answer, finds its
    best-matching passage anywhere in the source text (cosine similarity) and takes the WEAKEST of those best-matches across all sentences —
    a lawyer needs to know about the worst-supported claim in the answer, not be reassured by strong matches elsewhere.

    This catches something d_score/n_score cannot: an answer where every individual date and name technically appears in the document, but the
    sentence stitching them together isn't actually what the source says.

    HARD LIMITATION, stated plainly: this measures topical/lexical overlap with the source, not whether the legal conclusion drawn is correct. A
    fluently-written, well-grounded-sounding answer that draws a wrong legal conclusion from correctly-cited text can still score well here.
    This raises the ceiling of what's automatically checkable — it does not replace a lawyer's own review of the answer.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return None

    chunks = _chunk_text(raw_text)
    if not chunks:
        return None

    model = _get_embedding_model()
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    weakest_best_match = 1.0
    for sentence_embedding in sentence_embeddings:
        similarities = util.cos_sim(sentence_embedding, chunk_embeddings)[0]
        best_match_for_this_sentence = float(similarities.max())
        weakest_best_match = min(weakest_best_match, best_match_for_this_sentence)

    return round(max(0.0, weakest_best_match), 3)


def extract_text_from_pdf(file_path: str) -> tuple:
    """
    Extracts raw text from PDF. 
    Returns (text, page_count, error)
    """
    try:
        reader = PdfReader(file_path)
        page_count = len(reader.pages)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text = " ".join(text.split())
        if len(text) < 100:
            return "", page_count, "Could not extract readable text from PDF. File may be scanned or image-based."

        return text, page_count, None

    except Exception as e:
        return "", 0, f"Error reading PDF: {str(e)}"


def extract_structured_facts(raw_text: str, file_name: str) -> dict:
    """
    Forces LLM to extract three specific objects:
    A. Chronological timeline of events and filings
    B. Cast of characters — names, titles, roles
    C. Core causes of action or procedural claims
    Uses only the first MAX_CHARS_FOR_LLM characters to stay within token limits.
    """
    text_sample = raw_text[:MAX_CHARS_FOR_LLM]

    prompt = f"""You are a legal document analyst. Extract structured facts from the following legal document.

DOCUMENT: {file_name}

SOURCE TEXT:
{text_sample}

Extract ONLY what is explicitly stated in the text above. Do not infer or add anything not present.

Return your extraction in exactly this format:

CASE NAME: [full case name if present, else "Not stated"]
COURT: [court name if present, else "Not stated"]
DOCUMENT TYPE: [complaint / opinion / brief / answer / motion / other]

TIMELINE:
[Every date or time reference found, with the event it describes. One bullet per entry — format each line as "- DATE — EVENT". If no dates found write "No explicit dates found."]

CAST OF CHARACTERS:
[Every named person or organisation with their role. One bullet per entry — format each line as "- NAME — ROLE". If none found write "No named parties found."]

CAUSES OF ACTION:
[Every legal claim or cause of action explicitly stated. One bullet per entry — format each line as "- CLAIM — BASIS". If none found write "No explicit claims found."]

KEY FACTS:
[3 to 5 most important factual assertions from the document, each in one sentence.]

Do NOT add closing notes or meta-commentary."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = _strip_think_tags(response.content)
        text += _truncation_notice(raw_text)
        return {"extraction": text, "error": None}
    except Exception as e:
        return {"extraction": "", "error": str(e)}


def compute_chronological_integrity(extraction: str, raw_text: str) -> float:
    """
    C — 50%
    Checks each TIMELINE line's date against the source document, AND checks that a distinctive word from that same line's event description
    appears near the date in the source — not just that the date string exists somewhere in the document. Existence-only checking previously
    scored a date as "verified" even when it was attached to the wrong event (the exact failure mode found in manual review of a prior case
    summary — a real date, verified as present, but mapped to the wrong fact).

    Per-line scoring: 1.0 if date exists and is near matching event words, 0.5 if the date exists but not clearly linked to this event, 0.0 if the
    date doesn't appear in the source at all.
    """
    raw_lower = raw_text.lower()

    timeline_section = ""
    if "TIMELINE:" in extraction:
        parts = extraction.split("TIMELINE:")
        if len(parts) > 1:
            next_section = parts[1]
            for section_header in ["CAST OF CHARACTERS:", "CAUSES OF ACTION:", "KEY FACTS:"]:
                if section_header in next_section:
                    next_section = next_section.split(section_header)[0]
            timeline_section = next_section

    if "no explicit dates found" in timeline_section.lower():
        return 1.0  

    lines = [l.strip() for l in timeline_section.split("\n") if l.strip()]
    total = 0
    verified_weight = 0.0

    for line in lines:
        if "—" in line:
            date_part, event_part = line.split("—", 1)
        elif "-" in line:
            date_part, event_part = line.split("-", 1)
        else:
            continue  

        date_str = _extract_first_date(date_part)
        if not date_str:
            continue
        total += 1

        occurrences = _find_all_occurrences(date_str, raw_lower)
        if not occurrences:
            continue  

        event_words = _significant_words(event_part)
        if not event_words:
            verified_weight += 1.0
        elif _proximity_match(occurrences, raw_lower, event_words):
            verified_weight += 1.0
        else:
            verified_weight += 0.5

    if total == 0:
        return 1.0

    return round(verified_weight / total, 3)


def compute_entity_grounding(extraction: str, raw_text: str) -> float:
    """
    E — 50%
    Same proximity-based approach as compute_chronological_integrity, applied to CAST OF CHARACTERS lines: checks the name exists AND that a
    distinctive word from the stated role appears near it in the source, rather than just checking the name exists anywhere in the document.
    Also fixes a word-boundary bug — the old substring check let a short name like "Roe" match inside unrelated words like "wardrobe".
    """
    raw_lower = raw_text.lower()

    cast_section = ""
    if "CAST OF CHARACTERS:" in extraction:
        parts = extraction.split("CAST OF CHARACTERS:")
        if len(parts) > 1:
            next_section = parts[1]
            for section_header in ["CAUSES OF ACTION:", "KEY FACTS:", "TIMELINE:"]:
                if section_header in next_section:
                    next_section = next_section.split(section_header)[0]
            cast_section = next_section

    if "no named parties found" in cast_section.lower():
        return 1.0

    lines = [l.strip() for l in cast_section.split("\n") if l.strip()]
    total = 0
    verified_weight = 0.0

    for line in lines:
        if "—" in line:
            name_part, role_part = line.split("—", 1)
        elif "-" in line:
            name_part, role_part = line.split("-", 1)
        else:
            continue

        name = name_part.strip().lstrip("-").strip().strip("*[]").strip()
        if len(name) <= 2:
            continue
        total += 1

        occurrences = _find_all_occurrences(name, raw_lower)
        if not occurrences:
            for word in [w for w in name.lower().split() if len(w) > 3]:
                occurrences.extend(_find_all_occurrences(word, raw_lower))

        if not occurrences:
            continue  

        role_words = _significant_words(role_part)
        if not role_words:
            verified_weight += 1.0
        elif _proximity_match(occurrences, raw_lower, role_words):
            verified_weight += 1.0
        else:
            verified_weight += 0.5

    if total == 0:
        return 1.0

    return round(verified_weight / total, 3)


def compute_summary_confidence(extraction: str, raw_text: str) -> dict:
    c_score = compute_chronological_integrity(extraction, raw_text)
    e_score = compute_entity_grounding(extraction, raw_text)

    key_facts = _extract_section(extraction, "KEY FACTS:")
    causes = _extract_section(extraction, "CAUSES OF ACTION:")
    combined = (key_facts + " " + causes).strip()
    s_score = compute_semantic_grounding(combined, raw_text) if combined else None

    scores = {"c_score": c_score, "e_score": e_score, "s_score": s_score}
    available = {k: v for k, v in scores.items() if v is not None}
    final_score = round(sum(available.values()) / len(available), 3)
    threshold = 0.75

    if final_score >= 0.90:
        rating = "HIGH INTEGRITY — Verified"
        flag = "HIGH INTEGRITY"
    elif final_score >= 0.75:
        rating = "MODERATE — Spot-check recommended"
        flag = "MODERATE"
    else:
        rating = "LOW INTEGRITY — Manual review required"
        flag = "LOW INTEGRITY"

    return {
        "final_score": final_score,
        "c_score": c_score,
        "e_score": e_score,
        "s_score": s_score,
        "rating": rating,
        "flag": flag,
        "above_threshold": final_score >= threshold,
        "threshold": threshold,
        "task": "summarizer",
        "explanation": (
            f"Chronological: {c_score} | Entity: {e_score} | Semantic (Key Facts/Causes): {s_score} "
            f"→ {final_score} (averaged over {len(available)} available signal(s))"
        )
    }


def compute_qa_confidence(answer: str, raw_text: str) -> dict:
    """
    Dedicated confidence check for the Q&A branch — deliberately NOT compute_summary_confidence, which silently returned ~1.0 for every
    Q&A answer regardless of correctness. That function only finds checkable content inside "TIMELINE:"/"CAST OF CHARACTERS:" headers,
    which free-text Q&A answers never contain.

    Three independent signals, averaged over whichever are actually present (not a fixed weight per signal — see the "weight" comment
    below, which matters for how this is displayed):
      - d_score: date grounding (existence-only, no structured lines to anchor a proximity check against in free-text prose)
      - n_score: proper-noun grounding (word-boundary safe, filters out ordinary sentence-initial capitalized words)
      - s_score: semantic grounding via embeddings (see compute_semantic_grounding) — catches answers that pass the
        first two checks on individual facts but aren't actually what the source says when stitched together

    checkable_claims_found is False only when ALL THREE are unavailable — with semantic grounding added, that's now rare, since it works even
    on answers with no dates or names at all.
    """
    raw_lower = raw_text.lower()

    if _looks_like_decline(answer):
        return {
            "final_score": 1.0,
            "d_score": None,
            "n_score": None,
            "s_score": None,
            "rating": "DECLINED — model reported the answer isn't in the document",
            "flag": "DECLINED",
            "above_threshold": True,
            "threshold": 0.75,
            "task": "qa",
            "checkable_claims_found": False,
            "explanation": "Model explicitly stated the information wasn't found — nothing to verify against source text."
        }

    raw_dates = set()
    for pattern in DATE_PATTERNS:
        raw_dates.update(re.findall(pattern, answer.lower()))
    valid_dates = [d for d in raw_dates if not (d.isdigit() and not (1900 <= int(d) <= 2099))]

    d_score = None
    if valid_dates:
        d_verified = sum(1 for d in valid_dates if d in raw_lower)
        d_score = round(d_verified / len(valid_dates), 3)

    candidates = set(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", answer))
    proper_nouns = [
        p for p in candidates
        if len(p) > 2 and (" " in p or p.lower() not in SENTENCE_STARTER_WORDS)
    ]

    n_score = None
    if proper_nouns:
        n_verified = sum(1 for p in proper_nouns if _find_all_occurrences(p, raw_lower))
        n_score = round(n_verified / len(proper_nouns), 3)

    s_score = compute_semantic_grounding(answer, raw_text)

    scores = {"d_score": d_score, "n_score": n_score, "s_score": s_score}
    available = {k: v for k, v in scores.items() if v is not None}

    if not available:
        return {
            "final_score": None,
            "d_score": None,
            "n_score": None,
            "s_score": None,
            "rating": "UNVERIFIABLE — no checkable content in the answer",
            "flag": "UNVERIFIABLE",
            "above_threshold": False,
            "threshold": 0.75,
            "task": "qa",
            "checkable_claims_found": False,
            "explanation": "No dates, named entities, or matchable sentence content in the answer — "
                           "this reflects a limitation of the check, not evidence the answer is wrong."
        }

    final_score = round(sum(available.values()) / len(available), 3)
    threshold = 0.75

    if final_score >= 0.90:
        rating, flag = "HIGH INTEGRITY — Verified", "HIGH INTEGRITY"
    elif final_score >= 0.75:
        rating, flag = "MODERATE — Spot-check recommended", "MODERATE"
    else:
        rating, flag = "LOW INTEGRITY — Manual review required", "LOW INTEGRITY"

    return {
        "final_score": final_score,
        "d_score": d_score,
        "n_score": n_score,
        "s_score": s_score,
        "rating": rating,
        "flag": flag,
        "above_threshold": final_score >= threshold,
        "threshold": threshold,
        "task": "qa",
        "checkable_claims_found": True,
        "explanation": f"Date: {d_score} | Entity: {n_score} | Semantic: {s_score} → {final_score} "
                       f"(averaged over {len(available)} available signal(s))"
    }


def answer_document_question(raw_text: str, question: str, file_name: str) -> str:
    text_sample = raw_text[:MAX_CHARS_FOR_LLM]

    prompt = f"""You are a legal document analyst.
The lawyer has uploaded: {file_name}

DOCUMENT TEXT:
{text_sample}

The lawyer asks: {question}

Answer using ONLY information explicitly stated in the document above.
Do not add anything from your own knowledge.
If the answer is not in the document say so clearly.
Do NOT add closing notes or meta-commentary."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = _strip_think_tags(response.content)
        return text + _truncation_notice(raw_text)
    except Exception as e:
        return f"Could not answer question: {str(e)}"

def run_document_analysis(file_path: str, question: str = None) -> dict:
    """
    Full pipeline:
    1. Extract text from PDF
    2. Extract structured facts OR answer a specific question
    3. Verify against source
    4. Compute confidence score
    Returns everything needed for main.py to build the response.
    """
    file_name = os.path.basename(file_path)

    # Text extraction
    raw_text, page_count, extract_error = extract_text_from_pdf(file_path)

    if extract_error:
        return {
            "error": extract_error,
            "file_name": file_name,
            "page_count": 0,
            "extraction": "",
            "raw_text": "",
            "confidence": {
                "final_score": 0.0,
                "flag": "FAILED",
                "rating": "Text extraction failed",
                "c_score": 0.0,
                "s_score": None,
                "e_score": 0.0,
                "above_threshold": False,
                "threshold": 0.75,
                "task": "summarizer"
            }
        }

    # If specific question asked — answer directly without full summary
    if question:
        answer = answer_document_question(raw_text, question, file_name)
        confidence = compute_qa_confidence(answer, raw_text)
        return {
            "error": None,
            "file_name": file_name,
            "page_count": page_count,
            "extraction": answer,
            "raw_text": raw_text[:MAX_CHARS_FOR_LLM],
            "question": question,
            "confidence": confidence
        }

    # Structured fact extraction
    result = extract_structured_facts(raw_text, file_name)

    if result["error"]:
        return {
            "error": result["error"],
            "file_name": file_name,
            "page_count": page_count,
            "extraction": "",
            "raw_text": raw_text,
            "confidence": {
                "final_score": 0.0,
                "flag": "FAILED",
                "rating": "Fact extraction failed",
                "c_score": 0.0,
                "s_score": None,
                "e_score": 0.0,
                "above_threshold": False,
                "threshold": 0.75,
                "task": "summarizer"
            }
        }

    extraction = result["extraction"]

    # Confidence scoring
    confidence = compute_summary_confidence(extraction, raw_text)

    return {
        "error": None,
        "file_name": file_name,
        "page_count": page_count,
        "extraction": extraction,
        "raw_text": raw_text[:MAX_CHARS_FOR_LLM],  # Store for Q&A follow-ups
        "confidence": confidence
    }