import os
import re
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

log = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1, 
    api_key=os.getenv("GROQ_API_KEY")
)

DEFENSE_KEYWORDS = {
    "good_faith": ["good faith", "willful", "intentional", "deliberate", "scienter", "knew of", "was aware of"],
    "lack_of_intent": ["mens rea", "criminal intent", "knowingly", "reckless disregard","willful blindness", "unintentional"],
    "statute_of_limitations": ["statute of limitations", "limitations period", "time-barred", "time barred", "untimely filing", "filing deadline",
        "equitable tolling"],
    "safe_harbour": ["safe harbor", "safe harbour", "statutory exemption", "immunity", "qualified immunity", "privilege", "protected activity"],
    "standing": ["standing to sue", "lack of standing", "injury in fact", "concrete and particularized", "aggrieved party", "actual damages"],
}

DEFENSE_LABELS = {
    "good_faith": "Good faith defence",
    "lack_of_intent": "Lack of intent",
    "statute_of_limitations": "Statute of limitations",
    "safe_harbour": "Safe harbour / exemption",
    "standing": "Standing / injury",
}


def _stem_or_exact_match(term: str, arg_lower: str, words: list) -> bool:
    """Multi-word phrases: exact substring match (stemming per-word in a phrase is unreliable). Long single words (>=8 chars): match by stem
    (first len-3 characters), so morphological variants — retaliation/retaliated/retaliatory, discrimination/discriminated/discriminatory —
    are caught without needing every variant of every term listed by hand, which is exactly what missed "retaliated" when only
    "retaliation" was in the list. Short single words (<8 chars, e.g. "act", "code"): exact whole-word match only — stemming something
    that short would match unrelated words too easily ("act" stemmed would match "action", "actual", "activity")."""
    if " " in term:
        return term in arg_lower
    if len(term) < 8:
        return bool(re.search(r'\b' + re.escape(term) + r'\b', arg_lower))
    stem = term[:len(term) - 3]
    return any(w.startswith(stem) for w in words)


def compute_statutory_grounding(argument: str) -> float:
    arg_lower = argument.lower()

    usc_match = re.search(
        r'(\d+)\s*(?:u\.s\.c\.?|usc)\s*[§s]?\s*(\d+[a-z]?)',
        arg_lower
    )
    if usc_match:
        return 1.0

    rule_match = re.search(
        r'(?:rule|fre|frcp|frap)\s*(\d+[a-z]?)',
        arg_lower
    )
    if rule_match:
        return 1.0

    legal_terms = [
        "title vii", "ada", "cfaa", "fmla", "flsa", "erisa", "first amendment", "fourth amendment", "fifth amendment", "due process", 
        "equal protection", "retaliation", "discrimination", "negligence", "breach of contract", "unconscionable", "fiduciary duty", 
        "tortious interference", "statute", "section", "code", "act", "regulation", "provision", "pursuant to", "violation of", 
        "civil rights act", "sherman act", "securities act", "fair labor", "amended by", "under the law", "federal law", "common law", 
        "constitutional", "statutory"
    ]
    words = re.findall(r"[a-z]+", arg_lower)
    if any(_stem_or_exact_match(term, arg_lower, words) for term in legal_terms):
        return 0.5

    return 0.0


def compute_precedent_support(retrieved_cases: list) -> float:
    """Takes already-retrieved cases instead of searching itself — see run_full_analysis, which now does ONE search_past_cases call shared
    across this function, analyze_counter_arguments, and evaluate_proof_strength. Previously each called search_past_cases
    independently with the identical query, running the same embedding + nearest-neighbor search three times for the same result."""
    if not retrieved_cases:
        return 0.1

    try:
        similarities = [r["similarity_score"] for r in retrieved_cases]
        avg_similarity = sum(similarities) / len(similarities)
    except Exception as e:
        log.error(f"Malformed retrieved_cases in compute_precedent_support: {e}")
        return 0.0

    if avg_similarity >= 0.55:
        return 1.0
    elif avg_similarity >= 0.40:
        return 0.7
    elif avg_similarity >= 0.30:
        return 0.4
    else:
        return 0.1


def compute_vulnerability_exposure(argument: str) -> float:
    arg_lower = argument.lower()
    word_count = len(argument.split())

    addressed = 0
    for defence, keywords in DEFENSE_KEYWORDS.items():
        if any(kw in arg_lower for kw in keywords):
            addressed += 1

    if word_count < 50:
        expected = 2
    elif word_count < 150:
        expected = 3
    else:
        expected = 5

    return min(round(addressed / expected, 2), 1.0)


def compute_strategic_strength(argument: str, retrieved_cases: list) -> dict:
    s = compute_statutory_grounding(argument)
    p = compute_precedent_support(retrieved_cases)
    v = compute_vulnerability_exposure(argument)

    final_score = round((s * 0.4) + (p * 0.4) + (v * 0.2), 3)

    if final_score >= 0.8:
        rating = "Strong Legal Position"
        flag = "STRONG"
    elif final_score >= 0.5:
        rating = "Moderate — Needs Strengthening"
        flag = "MODERATE"
    else:
        rating = "High Vulnerability Warning"
        flag = "HIGH RISK"

    return {
        "final_score": final_score,
        "rating": rating,
        "flag": flag,
        "s_score": s,
        "p_score": p,
        "v_score": v,
        "above_threshold": final_score >= 0.8,
        "threshold": 0.8,
        "task": "mock_court",
        "explanation": (
            f"Statutory Grounding: {s} × 0.4 + "
            f"Precedent Support: {p} × 0.4 + "
            f"Vulnerability Exposure: {v} × 0.2 = {final_score}"
        )
    }


def analyze_counter_arguments(argument_text: str, retrieved_cases: list) -> dict:
    case_context = ""
    if retrieved_cases:
        case_context = "\n\n".join([
            f"Case: {r['case_name']} ({r['citation']})\n{r['text'][:500]}"
            for r in retrieved_cases
        ])

    prompt = f"""You are a senior litigation analyst stress-testing a lawyer's argument.

LAWYER'S ARGUMENT: {argument_text}

RELEVANT CASE LAW FROM DATABASE: {case_context if case_context else "No directly relevant cases found in database."}

Identify the TOP 3 most aggressive and specific counter-arguments opposing counsel will raise against this argument.

For each counter-argument:
1. State the counter-argument precisely
2. Identify the legal basis — statute, case, or doctrine
3. Explain why it weakens the lawyer's position

Use ONLY legal reasoning grounded in the cases above or established legal doctrine.
Do not invent cases. If no case supports a counter-argument, state the doctrinal basis.
Be clinical and specific. Do NOT add meta-commentary."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return {"analysis": text, "retrieved_cases": retrieved_cases}
    except Exception as e:
        log.error(f"analyze_counter_arguments LLM call failed: {e}")
        return {"analysis": f"Could not generate analysis: {str(e)}", "retrieved_cases": retrieved_cases}


def evaluate_proof_strength(argument_text: str, retrieved_cases: list) -> dict:
    s_score = compute_statutory_grounding(argument_text)

    case_context = ""
    if retrieved_cases:
        case_context = "\n\n".join([
            f"Case: {r['case_name']} ({r['citation']}, similarity: {r['similarity_score']:.2f})\n{r['text'][:500]}"
            for r in retrieved_cases
        ])

    if s_score == 1.0:
        statutory_note = (
            "A statute or rule citation in the correct FORMAT was found in the argument. "
            "This does NOT verify the citation is the correct or legally accurate section "
            "for this claim — only that citation-shaped text is present."
        )
    elif s_score == 0.5:
        statutory_note = "Legal terminology present but no specific citation found."
    else:
        statutory_note = "No statutory anchor detected. Argument relies on facts alone."

    prompt = f"""You are a legal evidence analyst evaluating the proof strength of a lawyer's argument.

LAWYER'S ARGUMENT:
{argument_text}

STATUTORY ASSESSMENT: {statutory_note}

RELEVANT CASES FROM DATABASE:
{case_context if case_context else "No directly relevant cases found in database."}

Evaluate the proof strength of this argument by answering:
1. Is the argument legally binding or speculative? Why?
2. Which parts are well-supported by the cases or statutes above?
3. Which parts are unsupported and could be dismissed as speculation?
4. What specific evidence would make this argument stronger?

Be precise and clinical. Ground every assessment in the cases or statutes provided.
Do NOT invent citations. Do NOT add meta-commentary."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return {
            "analysis": text,
            "statutory_note": statutory_note,
            "retrieved_cases": retrieved_cases
        }
    except Exception as e:
        log.error(f"evaluate_proof_strength LLM call failed: {e}")
        return {
            "analysis": f"Could not generate analysis: {str(e)}",
            "statutory_note": statutory_note,
            "retrieved_cases": retrieved_cases
        }


def identify_judicial_gaps(argument_text: str) -> dict:
    v_score = compute_vulnerability_exposure(argument_text)

    arg_lower = argument_text.lower()
    unaddressed = [
        DEFENSE_LABELS[defence]
        for defence, keywords in DEFENSE_KEYWORDS.items()
        if not any(kw in arg_lower for kw in keywords)
    ]

    prompt = f"""You are a hostile federal judge reviewing a lawyer's argument for weaknesses.

LAWYER'S ARGUMENT:
{argument_text}

UNADDRESSED LEGAL DEFENCES DETECTED:
{chr(10).join(f"- {d}" for d in unaddressed) if unaddressed else "All standard defences appear addressed."}

Identify the specific judicial gaps in this argument:
1. What logical leaps or unsupported assumptions does the argument make?
2. What questions will a hostile judge ask from the bench?
3. What elements of the legal standard are missing from this argument?
4. What is the single most dangerous weakness opposing counsel will exploit?

Be aggressive and specific. Think like a judge who is skeptical of this argument.
Do NOT add meta-commentary or closing notes."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return {
            "analysis": text,
            "unaddressed_defences": unaddressed,
            "v_score": v_score
        }
    except Exception as e:
        log.error(f"identify_judicial_gaps LLM call failed: {e}")
        return {
            "analysis": f"Could not generate analysis: {str(e)}",
            "unaddressed_defences": unaddressed,
            "v_score": v_score
        }


def _deduplicate_cases(cases: list) -> list:
    """ChromaDB stores cases in chunks — the same case can be returned as multiple separate 'results' if several of its chunks each match the
    query well. Left undeduplicated, this both inflates compute_precedent_support's average similarity (three chunks of one case reads as 
    three independent supporting cases) and makes the 'Cases cross-referenced' list show the same case repeated. Keeps only the 
    highest-similarity chunk per unique case (grouped by citation, falling back to case_name if citation is missing), sorted
    by similarity descending."""
    best_by_case = {}
    for c in cases:
        key = c.get("citation") or c.get("case_name") or repr(c)
        if key not in best_by_case or c.get("similarity_score", 0) > best_by_case[key].get("similarity_score", 0):
            best_by_case[key] = c
    return sorted(best_by_case.values(), key=lambda c: c.get("similarity_score", 0), reverse=True)


def run_full_analysis(argument_text: str) -> dict:
    try:
        from tasks.past_cases import search_past_cases
        results = search_past_cases(query=argument_text, n_results=3)
        retrieved_cases = _deduplicate_cases(results.get("results", []))
    except Exception as e:
        log.error(f"search_past_cases failed in run_full_analysis: {e}")
        retrieved_cases = []

    counter = analyze_counter_arguments(argument_text, retrieved_cases)
    proof = evaluate_proof_strength(argument_text, retrieved_cases)
    gaps = identify_judicial_gaps(argument_text)
    score = compute_strategic_strength(argument_text, retrieved_cases)

    return {
        "argument": argument_text,
        "counter_arguments": counter["analysis"],
        "proof_strength": proof["analysis"],
        "judicial_gaps": gaps["analysis"],
        "unaddressed_defences": gaps["unaddressed_defences"],
        "statutory_note": proof["statutory_note"],
        "retrieved_cases": retrieved_cases,
        "score": score
    }