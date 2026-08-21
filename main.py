import os
import re
import logging
from pathlib import Path
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tasks.past_cases import search_past_cases, compute_past_cases_confidence
from tasks.multi_tool_runner import run_multi_tool_query

from tasks.statutory_api import (
    fetch_statute,
    fetch_legal_definition,
    fetch_federal_rule,
    fetch_constitutional_amendment,
    AMENDMENT_ORDINALS,
    calculate_legal_confidence,
    is_content_truncated,
    STATUTE_CHAR_LIMIT,
    CORNELL_CHAR_LIMIT
)
from tasks.mock_court import run_full_analysis
from tasks.summarize import run_document_analysis, compute_semantic_grounding, _looks_like_decline
from router_llm import router_node_llm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger(__name__)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)

LEGAL_PRESENTATION_PROMPT = """You are a legal information assistant for a law firm.
You have retrieved the following authenticated legal text from an official source.
Your job is to present this information clearly and professionally to the lawyer.

STRICT RULES:
- Use ONLY the text provided below. Do not add anything from your own knowledge.
- Do not add case examples, interpretations, or context not present in the retrieved text.
- Do not paraphrase the core legal text — present it accurately.
- You may add clear headers and structure to make it readable.
- Every legal claim in your response must come directly from the retrieved text.
- Do NOT add any closing notes, disclaimers, or meta-commentary about your process.
- Do NOT include page metadata, keywords, navigation tags, or website labels from the source page.

RETRIEVED LEGAL TEXT:
{raw_text}

SOURCE: {source_url}

Present this clearly to the lawyer now."""

FOLLOWUP_PROMPT = """You are a legal information assistant for a law firm.
The lawyer previously retrieved the following authenticated legal text:

SOURCE: {source_url}

RETRIEVED LEGAL TEXT:
{raw_text}

CONVERSATION SO FAR (a compressed summary, for resolving references like
"the first one" / "the second one" / "it" back to the correct earlier
topic -- use this ONLY to figure out WHAT the lawyer is referring to,
not as a source of legal facts to state as fact):
{thread_summary}

The lawyer now asks: {question}

Answer using ONLY the information in the RETRIEVED LEGAL TEXT above.
Use the conversation summary only to understand what "it"/"the first
one"/"the second one" refers to -- if the retrieved text above doesn't
actually cover that specific thing, say so clearly rather than guessing.
Do not add anything from your own knowledge.
Do NOT add any closing notes or meta-commentary."""


class AgentState(TypedDict):
    query: str
    intent: str
    parsed_parameters: dict
    raw_api_text: Optional[str]
    source_url: Optional[str]
    source_domain: Optional[str]
    llm_output: str
    confidence_score: float
    final_response: str
    error: Optional[str]
    last_raw_text: Optional[str]
    last_source_url: Optional[str]
    last_intent: Optional[str]
    last_params: Optional[dict]
    thread_summary: Optional[str]
    retrieved_cases: Optional[list]
    mock_result: Optional[dict]
    document_result: Optional[dict]
    retry_feedback: Optional[str]
    best_response: Optional[str]
    best_intent: Optional[str]
    best_confidence: Optional[float]
    user_threshold: Optional[float]
    is_possibly_compound: Optional[bool]

def _validate_pdf_path(file_path: str, allowed_root: Optional[str] = None) -> tuple:
    """Guardrail: rejects path traversal, non-PDF, and non-existent files before anything tries to open them."""
    if not file_path:
        return False, "No file path provided."
    if ".." in Path(file_path).parts:
        return False, "Path traversal detected — rejected."
    try:
        candidate = Path(file_path).expanduser().resolve()
    except Exception as e:
        return False, f"Invalid path: {e}"
    if not candidate.exists():
        return False, f"File not found: {candidate}"
    if not candidate.is_file():
        return False, f"Not a file: {candidate}"
    if candidate.suffix.lower() != ".pdf":
        return False, f"Expected a .pdf file, got: {candidate.suffix}"
    if allowed_root:
        allowed_root_resolved = Path(allowed_root).expanduser().resolve()
        if allowed_root_resolved not in candidate.parents and candidate != allowed_root_resolved:
            return False, f"Path outside allowed directory: {candidate}"
    return True, str(candidate)
 

def api_execution_node(state: AgentState) -> dict:
    intent = state["intent"]
    params = state["parsed_parameters"]
    query = state["query"]
    api_key = os.getenv("GOVINFO_API_KEY", "DEMO_KEY")

    if intent == "summarizer":
        file_path = params.get("file_path", "").strip()
        question = params.get("question")
        wants_summary = params.get("wants_summary", True)

        is_valid, resolved_or_error = _validate_pdf_path(file_path)
        if not is_valid:
            log.warning(f"Rejected file path for summarizer intent: {resolved_or_error}")
            return {
                "raw_api_text": "",
                "source_url": "",
                "source_domain": "",
                "document_result": {"error": resolved_or_error},
                "error": resolved_or_error
            }
        file_path = resolved_or_error

        result = run_document_analysis(file_path, question=question)
        return {
            "raw_api_text": result.get("raw_text", ""),
            "source_url": f"Document: {os.path.basename(file_path)}",
            "source_domain": "local",
            "document_result": result,
            "error": result.get("error")
        }
    
    elif intent == "followup":
        return {
            "raw_api_text": state.get("last_raw_text") or "",
            "source_url": state.get("last_source_url") or "",
            "source_domain": "law.cornell.edu" if "cornell" in (state.get("last_source_url") or "") else "api.govinfo.gov",
            "error": None
        }

    elif intent == "past_cases":
        results = search_past_cases(
            query=query,
            n_results=5,
            jurisdiction_filter=params.get("jurisdiction")
        )
        return {
            "raw_api_text": "",
            "source_url": "",
            "source_domain": "",
            "retrieved_cases": results.get("results", []),
            "error": results.get("error")
        }
    
    elif intent == "statute":
        raw_text, source_url, domain = fetch_statute(
            title=params["title"],
            section=params["section"],
            api_key=api_key
        )

    elif intent == "mock_court":
        result = run_full_analysis(query)
        return {
            "raw_api_text": "",
            "source_url": "",
            "source_domain": "",
            "mock_result": result,
            "error": None
        }
    
    elif intent == "rule":
        raw_text, source_url, domain = fetch_federal_rule(
            rule_type=params["rule_type"],
            rule_number=params["rule_number"]
        )
    elif intent == "constitution":
        raw_text, source_url, domain = fetch_constitutional_amendment(
            amendment_number=params["amendment_number"]
        )
    elif intent == "definition":
        raw_text, source_url, domain = fetch_legal_definition(
            term=params["term"]
        )
    else:
        return {
            "raw_api_text": "",
            "source_url": "",
            "source_domain": "",
            "error": f"Unknown intent: {intent}"
        }

    return {
        "raw_api_text": raw_text,
        "source_url": source_url,
        "source_domain": domain,
        "error": None
    }


def _strip_think_tags(text: str) -> str:
    """Some models wrap reasoning in <think>...</think> before the real
    answer. Strip it if present. """
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def detect_compound_question_node(state: AgentState) -> dict:
    """Cheap heuristic (no LLM call) that flags when a question likely has
    more than one part, using simple signals like "and"/"also"/a second
    question mark near a second question word. The system still only
    answers ONE part per query — this node doesn't fix that, it makes the
    limitation visible to the lawyer instead of silently returning a
    half-answer that reads as complete. Real multi-part support (splitting
    into sub-questions, fetching each, merging results) is a larger change
    spanning router_llm.py, api_execution_node, and verification_node —
    tracked as separate future work, not done here."""
    query = state.get("query", "")
    query_lower = query.lower()

    second_question_markers = [
        " and how ", " and does ", " and what ", " and can ", " and is ",
        " and are ", " and which ", " and who ", " and when ", " and why ",
        "? also,", "? also ", "; also,", "; also ",
    ]
    has_second_question_mark = query_lower.count("?") >= 2
    has_conjunction_marker = any(m in query_lower for m in second_question_markers)

    return {"is_possibly_compound": has_second_question_mark or has_conjunction_marker}


def _verification_node_core(state: AgentState) -> dict:
    if state.get("error"):
        return {
            "final_response": f"Error: {state['error']}",
            "confidence_score": 0.0,
            "llm_output": "",
            "last_raw_text": state.get("last_raw_text"),
            "last_source_url": state.get("last_source_url"),
            "last_intent": state.get("last_intent"),
            "last_params": state.get("last_params")
        }

    raw_text = state["raw_api_text"]
    source_url = state["source_url"]
    source_domain = state["source_domain"]
    intent = state["intent"]
    params = state["parsed_parameters"]
    query = state["query"]

    if intent == "summarizer":
        doc = state.get("document_result", {})
        if not doc or doc.get("error"):
            error_msg = doc.get("error", "Document analysis failed") if doc else "No result"
            return {
                "final_response": f"Document analysis failed: {error_msg}",
                "confidence_score": 0.0,
                "last_raw_text": state.get("last_raw_text"),
                "last_source_url": state.get("last_source_url"),
                "last_intent": state.get("last_intent"),
                "last_params": state.get("last_params"),
                "document_result": doc
            }

        confidence = doc["confidence"]
        extraction = doc["extraction"]
        file_name = doc["file_name"]
        page_count = doc["page_count"]

        final_score = confidence.get("final_score")
        score_pct = f"{round(final_score * 100)}%" if final_score is not None else "N/A"

        if confidence.get("task") == "qa":
            signal_labels = {
                "d_score": "Date Grounding",
                "n_score": "Entity Grounding",
                "s_score": "Semantic Grounding",
            }
        else:
            signal_labels = {
                "c_score": "Chronological Integrity",
                "e_score": "Entity Grounding",
                "s_score": "Semantic Grounding (Key Facts/Causes)",
            }

        present = {
            signal_labels[key]: confidence.get(key)
            for key in signal_labels
            if confidence.get(key) is not None
        }
        if present:
            weight = round(1.0 / len(present), 3)
            parts = [f"**{label}:** {value} × {weight}" for label, value in present.items()]
            missing_labels = [label for key, label in signal_labels.items() if confidence.get(key) is None]
            note = f" *({' and '.join(missing_labels)} not checkable for this response)*" if missing_labels else ""
            breakdown = " | ".join(parts) + note
        else:
            breakdown = "*No checkable signals available for this response.*"

        final_response = f"""### Legal Document Analysis: {file_name}
**Pages processed:** {page_count} | **Integrity Rating:** {confidence['flag']} {score_pct}

---

{extraction}

---

### Summary Integrity Score: {score_pct}
{breakdown}

**Assessment:** {confidence['rating']}

> This summary extracts only what is explicitly stated in the document. Verify all critical facts against the source before relying on this analysis."""

        return {
            "final_response": final_response,
            "confidence_score": final_score if final_score is not None else 0.0,
            "last_raw_text": doc.get("raw_text", ""),
            "last_source_url": f"Document: {file_name}",
            "last_intent": intent,
            "last_params": params,
            "document_result": doc
        }
    
    if intent == "mock_court":
        mock = state.get("mock_result", {})
        if not mock:
            return {
                "final_response": "Could not run argument analysis.",
                "confidence_score": 0.0,
                "last_raw_text": state.get("last_raw_text"),
                "last_source_url": state.get("last_source_url"),
                "last_intent": state.get("last_intent"),
                "last_params": state.get("last_params"),
                "mock_result": None
            }

        score = mock["score"]
        cases = mock.get("retrieved_cases", [])
        case_citations = "\n".join([
            f"- {r['case_name']} ({r['citation']}) — Similarity: {r['similarity_score']:.2f}"
            for r in cases
        ]) if cases else "No directly relevant cases in database."

        unaddressed = mock.get("unaddressed_defences", [])
        unaddressed_text = "\n".join([f"- {d}" for d in unaddressed]) if unaddressed else "None detected."

        final_response = f"""## Argument Vulnerability Analysis

**Argument submitted:**
{mock['argument']}

---

### 1. Counter-Arguments Opposing Counsel Will Raise
{mock['counter_arguments']}

---

### 2. Proof Strength Evaluation
**Statutory Assessment:** {mock['statutory_note']}

{mock['proof_strength']}

---

### 3. Judicial Gaps
**Unaddressed defences:**
{unaddressed_text}

{mock['judicial_gaps']}

---

### Strategic Strength Rating: {score['flag']} {round(score['final_score'] * 100)}% — {score['rating']}

**Score Breakdown:**
Statutory Grounding: {score['s_score']} × 0.4 | Precedent Support: {score['p_score']} × 0.4 | Vulnerability Exposure: {score['v_score']} × 0.2

**Cases cross-referenced:**
{case_citations}"""

        return {
            "final_response": final_response,
            "confidence_score": score["final_score"],
            "last_raw_text": mock["counter_arguments"],
            "last_source_url": "MockCourt — Vulnerability Analyzer",
            "last_intent": intent,
            "last_params": params,
            "mock_result": mock
        }
    
    # Past cases — different flow
    if intent == "past_cases":
        retrieved_cases = state.get("retrieved_cases", [])
        if not retrieved_cases:
            return {
                "final_response": "No relevant cases found in the database for this query.",
                "confidence_score": 0.0,
                "llm_output": "",
                "last_raw_text": state.get("last_raw_text"),
                "last_source_url": state.get("last_source_url"),
                "last_intent": state.get("last_intent"),
                "last_params": state.get("last_params"),
                "retrieved_cases": []
            }

        context = "\n\n".join([
            f"Case: {r['case_name']} ({r['citation']}, {r['year']})\n"
            f"Court: {r['court']}\n"
            f"Issues: {r['legal_issues']}\n"
            f"Relevant excerpt:\n{r['text']}"
            for r in retrieved_cases
        ])

        past_cases_prompt = f"""You are a legal research assistant for a law firm.
The lawyer asked: {query}

The following past case excerpts were retrieved from the firm's case database:

{context}

Using ONLY the information in the excerpts above:
1. Answer the lawyer's question directly
2. Reference specific cases by name and citation
3. Do not add anything from your own knowledge
4. If the answer is not in the excerpts say so clearly
Do NOT add closing notes or meta-commentary."""

        try:
            response = llm.invoke([HumanMessage(content=past_cases_prompt)])
            llm_text = _strip_think_tags(response.content)
        except Exception as e:
            llm_text = f"Could not generate answer: {str(e)}"

        confidence = compute_past_cases_confidence(
            query=query,
            retrieved_results=retrieved_cases,
            llm_answer=llm_text,
            requested_jurisdiction=params.get("jurisdiction")
        )

        citations = "\n".join([
            f"- {r['case_name']} ({r['citation']}, {r['year']}) — Similarity: {r['similarity_score']:.2f}"
            for r in retrieved_cases
        ])

        final_response = (
            f"**Past Cases: {query}**\n\n"
            f"{llm_text}\n\n"
            f"---\n"
            f"**Cases Retrieved:**\n{citations}\n\n"
            f"**Confidence Score: {round(confidence['final_score'] * 100)}%**\n"
            f"Vector match: {confidence['m_score']} × 0.4 | "
            f"Jurisdiction: {confidence['j_score']} × 0.3 | "
            f"Grounding: {confidence['g_score']} × 0.3"
        )

        return {
            "final_response": final_response,
            "confidence_score": confidence["final_score"],
            "llm_output": llm_text,
            "last_raw_text": context,
            "last_source_url": "ChromaDB — local case database",
            "last_intent": intent,
            "last_params": params,
            "retrieved_cases": retrieved_cases
        }
    
    if not raw_text or raw_text.startswith("Could not") or raw_text.startswith("Error"):
        return {
            "final_response": raw_text or "No content retrieved.",
            "confidence_score": 0.0,
            "llm_output": "",
            "last_raw_text": state.get("last_raw_text"),
            "last_source_url": state.get("last_source_url"),
            "last_intent": state.get("last_intent"),
            "last_params": state.get("last_params")
        }

    if intent == "statute":
        header = f"**{params['title']} U.S.C. § {params['section']}**"
    elif intent == "rule":
        labels = {
            "fre": "Federal Rules of Evidence",
            "frcp": "Federal Rules of Civil Procedure",
            "frap": "Federal Rules of Appellate Procedure"
        }
        header = f"**{labels.get(params['rule_type'], 'Federal Rules')} — Rule {params['rule_number']}**"
    elif intent == "constitution":
        ordinal = AMENDMENT_ORDINALS.get(params.get("amendment_number"), "").title()
        header = f"**{ordinal} Amendment to the U.S. Constitution**"
    elif intent == "followup":
        header = f"**Follow-up: {query}**"
    else:
        header = f"**Legal Definition: {params.get('term', '').replace('_', ' ').title()}**"

    if intent == "followup":
        prompt = FOLLOWUP_PROMPT.format(
            source_url=source_url,
            raw_text=raw_text,
            question=query,
            thread_summary=state.get("thread_summary") or "(no earlier conversation summary available)",
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            llm_text = _strip_think_tags(response.content)
        except Exception as e:
            llm_text = f"Could not answer follow-up: {str(e)}"

        if _looks_like_decline(llm_text):
            followup_confidence = 1.0
            confidence_label = (
                f"**Confidence Score: 100%** "
                f"(DECLINED — the retrieved source doesn't cover this; "
                f"nothing here to independently verify)"
            )
            authority_label = "✓ (honest decline)"
        else:
            grounding_score = compute_semantic_grounding(llm_text, raw_text)
            followup_confidence = grounding_score if grounding_score is not None else 0.7
            declined_by_grounding = followup_confidence < 0.5
            confidence_label = (
                f"**Confidence Score: {round(followup_confidence * 100)}%** "
                + ("(this answer does not appear well-supported by the previously retrieved text -- verify independently)"
                   if declined_by_grounding else
                   "(answered from previously retrieved authenticated source)")
            )
            authority_label = "⚠" if declined_by_grounding else "✓"

        final_response = (
            f"{header}\n\n{llm_text}\n\n"
            f"**Source:** {source_url}\n\n"
            f"---\n"
            f"{confidence_label}\n"
            f"Authority: {authority_label} | Source: {source_url}"
        )

        return {
            "final_response": final_response,
            "confidence_score": followup_confidence,
            "llm_output": llm_text,
            "last_raw_text": raw_text,
            "last_source_url": source_url,
            "last_intent": intent,
            "last_params": params
        }

    prompt = LEGAL_PRESENTATION_PROMPT.format(
        raw_text=raw_text,
        source_url=source_url
    )
    presentation_failed = False
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        llm_text = _strip_think_tags(response.content)
    except Exception as e:
        log.error(f"Presentation LLM call failed for intent='{intent}', source={source_url}: {e}")
        llm_text = raw_text
        presentation_failed = True

    formatted_output = f"{header}\n\n{llm_text}\n\n**Source:** {source_url}"
    if presentation_failed:
        formatted_output += (
            "\n\n*Note: formatting step failed — showing unformatted source text above.*"
        )

    truncation_limit = STATUTE_CHAR_LIMIT if intent == "statute" else CORNELL_CHAR_LIMIT
    if is_content_truncated(raw_text, truncation_limit):
        formatted_output += (
            f"\n\n*Note: the retrieved source text was {len(raw_text):,} characters "
            f"and may have been cut off at the retrieval limit — verify against the "
            f"full source at the link above if completeness matters.*"
        )

    score = calculate_legal_confidence(
        source_domain=source_domain,
        raw_api_text=raw_text,
        llm_output_text=llm_text,
        source_url=source_url
    )

    final_response = (
        f"{formatted_output}\n\n"
        f"---\n"
        f"**Confidence Score: {round(score * 100)}%**\n"
        f"Authority: {'VERIFIED' if source_domain.endswith('.gov') or source_domain.endswith('.edu') else 'UNVERIFIED'} "
        f"| Source: {source_url}"
    )

    return {
        "final_response": final_response,
        "confidence_score": score,
        "llm_output": llm_text,
        "last_raw_text": raw_text,
        "last_source_url": source_url,
        "last_intent": intent,
        "last_params": params
    }

MAX_ATTEMPTS = int(os.getenv("AGENT_MAX_ATTEMPTS", "1"))  
DEFAULT_RETRY_THRESHOLDS = {
    "statute": 0.5,
    "rule": 0.5,
    "definition": 0.5,
    "past_cases": 0.70,
    "mock_court": 0.8,
    "summarizer": 0.75,
    "followup": 0.0,  
}

def verification_node(state: AgentState) -> dict:
    """Thin wrapper around _verification_node_core — added so the
    compound-question disclaimer can be appended in ONE place, 
    instead of editing every one of the ~20
    intent-specific return statements inside the core function, which
    would be far more error-prone for a one-line addition."""
    result = _verification_node_core(state)

    if state.get("is_possibly_compound") and not state.get("error") and result.get("final_response"):
        result["final_response"] = result["final_response"] + (
            "\n\n---\n*Note: this question looks like it may have more than "
            "one part. The answer above covers what a single lookup could "
            "address — if part of your question isn't covered, please ask "
            "it as a separate follow-up.*"
        )

    return result


def prepare_retry_node(state: AgentState) -> dict:
    """Runs only when should_retry decides to loop back. Clears the
    stale per-attempt fields and writes feedback the router can actually
    reason about on the next pass, instead of blindly repeating the same
    call."""
    intent = state.get("intent")
    error = state.get("error")
    confidence = state.get("confidence_score")
 
    if error:
        reason = f"failed with error: {error}"
    else:
        reason = f"had low confidence ({confidence})"
 
    feedback = (
        f"Previous attempt chose intent='{intent}' and {reason}. "
        "Reconsider whether a different tool fits better, whether the "
        "extracted parameters were correct, or retry the same tool if "
        "this looks like a transient failure."
    )
    new_count = state.get("attempt_count", 0) + 1
    log.info(f"[prepare_retry_node] attempt_count IN={state.get('attempt_count', 0)} -> OUT={state.get('attempt_count', 0) + 1} : {reason}")
 
    return {
        "attempt_count": state.get("attempt_count", 0) + 1,
        "retry_feedback": feedback,
        "error": None,
        "confidence_score": None,
        "raw_api_text": "",
        "source_url": "",
        "source_domain": "",
        "retrieved_cases": None,
        "mock_result": None,
        "document_result": None,
    }

def track_best_node(state: AgentState) -> dict:
    """Runs after every verifier pass, before the retry decision. Keeps
    whichever attempt (so far) had the highest confidence_score — so if a
    retry produces a *worse* answer than an earlier attempt, the earlier,
    better one still wins in the end instead of being silently discarded.

    best_confidence starts as None (not 0.0) specifically so that a first
    attempt scoring exactly 0.0 confidence still gets captured as "the
    best we've seen" rather than being skipped by a >  comparison against
    a 0.0 default.

    best_intent is tracked alongside response/confidence for a reason
    found the hard way: without it, a later WORSE attempt (e.g. a router
    failure on the retry) could leave state["intent"] showing that failed
    attempt's intent, while best_response/best_confidence correctly show
    the earlier, better attempt's content — a real mismatch where the
    displayed "Task:" label didn't match the actual displayed content."""
    current_response = state.get("final_response", "")
    current_confidence = state.get("confidence_score") or 0.0
    current_intent = state.get("intent")
    best_confidence = state.get("best_confidence")

    if best_confidence is None or current_confidence > best_confidence:
        return {
            "best_response": current_response,
            "best_confidence": current_confidence,
            "best_intent": current_intent
        }
    return {}


def should_retry(state: AgentState) -> str:
    """Routing decision after verification: retry or accept the result."""
    intent = state.get("intent")
    attempt_count = state.get("attempt_count", 0)
    log.info(f"[should_retry] ENTRY attempt_count={attempt_count} MAX_ATTEMPTS={MAX_ATTEMPTS} "
             f"intent={intent!r} error={state.get('error')!r} confidence={state.get('confidence_score')!r}")
 
    if intent == "followup":
        return "done" 
 
    if attempt_count >= MAX_ATTEMPTS:
        if state.get("error") or (state.get("confidence_score") or 1.0) < 1.0:
            log.warning(f"Max attempts ({MAX_ATTEMPTS}) reached for intent='{intent}' — accepting result as-is.")
        log.info(f"[should_retry] DECISION=done (attempt_count {attempt_count} >= MAX_ATTEMPTS {MAX_ATTEMPTS})")
        return "done"
 
    threshold = state.get("user_threshold")
    if threshold is None:
        threshold = DEFAULT_RETRY_THRESHOLDS.get(intent, 0.5)
    else:
        threshold = max(0.0, min(1.0, threshold))  
 
    if state.get("error"):
        permanent = [
            "path traversal",
            "file not found",
            "not a file",
            "expected a .pdf",
            "unknown intent",
            "could not find",
            "413",
            "payload too large",
            "rate_limit_exceeded",
            "tokens per minute",
            "tokens per day",
            "rate limit was reached",
            "ai service configuration has a problem",
            "model_not_found",
            "does not exist or you do not have access",
            "invalid_request_error",
        ]
        if any(p in state["error"].lower() for p in permanent):
            log.info(f"[should_retry] DECISION=done (error matched permanent-failure list)")
            return "done"
        log.info(f"[should_retry] DECISION=retry (transient-looking error, attempt_count={attempt_count})")
        return "retry"
    
    if state.get("confidence_score") == 0.0 and not state.get("error"):
        return "done"
 
    if state.get("confidence_score") is not None and state["confidence_score"] < threshold:
        log.info(f"Confidence {state['confidence_score']} < threshold {threshold} for intent='{intent}'.")
        return "retry"
 
    return "done"

def start_node(state: AgentState) -> dict:

    return {
        "attempt_count": 0,
        "retry_feedback": None,
        "best_response": None,
        "best_confidence": None,
        "best_intent": None,
    }

workflow = StateGraph(AgentState)
workflow.add_node("start", start_node)
workflow.add_node("detect_compound", detect_compound_question_node)
workflow.add_node("router", router_node_llm)
workflow.add_node("api_runner", api_execution_node)
workflow.add_node("verifier", verification_node)
workflow.add_node("track_best", track_best_node)

workflow.set_entry_point("start")
workflow.add_edge("start", "detect_compound")
workflow.add_edge("detect_compound", "router")
workflow.add_conditional_edges(
    "router",
    lambda s: "stop" if s.get("error") else "execute",
    {"execute": "api_runner", "stop": "verifier"}
)
workflow.add_edge("api_runner", "verifier")
workflow.add_edge("verifier", "track_best")

workflow.add_edge("track_best", END)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL missing. Check .env file")

try:
    from psycopg_pool import ConnectionPool
    pool = ConnectionPool(
        DB_URL,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": None}
    )
    checkpointer = PostgresSaver(pool)
    legal_agent_graph = workflow.compile(checkpointer=checkpointer)
    print("Connected to Supabase PostgreSQL")
except Exception as e:
    print(f"Database connection failed: {e}")
    raise


def _result_is_retry_worthy(result: dict, user_threshold: Optional[float] = None) -> tuple:
    """Plain Python decision — same logic should_retry used, but as an
    ordinary function call with an ordinary dict, not a LangGraph
    channel. Returns (should_retry: bool, reason: str)."""
    intent = result.get("intent")
    if intent == "followup":
        return False, ""

    threshold = user_threshold
    if threshold is None:
        threshold = DEFAULT_RETRY_THRESHOLDS.get(intent, 0.5)
    else:
        threshold = max(0.0, min(1.0, threshold))

    error = result.get("error")
    confidence = result.get("confidence_score")

    if error:
        permanent = [
            "path traversal", "file not found", "not a file", "expected a .pdf",
            "unknown intent", "could not find", "413", "payload too large",
            "rate_limit_exceeded", "tokens per minute", "tokens per day",
            "rate limit was reached", "ai service configuration has a problem",
            "model_not_found", "does not exist or you do not have access",
            "invalid_request_error",
        ]
        if any(p in error.lower() for p in permanent):
            return False, ""
        return True, f"failed with error: {error}"

    if confidence == 0.0:
        return False, ""

    if confidence is not None and confidence < threshold:
        return True, f"had low confidence ({confidence})"

    return False, ""


def run_query_with_retry(query: str, thread_id: str, user_threshold: Optional[float] = None,
                          thread_summary: Optional[str] = None) -> dict:
    """Replaces the old in-graph retry cycle (should_retry/prepare_retry_node
    as graph nodes), which is NOT used anymore.

    Hard cap: at most 2 total calls to legal_agent_graph.invoke(), no
    exceptions, enforced by this loop's own range(2) — cannot loop more
    than that no matter what any node returns.
    """
    best_result = None
    best_confidence = None
    feedback_for_next_attempt = None
    if thread_summary is None:
        thread_summary = get_thread_summary(thread_id)

    for attempt in range(2):  # hard cap: at most 2 total attempts, always
        log.info(f"[run_query_with_retry] attempt {attempt + 1}/2 for thread_id={thread_id}")

        initial_state = {
            "query": query, "intent": "", "parsed_parameters": {},
            "raw_api_text": "", "source_url": "", "source_domain": "",
            "llm_output": "", "confidence_score": 0.0, "final_response": "",
            "error": None, "retrieved_cases": None, "mock_result": None,
            "document_result": None, "attempt_count": 0,
            "retry_feedback": feedback_for_next_attempt,
            "user_threshold": user_threshold,
            "thread_summary": thread_summary,
        }
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
        result = legal_agent_graph.invoke(initial_state, config=config)

        current_confidence = result.get("confidence_score") or 0.0
        if best_confidence is None or current_confidence > best_confidence:
            best_result = result
            best_confidence = current_confidence
            log.info(f"[run_query_with_retry] attempt {attempt + 1} is the new best "
                     f"(confidence={current_confidence})")

        should_retry_now, reason = _result_is_retry_worthy(result, user_threshold)
        if not should_retry_now or attempt == 1:  # attempt 1 is the last allowed (0-indexed, 2 total)
            if should_retry_now:
                log.warning(f"[run_query_with_retry] hit the 2-attempt cap while still "
                            f"retry-worthy ({reason}) — accepting best result as-is.")
            break
        log.info(f"[run_query_with_retry] retrying — {reason}")
        feedback_for_next_attempt = (
            f"Previous attempt chose intent='{result.get('intent')}' and {reason}. "
            "Reconsider whether a different tool fits better, whether the "
            "extracted parameters were correct, or retry the same tool if "
            "this looks like a transient failure."
        )

    return best_result


COMPOUND_SPLIT_PROMPT = """A lawyer asked a question that may contain two distinct legal sub-questions joined together. Rewrite it as exactly two SEPARATE, FULLY SELF-CONTAINED questions — each one must make complete sense entirely on its own, with no pronouns or implicit references back to the other question (e.g. replace "either" or "it" with the actual term or subject it refers to).

If the question is genuinely only ONE question (not actually compound), respond with EXACTLY: NOT_COMPOUND

Otherwise, respond with EXACTLY two lines and nothing else — no numbering, no explanation, no extra text:
<first self-contained question>
<second self-contained question>

LAWYER'S QUESTION: {query}"""

MERGEABLE_COMPOUND_INTENTS = {"statute", "rule", "definition", "constitution"}

MULTI_PART_PRESENTATION_PROMPT = """You are a legal information assistant for a law firm.
The lawyer asked a question with two parts. You have retrieved authenticated legal text answering EACH part separately.

STRICT RULES:
- Use ONLY the text provided below for each part. Do not add anything from your own knowledge.
- Structure your answer with clear headers so BOTH parts are addressed distinctly.
- Every legal claim must come directly from the retrieved text for that specific part.
- Do NOT add closing notes, disclaimers, or meta-commentary about your process.

PART 1 — "{part1_query}"
RETRIEVED TEXT: {raw1}
SOURCE: {source1}

PART 2 — "{part2_query}"
RETRIEVED TEXT: {raw2}
SOURCE: {source2}

Present a single, well-organized answer covering both parts now."""


def _split_compound_query_via_llm(query: str) -> Optional[tuple]:
    """One plain text-generation LLM call (NOT forced tool-calling, so it
    doesn't share the malformed-tool-call reliability issue seen
    elsewhere) that rewrites a likely-compound question into two fully
    self-contained sub-questions. Returns None — meaning "don't attempt
    the compound path, fall back to normal single-lookup" — whenever the
    split isn't clean, rather than guessing."""
    prompt = COMPOUND_SPLIT_PROMPT.format(query=query)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = _strip_think_tags(response.content).strip()
    except Exception as e:
        log.error(f"[run_compound_query] split LLM call failed: {e}")
        return None

    if text.upper().startswith("NOT_COMPOUND"):
        return None

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) != 2:
        log.warning(f"[run_compound_query] split produced {len(lines)} lines, expected "
                    f"exactly 2 — falling back to single lookup. Raw output: {text!r}")
        return None

    return lines[0], lines[1]


def run_compound_query(query: str, thread_id: str, user_threshold: Optional[float] = None) -> Optional[dict]:
    """Attempts to actually answer BOTH parts of a genuinely two-part
    question, instead of the older behavior of silently answering only
    the first half with a disclaimer tacked on.

    Design, and why it's built this way: reuses run_query_with_retry()
    (already tested, already fixed) for EACH sub-question independently,
    on separate sub-thread-ids — no new graph nodes, no new cycles. After
    the attempt_count bug this session, that's a deliberate choice: this
    stays in plain Python, where a local variable can't have the same
    class of persistence bug a LangGraph channel just did.

    Returns None whenever anything doesn't fit cleanly — the split fails,
    either half resolves to a non-mergeable intent, or either half
    errors. The caller (api.py) falls back to the existing single-lookup
    path (which still includes the honest "this may have multiple parts"
    disclaimer) in that case. A confusing or wrong merged answer is worse
    than that honest fallback, so this is intentionally conservative
    rather than forcing a merge that doesn't actually fit.

    NOTE, a real known gap: sub-thread-ids are used only internally for
    running each half through the graph — this means a follow-up
    question asked right after a compound answer won't have last_raw_text
    context from either half, since that's tracked per-sub-thread, not
    on the main conversation thread. Worth knowing; not fixed here.
    """
    split = _split_compound_query_via_llm(query)
    if split is None:
        return None
    part1_query, part2_query = split
    log.info(f"[run_compound_query] split into:\n  1) {part1_query}\n  2) {part2_query}")

    real_thread_summary = get_thread_summary(thread_id)
    result1 = run_query_with_retry(part1_query, thread_id=f"{thread_id}-part1", user_threshold=user_threshold,
                                    thread_summary=real_thread_summary)
    result2 = run_query_with_retry(part2_query, thread_id=f"{thread_id}-part2", user_threshold=user_threshold,
                                    thread_summary=real_thread_summary)

    intent1, intent2 = result1.get("intent"), result2.get("intent")
    if intent1 not in MERGEABLE_COMPOUND_INTENTS or intent2 not in MERGEABLE_COMPOUND_INTENTS:
        log.info(f"[run_compound_query] non-mergeable intent(s) ({intent1!r}, {intent2!r}) — falling back.")
        return None
    if result1.get("error") or result2.get("error"):
        log.info(f"[run_compound_query] a sub-lookup errored "
                 f"({result1.get('error')!r}, {result2.get('error')!r}) — falling back.")
        return None

    raw1, source1 = result1.get("raw_api_text") or "", result1.get("source_url") or ""
    raw2, source2 = result2.get("raw_api_text") or "", result2.get("source_url") or ""
    if not raw1 or not raw2:
        log.info("[run_compound_query] one sub-lookup had no retrieved text — falling back.")
        return None

    prompt = MULTI_PART_PRESENTATION_PROMPT.format(
        part1_query=part1_query, raw1=raw1, source1=source1,
        part2_query=part2_query, raw2=raw2, source2=source2,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        merged_text = _strip_think_tags(response.content)
    except Exception as e:
        log.error(f"[run_compound_query] merge presentation LLM call failed: {e}")
        return None

    conf1 = result1.get("confidence_score") or 0.0
    conf2 = result2.get("confidence_score") or 0.0

    combined_raw_lower = (raw1 + " " + raw2).lower()
    merge_answer_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', merged_text.lower()))
    if merge_answer_words:
        merge_grounded = sum(1 for w in merge_answer_words if w in combined_raw_lower)
        merge_grounding_score = round(merge_grounded / len(merge_answer_words), 4)
    else:
        merge_grounding_score = 1.0
    log.info(f"[run_compound_query] merge_grounding_score={merge_grounding_score} "
             f"(fraction of the merged answer's words actually found in either source)")

    combined_confidence = min(conf1, conf2, merge_grounding_score)  # conservative: weakest of all three signals

    final_response = (
        f"{merged_text}\n\n"
        f"---\n"
        f"**Sources:**\n- Part 1: {source1}\n- Part 2: {source2}\n\n"
        f"**Confidence Score: {round(combined_confidence * 100)}%** "
        f"(weakest of: part 1 grounding {round(conf1 * 100)}%, "
        f"part 2 grounding {round(conf2 * 100)}%, "
        f"merge grounding {round(merge_grounding_score * 100)}%)"
    )

    return {
        "intent": "compound",
        "confidence_score": combined_confidence,
        "final_response": final_response,
        "error": None,
    }


def _ensure_thread_metadata_table() -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_metadata (
                thread_id TEXT PRIMARY KEY,
                lawyer_name TEXT NOT NULL,
                label TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            ALTER TABLE thread_metadata
            ADD COLUMN IF NOT EXISTS conversation_summary TEXT DEFAULT ''
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_thread_metadata_lawyer ON thread_metadata (lawyer_name, created_at DESC)
            """
        )

_ensure_thread_metadata_table()


def get_thread_summary(thread_id: str) -> str:
    """Reads the current running summary for a thread -- empty string
    if none exists yet (brand new thread, or the summary update failed
    on a previous turn and was skipped)."""
    if not thread_id:
        return ""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT conversation_summary FROM thread_metadata WHERE thread_id = %s",
            (thread_id,)
        ).fetchone()
    return row[0] if row and row[0] else ""


SUMMARY_UPDATE_PROMPT = """You are maintaining a running summary of an ongoing legal research conversation between a lawyer and an AI assistant, so that later questions in the SAME conversation can reference earlier topics without needing the full text of every past answer.

EXISTING SUMMARY (empty if this is the first exchange):
{existing_summary}

NEW EXCHANGE JUST COMPLETED:
Lawyer asked: {user_message}
Agent answered (condensed): {assistant_response}

Update the summary to include this new exchange. Keep it concise -- aim for under 400 words total, even as the conversation grows longer. Focus on WHAT TOPICS/QUESTIONS have been covered and WHAT KEY CONCLUSIONS were given (statute numbers, case names, definitions, holdings) -- not the full text of any answer. Write it as plain prose, oldest topic first.

Respond with ONLY the updated summary text, nothing else -- no preamble, no headers."""


def update_thread_summary(thread_id: str, user_message: str, assistant_response: str) -> None:
    """Incrementally updates the running summary -- ONE small LLM call
    per turn, regardless of how long the conversation gets, rather than
    re-summarizing the entire growing history every time (which would
    make cost grow with conversation length, defeating the point).

    Deliberately non-fatal: if this fails for any reason (Groq error,
    DB error), it's logged and skipped -- the actual answer the lawyer
    already got is unaffected either way. Losing one turn's worth of
    summary update is a minor, recoverable gap, not worth failing the
    whole request over.
    """
    if not thread_id:
        return
    existing_summary = get_thread_summary(thread_id)
    condensed_response = (assistant_response or "")[:1500]

    prompt = SUMMARY_UPDATE_PROMPT.format(
        existing_summary=existing_summary or "(no prior conversation yet)",
        user_message=user_message,
        assistant_response=condensed_response,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        new_summary = _strip_think_tags(response.content).strip()
    except Exception as e:
        log.error(f"Thread summary update failed for thread_id={thread_id}: {e}")
        return

    try:
        with pool.connection() as conn:
            conn.execute(
                "UPDATE thread_metadata SET conversation_summary = %s WHERE thread_id = %s",
                (new_summary, thread_id)
            )
    except Exception as e:
        log.error(f"Saving updated thread summary failed for thread_id={thread_id}: {e}")


def save_thread_metadata(thread_id: str, lawyer_name: str, label: str) -> None:
    """Record which lawyer owns a thread. Written on the first message of a thread only — ON CONFLICT DO NOTHING means later calls for the same
    thread_id are no-ops, so the label reflects the opening message."""
    if not thread_id or not lawyer_name:
        return
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO thread_metadata (thread_id, lawyer_name, label)
            VALUES (%s, %s, %s)
            ON CONFLICT (thread_id) DO NOTHING
            """,
            (thread_id, lawyer_name, (label or "")[:200])
        )


def get_threads_for_lawyer(lawyer_name: str) -> list:
    """Returns this lawyer's threads only, most recent first."""
    if not lawyer_name:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, label, created_at
            FROM thread_metadata
            WHERE lawyer_name = %s
            ORDER BY created_at DESC
            """,
            (lawyer_name,)
        ).fetchall()
    return [
        {
            "thread_id": row[0],
            "label": row[1],
            "created_at": row[2].isoformat() if row[2] else None
        }
        for row in rows
    ]


def thread_belongs_to_lawyer(thread_id: str, lawyer_name: str) -> bool:
    """Ownership check — a lawyer can only pull message history for their own threads, not by guessing/reusing someone else's thread_id."""
    if not thread_id or not lawyer_name:
        return False
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM thread_metadata WHERE thread_id = %s AND lawyer_name = %s",
            (thread_id, lawyer_name)
        ).fetchone()
    return row is not None


def _ensure_chat_messages_table() -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL,
                intent TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
            ON chat_messages (thread_id, created_at ASC)
            """
        )

_ensure_chat_messages_table()


def save_message(thread_id: str, role: str, content: str,
                  confidence: Optional[float] = None, intent: Optional[str] = None) -> None:
    if not thread_id or not content:
        return
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (thread_id, role, content, confidence, intent)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (thread_id, role, content, confidence, intent)
        )


def get_messages_for_thread(thread_id: str) -> list:
    """Full transcript for a thread, oldest first — what the frontend
    replays into the chat window when a lawyer reopens a past thread."""
    if not thread_id:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, confidence, intent, created_at
            FROM chat_messages
            WHERE thread_id = %s
            ORDER BY created_at ASC, id ASC
            """,
            (thread_id,)
        ).fetchall()
    return [
        {
            "role": row[0],
            "content": row[1],
            "confidence": row[2],
            "intent": row[3],
            "created_at": row[4].isoformat() if row[4] else None
        }
        for row in rows
    ]