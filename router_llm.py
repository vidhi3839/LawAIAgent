import os
import logging
from typing import Optional, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

router_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
)


class RouterResult(TypedDict):
    """Exact shape router_node_llm returns — merged into AgentState by LangGraph."""
    intent: str
    parsed_parameters: dict
    error: Optional[str]

# Tool schemas
class StatuteLookup(BaseModel):
    """Use when the lawyer cites a specific U.S. Code statute, e.g. '42 U.S.C. § 1983' or '15 USC 78j'."""
    title: int = Field(description="The U.S.C. title number, e.g. 42")
    section: str = Field(description="The section exactly as written, e.g. '1983' or '78j'")

class ConstitutionLookup(BaseModel):
    """Use when the lawyer asks what a constitutional AMENDMENT says or means,
    e.g. 'What does the Fourth Amendment say?', 'What is the First Amendment?',
    'Explain the Fifth Amendment'. Do NOT use StatuteLookup for this —
    amendments have no U.S. Code title/section, they are NOT statutes.
    Only for numbered amendments (1st through 27th), not general constitutional
    law questions without a specific amendment named."""
    amendment_number: int = Field(
        description="The amendment number as a plain integer, e.g. 'Fourth Amendment' -> 4, "
                    "'First Amendment' -> 1. Convert the written ordinal/number word to an int."
    )

class RuleLookup(BaseModel):
    """Use when the lawyer cites a Federal Rule of Evidence, Civil Procedure, or Appellate Procedure, e.g. 'Rule 56', 'FRE 401', 'FRCP 12(b)(6)'."""
    rule_type: Literal["fre", "frcp", "frap"] = Field(
        description="fre=Evidence, frcp=Civil Procedure, frap=Appellate Procedure. "
                    "Infer from context if not stated explicitly; default frcp for bare 'Rule N'."
    )
    rule_number: str = Field(description="The rule number exactly as written, e.g. '56' or '12b6'")


class DefinitionLookup(BaseModel):
    """Use when the lawyer is asking what a legal term MEANS or legal definition— a dictionary-
    style question, not a request for cases or their own argument analyzed."""
    term: str = Field(description="The legal term only. 'What is habeas corpus?' → 'habeas corpus'. 'Define estoppel' → 'estoppel'. Never the full question.")


class PastCasesSearch(BaseModel):
    """Use when the lawyer wants to find or reference PAST CASES / PRECEDENT from the firm's case database — not their own argument, and not asking
    to continue a prior answer."""
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Jurisdiction ONLY if the lawyer explicitly mentions one, e.g. 'ninth circuit', 'state court', 'supreme court'. Leave blank/null if no jurisdiction is mentioned — do NOT guess or default to federal."
    )


class MockCourtAnalysis(BaseModel):
    """Use when the lawyer's message begins with phrases like 'My argument is',
    'I argue', 'My position is', 'My client argues', 'We contend' — even if
    the message contains statute citations or case references. The presence of
    a legal citation does NOT override this tool if the lawyer is presenting
    their own argument for stress-testing."""
    pass


class DocumentSummarize(BaseModel):
    """Use when the lawyer's message references an uploaded PDF file path
    and wants it summarized, or asks a specific question about it."""
    file_path: str = Field(description="The exact file path to the PDF as written in the message")
    question: Optional[str] = Field(
        default=None,
        description="Specific question about the document, if the lawyer asked one beyond a general summary. "
                    "Leave null for a plain summary request."
    )


class FollowUp(BaseModel):
    """Use when the lawyer's message continues the PREVIOUS TURN's topic.
    Strong signals: 'how does this', 'what did the court hold', 'what was
    the holding', 'what is the penalty under this', 'what are the elements',
    'interact with', 'under this', 'within this', 'subsection'.
    IMPORTANT: If previous context exists and the message asks about something
    related to that context — even if it mentions a rule number like Rule 403
    — choose FollowUp over RuleLookup. A question about how two rules interact
    is a follow-up on the previously retrieved rule, not a new rule lookup."""
    pass


TOOLS = [
    StatuteLookup, RuleLookup, ConstitutionLookup, DefinitionLookup, PastCasesSearch, MockCourtAnalysis, DocumentSummarize, FollowUp
]
TOOL_NAME_TO_INTENT = {
    "StatuteLookup": "statute",
    "RuleLookup": "rule",
    "ConstitutionLookup": "constitution",
    "DefinitionLookup": "definition",
    "PastCasesSearch": "past_cases",
    "MockCourtAnalysis": "mock_court",
    "DocumentSummarize": "summarizer",
    "FollowUp": "followup",
}

router_llm_with_tools = router_llm.bind_tools(TOOLS, tool_choice="required")

NON_FOLLOWUP_TOOLS = [t for t in TOOLS if t is not FollowUp]
router_llm_no_followup = router_llm.bind_tools(NON_FOLLOWUP_TOOLS, tool_choice="required")


def _citation_matches_query(query: str, intent: str, parsed_parameters: dict) -> bool:
    """Guardrail: catches hallucinated digits by checking the extracted
    citation actually appears in the lawyer's original text."""
    if intent == "statute":
        return str(parsed_parameters["title"]) in query and str(parsed_parameters["section"]) in query
    if intent == "rule":
        return str(parsed_parameters["rule_number"]) in query
    return True


def _build_system_prompt(state: dict, has_previous_context: bool) -> str:
    base = (
        "You are the intent router for a law firm's legal assistant. "
        "Pick exactly one tool that matches what the lawyer is asking for. "
        "Extract every citation number, term, and path EXACTLY as the lawyer wrote it — "
        "do not correct, round, or reinterpret numbers."
    )

    thread_summary = state.get("thread_summary")
    if thread_summary:
        base += (
            f"\n\nCONVERSATION SO FAR (a compressed summary of the whole conversation, for background awareness only): {thread_summary}\n"
            f"Use this ONLY to understand what topics have already come up. "
            f"Do NOT choose FollowUp just because a topic sounds familiar from "
            f"this summary -- FollowUp is reserved for continuing the EXACT "
            f"immediately previous turn (see below). If the lawyer is asking "
            f"about an earlier topic again, pick the correct tool fresh and "
            f"extract its parameters from their CURRENT message."
        )

    if not has_previous_context:
        return base + "\n\nThere is NO previous turn context. Do not choose FollowUp."

    if state.get("retry_feedback"):
        base += f"\n\nRETRY CONTEXT: {state['retry_feedback']} Adjust your tool selection or parameters accordingly."

    prior_intent = state.get("last_intent", "unknown")
    prior_source = state.get("last_source_url", "unknown")
    prior_snippet = (state.get("last_raw_text") or "")[:400]

    return base + (
        f"\n\nPREVIOUS TURN CONTEXT:\n"
        f"- Previous intent handled: {prior_intent}\n"
        f"- Previous source: {prior_source}\n"
        f"- Snippet of what was retrieved: {prior_snippet}\n\n"
        f"Use this ONLY to judge whether the current message is a FollowUp continuing that exact topic. If you choose ANY tool other than FollowUp, this previous "
        f"context is IRRELEVANT to it — extract that tool's parameters (term, title, section, rule_number, etc.) SOLELY from the current message below, as if the "
        f"previous context did not exist. Do not let the previous topic influence or carry over into a new, unrelated question's parameters."
    )


def router_node_llm(state: dict) -> RouterResult:
    """
    Graph node: reads state["query"] (and state["retry_feedback"] on a retry),
    returns {"intent", "parsed_parameters", "error"}.
    """
    query = state["query"]
    has_previous_context = bool(state.get("last_raw_text"))

    system = SystemMessage(content=_build_system_prompt(state, has_previous_context))
    human = HumanMessage(content=query)

    try:
        response = router_llm_with_tools.invoke([system, human])
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            raise ValueError("No tool call returned by router LLM")

        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        args = tool_call.get("args", {})
        intent = TOOL_NAME_TO_INTENT.get(tool_name)

        if intent == "followup" and not has_previous_context:
            correction_system = SystemMessage(content=(
                _build_system_prompt(state, has_previous_context)
                + "\n\nNOTE: FollowUp is not a valid choice here — there is "
                  "no previous turn context. Choose the best fit among the "
                  "remaining options."
            ))
            retry_response = router_llm_no_followup.invoke([correction_system, human])
            retry_calls = getattr(retry_response, "tool_calls", None)
            if not retry_calls:
                raise ValueError("No tool call returned on FollowUp-recovery re-ask")

            tool_call = retry_calls[0]
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})
            intent = TOOL_NAME_TO_INTENT.get(tool_name)

        # Map the tool's raw args to the shape api_execution_node expects for that intent.
        if intent == "statute":
            parsed_parameters = {"title": int(args["title"]), "section": args["section"]}
        elif intent == "rule":
            parsed_parameters = {"rule_type": args.get("rule_type", "frcp"), "rule_number": args["rule_number"]}
        elif intent == "constitution":
            parsed_parameters = {"amendment_number": int(args["amendment_number"])}
        elif intent == "definition":
            parsed_parameters = {"term": args.get("term", query)}
        elif intent == "past_cases":
            parsed_parameters = {"jurisdiction": args.get("jurisdiction")}
        elif intent == "summarizer":
            parsed_parameters = {
                "file_path": args["file_path"],
                "question": args.get("question"),
                "wants_summary": not args.get("question"),
            }
        elif intent == "followup":

            parsed_parameters = state.get("last_params") or {}
        else:  # mock_court
            parsed_parameters = {}

        if not _citation_matches_query(query, intent, parsed_parameters):
            correction_system = SystemMessage(content=(
                _build_system_prompt(state, has_previous_context)
                + f"\n\nNOTE: You previously extracted {parsed_parameters} for this "
                  "query, but those numbers don't appear in the lawyer's text. "
                  "Re-read the query carefully and extract the citation EXACTLY "
                  "as written."
            ))
            retry_response = router_llm_with_tools.invoke([correction_system, human])
            retry_calls = getattr(retry_response, "tool_calls", None)
            if retry_calls:
                tool_call = retry_calls[0]
                intent = TOOL_NAME_TO_INTENT.get(tool_call["name"])
                args = tool_call.get("args", {})
                if intent == "statute":
                    parsed_parameters = {"title": int(args["title"]), "section": args["section"]}
                elif intent == "rule":
                    parsed_parameters = {"rule_type": args.get("rule_type", "frcp"), "rule_number": args["rule_number"]}

            if not _citation_matches_query(query, intent, parsed_parameters):
                return {
                    "intent": intent,
                    "parsed_parameters": parsed_parameters,
                    "error": f"Could not reliably extract the citation from '{query}' — "
                             "extracted numbers don't match the query text."
                }

        return {"intent": intent, "parsed_parameters": parsed_parameters, "error": None}

    except Exception as e:
        # Router itself failed (Groq API error, malformed tool call, etc).
        log.error(f"Router LLM failed for query={query!r}: {e}")

        error_str = str(e).lower()

        is_malformed_tool_call = any(marker in error_str for marker in
                                      ["tool_use_failed", "tool call validation failed"])

        if is_malformed_tool_call:
            log.warning(f"Malformed tool-call generation for query={query!r} — retrying once.")
            try:
                retry_response = router_llm_with_tools.invoke([system, human])
                retry_calls = getattr(retry_response, "tool_calls", None)
                if retry_calls:
                    tool_call = retry_calls[0]
                    tool_name = tool_call["name"]
                    args = tool_call.get("args", {})
                    intent = TOOL_NAME_TO_INTENT.get(tool_name)

                    if intent == "statute":
                        parsed_parameters = {"title": int(args["title"]), "section": args["section"]}
                    elif intent == "rule":
                        parsed_parameters = {"rule_type": args.get("rule_type", "frcp"), "rule_number": args["rule_number"]}
                    elif intent == "definition":
                        parsed_parameters = {"term": args.get("term", query)}
                    elif intent == "past_cases":
                        parsed_parameters = {"jurisdiction": args.get("jurisdiction")}
                    elif intent == "summarizer":
                        parsed_parameters = {
                            "file_path": args["file_path"],
                            "question": args.get("question"),
                            "wants_summary": not args.get("question"),
                        }
                    elif intent == "followup":
                        parsed_parameters = state.get("last_params") or {}
                    else:
                        parsed_parameters = {}

                    if _citation_matches_query(query, intent, parsed_parameters):
                        return {"intent": intent, "parsed_parameters": parsed_parameters, "error": None}
            except Exception as retry_e:
                log.error(f"Retry after malformed tool call also failed for query={query!r}: {retry_e}")
            error_message = ("Could not process this request right now — the AI produced an "
                              "invalid response twice in a row for this question. Try rephrasing "
                              "it more simply, or asking about one thing at a time.")
            return {"intent": "router_error", "parsed_parameters": {}, "error": error_message}

        is_rate_limit = any(marker in error_str for marker in
                             ["429", "rate_limit_exceeded", "tokens per minute", "tokens per day"])
        is_client_error = any(marker in error_str for marker in
                               ["404", "model_not_found", "does not exist",
                                "400", "401", "403", "422",
                                "invalid_request_error", "unprocessable"])

        if is_rate_limit:
            error_message = "Could not process this request right now — the AI service rate limit was reached. Please wait a few minutes and try again."
        elif is_client_error:
            error_message = "Could not process this request right now — the AI service configuration has a problem. Please contact your administrator."
        else:
            error_message = "Could not process this request right now — the routing step failed. Please try again."

        return {
            "intent": "router_error",
            "parsed_parameters": {},
            "error": error_message
        }