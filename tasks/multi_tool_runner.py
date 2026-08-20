import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tasks.statutory_api import (
    fetch_statute,
    fetch_legal_definition,
    fetch_federal_rule,
    fetch_constitutional_amendment,
    calculate_legal_confidence,
    AMENDMENT_ORDINALS,
    CORNELL_CHAR_LIMIT,
)
from tasks.past_cases import search_past_cases, compute_past_cases_confidence
from tasks.summarize import compute_semantic_grounding

log = logging.getLogger(__name__)

_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)

PRIMARY_TOOL_PROMPT = """You are the intent router for a law firm legal assistant.
Pick exactly ONE tool that best answers this question.
Extract every citation, term, and number EXACTLY as written.

Tools:
- StatuteLookup(title: int, section: str): for US Code citations like "18 U.S.C. 1030"
- RuleLookup(rule_type: str [fre/frcp/frap], rule_number: str): for FRE/FRCP/FRAP rules
- ConstitutionLookup(amendment_number: int): for constitutional amendments, NOT StatuteLookup
- DefinitionLookup(term: str): for legal terms and definitions
- PastCasesSearch(jurisdiction: str): for how courts have ruled, precedents

Respond with EXACTLY one line in this format and nothing else:
TOOL: <ToolName> | ARGS: <key>=<value>, <key>=<value>

Examples:
TOOL: DefinitionLookup | ARGS: term=negligence
TOOL: StatuteLookup | ARGS: title=18, section=1030
TOOL: RuleLookup | ARGS: rule_type=fre, rule_number=401
TOOL: ConstitutionLookup | ARGS: amendment_number=4
TOOL: PastCasesSearch | ARGS: jurisdiction=federal"""

SECONDARY_TOOL_PROMPT = """A lawyer asked: {query}

The primary tool selected was {primary_tool} which will retrieve: {primary_desc}

Is there a SECOND tool that would meaningfully add to the answer? Only say yes if the question genuinely needs a second type of information.

Rules:
- If the question asks BOTH for a definition AND how courts ruled: add PastCasesSearch
- If comparing two legal concepts (e.g. negligence vs gross negligence): add DefinitionLookup for the second term
- If a statute question would benefit from the legal definition of a key term: add DefinitionLookup
- Otherwise: NO

Respond with EXACTLY one of these and nothing else:
NO
TOOL: <ToolName> | ARGS: <key>=<value>, <key>=<value>"""

SYNTHESIS_PROMPT = """You are a legal research assistant for a law firm.
The lawyer asked: {query}

Retrieved from {n} sources:

{sources_block}

Using ONLY the information above:
1. Answer comprehensively drawing on ALL sources
2. Reference sources by name when citing them
3. State clearly if any part is not covered by the retrieved sources
4. Do not add anything from your own knowledge
5. No closing notes or meta-commentary"""


def _parse_tool_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.upper().startswith("TOOL:"):
        return None
    try:
        parts = line.split("|")
        tool_name = parts[0].replace("TOOL:", "").strip()
        args = {}
        if len(parts) > 1:
            args_str = parts[1].replace("ARGS:", "").strip()
            for pair in args_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    args[k.strip()] = v.strip()
        return {"tool_name": tool_name, "args": args}
    except Exception:
        return None


def _route_primary(query: str, thread_summary: Optional[str]) -> Optional[dict]:
    system = PRIMARY_TOOL_PROMPT
    if thread_summary:
        system += f"\n\nConversation so far: {thread_summary}"
    try:
        response = _llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=query)
        ])
        text = response.content.strip()
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>")[-1].strip()
        for line in text.split("\n"):
            result = _parse_tool_line(line)
            if result:
                return result
    except Exception as e:
        log.error(f"[multi_tool] primary routing failed: {e}")
    return None


def _route_secondary(query: str, primary_tool: str, primary_desc: str) -> Optional[dict]:
    prompt = SECONDARY_TOOL_PROMPT.format(
        query=query,
        primary_tool=primary_tool,
        primary_desc=primary_desc
    )
    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if text.upper().startswith("NO"):
            return None
        for line in text.split("\n"):
            result = _parse_tool_line(line)
            if result:
                return result
    except Exception as e:
        log.error(f"[multi_tool] secondary routing failed: {e}")
    return None


def _fetch_one(tool_name: str, args: dict, query: str) -> Optional[dict]:
    api_key = os.getenv("GOVINFO_API_KEY", "DEMO_KEY")
    try:
        if tool_name == "StatuteLookup":
            text, url, domain = fetch_statute(
                title=int(args["title"]),
                section=str(args["section"]),
                api_key=api_key
            )
            name = f"{args['title']} U.S.C. § {args['section']}"

        elif tool_name == "RuleLookup":
            text, url, domain = fetch_federal_rule(
                rule_type=args["rule_type"],
                rule_number=str(args["rule_number"])
            )
            labels = {"fre": "FRE", "frcp": "FRCP", "frap": "FRAP"}
            name = f"{labels.get(args['rule_type'], 'Rule')} {args['rule_number']}"

        elif tool_name == "ConstitutionLookup":
            num = int(args["amendment_number"])
            text, url, domain = fetch_constitutional_amendment(amendment_number=num)
            ordinal = AMENDMENT_ORDINALS.get(num, str(num)).title()
            name = f"{ordinal} Amendment"

        elif tool_name == "DefinitionLookup":
            text, url, domain = fetch_legal_definition(term=args["term"])
            name = f"Definition: {args['term']}"

        elif tool_name == "PastCasesSearch":
            results = search_past_cases(
                query=query,
                n_results=5,
                jurisdiction_filter=args.get("jurisdiction")
            )
            if results.get("error") or not results.get("results"):
                return None
            cases = results["results"]
            text = "\n\n".join([
                f"Case: {r['case_name']} ({r['citation']}, {r['year']})\n"
                f"Court: {r['court']}\nExcerpt: {r['text']}"
                for r in cases
            ])
            url = "ChromaDB — local case database"
            domain = "internal"
            name = "Past Cases"
            return {"name": name, "text": text, "url": url,
                    "domain": domain, "tool_name": tool_name, "cases": cases}
        else:
            return None

        if not text or text.startswith("Could not") or text.startswith("Error"):
            return None

        return {"name": name, "text": text[:CORNELL_CHAR_LIMIT],
                "url": url, "domain": domain, "tool_name": tool_name}

    except Exception as e:
        log.error(f"[multi_tool] _fetch_one {tool_name}: {e}")
        return None


def _primary_description(tool_name: str, args: dict) -> str:
    if tool_name == "StatuteLookup":
        return f"the text of {args.get('title')} U.S.C. § {args.get('section')}"
    if tool_name == "RuleLookup":
        return f"Rule {args.get('rule_number')} ({args.get('rule_type', '').upper()})"
    if tool_name == "ConstitutionLookup":
        ordinal = AMENDMENT_ORDINALS.get(int(args.get("amendment_number", 0)), "").title()
        return f"the {ordinal} Amendment"
    if tool_name == "DefinitionLookup":
        return f"the legal definition of '{args.get('term')}'"
    if tool_name == "PastCasesSearch":
        return "relevant past case law"
    return tool_name


def _compute_confidence(query: str, sources: list, answer: str) -> dict:
    if not sources or not answer:
        return {"final_score": 0.0, "authority": 0.0, "grounding": 0.0, "coverage": 0.0}

    authority_scores = []
    for src in sources:
        if src["tool_name"] == "PastCasesSearch":
            cases = src.get("cases", [])
            if cases:
                conf = compute_past_cases_confidence(
                    query=query,
                    retrieved_results=cases,
                    llm_answer=answer,
                    requested_jurisdiction=None
                )
                authority_scores.append(conf["final_score"])
            else:
                authority_scores.append(0.0)
        else:
            domain = src["domain"].lower()
            if domain.endswith(".gov"):
                auth = 1.0
            elif "cornell" in domain or domain.endswith(".edu"):
                auth = 0.95
            else:
                auth = 0.7
            authority_scores.append(auth)

    source_authority = sum(authority_scores) / len(authority_scores)

    combined_text = " ".join(src["text"] for src in sources)
    grounding = compute_semantic_grounding(answer, combined_text)
    grounding = grounding if grounding is not None else 0.7

    answer_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower()))
    if answer_words:
        overlap = sum(1 for w in answer_words if w in combined_text.lower())
        if (overlap / len(answer_words)) >= 0.7:
            grounding = max(grounding, 0.65)

    coverage_scores = []
    answer_lower = answer.lower()
    for src in sources:
        src_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', src["text"].lower()))
        if not src_words:
            coverage_scores.append(1.0)
            continue
        covered = sum(1 for w in src_words if w in answer_lower)
        raw = covered / len(src_words)
        coverage_scores.append(0.5 + (raw * 0.5))

    coverage = sum(coverage_scores) / len(coverage_scores)

    final = round(min(source_authority, grounding, coverage), 4)
    return {
        "final_score": final,
        "authority": round(source_authority, 4),
        "grounding": round(grounding, 4),
        "coverage": round(coverage, 4),
    }


def run_multi_tool_query(
    query: str,
    thread_id: str,
    thread_summary: Optional[str] = None,
    user_threshold: Optional[float] = None,
) -> Optional[dict]:
    primary = _route_primary(query=query, thread_summary=thread_summary)
    if not primary:
        log.warning("[multi_tool] primary routing failed")
        return None

    log.info(f"[multi_tool] primary: {primary['tool_name']} args={primary['args']}")

    primary_desc = _primary_description(primary["tool_name"], primary["args"])
    secondary = _route_secondary(
        query=query,
        primary_tool=primary["tool_name"],
        primary_desc=primary_desc
    )

    tool_calls = [primary]
    if secondary and secondary["tool_name"] != primary["tool_name"]:
        log.info(f"[multi_tool] secondary: {secondary['tool_name']} args={secondary['args']}")
        tool_calls.append(secondary)
    else:
        log.info("[multi_tool] no secondary tool selected")

    sources = []
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        futures = {
            executor.submit(_fetch_one, tc["tool_name"], tc["args"], query): tc
            for tc in tool_calls
        }
        for future, tc in futures.items():
            try:
                result = future.result(timeout=15)
                if result is not None:
                    sources.append(result)
            except FuturesTimeout:
                log.warning(f"[multi_tool] {tc['tool_name']} timed out")
            except Exception as e:
                log.error(f"[multi_tool] future error {tc['tool_name']}: {e}")

    if not sources:
        log.warning("[multi_tool] all fetches failed")
        return None

    sources_block = ""
    for i, src in enumerate(sources, 1):
        sources_block += f"--- SOURCE {i}: {src['name']} ---\nURL: {src['url']}\n{src['text']}\n\n"

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        n=len(sources),
        sources_block=sources_block.strip()
    )

    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
        if "<think>" in answer and "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
    except Exception as e:
        log.error(f"[multi_tool] synthesis failed: {e}")
        return None

    confidence = _compute_confidence(query=query, sources=sources, answer=answer)

    source_lines = "\n".join([f"- {s['name']}: {s['url']}" for s in sources])
    final_response = (
        f"**Multi-source Legal Research**\n\n"
        f"{answer}\n\n"
        f"---\n"
        f"**Sources ({len(sources)}):**\n{source_lines}\n\n"
        f"**Confidence Score: {round(confidence['final_score'] * 100)}%**\n"
        f"Authority: {round(confidence['authority'] * 100)}% | "
        f"Grounding: {round(confidence['grounding'] * 100)}% | "
        f"Coverage: {round(confidence['coverage'] * 100)}%"
    )

    return {
        "intent": "multi_tool",
        "confidence_score": confidence["final_score"],
        "final_response": final_response,
        "error": None,
        "last_raw_text": sources[0]["text"] if sources else "",
        "last_source_url": sources[0]["url"] if sources else "",
        "last_intent": "multi_tool",
        "last_params": {},
    }