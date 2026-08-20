import re
import logging
import urllib.parse
import requests
import time
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

STATUTE_CHAR_LIMIT = 5000
CORNELL_CHAR_LIMIT = 3000


def _request_with_retry(method: str, url: str, max_attempts: int = 3, backoff: float = 1.0, **kwargs):
    """Retries transient network failures (timeouts, connection resets) on the actual HTTP call. Does not swallow non-transient failures —
    a 404 still comes back as a normal response for the caller to handle, only requests.RequestException triggers a retry.

    Previously defined but never actually called anywhere in this file — every function below made its requests.get/head call directly. Now
    wired in everywhere a network call happens."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return requests.request(method, url, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            log.error(f"{method} {url} attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(backoff * attempt)
    raise last_exc


def is_content_truncated(text: str, limit: int) -> bool:
    """True if text was cut off at exactly this fetch's character limit. Used by main.py's verification_node to add a truncation notice AFTER
    the presentation LLM call — not embedded in the raw text passed INTO that call, since LEGAL_PRESENTATION_PROMPT explicitly instructs the
    model to omit meta-commentary, and it was silently stripping an earlier version of this notice that was embedded in the input text."""
    return len(text) >= limit


def calculate_legal_confidence(
    source_domain: str,
    raw_api_text: str,
    llm_output_text: str,
    source_url: str
) -> float:
    if source_domain.endswith(".gov") or source_domain.endswith(".edu"):
        authority_score = 1.0
    elif "courtlistener.com" in source_domain:
        authority_score = 0.9
    else:
        authority_score = 0.5

    raw_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', raw_api_text.lower()))
    raw_words |= set(re.findall(r'\b(?:not|no)\b', raw_api_text.lower()))
    out_lower = llm_output_text.lower()
    if raw_words:
        matched = sum(1 for w in raw_words if w in out_lower)
        integrity_score = round(matched / len(raw_words), 2)
    else:
        integrity_score = 1.0

    try:
        response = _request_with_retry("HEAD", source_url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            link_score = 1.0
        elif response.status_code in [301, 302]:
            link_score = 0.5
        else:
            link_score = 0.0
    except requests.RequestException as e:
        log.error(f"Link-reachability check failed for {source_url}: {e}")
        link_score = 0.0

    return round(
        (authority_score * 0.4) +
        (integrity_score * 0.4) +
        (link_score * 0.2),
        2
    )


def fetch_statute(title: int, section: str, api_key: str) -> tuple:

    section_for_granule_match = re.sub(r'(\([^)]*\))+$', '', section).strip()

    package_id = f"USCODE-2023-title{title}"
    detail_url = f"https://www.govinfo.gov/app/details/{package_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    target_granule_id = None
    offset = "*"
    max_pages = 50
    pages_fetched = 0

    while pages_fetched < max_pages:
        pages_fetched += 1
        try:
            response = _request_with_retry(
                "GET",
                f"https://api.govinfo.gov/packages/{package_id}/granules",
                params={"api_key": api_key, "pageSize": 100, "offsetMark": offset},
                headers=headers,
                timeout=15
            )
            if response.status_code != 200:
                break

            data = response.json()
            granules = data.get("granules", [])

            for g in granules:
                granule_id = g.get("granuleId", "")
                if f"sec{section_for_granule_match}" in granule_id:
                    target_granule_id = granule_id
                    break

            if target_granule_id:
                break

            next_page = data.get("nextPage")
            if not next_page or not granules:
                break

            match = re.search(r'offsetMark=([^&]+)', next_page)
            if match:
                offset = urllib.parse.unquote(match.group(1))
            else:
                break

        except Exception as e:
            log.error(f"Granule search failed for {package_id}, section {section}: {e}")
            return (
                f"Error searching granules: {str(e)}",
                detail_url,
                "api.govinfo.gov"
            )

    if not target_granule_id:
        if pages_fetched >= max_pages:
            log.error(f"Hit max_pages cap ({max_pages}) searching {package_id} for section {section} without a match.")
        return (
            f"Could not find Title {title} U.S.C. Section {section} in GovInfo.",
            detail_url,
            "api.govinfo.gov"
        )

    try:
        content_url = f"https://api.govinfo.gov/packages/{package_id}/granules/{target_granule_id}/htm"
        content_response = _request_with_retry(
            "GET",
            content_url,
            params={"api_key": api_key},
            headers=headers,
            timeout=15
        )
        if content_response.status_code == 200:
            soup = BeautifulSoup(content_response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())
            result_text = text[:STATUTE_CHAR_LIMIT]
            section_url = f"https://www.govinfo.gov/app/details/{package_id}/granules/{target_granule_id}"
            return result_text, section_url, "api.govinfo.gov"

        return (
            f"Could not retrieve content for Title {title} U.S.C. Section {section}.",
            detail_url,
            "api.govinfo.gov"
        )

    except Exception as e:
        log.error(f"Error fetching granule content for {package_id}/{target_granule_id}: {e}")
        return (
            f"Error fetching section content: {str(e)}",
            detail_url,
            "api.govinfo.gov"
        )


def fetch_legal_definition(term: str) -> tuple:
    term_for_url = re.sub(r'[^\w\s]', '', term).strip().lower().replace(" ", "_")
    headers = {"User-Agent": "Mozilla/5.0"}

    terms_to_try = [term_for_url]
    parts = term_for_url.split("_")
    if len(parts) > 2:
        terms_to_try.append("_".join(parts[:2]))
    if len(parts) > 1:
        terms_to_try.append(parts[0])

    for term_to_try in terms_to_try:
        url = f"https://www.law.cornell.edu/wex/{term_to_try}"
        try:
            response = _request_with_retry("GET", url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                content = (
                    soup.find("div", {"class": "field-items"}) or
                    soup.find("div", {"id": "content"}) or
                    soup.find("main")
                )
                if content:
                    text = content.get_text(separator=" ", strip=True)
                    text = " ".join(text.split())
                    if len(text) > 100 and "wex definitions" not in text.lower()[:200]:
                        result_text = text[:CORNELL_CHAR_LIMIT]
                        return result_text, url, "law.cornell.edu"
        except requests.RequestException as e:
            log.error(f"Definition fetch failed for term '{term_to_try}' at {url}: {e}")
            continue

    return (
        f"Could not find a legal definition for '{term}' on Cornell LII.",
        f"https://www.law.cornell.edu/wex/{term_for_url}",
        "law.cornell.edu"
    )

AMENDMENT_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh",
    12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
    16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
    20: "twentieth", 21: "twenty-first", 22: "twenty-second",
    23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth",
    26: "twenty-sixth", 27: "twenty-seventh",
}
 
 
def fetch_constitutional_amendment(amendment_number: int) -> tuple:
    """Fetches the plain-text overview of a constitutional amendment from
    Cornell -- confirmed real URL pattern: law.cornell.edu/constitution/fourth_amendment."""
    ordinal = AMENDMENT_ORDINALS.get(amendment_number)
    if not ordinal:
        return (
            f"'{amendment_number}' is not a valid amendment number (the U.S. Constitution has 27 amendments).",
            "https://www.law.cornell.edu/constitution",
            "law.cornell.edu",
        )
 
    url = f"https://www.law.cornell.edu/constitution/{ordinal}_amendment"
    headers = {"User-Agent": "Mozilla/5.0"}
 
    try:
        response = _request_with_retry("GET", url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            content = (
                soup.find("div", {"class": "field-items"}) or
                soup.find("div", {"id": "content"}) or
                soup.find("main")
            )
            if content:
                text = content.get_text(separator=" ", strip=True)
                text = " ".join(text.split())
                if len(text) > 100:
                    result_text = text[:CORNELL_CHAR_LIMIT]
                    return result_text, url, "law.cornell.edu"
 
        return (
            f"Could not retrieve the {ordinal.title()} Amendment from Cornell LII.",
            url,
            "law.cornell.edu",
        )
    except Exception as e:
        log.error(f"Constitutional amendment fetch failed for amendment {amendment_number} at {url}: {e}")
        return (
            f"Error fetching amendment: {str(e)}",
            url,
            "law.cornell.edu",
        )
    
def fetch_federal_rule(rule_type: str, rule_number: str) -> tuple:
    rule_labels = {
        "fre": "Federal Rules of Evidence",
        "frcp": "Federal Rules of Civil Procedure",
        "frap": "Federal Rules of Appellate Procedure"
    }
    url = f"https://www.law.cornell.edu/rules/{rule_type}/rule_{rule_number}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = _request_with_retry("GET", url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            content = (
                soup.find("div", {"class": "field-items"}) or
                soup.find("div", {"id": "content"}) or
                soup.find("main")
            )
            if content:
                text = content.get_text(separator=" ", strip=True)
                text = " ".join(text.split())
                if len(text) > 100:
                    result_text = text[:CORNELL_CHAR_LIMIT]
                    return result_text, url, "law.cornell.edu"

        return (
            f"Could not retrieve {rule_type.upper()} Rule {rule_number} from Cornell LII.",
            url,
            "law.cornell.edu"
        )
    except Exception as e:
        log.error(f"Rule fetch failed for {rule_type} rule {rule_number} at {url}: {e}")
        return (
            f"Error fetching rule: {str(e)}",
            url,
            "law.cornell.edu"
        )