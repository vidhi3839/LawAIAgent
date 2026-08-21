"""
It hits the REAL GovInfo and Cornell endpoints your app already calls and
saves the raw responses under tests/fixtures/. After that, the fixture-based
tests in test_statutory_api_fixtures.py replay these saved responses through
your ACTUAL parsing code with zero network calls — so they run instantly.

what this does and doesn't catch:
- DOES catch: you (or an LLM) accidentally breaking the BeautifulSoup
  selectors, the character limits, the retry logic, etc.
- Does NOT catch: Cornell or GovInfo changing their page structure AFTER
  you captured these fixtures. 

"""
import os
import requests

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def save(filename, content):
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"saved {filename} ({len(content):,} chars)")


def main():
    # 1. Cornell Wex definition page (used by fetch_legal_definition)
    r = requests.get("https://www.law.cornell.edu/wex/estoppel", headers=HEADERS, timeout=15)
    print("cornell wex estoppel:", r.status_code)
    if r.status_code == 200:
        save("cornell_wex_estoppel.html", r.text)

    # 2. Cornell Federal Rule page (used by fetch_federal_rule)
    r = requests.get("https://www.law.cornell.edu/rules/fre/rule_401", headers=HEADERS, timeout=15)
    print("cornell fre rule 401:", r.status_code)
    if r.status_code == 200:
        save("cornell_fre_rule_401.html", r.text)

    # 3. GovInfo granules list for Title 18 (used by fetch_statute, step 1)
    api_key = os.getenv("GOVINFO_API_KEY", "DEMO_KEY")
    r = requests.get(
        "https://api.govinfo.gov/packages/USCODE-2023-title18/granules",
        params={"api_key": api_key, "pageSize": 100, "offsetMark": "*"},
        headers=HEADERS,
        timeout=15,
    )
    print("govinfo title18 granules:", r.status_code)
    if r.status_code == 200:
        save("govinfo_title18_granules_page1.json", r.text)

        import json
        try:
            data = json.loads(r.text)
            granule_id = next(
                (g["granuleId"] for g in data.get("granules", []) if "sec1030" in g.get("granuleId", "")),
                None,
            )
            if granule_id:
                r2 = requests.get(
                    f"https://api.govinfo.gov/packages/USCODE-2023-title18/granules/{granule_id}/htm",
                    params={"api_key": api_key},
                    headers=HEADERS,
                    timeout=15,
                )
                print(f"govinfo granule content ({granule_id}):", r2.status_code)
                if r2.status_code == 200:
                    save("govinfo_title18_sec1030_content.html", r2.text)
            else:
                print("Section 1030 not found on granules page 1 — DEMO_KEY rate limits or "
                      "pagination may require a real (non-DEMO_KEY) API key. Skipping content capture.")
        except Exception as e:
            print(f"Could not parse granules JSON to find section 1030: {e}")


if __name__ == "__main__":
    main()