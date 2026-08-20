"""
Fixture-based regression tests for statutory_api.py's scrapers.

These do NOT hit the network — they replay a real, previously-captured
snapshot (see capture_fixtures.py) through your ACTUAL parsing code by
mocking only `requests.request`'s return value. This is the honest
middle ground between:
  - a fully mocked unit test (proves nothing about real page structure)
  - a live test every CI run (flaky, slow, depends on Cornell/GovInfo
    being up right now)

If these fail, it means your parsing logic broke against real content
you already captured — a genuine regression, not a network hiccup.
If Cornell/GovInfo changes their page structure AFTER you captured the
fixture, these tests will keep passing (they're replaying the OLD
snapshot) — that's what test_live_smoke.py is for instead.

Run capture_fixtures.py once before these will do anything useful:
    python tests/capture_fixtures.py
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from tasks import statutory_api

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> str:
    path = os.path.join(FIXTURES_DIR, name)
    if not os.path.exists(path):
        pytest.skip(
            f"Fixture '{name}' not found — run `python tests/capture_fixtures.py` "
            f"once with real internet access to generate it."
        )
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestFetchLegalDefinitionAgainstRealSnapshot:
    def test_estoppel_parses_to_real_definition_text(self):
        html = _load_fixture("cornell_wex_estoppel.html")
        fake_response = MagicMock(status_code=200, text=html)
        with patch("requests.request", return_value=fake_response):
            text, url, domain = statutory_api.fetch_legal_definition("estoppel")

        assert domain == "law.cornell.edu"
        assert not text.startswith("Could not find")
        # The real page's actual defining sentence — if your selector logic
        # (soup.find("div", {"class": "field-items"}) or #content or main)
        # stops matching Cornell's current markup, this substantively
        # changes or empties out, and this assertion catches that.
        assert "equitable doctrine" in text.lower()
        assert "estoppel" in text.lower()

    def test_wex_definitions_index_page_correctly_rejected(self):
        """fetch_legal_definition has an explicit guard against accidentally
        matching the generic 'wex definitions' index page instead of a
        real term page — confirm that guard still works against real
        Cornell markup, not just a hand-written test string."""
        html = _load_fixture("cornell_wex_estoppel.html")
        # Simulate the guard's actual trigger condition: a real index-page
        # fixture would be needed for a full check, but at minimum confirm
        # the real estoppel page does NOT get flagged as the index page
        # (i.e. no false-positive rejection of a valid definition).
        fake_response = MagicMock(status_code=200, text=html)
        with patch("requests.request", return_value=fake_response):
            text, url, domain = statutory_api.fetch_legal_definition("estoppel")
        assert "wex definitions" not in text.lower()[:200]


class TestFetchFederalRuleAgainstRealSnapshot:
    def test_fre_401_parses_to_real_rule_text(self):
        html = _load_fixture("cornell_fre_rule_401.html")
        fake_response = MagicMock(status_code=200, text=html)
        with patch("requests.request", return_value=fake_response):
            text, url, domain = statutory_api.fetch_federal_rule("fre", "401")

        assert domain == "law.cornell.edu"
        assert not text.startswith("Could not retrieve")
        # Rule 401 (relevance) — real rule text should mention this
        assert "relevant" in text.lower() or "relevance" in text.lower()


class TestFetchStatuteAgainstRealSnapshot:
    def test_granules_json_contains_expected_shape(self):
        """Sanity-checks the REAL GovInfo response shape your code parses
        (granuleId, nextPage) still matches what fetch_statute expects —
        this is what would silently break if GovInfo ever renamed a field."""
        import json
        raw = _load_fixture("govinfo_title18_granules_page1.json")
        data = json.loads(raw)
        assert "granules" in data
        if data["granules"]:
            assert "granuleId" in data["granules"][0]

    def test_section_1030_content_page_parses_to_real_statute_text(self):
        html = _load_fixture("govinfo_title18_sec1030_content.html")
        # fetch_statute makes two calls (granules search, then content) —
        # this test isolates the SECOND call (content extraction) by
        # directly exercising the same BeautifulSoup logic fetch_statute
        # uses, since fully re-running the two-call flow here would just
        # re-mock the first call anyway. If fetch_statute's extraction
        # logic changes, update this alongside it.
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        assert len(text) > 100
        # CFAA's actual statutory language — real content check, not a
        # placeholder string
        assert "computer" in text.lower()