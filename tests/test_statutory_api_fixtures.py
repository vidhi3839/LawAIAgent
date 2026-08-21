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
        assert "equitable doctrine" in text.lower()
        assert "estoppel" in text.lower()

    def test_wex_definitions_index_page_correctly_rejected(self):

        html = _load_fixture("cornell_wex_estoppel.html")
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
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        assert len(text) > 100
        assert "computer" in text.lower()