"""
Tests for tasks/statutory_api.py

fetch_statute / fetch_legal_definition / fetch_federal_rule are not unit
tested here — they're thin wrappers around live GovInfo/Cornell HTTP calls
and HTML scraping. Cover those with a recorded-response integration test
(e.g. vcrpy / responses library) against a fixture HTML page, separate
from this fast unit suite.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from tasks import statutory_api


# ── _request_with_retry ───────────────────────────────────────────────────

class TestRequestWithRetry:
    def test_succeeds_first_try_no_retry_needed(self):
        with patch("requests.request") as mock_req:
            mock_req.return_value = MagicMock(status_code=200)
            result = statutory_api._request_with_retry("GET", "https://example.com")
            assert mock_req.call_count == 1
            assert result.status_code == 200

    def test_retries_on_transient_exception_then_succeeds(self):
        with patch("requests.request") as mock_req, patch("time.sleep"):
            mock_req.side_effect = [
                requests.exceptions.ConnectionError("reset"),
                MagicMock(status_code=200),
            ]
            result = statutory_api._request_with_retry("GET", "https://example.com", max_attempts=3)
            assert mock_req.call_count == 2
            assert result.status_code == 200

    def test_raises_after_exhausting_max_attempts(self):
        with patch("requests.request") as mock_req, patch("time.sleep"):
            mock_req.side_effect = requests.exceptions.Timeout("too slow")
            with pytest.raises(requests.exceptions.Timeout):
                statutory_api._request_with_retry("GET", "https://example.com", max_attempts=2)
            assert mock_req.call_count == 2

    def test_a_real_404_response_is_returned_not_retried(self):
        """A 404 is a normal HTTP response, not a RequestException — must
        NOT trigger a retry loop."""
        with patch("requests.request") as mock_req:
            mock_req.return_value = MagicMock(status_code=404)
            result = statutory_api._request_with_retry("GET", "https://example.com")
            assert mock_req.call_count == 1
            assert result.status_code == 404


# ── is_content_truncated ──────────────────────────────────────────────────

class TestIsContentTruncated:
    def test_text_under_limit_is_not_truncated(self):
        assert statutory_api.is_content_truncated("short text", limit=100) is False

    def test_text_exactly_at_limit_is_truncated(self):
        assert statutory_api.is_content_truncated("x" * 100, limit=100) is True

    def test_text_over_limit_is_truncated(self):
        assert statutory_api.is_content_truncated("x" * 150, limit=100) is True


# ── fetch_statute subsection-matching regression ────────────────────────────

class TestFetchStatuteSubsectionMatching:
    """Regression tests for the fix: GovInfo granule IDs exist at the
    SECTION level only (verified against a real record:
    USCODE-2014-title15-chap2B-sec78j — no subsection letter in the id).
    A citation like '78j(b)' must still resolve to the 'sec78j' granule."""

    def _mock_granules_response(self, granule_id="USCODE-2014-title15-chap2B-sec78j"):
        return MagicMock(status_code=200, json=lambda: {
            "granules": [{"granuleId": granule_id}],
            "nextPage": None,
        })

    def _mock_content_response(self):
        return MagicMock(status_code=200, text="<html><body>Manipulative and deceptive devices statutory text.</body></html>")

    def test_subsection_citation_now_resolves_to_section_level_granule(self):
        granules_resp = self._mock_granules_response()
        content_resp = self._mock_content_response()
        with patch("requests.request", side_effect=[granules_resp, content_resp]):
            text, url, domain = statutory_api.fetch_statute(title=15, section="78j(b)", api_key="DEMO_KEY")

        assert not text.startswith("Could not find")
        assert "manipulative" in text.lower()
        assert domain == "api.govinfo.gov"

    def test_bare_section_without_subsection_still_works(self):
        """Confirms the fix didn't break the normal case of a citation
        with no subsection at all."""
        granules_resp = self._mock_granules_response()
        content_resp = self._mock_content_response()
        with patch("requests.request", side_effect=[granules_resp, content_resp]):
            text, url, domain = statutory_api.fetch_statute(title=15, section="78j", api_key="DEMO_KEY")
        assert not text.startswith("Could not find")

    def test_nested_subsection_citation_also_resolves(self):
        """Multiple parenthetical groups, e.g. '230(c)(2)(A)', must also
        strip down to the base section 230 for granule matching."""
        granules_resp = self._mock_granules_response(granule_id="USCODE-2024-title47-chap5-sec230")
        content_resp = self._mock_content_response()
        with patch("requests.request", side_effect=[granules_resp, content_resp]):
            text, url, domain = statutory_api.fetch_statute(title=47, section="230(c)(2)(A)", api_key="DEMO_KEY")
        assert not text.startswith("Could not find")


# ── calculate_legal_confidence ─────────────────────────────────────────────

class TestCalculateLegalConfidence:
    def _mock_head(self, status_code=200):
        return patch("requests.request", return_value=MagicMock(status_code=status_code))

    def test_gov_domain_gets_full_authority_score(self):
        with self._mock_head(200):
            score = statutory_api.calculate_legal_confidence(
                source_domain="api.govinfo.gov",
                raw_api_text="the term shall not exceed thirty days",
                llm_output_text="the term shall not exceed thirty days",
                source_url="https://api.govinfo.gov/x",
            )
        # authority(1.0*0.4) + integrity(1.0*0.4) + link(1.0*0.2) = 1.0
        assert score == 1.0

    def test_unrecognized_domain_gets_partial_authority_score(self):
        with self._mock_head(200):
            score = statutory_api.calculate_legal_confidence(
                source_domain="random-legal-blog.com",
                raw_api_text="some legal text here",
                llm_output_text="some legal text here",
                source_url="https://random-legal-blog.com/x",
            )
        assert score < 1.0

    def test_negation_flip_is_caught_by_integrity_score(self):
        """Regression check: dropping/flipping 'not'/'no' must lower the
        integrity score, since those words are explicitly tracked."""
        with self._mock_head(200):
            faithful = statutory_api.calculate_legal_confidence(
                source_domain="api.govinfo.gov",
                raw_api_text="the payment shall not exceed the statutory cap",
                llm_output_text="the payment shall not exceed the statutory cap",
                source_url="https://api.govinfo.gov/x",
            )
            flipped = statutory_api.calculate_legal_confidence(
                source_domain="api.govinfo.gov",
                raw_api_text="the payment shall not exceed the statutory cap",
                llm_output_text="the payment shall exceed the statutory cap",
                source_url="https://api.govinfo.gov/x",
            )
        assert flipped < faithful

    def test_redirect_gives_partial_link_score(self):
        with self._mock_head(301):
            score = statutory_api.calculate_legal_confidence(
                source_domain="api.govinfo.gov",
                raw_api_text="text",
                llm_output_text="text",
                source_url="https://api.govinfo.gov/x",
            )
        # authority 1.0*0.4 + integrity 1.0*0.4 + link 0.5*0.2 = 0.9
        assert score == 0.9

    def test_unreachable_link_scores_zero_for_link_component(self):
        with patch("requests.request", side_effect=requests.exceptions.ConnectionError("down")):
            score = statutory_api.calculate_legal_confidence(
                source_domain="api.govinfo.gov",
                raw_api_text="text",
                llm_output_text="text",
                source_url="https://api.govinfo.gov/x",
            )
        # authority 0.4 + integrity 0.4 + link 0.0 = 0.8
        assert score == 0.8