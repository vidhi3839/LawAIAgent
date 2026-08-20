"""
Opt-in LIVE tests — these hit the real internet (real Cornell/GovInfo).

Excluded from the default `pytest tests/` run (see pytest.ini's
`-m "not integration"`). Run explicitly and occasionally — e.g. weekly, or
whenever you suspect a site changed — with:

    pytest tests/test_live_smoke.py -m integration -v

Why these exist separately from test_statutory_api_fixtures.py: the
fixture tests replay a SAVED snapshot and will keep passing forever even
if Cornell/GovInfo change their page structure tomorrow. These live tests
are the only thing that actually re-checks against the CURRENT real site.
If one of these fails, re-run capture_fixtures.py and go look at what
actually changed on the page before assuming it's just a network blip.
"""
import pytest
from tasks import statutory_api

pytestmark = pytest.mark.integration


class TestLiveCornell:
    def test_estoppel_definition_still_resolves(self):
        text, url, domain = statutory_api.fetch_legal_definition("estoppel")
        assert domain == "law.cornell.edu"
        assert not text.startswith("Could not find")
        assert len(text) > 100

    def test_fre_401_still_resolves(self):
        text, url, domain = statutory_api.fetch_federal_rule("fre", "401")
        assert domain == "law.cornell.edu"
        assert not text.startswith("Could not retrieve")
        assert len(text) > 100


class TestLiveGovInfo:
    def test_cfaa_1030_still_resolves(self):
        import os
        api_key = os.getenv("GOVINFO_API_KEY", "DEMO_KEY")
        text, url, domain = statutory_api.fetch_statute(title=18, section="1030", api_key=api_key)
        assert domain == "api.govinfo.gov"
        assert not text.startswith("Could not find")
        assert not text.startswith("Error")
        assert len(text) > 100