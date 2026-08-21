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