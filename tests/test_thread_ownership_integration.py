"""
Tests the ONE thing that matters most and was never actually tested:
can Lawyer B see Lawyer A's private conversation? This must always be NO.

WHAT THIS TEST DOES, IN PLAIN TERMS:
1. Creates two pretend lawyers: "TEST_ISOLATION_LAWYER_A" and "...LAWYER_B".
2. Saves a fake conversation under Lawyer A's name.
3. Checks that Lawyer A CAN see it, and Lawyer B CANNOT.
4. Deletes all the pretend data afterward, so nothing fake is left behind
   in your real database.

WHY THIS RUNS AGAINST YOUR REAL DATABASE (not a mock):
Every other test in this suite mocks the database away, which is correct
for testing app LOGIC fast and without side effects. But that means the
actual SQL queries that enforce "lawyers can't see each other's threads"
have never actually been run against a real database in any test until
this one. A privacy guarantee is exactly the kind of thing you don't want
to trust to a mock — this test uses your real DATABASE_URL (from .env,
same as your running app) so the real SQL actually executes.

SAFETY: all data this test writes is tagged with an unmistakable
"TEST_ISOLATION_..." prefix and deleted in teardown, whether the test
passes or fails. It never touches or reads any of your real lawyers'
data. Skipped automatically (with a clear message) if DATABASE_URL isn't
configured.

Excluded from the default `pytest tests/` run. Run explicitly with:
    pytest tests/test_thread_ownership_integration.py -m integration -v
"""
import os
import sys
import uuid
import importlib
import pytest

pytestmark = pytest.mark.integration

LAWYER_A = "TEST_ISOLATION_LAWYER_A"
LAWYER_B = "TEST_ISOLATION_LAWYER_B"


@pytest.fixture(scope="module")
def real_main():
    """Imports main.py fresh, using your REAL DATABASE_URL from .env — not
    the fake placeholder other test files substitute. If test_main.py ran
    earlier in the same pytest session, it may have already replaced
    sys.modules["main"] with a mocked version; this forces a genuinely
    fresh import here regardless."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        pytest.skip("python-dotenv not installed — can't read .env directly.")

    env_path = os.path.join(os.getcwd(), ".env")
    real_values = dotenv_values(env_path)
    real_db_url = real_values.get("DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not real_db_url or "fake:fake" in real_db_url:
        pytest.skip(
            "No real DATABASE_URL found in .env — skipping the real-database "
            "privacy check. This test needs your actual Postgres connection "
            "string to test the real SQL, not a mock."
        )

    os.environ["DATABASE_URL"] = real_db_url
    if "main" in sys.modules:
        del sys.modules["main"]

    main = importlib.import_module("main")
    yield main


@pytest.fixture
def cleanup_test_rows(real_main):
    """Runs after the test (pass or fail) — deletes every row this test
    created, identified unambiguously by the TEST_ISOLATION_ prefix."""
    yield
    with real_main.pool.connection() as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE thread_id LIKE 'test-isolation-%'"
        )
        conn.execute(
            "DELETE FROM thread_metadata WHERE lawyer_name LIKE 'TEST_ISOLATION_%'"
        )


class TestLawyerDataIsolation:
    def test_lawyer_b_cannot_see_lawyer_a_thread(self, real_main, cleanup_test_rows):
        thread_id = f"test-isolation-{uuid.uuid4().hex[:8]}"

        real_main.save_thread_metadata(
            thread_id=thread_id, lawyer_name=LAWYER_A, label="Test privacy check"
        )

        # The actual privacy guarantee: A can see their own thread...
        assert real_main.thread_belongs_to_lawyer(thread_id, LAWYER_A) is True
        # ...but B, a different lawyer, absolutely cannot.
        assert real_main.thread_belongs_to_lawyer(thread_id, LAWYER_B) is False

    def test_lawyer_b_thread_list_never_includes_lawyer_a_threads(self, real_main, cleanup_test_rows):
        thread_id = f"test-isolation-{uuid.uuid4().hex[:8]}"
        real_main.save_thread_metadata(
            thread_id=thread_id, lawyer_name=LAWYER_A, label="Test privacy check"
        )

        a_threads = [t["thread_id"] for t in real_main.get_threads_for_lawyer(LAWYER_A)]
        b_threads = [t["thread_id"] for t in real_main.get_threads_for_lawyer(LAWYER_B)]

        assert thread_id in a_threads
        assert thread_id not in b_threads

    def test_saved_messages_round_trip_correctly(self, real_main, cleanup_test_rows):
        thread_id = f"test-isolation-{uuid.uuid4().hex[:8]}"
        real_main.save_thread_metadata(thread_id=thread_id, lawyer_name=LAWYER_A, label="msg test")
        real_main.save_message(thread_id, role="user", content="What is habeas corpus?")
        real_main.save_message(thread_id, role="assistant", content="Habeas corpus is...", confidence=0.9, intent="definition")

        messages = real_main.get_messages_for_thread(thread_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["confidence"] == 0.9

    def test_second_call_to_save_thread_metadata_does_not_overwrite_owner(self, real_main, cleanup_test_rows):
        """Regression check for the ON CONFLICT DO NOTHING behavior: once
        a thread is owned by Lawyer A, a later call claiming Lawyer B
        owns the same thread_id must NOT silently reassign ownership."""
        thread_id = f"test-isolation-{uuid.uuid4().hex[:8]}"
        real_main.save_thread_metadata(thread_id=thread_id, lawyer_name=LAWYER_A, label="first")
        real_main.save_thread_metadata(thread_id=thread_id, lawyer_name=LAWYER_B, label="second, should be ignored")

        assert real_main.thread_belongs_to_lawyer(thread_id, LAWYER_A) is True
        assert real_main.thread_belongs_to_lawyer(thread_id, LAWYER_B) is False