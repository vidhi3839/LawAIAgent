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

        thread_id = f"test-isolation-{uuid.uuid4().hex[:8]}"
        real_main.save_thread_metadata(thread_id=thread_id, lawyer_name=LAWYER_A, label="first")
        real_main.save_thread_metadata(thread_id=thread_id, lawyer_name=LAWYER_B, label="second, should be ignored")

        assert real_main.thread_belongs_to_lawyer(thread_id, LAWYER_A) is True
        assert real_main.thread_belongs_to_lawyer(thread_id, LAWYER_B) is False