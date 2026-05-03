"""
Comprehensive tests for StudySearch FastAPI backend.
Run with: pytest test_api.py -v
"""

import os
import time
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Force local SQLite (not Turso) and disable Anthropic key before import
os.environ["TURSO_DATABASE_URL"] = ""
os.environ["TURSO_AUTH_TOKEN"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

# Must import after env vars are set so module-level reads pick them up
import studytool
studytool.TURSO_DATABASE_URL = ""
studytool.TURSO_AUTH_TOKEN = ""
studytool.ANTHROPIC_API_KEY = ""

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DB_FILE = "test_studysearch.db"


@pytest.fixture(autouse=True)
def _use_test_db(tmp_path):
    """Point the app at a fresh per-test database."""
    db_path = str(tmp_path / "test.db")
    studytool.DB_PATH = db_path
    studytool.init_db()
    yield
    # cleanup handled by tmp_path


@pytest.fixture()
def client():
    """TestClient that bypasses rate limiting."""
    # Disable rate limits for tests
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    original_limiter = studytool.limiter
    studytool.limiter = Limiter(key_func=get_remote_address, enabled=False)
    studytool.app.state.limiter = studytool.limiter

    with TestClient(studytool.app, raise_server_exceptions=False) as c:
        yield c

    studytool.limiter = original_limiter
    studytool.app.state.limiter = original_limiter


@pytest.fixture()
def auth_user(client):
    """Create a test user and return (token, user_id, session_id)."""
    resp = client.post("/api/auth/signup", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123",
    })
    assert resp.status_code == 200
    data = resp.json()
    return data["token"], data["user_id"], data["session_id"]


@pytest.fixture()
def auth_header(auth_user):
    """Return Authorization header dict for the test user."""
    token, _, _ = auth_user
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def session_with_doc(client, auth_user):
    """Create a session with an uploaded .txt document.
    Returns (session_id, doc_id).
    """
    _, _, session_id = auth_user
    content = (
        "Photosynthesis is the process by which green plants convert sunlight "
        "into chemical energy stored in glucose. It occurs primarily in the "
        "chloroplasts of plant cells. The light-dependent reactions take place "
        "in the thylakoid membranes and produce ATP and NADPH. The Calvin cycle "
        "uses carbon dioxide and the products of the light reactions to synthesize "
        "glucose molecules. Chlorophyll is the main pigment that absorbs light energy. "
        "Water molecules are split during the light reactions releasing oxygen as a byproduct. "
        "The overall equation is 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. "
        "Photosynthesis is essential for life on Earth as it produces oxygen and organic compounds."
    )
    resp = client.post(
        f"/api/sessions/{session_id}/documents",
        files={"file": ("biology_notes.txt", content.encode(), "text/plain")},
    )
    assert resp.status_code == 200
    doc_id = resp.json()["doc_id"]
    return session_id, doc_id


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


# ---------------------------------------------------------------------------
# 2. Auth flow
# ---------------------------------------------------------------------------

class TestAuth:
    def test_signup_valid(self, client):
        resp = client.post("/api/auth/signup", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "secret123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "user_id" in data
        assert data["username"] == "alice"
        assert "session_id" in data

    def test_signup_duplicate_email(self, client):
        payload = {
            "username": "bob",
            "email": "dup@example.com",
            "password": "secret123",
        }
        client.post("/api/auth/signup", json=payload)
        resp = client.post("/api/auth/signup", json=payload)
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"].lower()

    def test_signup_short_password(self, client):
        resp = client.post("/api/auth/signup", json={
            "username": "carol",
            "email": "carol@example.com",
            "password": "abc",
        })
        assert resp.status_code == 400
        assert "6 characters" in resp.json()["detail"]

    def test_login_valid(self, client, auth_user):
        resp = client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "session_id" in data

    def test_login_wrong_password(self, client, auth_user):
        resp = client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

    def test_get_me_with_token(self, client, auth_user, auth_header):
        resp = client.get("/api/auth/me", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert "session_id" in data

    def test_get_me_without_token(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_logout_invalidates_token(self, client, auth_user):
        token, _, _ = auth_user
        headers = {"Authorization": f"Bearer {token}"}

        # Token works before logout
        assert client.get("/api/auth/me", headers=headers).status_code == 200

        # Logout
        resp = client.post("/api/auth/logout", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Token should be invalid now
        resp = client.get("/api/auth/me", headers=headers)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 3. Password Reset
# ---------------------------------------------------------------------------

class TestPasswordReset:
    def test_request_reset_returns_token(self, client, auth_user):
        resp = client.post("/api/auth/reset-request", json={
            "email": "test@example.com",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert len(data["token"]) > 0

    def test_request_reset_unknown_email_returns_empty_token(self, client):
        resp = client.post("/api/auth/reset-request", json={
            "email": "nobody@example.com",
        })
        assert resp.status_code == 200
        assert resp.json()["token"] == ""

    def test_reset_with_valid_token(self, client, auth_user):
        # Request a reset token
        resp = client.post("/api/auth/reset-request", json={
            "email": "test@example.com",
        })
        reset_token = resp.json()["token"]

        # Use the token to change password
        resp = client.post("/api/auth/reset", json={
            "token": reset_token,
            "new_password": "newpassword456",
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Old password should no longer work
        resp = client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
        assert resp.status_code == 401

        # New password should work
        resp = client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "newpassword456",
        })
        assert resp.status_code == 200

    def test_reset_with_used_token_fails(self, client, auth_user):
        resp = client.post("/api/auth/reset-request", json={
            "email": "test@example.com",
        })
        reset_token = resp.json()["token"]

        # Use it once
        client.post("/api/auth/reset", json={
            "token": reset_token,
            "new_password": "newpassword456",
        })

        # Try to use it again
        resp = client.post("/api/auth/reset", json={
            "token": reset_token,
            "new_password": "anotherpassword",
        })
        assert resp.status_code == 400
        assert "expired" in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()

    def test_reset_with_expired_token_fails(self, client, auth_user):
        # Create a reset token and manually expire it
        resp = client.post("/api/auth/reset-request", json={
            "email": "test@example.com",
        })
        reset_token = resp.json()["token"]

        # Manually set the expiry in the past
        conn = studytool.get_db()
        conn.execute(
            "UPDATE password_reset_tokens SET expires_at=? WHERE token=?",
            (time.time() - 3600, reset_token),
        )
        conn.commit()
        conn.close()

        resp = client.post("/api/auth/reset", json={
            "token": reset_token,
            "new_password": "newpassword456",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 4. Sessions
# ---------------------------------------------------------------------------

class TestSessions:
    def test_create_session(self, client, auth_header):
        resp = client.post("/api/sessions", json={"name": "Bio Notes"}, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["name"] == "Bio Notes"

    def test_get_session_returns_docs(self, client, session_with_doc):
        sid, doc_id = session_with_doc
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "biology_notes.txt"

    def test_get_nonexistent_session(self, client):
        resp = client.get("/api/sessions/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. Document upload
# ---------------------------------------------------------------------------

class TestDocumentUpload:
    def test_upload_txt_file(self, client, auth_user):
        _, _, sid = auth_user
        content = b"This is a test document with enough words to be parsed as content for the study tool application."
        resp = client.post(
            f"/api/sessions/{sid}/documents",
            files={"file": ("notes.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "doc_id" in data
        assert data["filename"] == "notes.txt"
        assert data["chunks"] >= 1

    def test_upload_creates_searchable_chunks(self, client, session_with_doc):
        sid, _ = session_with_doc
        # Verify chunks exist by searching
        resp = client.post(f"/api/sessions/{sid}/search", json={"query": "photosynthesis"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0
        # Results should contain relevant text
        found_text = " ".join(r["text"].lower() for r in results)
        assert "photosynthesis" in found_text

    def test_upload_unsupported_format(self, client, auth_user):
        _, _, sid = auth_user
        resp = client.post(
            f"/api/sessions/{sid}/documents",
            files={"file": ("image.jpg", b"fake-image-data", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_delete_document(self, client, session_with_doc):
        sid, doc_id = session_with_doc
        resp = client.delete(f"/api/sessions/{sid}/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # After deletion, search should show no documents
        resp = client.post(f"/api/sessions/{sid}/search", json={"query": "photosynthesis"})
        assert resp.status_code == 400  # "No documents uploaded yet"


# ---------------------------------------------------------------------------
# 6. Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_bm25_search_returns_relevant_results(self, client, session_with_doc):
        sid, _ = session_with_doc
        resp = client.post(f"/api/sessions/{sid}/search", json={"query": "chlorophyll light energy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "chlorophyll light energy"
        results = data["results"]
        assert len(results) > 0
        # Each result should have expected fields
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "highlighted" in r
            assert r["score"] > 0

    def test_search_with_no_docs_returns_400(self, client, auth_user):
        _, _, sid = auth_user
        resp = client.post(f"/api/sessions/{sid}/search", json={"query": "anything"})
        assert resp.status_code == 400

    def test_search_highlights_terms(self, client, session_with_doc):
        sid, _ = session_with_doc
        resp = client.post(f"/api/sessions/{sid}/search", json={"query": "glucose"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        if results:
            assert "<mark>" in results[0]["highlighted"]


# ---------------------------------------------------------------------------
# 7. Community
# ---------------------------------------------------------------------------

class TestCommunity:
    def test_list_ap_classes(self, client):
        resp = client.get("/api/community/classes")
        assert resp.status_code == 200
        classes = resp.json()["classes"]
        assert len(classes) > 30
        assert "AP Biology" in classes
        assert "AP Calculus AB" in classes

    def test_create_post_requires_auth(self, client):
        resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "Test Post",
            "body": "Some body text",
        })
        assert resp.status_code == 401

    def test_create_post(self, client, auth_header):
        resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "Mitosis Study Guide",
            "body": "Here are my notes on mitosis...",
            "category": "study-guide",
        }, headers=auth_header)
        assert resp.status_code == 200
        assert "id" in resp.json()

    def test_create_post_invalid_class(self, client, auth_header):
        resp = client.post("/api/community/posts", json={
            "ap_class": "AP Underwater Basketweaving",
            "title": "Title",
            "body": "Body",
        }, headers=auth_header)
        assert resp.status_code == 400

    def test_list_posts(self, client, auth_header):
        # Create a post first
        client.post("/api/community/posts", json={
            "ap_class": "AP Chemistry",
            "title": "Moles and Stoichiometry",
            "body": "Study guide for unit 3",
            "category": "general",
        }, headers=auth_header)

        resp = client.get("/api/community/posts")
        assert resp.status_code == 200
        posts = resp.json()["posts"]
        assert len(posts) >= 1

    def test_list_posts_filter_by_class(self, client, auth_header):
        client.post("/api/community/posts", json={
            "ap_class": "AP Physics 1: Algebra-Based",
            "title": "Kinematics",
            "body": "Motion equations",
            "category": "notes",
        }, headers=auth_header)

        resp = client.get("/api/community/posts?ap_class=AP Physics 1: Algebra-Based")
        assert resp.status_code == 200
        for p in resp.json()["posts"]:
            assert p["ap_class"] == "AP Physics 1: Algebra-Based"

    def test_get_single_post_with_replies(self, client, auth_header):
        # Create post
        post_resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "Genetics Help",
            "body": "Can someone explain Punnett squares?",
        }, headers=auth_header)
        post_id = post_resp.json()["id"]

        # Create reply
        client.post(f"/api/community/posts/{post_id}/replies", json={
            "body": "Sure! A Punnett square is a diagram used to predict genotypes.",
        }, headers=auth_header)

        # Get the post
        resp = client.get(f"/api/community/posts/{post_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["post"]["title"] == "Genetics Help"
        assert len(data["replies"]) == 1
        assert "Punnett square" in data["replies"][0]["body"]

    def test_create_reply(self, client, auth_header):
        # Create a post
        post_resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "Question",
            "body": "What is DNA?",
        }, headers=auth_header)
        post_id = post_resp.json()["id"]

        resp = client.post(f"/api/community/posts/{post_id}/replies", json={
            "body": "DNA stands for deoxyribonucleic acid.",
        }, headers=auth_header)
        assert resp.status_code == 200
        assert "id" in resp.json()

    def test_create_reply_requires_auth(self, client, auth_header):
        post_resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "Question",
            "body": "What is RNA?",
        }, headers=auth_header)
        post_id = post_resp.json()["id"]

        resp = client.post(f"/api/community/posts/{post_id}/replies", json={
            "body": "RNA stands for ribonucleic acid.",
        })
        assert resp.status_code == 401

    def test_upvote_toggle(self, client, auth_header):
        # Create post
        post_resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "Upvote Test",
            "body": "Test body",
        }, headers=auth_header)
        post_id = post_resp.json()["id"]

        # First vote -- adds upvote
        resp = client.post(f"/api/community/posts/{post_id}/upvote", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["voted"] is True
        assert resp.json()["upvotes"] == 1

        # Second vote -- removes upvote (toggle off)
        resp = client.post(f"/api/community/posts/{post_id}/upvote", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["voted"] is False
        assert resp.json()["upvotes"] == 0

        # Third vote -- adds upvote again (re-vote)
        resp = client.post(f"/api/community/posts/{post_id}/upvote", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["voted"] is True
        assert resp.json()["upvotes"] == 1

    def test_delete_own_post(self, client, auth_header):
        post_resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "To Delete",
            "body": "This will be deleted",
        }, headers=auth_header)
        post_id = post_resp.json()["id"]

        resp = client.delete(f"/api/community/posts/{post_id}", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Post should no longer exist
        resp = client.get(f"/api/community/posts/{post_id}")
        assert resp.status_code == 404

    def test_delete_others_post_forbidden(self, client, auth_header):
        # Create post as first user
        post_resp = client.post("/api/community/posts", json={
            "ap_class": "AP Biology",
            "title": "Protected Post",
            "body": "Cannot delete this",
        }, headers=auth_header)
        post_id = post_resp.json()["id"]

        # Create a second user
        resp2 = client.post("/api/auth/signup", json={
            "username": "otheruser",
            "email": "other@example.com",
            "password": "password123",
        })
        other_header = {"Authorization": f"Bearer {resp2.json()['token']}"}

        # Try to delete as second user
        resp = client.delete(f"/api/community/posts/{post_id}", headers=other_header)
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 8. Flashcards (mocked AI)
# ---------------------------------------------------------------------------

class TestFlashcards:
    def _mock_anthropic_response(self, content_text):
        """Create a mock response object for httpx.AsyncClient.post."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": content_text}],
        }
        return mock_resp

    def test_flashcards_returns_array(self, client, session_with_doc):
        sid, _ = session_with_doc
        studytool.ANTHROPIC_API_KEY = "test-key"

        flashcards_json = json.dumps([
            {"front": "What is photosynthesis?", "back": "The process by which plants convert sunlight into glucose."},
            {"front": "Where does photosynthesis occur?", "back": "In the chloroplasts of plant cells."},
        ])

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_anthropic_response(flashcards_json)
            resp = client.post(f"/api/sessions/{sid}/flashcards", json={"topic": "photosynthesis", "count": 2})

        studytool.ANTHROPIC_API_KEY = ""

        assert resp.status_code == 200
        data = resp.json()
        assert "flashcards" in data
        assert len(data["flashcards"]) == 2
        assert "front" in data["flashcards"][0]
        assert "back" in data["flashcards"][0]

    def test_flashcards_no_docs_returns_400(self, client, auth_user):
        _, _, sid = auth_user
        studytool.ANTHROPIC_API_KEY = "test-key"
        resp = client.post(f"/api/sessions/{sid}/flashcards", json={"topic": "anything"})
        studytool.ANTHROPIC_API_KEY = ""
        assert resp.status_code == 400

    def test_flashcards_no_api_key_returns_500(self, client, session_with_doc):
        sid, _ = session_with_doc
        studytool.ANTHROPIC_API_KEY = ""
        resp = client.post(f"/api/sessions/{sid}/flashcards", json={"topic": "photosynthesis"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# 9. Quiz (mocked AI)
# ---------------------------------------------------------------------------

class TestQuiz:
    def _mock_anthropic_response(self, content_text):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": content_text}],
        }
        return mock_resp

    def test_quiz_returns_questions(self, client, session_with_doc):
        sid, _ = session_with_doc
        studytool.ANTHROPIC_API_KEY = "test-key"

        quiz_json = json.dumps([
            {
                "question": "What is the main pigment in photosynthesis?",
                "options": ["Chlorophyll", "Carotene", "Xanthophyll", "Melanin"],
                "correct": 0,
                "explanation": "Chlorophyll is the primary pigment that absorbs light energy.",
            },
            {
                "question": "What gas is released during photosynthesis?",
                "options": ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"],
                "correct": 2,
                "explanation": "Oxygen is released as a byproduct when water molecules are split.",
            },
        ])

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_anthropic_response(quiz_json)
            resp = client.post(f"/api/sessions/{sid}/quiz", json={"topic": "photosynthesis", "count": 2})

        studytool.ANTHROPIC_API_KEY = ""

        assert resp.status_code == 200
        data = resp.json()
        assert "questions" in data
        assert len(data["questions"]) == 2
        q = data["questions"][0]
        assert "question" in q
        assert "options" in q
        assert "correct" in q
        assert "explanation" in q

    def test_quiz_no_docs_returns_400(self, client, auth_user):
        _, _, sid = auth_user
        studytool.ANTHROPIC_API_KEY = "test-key"
        resp = client.post(f"/api/sessions/{sid}/quiz", json={"topic": "anything"})
        studytool.ANTHROPIC_API_KEY = ""
        assert resp.status_code == 400

    def test_quiz_no_api_key_returns_500(self, client, session_with_doc):
        sid, _ = session_with_doc
        studytool.ANTHROPIC_API_KEY = ""
        resp = client.post(f"/api/sessions/{sid}/quiz", json={"topic": "photosynthesis"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# 10. Progress
# ---------------------------------------------------------------------------

class TestProgress:
    def test_log_activity(self, client, auth_user, auth_header):
        _, _, sid = auth_user
        resp = client.post(f"/api/sessions/{sid}/progress", json={
            "activity_type": "search",
            "topic": "photosynthesis",
        }, headers=auth_header)
        assert resp.status_code == 200
        assert "id" in resp.json()

    def test_log_quiz_score(self, client, auth_user, auth_header):
        _, _, sid = auth_user
        resp = client.post(f"/api/sessions/{sid}/progress", json={
            "activity_type": "quiz",
            "topic": "biology",
            "score": 4,
            "total": 5,
        }, headers=auth_header)
        assert resp.status_code == 200

    def test_get_progress_stats(self, client, auth_user, auth_header):
        _, _, sid = auth_user

        # Log some activities
        client.post(f"/api/sessions/{sid}/progress", json={
            "activity_type": "search",
            "topic": "cell division",
        }, headers=auth_header)
        client.post(f"/api/sessions/{sid}/progress", json={
            "activity_type": "quiz",
            "topic": "genetics",
            "score": 3,
            "total": 5,
        }, headers=auth_header)
        client.post(f"/api/sessions/{sid}/progress", json={
            "activity_type": "flashcard",
            "topic": "proteins",
        }, headers=auth_header)

        resp = client.get(f"/api/sessions/{sid}/progress", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "activities" in data
        assert "stats" in data
        stats = data["stats"]
        assert stats["total_searches"] == 1
        assert stats["total_quizzes"] == 1
        assert stats["total_flashcards"] == 1
        assert stats["avg_quiz_score"] == 60.0  # 3/5 = 60%
        assert "streak_days" in stats

    def test_progress_requires_auth(self, client, auth_user):
        _, _, sid = auth_user
        resp = client.post(f"/api/sessions/{sid}/progress", json={
            "activity_type": "search",
        })
        assert resp.status_code == 401

        resp = client.get(f"/api/sessions/{sid}/progress")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 11. Sharing
# ---------------------------------------------------------------------------

class TestSharing:
    def test_create_share_link(self, client, auth_user, auth_header):
        _, _, sid = auth_user
        resp = client.post(f"/api/sessions/{sid}/share", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "share_url" in data
        assert "token" in data
        assert data["share_url"].startswith("/shared/")

    def test_access_shared_session(self, client, session_with_doc, auth_header):
        sid, _ = session_with_doc

        # Create a share link
        share_resp = client.post(f"/api/sessions/{sid}/share", headers=auth_header)
        share_token = share_resp.json()["token"]

        # Access the shared session publicly (no auth needed)
        resp = client.get(f"/api/shared/{share_token}")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_name" in data
        assert "documents" in data
        assert len(data["documents"]) >= 1
        assert data["shared_by"] == "testuser"

    def test_access_invalid_share_token(self, client):
        resp = client.get("/api/shared/nonexistent-token")
        assert resp.status_code == 404

    def test_share_requires_auth(self, client, auth_user):
        _, _, sid = auth_user
        resp = client.post(f"/api/sessions/{sid}/share")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Extra: Explain endpoint (mocked AI)
# ---------------------------------------------------------------------------

class TestExplain:
    def _mock_anthropic_response(self, content_text):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": content_text}],
        }
        return mock_resp

    def test_explain_returns_answer(self, client, session_with_doc):
        sid, _ = session_with_doc
        studytool.ANTHROPIC_API_KEY = "test-key"

        # First get chunk IDs via search
        search_resp = client.post(f"/api/sessions/{sid}/search", json={"query": "photosynthesis"})
        chunk_ids = [r["id"] for r in search_resp.json()["results"][:2]]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_anthropic_response(
                "Photosynthesis converts sunlight into glucose in plant chloroplasts."
            )
            resp = client.post(f"/api/sessions/{sid}/explain", json={
                "query": "What is photosynthesis?",
                "chunk_ids": chunk_ids,
            })

        studytool.ANTHROPIC_API_KEY = ""

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "conversation_id" in data
        assert "sources" in data

    def test_explain_no_api_key(self, client, session_with_doc):
        sid, _ = session_with_doc
        studytool.ANTHROPIC_API_KEY = ""
        resp = client.post(f"/api/sessions/{sid}/explain", json={
            "query": "What is photosynthesis?",
            "chunk_ids": ["fake-id"],
        })
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Extra: Conversations
# ---------------------------------------------------------------------------

class TestConversations:
    def test_list_conversations_empty(self, client, auth_user):
        _, _, sid = auth_user
        resp = client.get(f"/api/sessions/{sid}/conversations")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Extra: Suggestions
# ---------------------------------------------------------------------------

class TestSuggestions:
    def test_suggestions_with_no_docs(self, client, auth_user):
        _, _, sid = auth_user
        resp = client.get(f"/api/sessions/{sid}/suggestions")
        assert resp.status_code == 200
        assert resp.json()["suggestions"] == []

    def test_suggestions_with_docs(self, client, session_with_doc):
        sid, _ = session_with_doc
        resp = client.get(f"/api/sessions/{sid}/suggestions")
        assert resp.status_code == 200
        suggestions = resp.json()["suggestions"]
        # Should have generated some suggestions from the biology content
        assert isinstance(suggestions, list)


# ---------------------------------------------------------------------------
# Extra: User Profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_get_user_profile(self, client, auth_user):
        _, user_id, _ = auth_user
        resp = client.get(f"/api/users/{user_id}/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"]["username"] == "testuser"
        assert "stats" in data
        assert "recent_posts" in data

    def test_update_bio(self, client, auth_user, auth_header):
        resp = client.put("/api/users/me/bio", json={"bio": "AP Bio student"}, headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["bio"] == "AP Bio student"

    def test_update_bio_requires_auth(self, client):
        resp = client.put("/api/users/me/bio", json={"bio": "No auth"})
        assert resp.status_code == 401

    def test_nonexistent_user_profile(self, client):
        resp = client.get("/api/users/fake-id-12345/profile")
        assert resp.status_code == 404
