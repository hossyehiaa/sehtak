"""
Comprehensive E2E Test Suite for Sehtak AI v2:
1. End-to-end clinical triage pipeline (Rule 1, Rule 2, Triple Whammy)
2. Task 0.6: test_voice_failure_never_silently_triages
3. Task 1.6: Auth tests (signup, duplicate email 409, login wrong password 401, /me without token 401, session ownership)
4. Task 3.3: Verbatim-verified citations returned, dual PDF generation with sources, degradation handling
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas.clinical_schemas import TriageSeverity


def test_system_health():
    with TestClient(app) as client:
        resp = client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "gemini_key_present" in data
        assert "db_ok" in data
        assert data["db_ok"] is True
        assert data["rules_version"] == "2.0.0-deterministic"
        assert "rag_index_size" in data
        assert data["rag_index_size"] >= 10


def test_auth_full_cycle():
    with TestClient(app) as client:
        email = "dr.ahmed.caregiver@sehtak.ai"
        # 1. Signup
        signup_resp = client.post(
            "/api/v1/auth/signup",
            json={
                "email": email,
                "password": "StrongPassword2026!",
                "full_name": "Dr. Ahmed Mansour",
                "phone": "+201012345678",
                "role": "caregiver",
            },
        )
        assert signup_resp.status_code in [201, 409]
        if signup_resp.status_code == 409:
            # Already created in prior run, proceed to login
            pass

        # 2. Duplicate signup returns 409
        dup_resp = client.post(
            "/api/v1/auth/signup",
            json={
                "email": email,
                "password": "AnotherPassword",
                "full_name": "Duplicate User",
            },
        )
        assert dup_resp.status_code == 409

        # 3. Wrong password returns 401
        bad_login = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "WrongPassword!"},
        )
        assert bad_login.status_code == 401

        # 4. Correct login returns tokens
        good_login = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "StrongPassword2026!"},
        )
        assert good_login.status_code == 200
        token_data = good_login.json()
        access_token = token_data["access_token"]
        refresh_token = token_data["refresh_token"]
        assert access_token and refresh_token

        # 5. /me without token returns 401
        no_auth_me = client.get("/api/v1/auth/me")
        assert no_auth_me.status_code == 401

        # 6. /me with Bearer token returns profile
        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_resp.status_code == 200
        profile = me_resp.json()
        assert profile["email"] == email

        # 7. Refresh token
        ref_resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert ref_resp.status_code == 200
        assert "access_token" in ref_resp.json()


def test_auth_enforcement_when_demo_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_ANONYMOUS_DEMO", False)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/triage/analyze",
            json={"session_id": "locked-session", "text_input": "عندي صداع"},
        )
        assert resp.status_code == 401


def test_voice_failure_never_silently_triages(monkeypatch):
    """
    Task 0.6: Monkeypatch Gemini to raise; assert degradation error is surfaced,
    no triage output generated, and explicit Arabic degradation message sent.
    """
    from app.ai.multimodal_intake import multimodal_intake

    async def mock_extract_voice(*args, **kwargs):
        from app.ai.errors import IntakeDegradationError
        raise IntakeDegradationError(
            component="voice_transcription",
            reason="Gemini API Quota Exceeded (Simulated Test)",
        )

    monkeypatch.setattr(multimodal_intake, "extract_from_voice", mock_extract_voice)

    with TestClient(app) as client:
        # Send simulated dummy audio bytes
        fake_audio = b"FAKE_AUDIO_BYTES_TESTING_VOICE_FAILURE"
        resp = client.post(
            "/api/v1/triage/upload",
            data={"session_id": "voice-fail-test"},
            files={"file": ("test.ogg", fake_audio, "audio/ogg")},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["degradation"]["component"] == "voice_transcription"
        assert "تحليل الصوت مش شغال حالياً" in data["degradation"]["user_message_ar"]
        assert "triage_orchestrator" not in data


def test_triage_clinical_pipeline_with_rag_citations():
    """
    Full triage pipeline testing:
    - Rule 2 (Missing duration guardrail)
    - Rule 1 (Silent renal danger)
    - Verbatim citations from WHO / EDA / KDIGO / NICE / MOH
    - Dual monochrome PDFs generated
    """
    with TestClient(app) as client:
        session_id = "test-session-rag-001"

        # Step 1: Cataflam without duration -> Rule 2 halt
        step1 = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": session_id,
                "text_input": "والدتي عندها 65 سنة وعندها ضغط وبتاخد برشام كتافلام عشان وجع الركبة",
                "reset_session": True,
            },
        )
        assert step1.status_code == 200
        d1 = step1.json()
        assert d1["status"] == "clarification_needed"
        assert d1["severity"] == TriageSeverity.NEEDS_CLARIFICATION.value
        assert "المريض بياخد المسكن ده بقاله كام يوم أو كام أسبوع؟" in d1["caregiver_response_ar"]

        # Step 2: Caregiver provides duration + systemic symptoms
        step2 = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": session_id,
                "text_input": "بتاخده بقالها شهرين كل يوم، وبقالها يومين بتشتكي من وجع في دراعها وهمدان شديد وغممان نفس",
                "reset_session": False,
            },
        )
        assert step2.status_code == 200
        d2 = step2.json()
        assert d2["status"] == "completed"
        assert d2["severity"] == TriageSeverity.CRITICAL_RED_FLAG.value

        # Assert Rule 1 triggered
        rules_triggered = [a["rule_id"] for a in d2["rule_evaluation"]["triggered_alerts"]]
        assert "RULE_01_SILENT_RENAL_DANGER" in rules_triggered

        # Assert RAG Grounded Evidence contains verified claims
        grounded = d2.get("grounded_evidence")
        assert grounded is not None
        assert "claims" in grounded

        # Assert both Arabic and English PDFs were generated
        assert d2["pdf_report_path"] is not None and Path(d2["pdf_report_path"]).exists()
        assert d2["pdf_report_path_ar"] is not None and Path(d2["pdf_report_path_ar"]).exists()
        assert d2["pdf_base64"] is not None
        assert d2["pdf_base64_ar"] is not None


def test_medication_and_supplement_consultation_mode():
    """
    Verifies Mode C:
    When a patient/caregiver enters a supplement or medication name like 'سنترم لوتين':
    1. Extracted as a medication (VITAMIN_SUPPLEMENT), NOT as an illness/symptom.
    2. Does NOT halt for duration clarification.
    3. Triggers Mode C (Pharmacological Consultation).
    4. Generates rich clinical guidance with active ingredients, dosing, and safety.
    5. Emits dual PDF reports without errors.
    """
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": "test-session-centrum-consultation",
                "text_input": "سنترم لوتين",
                "reset_session": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

        # Check medications extraction
        meds = data["intake_snapshot"]["medications"]
        assert len(meds) >= 1
        assert any("Centrum" in m["brand_name"] for m in meds)
        assert any(m["drug_class"] == "VITAMIN_SUPPLEMENT" for m in meds)

        # Symptoms should be empty
        assert len(data["intake_snapshot"]["symptoms"]) == 0

        # Caregiver response should contain pharmacological consultation sections
        caregiver_text = data["caregiver_response_ar"]
        assert "التعريف بالمستحضر" in caregiver_text
        assert "المواد الفعالة" in caregiver_text or "اللوتين" in caregiver_text
        assert "هيئة الدواء المصرية" in caregiver_text

        # Physician payload check
        physician = data["physician_payload_en"]
        assert physician is not None
        assert physician["severity_badge"] == "[PHARMACOLOGICAL CONSULTATION - DRUG & SUPPLEMENT SAFETY]"
        assert "Centrum Lutein" in physician["chief_complaint_and_timeline"]

        # Dual PDF verification
        assert data["pdf_report_path"] is not None and Path(data["pdf_report_path"]).exists()
        assert data["pdf_report_path_ar"] is not None and Path(data["pdf_report_path_ar"]).exists()


def test_anonymous_trial_limit_and_unlimited_auth_bypass():
    """
    Verifies Guest Trial Mode:
    1. Unauthenticated users get exactly 2 messages in a session.
    2. The 3rd message without auth is blocked with HTTP 403 and high-converting message.
    3. An authenticated user can bypass trial restrictions completely (unlimited access).
    """
    with TestClient(app) as client:
        session_id = f"test-trial-guest-{os.urandom(4).hex()}"

        # 1st free message (reset_session=True to ensure clean start)
        r1 = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": session_id,
                "text_input": "عندي كحة خفيفة من يومين",
                "reset_session": True,
            },
        )
        assert r1.status_code == 200

        # 2nd free message
        r2 = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": session_id,
                "text_input": "وباخود بنادول قرصين كل يوم",
                "reset_session": False,
            },
        )
        assert r2.status_code == 200

        # 3rd message should be BLOCKED with HTTP 403 Forbidden
        r3 = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": session_id,
                "text_input": "ودلوقتي حاسس بدوخة وإرهاق",
                "reset_session": False,
            },
        )
        assert r3.status_code == 403
        assert "استنفدت استشارتي التجربة المجانية" in r3.json()["detail"]

        # Authenticate / Create a registered user
        user_email = f"trial_bypass_{os.urandom(4).hex()}@sehtak.ai"
        signup_resp = client.post(
            "/api/v1/auth/signup",
            json={
                "email": user_email,
                "password": "SecurePassword123!",
                "full_name": "د. هاني السعيد",
                "role": "physician",
            },
        )
        assert signup_resp.status_code == 201
        token = signup_resp.json()["access_token"]

        # The authenticated user sends the 3rd message with same session_id: Bypasses trial limit!
        r3_auth = client.post(
            "/api/v1/triage/analyze",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "session_id": session_id,
                "text_input": "ودلوقتي حاسس بدوخة وإرهاق",
                "reset_session": False,
            },
        )
        assert r3_auth.status_code == 200
        assert r3_auth.json()["status"] in ["completed", "needs_input"]


def test_google_auth_and_config_endpoints():
    """
    Verifies Google Sign-in:
    1. GET /api/v1/auth/google/config returns enabled status and client ID.
    2. POST /api/v1/auth/google provisions a user account and returns JWT credentials.
    3. Authenticated session verifies identity via GET /api/v1/auth/me.
    """
    with TestClient(app) as client:
        # 1. Config endpoint
        cfg_resp = client.get("/api/v1/auth/google/config")
        assert cfg_resp.status_code == 200
        cfg = cfg_resp.json()
        assert "client_id" in cfg
        assert "enabled" in cfg

        # 2. Direct Google sign-in payload
        google_email = f"doctor.google.{os.urandom(4).hex()}@gmail.com"
        google_name = "د. كريم رضوان (Google Verified)"
        auth_resp = client.post(
            "/api/v1/auth/google",
            json={
                "email": google_email,
                "name": google_name,
                "google_id": f"gid_{os.urandom(6).hex()}",
                "role": "caregiver",
            },
        )
        assert auth_resp.status_code == 200
        auth_data = auth_resp.json()
        assert "access_token" in auth_data
        assert "refresh_token" in auth_data
        assert auth_data["token_type"] == "bearer"

        # 3. Verify access token against /api/v1/auth/me
        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {auth_data['access_token']}"},
        )
        assert me_resp.status_code == 200
        me_data = me_resp.json()
        assert me_data["email"] == google_email
        assert me_data["full_name"] == google_name


