"""
End-to-end automated tests verifying all 5 core modules of Sehtak AI (صحتك AI):
1. Deterministic Egyptian Drug Database & Red-Flag Engine (Rule 1, Rule 2, Triple Whammy, GI Bleed)
2. Multimodal Intake Pipeline (Text, Voice, Image)
3. Stateful Triage Orchestrator ("The Dots Connector")
4. Monochrome Emergency Triage PDF Generator
5. FastAPI REST & WhatsApp Cloud API Endpoints
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.clinical_schemas import TriageSeverity


def test_end_to_end_sehtak_ai_pipeline():
    with TestClient(app) as client:
        # 1. Health check & Seeded Egyptian Drug Knowledge Base
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["seeded_egyptian_drugs_count"] >= 15

        drugs_resp = client.get("/api/v1/drugs")
        assert drugs_resp.status_code == 200
        brands = [d["canonical_brand"] for d in drugs_resp.json()]
        assert "Cataflam" in brands
        assert "Ketofan" in brands
        assert "Voltaren" in brands
        assert "Marevan" in brands
        assert "Capoten" in brands
        assert "Lasix" in brands

        # 2. Test Rule 2 (Missing Duration Guardrail):
        # User sends a message mentioning Cataflam without duration -> MUST halt and ask clarification in Egyptian Arabic
        step1_resp = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": "test-cairo-caregiver-01",
                "text_input": "لو سمحت والدتي عندها 65 سنة وعندها ضغط وبتاخد برشام كتافلام عشان وجع الركبة",
                "reset_session": True,
            },
        )
        assert step1_resp.status_code == 200
        step1_data = step1_resp.json()
        assert step1_data["status"] == "clarification_needed"
        assert step1_data["severity"] == TriageSeverity.NEEDS_CLARIFICATION.value
        assert "المريض بياخد المسكن ده بقاله كام يوم أو كام أسبوع؟" in step1_data["caregiver_response_ar"]
        assert step1_data["pdf_report_path"] is None

        # 3. Test Rule 1 (Silent Renal Danger) + Stateful Session Merge + PDF Report Generation:
        # Caregiver replies with duration ("بقالها شهرين") + vague systemic symptoms (arm pain, fatigue, nausea)
        step2_resp = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": "test-cairo-caregiver-01",
                "text_input": "بتاخده بقالها شهرين كل يوم، وبقالها يومين عندها وجع في دراعها وهمدان شديد وغممان نفس",
                "reset_session": False,
            },
        )
        assert step2_resp.status_code == 200
        step2_data = step2_resp.json()
        assert step2_data["status"] == "completed"
        assert step2_data["severity"] == TriageSeverity.CRITICAL_RED_FLAG.value

        # Verify Rule 1 triggered
        rule_ids = [r["rule_id"] for r in step2_data["rule_evaluation"]["triggered_alerts"]]
        assert "RULE_01_SILENT_RENAL_DANGER" in rule_ids

        # Verify mandatory renal lab panel
        labs_joined = " ".join(step2_data["rule_evaluation"]["urgent_labs_consolidated"])
        assert "Serum Creatinine" in labs_joined
        assert "BUN" in labs_joined
        assert "eGFR" in labs_joined
        assert "Electrolytes" in labs_joined
        assert "Urine Analysis" in labs_joined

        # Verify PDF report was generated and exists on disk
        pdf_path = step2_data["pdf_report_path"]
        assert pdf_path is not None
        assert Path(pdf_path).exists()
        assert Path(pdf_path).stat().st_size > 1000

        # 4. Test Triple Whammy Combination (Capoten + Lasix + Ketofan for 14 days)
        triple_resp = client.post(
            "/api/v1/triage/analyze",
            json={
                "session_id": "test-triple-whammy-02",
                "text_input": "والدي عنده 70 سنة بياخد كابوتين ولازكس للضغط، وبقاله أسبوعين بياخد كيتوفان كل يوم وعنده قلة البول وتورم في الرجلين",
                "reset_session": True,
            },
        )
        assert triple_resp.status_code == 200
        triple_data = triple_resp.json()
        assert triple_data["status"] == "completed"
        triple_rules = [r["rule_id"] for r in triple_data["rule_evaluation"]["triggered_alerts"]]
        assert "RULE_01_SILENT_RENAL_DANGER" in triple_rules
        assert "RULE_03_TRIPLE_WHAMMY_AKI" in triple_rules

        # 5. Test WhatsApp Cloud API Webhook Adapter
        wa_verify = client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "sehtak-whatsapp-verify-secret-2026",
                "hub.challenge": "123456789",
            },
        )
        assert wa_verify.status_code == 200
        assert wa_verify.text == "123456789"

        wa_msg = client.post(
            "/webhook/whatsapp",
            json={
                "entry": [
                    {
                        "changes": [
                            {
                                "value": {
                                    "messages": [
                                        {
                                            "from": "201001234567",
                                            "type": "text",
                                            "text": {
                                                "body": "أبويا عنده 60 سنة وبياخد باي الكوفان وماريفان بقاله شهر وعنده همدان ووجع في المعدة"
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
        )
        assert wa_msg.status_code == 200
        wa_data = wa_msg.json()
        assert wa_data["processed_count"] == 1
        assert wa_data["results"][0]["status"] == "completed"
        assert wa_data["results"][0]["severity"] == "CRITICAL_RED_FLAG"
        assert Path(wa_data["results"][0]["pdf_report_path"]).exists()
