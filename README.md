# Sehtak AI (صحتك AI) — Proactive Medical Safety & Triage Guardian 🇪🇬🩺

**Sehtak AI (صحتك AI)** is a production-ready, channel-decoupled Medical Safety & Triage Backend tailored for the Egyptian healthcare ecosystem. Its primary clinical mission is preventing **Fragmented Care** and **Silent Drug-Induced Complications**—specifically **NSAID-Induced Nephrotoxicity / Acute Kidney Injury (AKI)** caused by chronic OTC use of medications like *Cataflam, Voltaren, Ketofan, Bai-Alcofan, and Brufen* presenting with vague systemic symptoms (unexplained arm/body pain, fatigue, nausea, oliguria).

---

## 🏗️ Core Architecture

```text
sehtak-ai/
├── .env.example
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    ├── config.py                    # Centralized Pydantic-Settings configuration
    ├── main.py                      # FastAPI entry point + lifespan + REST & Webhook endpoints
    ├── schemas/
    │   └── clinical_schemas.py      # Strict Pydantic v2 models for Intake, Rules, and ER Triage Output
    ├── database/
    │   ├── db.py                    # Async SQLAlchemy (SQLite/PostgreSQL) + Stateful Session Merger
    │   └── egyptian_drugs_seed.py   # Seeded Egyptian OTC/Rx Drug Knowledge Base (Arabic & English aliases)
    ├── medical_engine/
    │   ├── rules_engine.py          # Deterministic Red-Flag & Nephrotoxicity Guardrail Engine
    │   └── triage_orchestrator.py   # "The Dots Connector" combining Rules + KDIGO/NICE Grounding + PDF
    ├── ai/
    │   └── multimodal_intake.py     # Google Gemini 2.5 Flash Voice/Image/Text extraction + Egyptian NLU
    ├── reports/
    │   └── pdf_generator.py         # Monochrome 1-Page B&W Emergency Triage Report PDF Generator
    └── bot/
        ├── telegram_bot.py          # Stateful aiogram 3.x Bot (Caregiver Mode, Voice, Photo, Text, PDF)
        └── whatsapp_webhook.py      # Decoupled Meta WhatsApp Cloud API Webhook Router
```

---

## 🚀 Quickstart

### 1. Create a Virtual Environment & Install Dependencies
```bash
cd sehtak-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and TELEGRAM_BOT_TOKEN
```

### 3. Run the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000/docs` to explore the interactive Swagger UI.

---

## 🧪 Testing the Clinical Scenarios via `curl`

### Scenario A: Missing Duration Guardrail (`Rule 2`)
When a patient or caregiver mentions taking **Cataflam (كتافلام)** without specifying for how many days, the deterministic guardrail halts final triage and asks for the duration in Egyptian Arabic:
```bash
curl -X POST http://localhost:8000/api/v1/triage/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "patient-101",
    "text_input": "والدتي عندها 62 سنة وبتاخد برشام كتافلام عشان المفاصل",
    "reset_session": true
  }'
```

### Scenario B: Completing the Timeline -> Triggers Silent Renal Danger (`Rule 1`) & Generates PDF
When the caregiver replies with the duration and symptoms (`بقالها شهرين وبتشتكي من وجع في دراعها وهمدان وغممان نفس`), the session merges the timeline, triggers `RULE_01_SILENT_RENAL_DANGER`, and generates the monochrome `Emergency_Triage_Report.pdf`:
```bash
curl -X POST http://localhost:8000/api/v1/triage/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "patient-101",
    "text_input": "بتاخده بقالها شهرين كل يوم، ودلوقتي عندها وجع شديد في دراعها وهمدان وغممان نفس"
  }'
```
