"""
Module 5C: FastAPI Entry Point & Lifespan Manager (`app/main.py`)
Exposes:
- `GET /health` -> System & DB readiness status
- `GET /api/v1/drugs` -> Seeded Egyptian high-risk medication catalog
- `POST /api/v1/triage/analyze` -> JSON / Multimodal REST endpoint for Web & Mobile apps
- `POST /api/v1/triage/upload` -> Multipart Voice Note (.ogg/.mp3) or Drug Strip Photo (.jpg/.png) endpoint
- `GET /api/v1/triage/report/{session_id}` -> Downloads the generated monochrome `Emergency_Triage_Report.pdf`
- `POST /webhook/telegram` -> Webhook endpoint for Telegram updates
- `/webhook/whatsapp` -> Mounted WhatsApp Cloud API router
"""

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiogram.types import Update
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.bot.telegram_bot import create_telegram_bot_and_dispatcher
from app.bot.whatsapp_webhook import whatsapp_router
from app.config import settings
from app.database.db import get_patient_intake_session, init_db, reset_patient_session
from app.database.egyptian_drugs_seed import EGYPTIAN_DRUG_KNOWLEDGE_BASE
from app.medical_engine.triage_orchestrator import triage_orchestrator
from app.schemas.clinical_schemas import TriageAnalyzeRequest, TriageOrchestratorResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sehtak_ai")

telegram_bot, telegram_dp = create_telegram_bot_and_dispatcher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite/PostgreSQL database, seeds Egyptian drug catalog, and manages Telegram polling if enabled."""
    logger.info("Initializing Sehtak AI Database & Egyptian Drug Knowledge Base...")
    await init_db()

    polling_task: Optional[asyncio.Task] = None
    if settings.ENABLE_TELEGRAM_POLLING and telegram_bot is not None:
        logger.info("Starting aiogram 3.x Telegram Bot polling in background...")
        polling_task = asyncio.create_task(telegram_dp.start_polling(telegram_bot))

    yield

    if polling_task is not None:
        polling_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await polling_task
    if telegram_bot is not None:
        await telegram_bot.session.close()


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "Sehtak AI (صحتك AI) — Proactive Medical Safety & Triage Guardian tailored for the Egyptian healthcare context. "
        "Prevents fragmented care and silent drug-induced complications (e.g., NSAID-induced Acute Kidney Injury)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(whatsapp_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_web_ui() -> HTMLResponse:
    """Serves the Sehtak AI Interactive Web UI & Caregiver Triage Simulator."""
    await init_db()
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>Sehtak AI Backend Running</h1>", status_code=200)


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint verifying engine & drug database readiness."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "seeded_egyptian_drugs_count": len(EGYPTIAN_DRUG_KNOWLEDGE_BASE),
        "telegram_bot_configured": telegram_bot is not None,
    }


@app.get("/api/v1/drugs", tags=["Egyptian Drug Knowledge Base"])
async def list_seeded_egyptian_drugs() -> List[Dict[str, Any]]:
    """Returns the deterministic Egyptian high-risk medication catalog."""
    return [
        {
            "canonical_brand": rec.canonical_brand,
            "active_ingredient": rec.active_ingredient,
            "drug_class": rec.drug_class.value,
            "aliases_en_ar": rec.aliases_en_ar,
            "common_dosages": rec.common_dosages,
            "nephrotoxic_risk": rec.nephrotoxic_risk,
            "gi_bleed_risk": rec.gi_bleed_risk,
            "clinical_notes": rec.clinical_notes,
        }
        for rec in EGYPTIAN_DRUG_KNOWLEDGE_BASE
    ]


@app.post(
    "/api/v1/triage/analyze",
    response_model=TriageOrchestratorResult,
    tags=["Medical Triage Engine"],
)
async def analyze_triage_request(payload: TriageAnalyzeRequest) -> TriageOrchestratorResult:
    """
    Primary REST API endpoint for analyzing patient text history or answering follow-up
    clarification questions (e.g. medication duration) in a stateful session.
    """
    if not payload.text_input or not payload.text_input.strip():
        raise HTTPException(status_code=400, detail="text_input is required.")

    result = await triage_orchestrator.handle_text_message(
        session_id=payload.session_id,
        text=payload.text_input,
        channel="web_api",
        reset=payload.reset_session,
    )
    return result


@app.post(
    "/api/v1/triage/upload",
    response_model=TriageOrchestratorResult,
    tags=["Medical Triage Engine"],
)
async def analyze_multimodal_upload(
    session_id: str = Form(default="web-default-session"),
    caption: Optional[str] = Form(default=None),
    reset_session: bool = Form(default=False),
    file: UploadFile = File(...),
) -> TriageOrchestratorResult:
    """
    Multipart upload endpoint supporting both Egyptian Medication Strip Photos (`.jpg`/`.png`)
    and Egyptian Arabic Voice Notes (`.ogg`/`.mp3`/`.wav`).
    """
    if reset_session:
        await reset_patient_session(session_id)

    content_bytes = await file.read()
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()

    if "audio" in content_type or filename.endswith((".ogg", ".mp3", ".wav", ".m4a")):
        return await triage_orchestrator.handle_voice_message(
            session_id=session_id,
            audio_bytes=content_bytes,
            mime_type=content_type or "audio/ogg",
            channel="web_api",
        )

    return await triage_orchestrator.handle_image_message(
        session_id=session_id,
        image_bytes=content_bytes,
        mime_type=content_type or "image/jpeg",
        caption=caption,
        channel="web_api",
    )


@app.get("/api/v1/triage/session/{session_id}", tags=["Medical Triage Engine"])
async def get_session_state(session_id: str) -> Dict[str, Any]:
    """Inspects the current accumulated clinical intake state for a given session_id."""
    intake = await get_patient_intake_session(session_id)
    return {
        "session_id": session_id,
        "intake": intake.model_dump(),
    }


@app.get("/api/v1/triage/report/{report_filename}", tags=["Emergency Reports"])
async def download_emergency_pdf_report(report_filename: str) -> FileResponse:
    """Downloads a generated monochrome 1-page Emergency Triage PDF report."""
    safe_name = os.path.basename(report_filename)
    file_path = settings.REPORTS_OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Triage PDF report not found.")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=safe_name,
    )


@app.post("/webhook/telegram", tags=["Telegram Webhook"])
async def telegram_webhook_endpoint(update_dict: Dict[str, Any]) -> Dict[str, str]:
    """Webhook receiver for aiogram 3.x Telegram updates when running in webhook mode."""
    if telegram_bot is None:
        raise HTTPException(
            status_code=503,
            detail="TELEGRAM_BOT_TOKEN is not configured in environment.",
        )
    update = Update.model_validate(update_dict, context={"bot": telegram_bot})
    await telegram_dp.feed_update(bot=telegram_bot, update=update)
    return {"status": "ok"}
