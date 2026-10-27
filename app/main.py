"""
Module 5C: FastAPI Entry Point & Lifespan Manager (`app/main.py`)
Exposes:
- `GET /health` -> Basic health check
- `GET /api/v1/system/health` -> Task 0.5: Detailed system health {gemini_key_present, gemini_ping_ok, db_ok, rules_version, rag_index_size}
- `GET /api/v1/drugs` -> Seeded Egyptian drug catalog
- `POST /api/v1/triage/analyze` -> Triage analysis with Auth / Demo mode enforcement (Task 1.3)
- `POST /api/v1/triage/upload` -> Multipart voice / image intake with degradation handling
- `POST /webhook/telegram` -> Webhook for Telegram updates
- Mounts `/api/v1/auth` and `/api/v1/sessions`
- Mounts `/webhook/whatsapp`
"""

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from aiogram.types import Update
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.ai.errors import IntakeDegradationError
from app.ai.rag.retriever import retriever
from app.auth.routes import auth_router, session_router
from app.auth.security import get_current_user_optional
from app.bot.telegram_bot import create_telegram_bot_and_dispatcher
from app.bot.whatsapp_webhook import whatsapp_router
from app.config import settings
from app.database.db import AsyncSessionLocal, UserModel, ensure_db_ready, get_patient_intake_session, init_db, reset_patient_session
from app.database.egyptian_drugs_seed import EGYPTIAN_DRUG_KNOWLEDGE_BASE
from app.medical_engine.triage_orchestrator import triage_orchestrator
from app.schemas.clinical_schemas import ComponentDegradation, TriageAnalyzeRequest, TriageOrchestratorResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sehtak_ai")

telegram_bot, telegram_dp = create_telegram_bot_and_dispatcher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Sehtak AI Database & Egyptian Drug Knowledge Base...")
    await ensure_db_ready()

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
    version="2.0.0",
    description="Sehtak AI v2 — Proactive Medical Safety & Triage Guardian with RAG Grounding & Auth.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(auth_router)
app.include_router(session_router)
app.include_router(whatsapp_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_web_ui() -> HTMLResponse:
    await ensure_db_ready()
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>Sehtak AI Backend Running</h1>", status_code=200)


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "seeded_egyptian_drugs_count": len(EGYPTIAN_DRUG_KNOWLEDGE_BASE),
        "telegram_bot_configured": telegram_bot is not None,
    }


# Task 0.5: Detailed System Health Endpoint
@app.get("/api/v1/system/health", tags=["System"])
async def detailed_system_health() -> Dict[str, Any]:
    gemini_key_present = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-google-gemini-api-key-here")
    gemini_ping_ok = False
    if gemini_key_present:
        try:
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            # Lightweight ping
            res = await client.aio.models.generate_content(
                model=settings.GEMINI_MODEL_NAME,
                contents="ping",
            )
            gemini_ping_ok = bool(res and res.text)
        except Exception as e:
            logger.warning("Gemini ping failed: %s", e)
            gemini_ping_ok = False

    db_ok = False
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        logger.error("DB check failed: %s", e)
        db_ok = False

    rag_size = retriever.count_indexed_chunks()

    return {
        "gemini_key_present": gemini_key_present,
        "gemini_ping_ok": gemini_ping_ok,
        "db_ok": db_ok,
        "rules_version": "2.0.0-deterministic",
        "rag_index_size": rag_size,
    }


@app.get("/api/v1/drugs", tags=["Egyptian Drug Knowledge Base"])
async def list_seeded_egyptian_drugs() -> List[Dict[str, Any]]:
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
async def analyze_triage_request(
    payload: TriageAnalyzeRequest,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
) -> TriageOrchestratorResult:
    # Task 1.3: Auth enforcement
    if not settings.ALLOW_ANONYMOUS_DEMO and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for triage evaluation. Set ALLOW_ANONYMOUS_DEMO=true for public demo.",
        )

    if not payload.text_input or not payload.text_input.strip():
        raise HTTPException(status_code=400, detail="text_input is required.")

    user_id = current_user.id if current_user else None
    result = await triage_orchestrator.handle_text_message(
        session_id=payload.session_id,
        text=payload.text_input,
        channel="web_api",
        reset=payload.reset_session,
        user_id=user_id,
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
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    if not settings.ALLOW_ANONYMOUS_DEMO and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Set ALLOW_ANONYMOUS_DEMO=true for public demo.",
        )

    if reset_session:
        await reset_patient_session(session_id)

    content_bytes = await file.read()
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    user_id = current_user.id if current_user else None

    # Handle voice upload
    if "audio" in content_type or filename.endswith((".ogg", ".mp3", ".wav", ".m4a")):
        try:
            return await triage_orchestrator.handle_voice_message(
                session_id=session_id,
                audio_bytes=content_bytes,
                mime_type=content_type or "audio/ogg",
                channel="web_api",
                user_id=user_id,
            )
        except IntakeDegradationError as err:
            logger.warning("Voice intake degraded: %s", err)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "degraded",
                    "degradation": {
                        "component": err.component,
                        "reason": err.reason,
                        "user_message_ar": err.user_message_ar,
                    },
                    "message": err.user_message_ar,
                },
            )

    # Handle image upload
    try:
        return await triage_orchestrator.handle_image_message(
            session_id=session_id,
            image_bytes=content_bytes,
            mime_type=content_type or "image/jpeg",
            caption=caption,
            channel="web_api",
            user_id=user_id,
        )
    except IntakeDegradationError as err:
        logger.warning("Image intake degraded: %s", err)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "degradation": {
                    "component": err.component,
                    "reason": err.reason,
                    "user_message_ar": err.user_message_ar,
                },
                "message": err.user_message_ar,
            },
        )


@app.get("/api/v1/triage/session/{session_id}", tags=["Medical Triage Engine"])
async def get_session_state(session_id: str) -> Dict[str, Any]:
    intake = await get_patient_intake_session(session_id)
    return {
        "session_id": session_id,
        "intake": intake.model_dump(),
    }


@app.get("/api/v1/triage/report/{report_filename}", tags=["Emergency Reports"])
async def download_emergency_pdf_report(report_filename: str) -> FileResponse:
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
    if telegram_bot is None:
        raise HTTPException(
            status_code=503,
            detail="TELEGRAM_BOT_TOKEN is not configured in environment.",
        )
    update = Update.model_validate(update_dict, context={"bot": telegram_bot})
    await telegram_dp.feed_update(bot=telegram_bot, update=update)
    return {"status": "ok"}
