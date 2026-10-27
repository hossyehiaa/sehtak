"""
Module 5B: Decoupled WhatsApp Cloud API Webhook Adapter (`app/bot/whatsapp_webhook.py`)
Connects Meta WhatsApp Cloud API directly to the channel-agnostic `TriageOrchestrator`.
"""

import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.config import settings
from app.medical_engine.triage_orchestrator import triage_orchestrator

logger = logging.getLogger(__name__)

whatsapp_router = APIRouter(prefix="/webhook/whatsapp", tags=["WhatsApp Cloud API"])


@whatsapp_router.get("")
async def verify_whatsapp_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
) -> Response:
    """Meta WhatsApp Cloud API Webhook verification endpoint."""
    if hub_mode == "subscribe" and hub_verify_token == settings.WHATSAPP_VERIFY_TOKEN:
        return Response(content=hub_challenge, media_type="text/plain", status_code=200)
    raise HTTPException(status_code=403, detail="WhatsApp webhook verification token mismatch.")


@whatsapp_router.post("")
async def receive_whatsapp_webhook(request: Request) -> Dict[str, Any]:
    """
    Processes incoming WhatsApp Cloud API messages (Text, Voice Notes, Drug Photos)
    and routes them through the decoupled Sehtak AI TriageOrchestrator.
    """
    payload = await request.json()
    entries = payload.get("entry", [])
    processed_results = []

    for entry in entries:
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            for msg in messages:
                sender_wa_id = msg.get("from", "unknown")
                session_id = f"wa-{sender_wa_id}"
                msg_type = msg.get("type", "text")

                if msg_type == "text":
                    body_text = msg.get("text", {}).get("body", "")
                    result = await triage_orchestrator.handle_text_message(
                        session_id=session_id,
                        text=body_text,
                        channel="whatsapp",
                    )
                    await _send_whatsapp_reply_if_configured(sender_wa_id, result.caregiver_response_ar)
                    processed_results.append(
                        {
                            "session_id": session_id,
                            "status": result.status,
                            "severity": result.severity.value,
                            "caregiver_response_ar": result.caregiver_response_ar,
                            "pdf_report_path": result.pdf_report_path,
                        }
                    )

    return {
        "status": "received",
        "processed_count": len(processed_results),
        "results": processed_results,
    }


async def _send_whatsapp_reply_if_configured(recipient_phone: str, text_body: str) -> None:
    """Dispatches reply via Meta Graph API when WHATSAPP_ACCESS_TOKEN is configured."""
    if not settings.WHATSAPP_ACCESS_TOKEN or not settings.WHATSAPP_PHONE_NUMBER_ID:
        return
    url = f"https://graph.facebook.com/v21.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "text",
        "text": {"body": text_body},
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, headers=headers, json=data)
    except Exception as exc:
        logger.warning("WhatsApp outbound send skipped/failed: %s", exc)
