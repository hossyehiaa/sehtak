"""
Module 5A: Telegram Bot Stateful Conversation Handler (`app/bot/telegram_bot.py`)
Built with `aiogram 3.x` supporting:
1. `/start` & `/reset` -> Warm Egyptian Arabic welcome explaining "Caregiver Mode".
2. Photo of a medicine strip -> Identifies the Egyptian drug, adds it to active session, and asks duration if missing.
3. Voice Note (`.ogg`) or Text -> Extracts clinical timeline, runs Deterministic Rules + AI Orchestrator.
4. Sends Arabic Actionable Alert Message AND uploads `Emergency_Triage_Report.pdf` directly into the chat.
"""

import io
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, FSInputFile, Message

from app.config import settings
from app.database.db import reset_patient_session
from app.medical_engine.triage_orchestrator import triage_orchestrator
from app.schemas.clinical_schemas import TriageOrchestratorResult

logger = logging.getLogger(__name__)

telegram_router = Router(name="sehtak_telegram_router")

WELCOME_MESSAGE_AR = (
    "👋 **أهلاً بيك في «صحتك AI» (Sehtak AI) — حارس السلامة الدوائية والفرز الطبي الذكي**\n\n"
    "احنا هنا عشان نساعدك تطمن على صحتك وصحة والدك أو والدتك (**Caregiver Mode**)، "
    "ونمنع المضاعفات الصامتة للأدوية والمسكنات قبل ما تسبب أي خطر على الكلى أو المعدة أو القلب.\n\n"
    "📌 **إزاي تستخدم البوت في خطوتين بس؟**\n"
    "1️⃣ **صوّر علبة أو شريط الدواء:** ابعت صورة واضحة لشنطة الأدوية أو شريط المسكن اللي المريض بياخده.\n"
    "2️⃣ **ابعت ريكورد صوتي (Voice Note) أو رسالة نصية:** احكيلنا بالمصري في دقيقة واحدة المريض عنده كام سنة، بياخد الدواء ده بقاله قد إيه، وإيه الأعراض اللي حاسس بيها (زي همدان، وجع في الدراع أو الجسم، غممان نفس، إلخ).\n\n"
    "🔄 لبدء حالة جديدة في أي وقت ابعت: `/reset`"
)


def _session_id_from_message(message: Message) -> str:
    user_id = message.from_user.id if message.from_user else message.chat.id
    return f"tg-{user_id}"


async def _dispatch_orchestrator_result(message: Message, result: TriageOrchestratorResult) -> None:
    """Sends the Egyptian Arabic response and attaches the monochrome PDF report when completed."""
    await message.answer(result.caregiver_response_ar, parse_mode="Markdown")

    if result.status == "completed" and result.pdf_report_path:
        try:
            pdf_file = FSInputFile(
                path=result.pdf_report_path,
                filename="Emergency_Triage_Report.pdf",
            )
            await message.answer_document(
                document=pdf_file,
                caption=(
                    "📄 **تقرير الفرز الطبي للطوارئ (Emergency_Triage_Report.pdf)**\n"
                    "اعرض التقرير ده فوراً على دكتور الطوارئ أو الباطنة (ملخص بالإنجليزية للطبيب وبالعربية للأسرة)."
                ),
                parse_mode="Markdown",
            )
        except Exception as exc:
            logger.error("Failed to upload PDF report to Telegram chat: %s", exc)


@telegram_router.message(CommandStart())
async def cmd_start_handler(message: Message) -> None:
    """Handles `/start` command: resets session and sends Caregiver Mode instructions in Egyptian Arabic."""
    session_id = _session_id_from_message(message)
    await reset_patient_session(session_id)
    await message.answer(WELCOME_MESSAGE_AR, parse_mode="Markdown")


@telegram_router.message(Command("reset"))
async def cmd_reset_handler(message: Message) -> None:
    """Handles `/reset` command to clear active patient session."""
    session_id = _session_id_from_message(message)
    await reset_patient_session(session_id)
    await message.answer(
        "🔄 تم مسح بيانات الجلسة السابقة بنجاح. تقدر دلوقتي تبعت صورة الدواء أو ريكورد صوتي لحالة جديدة.",
        parse_mode="Markdown",
    )


@telegram_router.message(F.photo)
async def photo_medication_handler(message: Message, bot: Bot) -> None:
    """Handles photos of medication boxes or blister strips."""
    session_id = _session_id_from_message(message)
    await message.answer("🔍 جاري فحص صورة الدواء والتعرف على المادة الفعالة...")

    largest_photo = message.photo[-1]
    file_info = await bot.get_file(largest_photo.file_id)
    buffer = io.BytesIO()
    await bot.download_file(file_info.file_path, destination=buffer)
    image_bytes = buffer.getvalue()

    result = await triage_orchestrator.handle_image_message(
        session_id=session_id,
        image_bytes=image_bytes,
        mime_type="image/jpeg",
        caption=message.caption,
        channel="telegram",
    )
    await _dispatch_orchestrator_result(message, result)


@telegram_router.message(F.voice | F.audio)
async def voice_note_handler(message: Message, bot: Bot) -> None:
    """Handles Egyptian Arabic Voice Notes (.ogg) or audio clips."""
    session_id = _session_id_from_message(message)
    await message.answer("🎙️ جاري الاستماع للتسجيل الصوتي وتحليل التاريخ الدوائي والأعراض...")

    media = message.voice or message.audio
    file_info = await bot.get_file(media.file_id)
    buffer = io.BytesIO()
    await bot.download_file(file_info.file_path, destination=buffer)
    audio_bytes = buffer.getvalue()
    mime_type = getattr(media, "mime_type", None) or "audio/ogg"

    result = await triage_orchestrator.handle_voice_message(
        session_id=session_id,
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        channel="telegram",
    )
    await _dispatch_orchestrator_result(message, result)


@telegram_router.message(F.text)
async def text_message_handler(message: Message) -> None:
    """Handles text messages (patient story or follow-up duration answers)."""
    session_id = _session_id_from_message(message)
    user_text = (message.text or "").strip()
    if not user_text:
        return

    result = await triage_orchestrator.handle_text_message(
        session_id=session_id,
        text=user_text,
        channel="telegram",
    )
    await _dispatch_orchestrator_result(message, result)


def create_telegram_bot_and_dispatcher() -> tuple[Optional[Bot], Dispatcher]:
    """Creates the aiogram 3.x Dispatcher and (if token configured) Bot instance."""
    dp = Dispatcher()
    dp.include_router(telegram_router)

    token = settings.TELEGRAM_BOT_TOKEN
    if token and token != "your-telegram-bot-token-from-botfather" and ":" in token:
        bot = Bot(token=token)
        return bot, dp
    return None, dp
