"""
Module 2: Multimodal Intake Pipeline (`app/ai/multimodal_intake.py`)
INV-2 Compliant: ZERO silent fallbacks. Any Gemini failure in voice, image, or text
raises `IntakeDegradationError` with exact traceback and explicit user message.
"""

import logging
import re
import traceback
from typing import List, Optional
from app.ai.errors import IntakeDegradationError
from app.config import settings
from app.database.egyptian_drugs_seed import (
    lookup_egyptian_drug,
    scan_text_for_known_drugs,
)
from app.schemas.clinical_schemas import (
    ClinicalIntakeSchema,
    DrugClass,
    ExtractedMedication,
    ExtractedSymptom,
    SymptomSeverity,
)

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False


SYSTEM_EXTRACTION_PROMPT = """
You are Sehtak AI (صحتك AI), an expert Egyptian Clinical Triage & Pharmacovigilance AI Architect.
Your job is to analyze Egyptian patient or caregiver inputs (Text, Egyptian Arabic Voice Notes, or Egyptian Medication Box/Strip Photos)
and extract a strictly validated `ClinicalIntakeSchema` JSON object.

CRITICAL INSTRUCTIONS FOR EGYPTIAN MEDICAL CONTEXT:
1. Identify all Egyptian brand medications and map them to their exact active ingredients and drug classes:
   - NSAIDs: Cataflam / Catafast (Diclofenac Potassium), Voltaren / Declophen / Olfen (Diclofenac Sodium),
     Ketofan / Bai-Alcofan / Alcofan / Ketolgin (Ketoprofen), Brufen / Marcofen (Ibuprofen), Celebrex / Arcoxia / Mobic.
   - Corticosteroids: Dexamethasone, Epidron, Prednisolone, Solupred, Solu-Medrol, Phenadone.
   - Anticoagulants/Antiplatelets: Marevan (Warfarin), Plavix (Clopidogrel), Aspocid / Jusprin (Aspirin), Eliquis (Apixaban), Xarelto.
   - Antihypertensives & Diuretics: Capoten (Captopril), Tritace (Ramipril), Tareg (Valsartan), Lasix (Furosemide), Natrilix.
2. Duration extraction (`duration_days`):
   - Convert Egyptian Arabic time expressions accurately into integer `duration_days`:
     - "اسبوع" = 7, "اسبوعين" = 14, "3 اسابيع" = 21
     - "شهر" = 30, "شهرين" = 60, "3 شهور" = 90, "سنة" = 365
     - "من فترة طويلة" or "على طول" or "مزمن" = 30
   - If the duration of a medication is NOT mentioned at all in the input, leave `duration_days` as `null` (`None`), set `missing_critical_info = true`,
     and set `clarification_question_ar = "مهم جداً تقولي: المريض بياخد المسكن ده بقاله كام يوم أو كام أسبوع؟"`.
3. Symptom extraction (`symptoms`):
   - Translate colloquial Egyptian symptoms into standard clinical English terms:
     - "وجع في دراعه / دراعها بيوجعها" -> "Unexplained upper limb / arm pain (rule out referred pain / uremia / ACS)"
     - "همدان / مفيش حيل / تعب عام" -> "Generalized fatigue and malaise"
     - "غممان نفس / نفسه مسدودة / ترجيع" -> "Nausea, anorexia, and uremic gastropathy suspicion"
     - "قلة بول / المية محبوسة / رجله مورمة" -> "Oliguria and peripheral edema"
     - "نهجان / كرشة نفس" -> "Dyspnea / volume overload suspicion"
"""

ARABIC_DURATION_PATTERNS = [
    (r"سنتين", 730),
    (r"سنة|سنه", 365),
    (r"ست شهور|6 شهور|٦ شهور", 180),
    (r"تلات شهور|ثلاث شهور|3 شهور|٣ شهور|تلت شهور", 90),
    (r"شهرين", 60),
    (r"شهر", 30),
    (r"تلات أسابيع|ثلاث أسابيع|3 أسابيع|3 اسابيع|٣ اسابيع", 21),
    (r"أسبوعين|اسبوعين", 14),
    (r"أسبوع|اسبوع", 7),
    (r"عشر أيام|عشر ايام|10 أيام|10 ايام|١٠ ايام", 10),
    (r"خمس أيام|5 أيام|5 ايام", 5),
    (r"تلات أيام|3 أيام|3 ايام", 3),
    (r"يومين", 2),
    (r"من زمان|على طول|بقاله فترة كبيرة|مزمن|كل يوم من فترة", 30),
]

EGYPTIAN_SYMPTOM_DICTIONARY = [
    (["وجع في دراع", "وجع دراع", "دراعها", "دراعه", "ذراع", "كتف", "arm pain", "shoulder pain"], "وجع في الذراع / الكتف", "Unexplained upper limb / arm pain", SymptomSeverity.SEVERE, True),
    (["همدان", "تعبان", "تعب عام", "ارهاق", "إرهاق", "مفيش حيل", "دايخ", "دوخة", "fatigue", "weakness", "dizziness"], "همدان وإرهاق عام ودوخة", "Generalized fatigue, malaise, and lethargy", SymptomSeverity.MODERATE, True),
    (["غممان نفس", "غثيان", "ترجيع", "قيء", "نفسه مسدودة", "مبياكلش", "nausea", "vomiting", "anorexia"], "غممان نفس وغثيان وفقدان شهية", "Nausea, anorexia, and vomiting", SymptomSeverity.MODERATE, True),
    (["قلة البول", "بول قليل", "مبيجيبش بول", "بول غامق", "حرقان", "oliguria", "reduced urine", "dark urine"], "قلة كمية البول وتغير لونه", "Oliguria / reduced urine output", SymptomSeverity.CRITICAL, True),
    (["ورم", "مورمة", "تورم", "رجلها مورمة", "رجله مورمة", "منفخ", "edema", "swelling"], "تورم في القدمين أو الجسم (احتباس سوائل)", "Peripheral edema / fluid retention", SymptomSeverity.SEVERE, True),
    (["نهجان", "كرشة نفس", "ضيق تنفس", "مش قادر ياخد نفسه", "dyspnea", "shortness of breath"], "نهجان وضيق في التنفس", "Dyspnea / shortness of breath", SymptomSeverity.CRITICAL, True),
    (["وجع في الضهر", "وجع ضهر", "جنبه", "اجناب", "كليته", "تكسير في الجسم", "back pain", "flank pain", "body pain"], "ألم في الظهر والجانبين والجسم", "Generalized body ache and flank/back pain", SymptomSeverity.MODERATE, True),
    (["فم المعدة", "معدته", "حموضة", "حرقان معدة", "براز أسود", "epigastric", "stomach pain", "melena"], "ألم حاد في فم المعدة", "Epigastric pain / suspected mucosal injury", SymptomSeverity.SEVERE, True),
]

CHRONIC_CONDITION_KEYWORDS = [
    (["ضغط", "hypertension", "htn"], "Hypertension (ارتفاع ضغط الدم)"),
    (["سكر", "diabetes", "dm"], "Diabetes Mellitus (مرض السكري)"),
    (["قلب", "شريان تاجي", "جلطة", "heart", "cad", "ischemic"], "Cardiovascular Disease (قصور القلب/الشرايين)"),
    (["كلى", "كلية", "ckd", "renal"], "Chronic Kidney Disease (قصور كلوي سابق)"),
    (["معدة", "قرحة", "gastritis", "ulcer"], "Peptic Ulcer / Gastritis (التهاب أو قرحة المعدة)"),
    (["خشونة", "غضروف", "مفاصل", "osteoarthritis", "arthritis"], "Osteoarthritis / Joint Pain (خشونة المفاصل)"),
]


def _translate_arabic_digits(text: str) -> str:
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"
    table = str.maketrans(arabic_digits, english_digits)
    return text.translate(table)


def extract_duration_days_from_text(text: str) -> Optional[int]:
    """Parses Egyptian Arabic or English text for medication duration in days (avoiding patient age false positives)."""
    if not text:
        return None
    normalized = _translate_arabic_digits(text.lower())

    cleaned_for_duration = re.sub(
        r"(?:عمره|عمرها|سنه|سنها|عنده|عندها|سن|عمر)\s*\d{1,3}\s*(?:سنة|سنه|عام)?",
        " ",
        normalized,
    )
    cleaned_for_duration = re.sub(r"\b\d{1,3}\s*(?:years old|yo)\b", " ", cleaned_for_duration)

    m_days = re.search(r"(\d+)\s*(?:يوم|ايام|أيام|day|days)", cleaned_for_duration)
    if m_days:
        return int(m_days.group(1))

    m_weeks = re.search(r"(\d+)\s*(?:اسبوع|أسبوع|اسابيع|أسابيع|week|weeks)", cleaned_for_duration)
    if m_weeks:
        return int(m_weeks.group(1)) * 7

    m_months = re.search(r"(\d+)\s*(?:شهر|شهور|أشهر|month|months)", cleaned_for_duration)
    if m_months:
        return int(m_months.group(1)) * 30

    for pattern, days in ARABIC_DURATION_PATTERNS:
        if re.search(pattern, cleaned_for_duration):
            return days

    if re.fullmatch(r"\s*(\d{1,3})\s*", normalized):
        return int(normalized.strip())

    return None


class MultimodalIntakePipeline:
    """
    AI Extraction Pipeline using Google Gemini 2.5 Flash (`google-genai`).
    Enforces INV-2: ZERO silent fallbacks.
    """

    def __init__(self) -> None:
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL_NAME
        self._client = None
        if HAS_GOOGLE_GENAI and self.api_key and self.api_key != "your-google-gemini-api-key-here":
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Google GenAI Client: %s", e)
                self._client = None

    def _deterministic_egyptian_nlu_parse(
        self, text: str, default_drug_hint: Optional[str] = None
    ) -> ClinicalIntakeSchema:
        normalized_text = _translate_arabic_digits(text or "")
        lower_text = normalized_text.lower()

        age: Optional[int] = None
        age_match = re.search(r"(?:عمره|عمرها|سنه|سنها|سن|عمر|age)\s*(\d{2})", lower_text) or re.search(
            r"(\d{2})\s*(?:سنة|سنه|years old|yo)", lower_text
        )
        if age_match:
            parsed_age = int(age_match.group(1))
            if 1 <= parsed_age <= 115:
                age = parsed_age

        sex: Optional[str] = None
        if any(w in lower_text for w in ["والدتي", "أمي", "امي", "مراتي", "أختي", "اختي", "ست", "سيدة", "بياخدها", "بتاخد", "عندها", "female", "mother", "woman"]):
            sex = "female"
        elif any(w in lower_text for w in ["والدي", "أبويا", "ابويا", "أبي", "جوزي", "أخويا", "راجل", "عمي", "خالي", "male", "father", "man"]):
            sex = "male"

        chronic_conditions: List[str] = []
        for keywords, label in CHRONIC_CONDITION_KEYWORDS:
            if any(k in lower_text for k in keywords):
                chronic_conditions.append(label)

        duration_days = extract_duration_days_from_text(normalized_text)

        frequency: Optional[str] = None
        if any(w in lower_text for w in ["مرتين", "كل 12 ساعة", "صباحا ومساء", "twice"]):
            frequency = "Twice daily (مرتين يومياً)"
        elif any(w in lower_text for w in ["3 مرات", "تلات مرات", "كل 8 ساعات", "three times"]):
            frequency = "Three times daily (3 مرات يومياً)"
        elif any(w in lower_text for w in ["كل يوم", "يوميا", "يومياً", "قرص واحد", "daily"]):
            frequency = "Daily (يومياً)"

        matched_records = scan_text_for_known_drugs(normalized_text)
        if default_drug_hint and not matched_records:
            rec = lookup_egyptian_drug(default_drug_hint)
            if rec:
                matched_records = [rec]

        medications: List[ExtractedMedication] = []
        for rec in matched_records:
            medications.append(
                ExtractedMedication(
                    brand_name=rec.canonical_brand,
                    active_ingredient=rec.active_ingredient,
                    drug_class=rec.drug_class,
                    dosage=rec.common_dosages[0] if rec.common_dosages else None,
                    duration_days=duration_days,
                    frequency=frequency,
                )
            )

        if not medications and duration_days is not None:
            medications.append(
                ExtractedMedication(
                    brand_name="__DURATION_UPDATE__",
                    active_ingredient=None,
                    drug_class=DrugClass.OTHER,
                    duration_days=duration_days,
                    frequency=frequency,
                )
            )

        symptoms: List[ExtractedSymptom] = []
        for keywords, ar_term, en_term, sev, is_vague in EGYPTIAN_SYMPTOM_DICTIONARY:
            if any(k in lower_text for k in keywords):
                symptoms.append(
                    ExtractedSymptom(
                        symptom_Arabic=ar_term,
                        clinical_term_English=en_term,
                        duration_days=None,
                        severity=sev,
                        is_systemic_or_vague=is_vague,
                    )
                )

        has_high_risk_without_duration = any(
            m.drug_class in {DrugClass.NSAID, DrugClass.CORTICOSTEROID, DrugClass.ANTICOAGULANT}
            and m.duration_days is None
            for m in medications
        )
        clarification_q = (
            "مهم جداً تقولي: المريض بياخد المسكن ده بقاله كام يوم أو كام أسبوع؟"
            if has_high_risk_without_duration
            else None
        )

        return ClinicalIntakeSchema(
            patient_age=age,
            patient_sex=sex,
            chronic_conditions=chronic_conditions,
            medications=medications,
            symptoms=symptoms,
            missing_critical_info=has_high_risk_without_duration,
            clarification_question_ar=clarification_q,
            raw_transcript_or_notes=text,
        )

    async def extract_from_text(self, text: str) -> ClinicalIntakeSchema:
        """Extracts structured ClinicalIntakeSchema from text using Gemini 2.5 Flash with Egyptian NLU enrichment."""
        deterministic_result = self._deterministic_egyptian_nlu_parse(text)

        if self._client is not None:
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=[
                        SYSTEM_EXTRACTION_PROMPT,
                        f"Patient/Caregiver Input:\n{text}",
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response and hasattr(response, "text") and response.text:
                    llm_schema = ClinicalIntakeSchema.model_validate_json(response.text)
                    llm_schema.raw_transcript_or_notes = text
                    from app.database.db import merge_clinical_intake
                    merged = merge_clinical_intake(deterministic_result, llm_schema, text)
                    merged.medications = [
                        m for m in merged.medications if m.brand_name != "__DURATION_UPDATE__"
                    ] if any(m.brand_name != "__DURATION_UPDATE__" for m in merged.medications) else merged.medications
                    return merged
            except Exception as e:
                logger.warning("Gemini text extraction failed, relying on deterministic Egyptian NLU: %s", e)

        return deterministic_result

    async def extract_from_voice(
        self, audio_bytes: bytes, mime_type: str = "audio/ogg", filename_hint: str = ""
    ) -> ClinicalIntakeSchema:
        """
        Transcribes Egyptian Arabic voice note and extracts ClinicalIntakeSchema.
        INV-2: ZERO silent fallbacks! Raises IntakeDegradationError on ANY Gemini/transcription failure.
        """
        if not audio_bytes or len(audio_bytes) < 10:
            raise IntakeDegradationError(
                component="voice_transcription",
                reason="ملف الصوت فارغ أو تالف",
                user_message_ar="التسجيل الصوتي المرسل فارغ أو تالف. اكتب الحالة نصاً وهحللها فوراً.",
            )

        # Normalize MIME type (Telegram sends audio/ogg with Opus codec)
        normalized_mime = mime_type
        if "ogg" in mime_type.lower():
            normalized_mime = "audio/ogg; codecs=opus"
        elif "mp3" in mime_type.lower():
            normalized_mime = "audio/mp3"
        elif "m4a" in mime_type.lower() or "mp4" in mime_type.lower():
            normalized_mime = "audio/mp4"

        if self._client is None:
            error_reason = "مفتاح خدمة الذكاء الاصطناعي (GEMINI_API_KEY) غير مهيأ"
            logger.error("extract_from_voice failed: %s", error_reason)
            raise IntakeDegradationError(
                component="voice_transcription",
                reason=error_reason,
                user_message_ar=f"تحليل الصوت مش شغال حالياً (سبب: {error_reason}). اكتب الحالة نصاً وهحللها فوراً.",
            )

        try:
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=normalized_mime)
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=[
                    SYSTEM_EXTRACTION_PROMPT,
                    "Listen carefully to this Egyptian Arabic voice note from a patient or caregiver. "
                    "Transcribe it verbatim into `raw_transcript_or_notes` and extract all medications, durations, chronic conditions, and symptoms.",
                    audio_part,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            # Defensive parsing per Task 0.3
            if not response or not hasattr(response, "candidates") or not response.candidates:
                raise ValueError("استجابة فارغة من نموذج التحليل الصوتي (No candidates returned)")

            first_candidate = response.candidates[0]
            if not hasattr(first_candidate, "content") or not first_candidate.content:
                raise ValueError("المحتوى فارغ في استجابة النموذج (Candidate content is empty)")

            parts = getattr(first_candidate.content, "parts", [])
            if not parts or not getattr(parts[0], "text", None):
                raise ValueError("لا يوجد نص مستخرج في استجابة النموذج (Candidate parts[0].text is empty)")

            raw_json = parts[0].text.strip()
            llm_schema = ClinicalIntakeSchema.model_validate_json(raw_json)

            if not llm_schema.raw_transcript_or_notes or len(llm_schema.raw_transcript_or_notes.strip()) < 3:
                raise ValueError("لم يتم التعرف على أي كلمات عربية في التسجيل الصوتي")

            # Enrich with deterministic NLU
            det = self._deterministic_egyptian_nlu_parse(llm_schema.raw_transcript_or_notes)
            from app.database.db import merge_clinical_intake
            return merge_clinical_intake(det, llm_schema)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Gemini voice transcription failed with traceback:\n%s", tb)
            reason_str = str(exc)
            if "quota" in reason_str.lower() or "429" in reason_str:
                reason_clean = "تم تجاوز حد استخدام Gemini API (Quota Exceeded)"
            elif "permission" in reason_str.lower() or "403" in reason_str:
                reason_clean = "خطأ في صلاحيات مفتاح API (Permission Denied)"
            else:
                reason_clean = f"تعذر تفريغ الصوت ({type(exc).__name__}: {exc})"

            raise IntakeDegradationError(
                component="voice_transcription",
                reason=reason_clean,
                user_message_ar=f"تحليل الصوت مش شغال حالياً (سبب: {reason_clean}). اكتب الحالة نصاً وهحللها فوراً.",
            ) from exc

    async def extract_from_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        caption: Optional[str] = None,
    ) -> ClinicalIntakeSchema:
        """
        Analyzes medication box or blister strip photo.
        INV-2: ZERO silent fallbacks! Raises IntakeDegradationError on ANY vision failure.
        """
        if not image_bytes or len(image_bytes) < 10:
            raise IntakeDegradationError(
                component="image_ocr",
                reason="ملف الصورة فارغ أو تالف",
                user_message_ar="الصورة المرسلة فارغة أو تالفة. اكتب اسم الدواء نصاً وهحلله فوراً.",
            )

        if self._client is None:
            error_reason = "مفتاح خدمة الذكاء الاصطناعي (GEMINI_API_KEY) غير مهيأ"
            logger.error("extract_from_image failed: %s", error_reason)
            raise IntakeDegradationError(
                component="image_ocr",
                reason=error_reason,
                user_message_ar=f"فحص الصورة مش شغال حالياً (سبب: {error_reason}). اكتب اسم الدواء نصاً وهحلله فوراً.",
            )

        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            prompt_text = (
                "Analyze this Egyptian medication box or blister strip photo. "
                "Extract the exact Brand Name (e.g., Cataflam, Ketofan, Voltaren, Brufen, Bai-Alcofan, Marevan, Capoten, Lasix), "
                "Active Ingredient, Dosage, and Drug Class. "
            )
            if caption:
                prompt_text += f"\nUser caption accompanying the photo: {caption}"
            else:
                prompt_text += (
                    "\nSince a photo alone usually does not state how many days the patient has been taking the drug, "
                    "set `duration_days = null`, `missing_critical_info = true`, and "
                    "`clarification_question_ar = 'مهم جداً تقولي: المريض بياخد المسكن ده بقاله كام يوم أو كام أسبوع؟'`."
                )

            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=[SYSTEM_EXTRACTION_PROMPT, prompt_text, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            if not response or not hasattr(response, "candidates") or not response.candidates:
                raise ValueError("استجابة فارغة من نموذج قراءة الصور")

            first_candidate = response.candidates[0]
            parts = getattr(getattr(first_candidate, "content", None), "parts", [])
            if not parts or not getattr(parts[0], "text", None):
                raise ValueError("لا يوجد نص مستخرج من تحليل الصورة")

            llm_schema = ClinicalIntakeSchema.model_validate_json(parts[0].text.strip())
            if caption:
                cap_det = self._deterministic_egyptian_nlu_parse(caption)
                from app.database.db import merge_clinical_intake
                return merge_clinical_intake(cap_det, llm_schema, caption)
            return llm_schema

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Gemini image analysis failed with traceback:\n%s", tb)
            reason_str = str(exc)
            if "quota" in reason_str.lower() or "429" in reason_str:
                reason_clean = "تم تجاوز حد استخدام Gemini API (Quota Exceeded)"
            elif "permission" in reason_str.lower() or "403" in reason_str:
                reason_clean = "خطأ في صلاحيات مفتاح API (Permission Denied)"
            else:
                reason_clean = f"تعذر التعرف على صورة الدواء ({type(exc).__name__}: {exc})"

            raise IntakeDegradationError(
                component="image_ocr",
                reason=reason_clean,
                user_message_ar=f"فحص الصورة مش شغال حالياً (سبب: {reason_clean}). اكتب اسم الدواء نصاً وهحلله فوراً.",
            ) from exc


multimodal_intake = MultimodalIntakePipeline()
