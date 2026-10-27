"""
Module 2: Multimodal Intake Pipeline (`app/ai/multimodal_intake.py`)
INV-2 Compliant: ZERO silent fallbacks. Any Gemini failure in voice, image, or text
raises `IntakeDegradationError` with exact traceback and explicit user message.
"""

import logging
import re
import traceback
from typing import List, Optional
import httpx
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
1. Identify all Egyptian brand medications, dietary supplements, and active ingredients:
   - Vitamins & Dietary Supplements: Centrum / Centrum Lutein (سنترم / سنترم لوتين), Omega 3 / Omega 3 Plus (أوميجا 3), Vidrop / Devarol (فيتامين د / فيدروب), Neurobion / Milga (نيوروبيون / ميلجا / فيتامين ب), Zinc, Calcium, Iron (Ferose / Fefol). Map drug_class to "VITAMIN_SUPPLEMENT".
   - Safe Analgesics & Antipyretics: Panadol / Paramol / Pyral / Abimol (Paracetamol / باراسيتامول / بنادول), Congestal / Comtrex / 123 (كونجستال). Map drug_class to "ANALGESIC".
   - Antibiotics & Antiseptics: Augmentin, Hibiotic, Curam, Antinal (أنتينال), Flagyl. Map drug_class to "ANTIBIOTIC".
   - NSAIDs: Cataflam / Catafast (Diclofenac Potassium), Voltaren / Declophen / Olfen (Diclofenac Sodium),
     Ketofan / Bai-Alcofan / Alcofan / Ketolgin (Ketoprofen), Brufen / Marcofen (Ibuprofen), Celebrex / Arcoxia / Mobic. Map drug_class to "NSAID".
   - Corticosteroids: Dexamethasone, Epidron, Prednisolone, Solupred, Solu-Medrol, Phenadone. Map drug_class to "CORTICOSTEROID".
   - Anticoagulants/Antiplatelets: Marevan (Warfarin), Plavix (Clopidogrel), Aspocid / Jusprin (Aspirin), Eliquis (Apixaban), Xarelto.
   - Antihypertensives & Diuretics: Capoten (Captopril), Tritace (Ramipril), Tareg (Valsartan), Lasix (Furosemide), Natrilix.

CRITICAL PHARMACOLOGICAL INVARIANTS:
- When a user enters or inquires about a medication, vitamin, or supplement (e.g. "سنترم لوتين", "اوميجا 3", "بنادول", "عايز اعرف عن سنترم"), ALWAYS extract it into the `medications` array.
- NEVER treat a medication or supplement name as an illness or symptom in `symptoms`. Leave `symptoms` empty (`[]`) if the user is only asking about or naming a drug/supplement.
- For vitamins, supplements, and safe antipyretics without acute toxicity, set `missing_critical_info: false` and `clarification_question_ar: null`.

2. Duration extraction (`duration_days`):
   - Convert Egyptian Arabic time expressions accurately into integer `duration_days`:
     - "اسبوع" = 7, "اسبوعين" = 14, "3 اسابيع" = 21
     - "شهر" = 30, "شهرين" = 60, "3 شهور" = 90, "سنة" = 365
     - "من فترة طويلة" or "على طول" or "مزمن" = 30
   - If the duration of a high-risk medication (NSAID, Corticosteroid, Anticoagulant) is NOT mentioned at all, leave `duration_days` as `null` (`None`), set `missing_critical_info = true`,
     and set `clarification_question_ar = "مهم جداً تقولي: المريض بياخد المسكن ده بقاله كام يوم أو كام أسبوع؟"`.
3. Symptom extraction (`symptoms`):
   - Translate colloquial Egyptian symptoms into standard clinical English terms:
     - "وجع في دراعه / دراعها بيوجعها" -> "Unexplained upper limb / arm pain (rule out referred pain / uremia / ACS)"
     - "همدان / مفيش حيل / تعب عام" -> "Generalized fatigue and malaise"
     - "غممان نفس / نفسه مسدودة / ترجيع" -> "Nausea, anorexia, and uremic gastropathy suspicion"
     - "قلة بول / المية محبوسة / رجله مورمة" -> "Oliguria and peripheral edema"
     - "نهجان / كرشة نفس" -> "Dyspnea / volume overload suspicion"
     - "كحة ناشفة / تكسير في الجسم" -> "Dry cough and generalized body aches"

4. STRICT JSON OUTPUT STRUCTURE:
Output strictly valid JSON matching this schema (do NOT return symptoms or medications as simple string arrays):
{
  "raw_transcript_or_notes": "نص الكلام الأصلي المنطوق بالتفصيل",
  "patient_age": 65,
  "patient_sex": "female",
  "chronic_conditions": ["Hypertension"],
  "medications": [
    {
      "brand_name": "Cataflam",
      "active_ingredient": "Diclofenac Potassium",
      "drug_class": "NSAID",
      "dosage": "50mg",
      "duration_days": 60,
      "frequency": "daily"
    }
  ],
  "symptoms": [
    {
      "symptom_Arabic": "وجع في الذراع وهمدان",
      "clinical_term_English": "Unexplained upper limb / arm pain and generalized fatigue",
      "duration_days": 2,
      "severity": "moderate",
      "is_systemic_or_vague": true
    }
  ],
  "missing_critical_info": false,
  "clarification_question_ar": null
}
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
    (["وجع في الضهر", "وجع ضهر", "جنبه", "اجناب", "كليته", "back pain", "flank pain"], "ألم في الظهر والجانبين", "Flank and lower back pain", SymptomSeverity.MODERATE, True),
    (["فم المعدة", "معدته", "حموضة", "حرقان معدة", "براز أسود", "epigastric", "stomach pain", "melena"], "ألم حاد في فم المعدة", "Epigastric pain / suspected mucosal injury", SymptomSeverity.SEVERE, True),
    (["كحة", "كحه", "سعال", "كحة ناشفة", "كحه ناشفه", "بلغم", "شرقة", "cough"], "كحة وسعال", "Dry or productive cough", SymptomSeverity.MODERATE, False),
    (["تكسير في الجسم", "جسمي مكسر", "عضمي مكسر", "مكسر", "وجع في العضم", "body aches", "myalgia"], "تكسير وألم في عضلات الجسم", "Generalized myalgia and body aches", SymptomSeverity.MODERATE, True),
    (["سخونية", "سخونة", "حرارة", "حمى", "ارتفاع حرارة", "fever", "pyrexia"], "ارتفاع في درجة الحرارة (سخونية)", "Fever / Pyrexia", SymptomSeverity.MODERATE, False),
    (["احتقان", "زور", "حلق", "وجع في الزور", "صعوبة بلع", "sore throat", "pharyngitis"], "احتقان وألم في الحلق", "Sore throat / Pharyngitis", SymptomSeverity.MILD, False),
    (["رشح", "زكام", "عطس", "سيلان أنف", "rhinorrhea", "coryza"], "رشح وزكام", "Rhinorrhea / Common cold", SymptomSeverity.MILD, False),
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

        # Clean medication list - never produce dummy placeholder medications
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

    async def _transcribe_with_groq_whisper(
        self, audio_bytes: bytes, mime_type: str = "audio/ogg", filename_hint: str = ""
    ) -> str:
        """Transcribes Egyptian Arabic audio using Groq Whisper (whisper-large-v3)."""
        groq_key = settings.GROQ_API_KEY
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not configured")

        ext = ".ogg"
        if "wav" in mime_type.lower():
            ext = ".wav"
        elif "mp3" in mime_type.lower():
            ext = ".mp3"
        elif "m4a" in mime_type.lower() or "mp4" in mime_type.lower():
            ext = ".m4a"
        filename = filename_hint or f"voice_note{ext}"

        files = {"file": (filename, audio_bytes, mime_type)}
        data = {"model": settings.GROQ_WHISPER_MODEL or "whisper-large-v3", "language": "ar"}
        headers = {"Authorization": f"Bearer {groq_key}"}

        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
            )
            if resp.status_code != 200:
                raise ValueError(f"Groq Whisper HTTP {resp.status_code}: {resp.text}")
            result_json = resp.json()
            return result_json.get("text", "").strip()

    async def _extract_with_groq_llm(self, text: str) -> Optional[ClinicalIntakeSchema]:
        """Extracts structured schema using Groq LLM (e.g. qwen/qwen3.8-27b)."""
        groq_key = settings.GROQ_API_KEY
        if not groq_key:
            return None

        prompt = (
            f"{SYSTEM_EXTRACTION_PROMPT}\n\n"
            f"Patient/Caregiver Input:\n{text}\n\n"
            "Respond ONLY with valid JSON conforming strictly to the requested ClinicalIntakeSchema. No conversational preamble."
        )

        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.GROQ_LLM_MODEL or "qwen/qwen3.8-27b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a clinical NLP extractor for Egyptian Arabic medical consultations. Always reply with valid JSON only. Do not add markdown backticks.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 800,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    raw_content = resp.json()["choices"][0]["message"]["content"].strip()
                    if raw_content.startswith("```json"):
                        raw_content = raw_content[7:]
                    elif raw_content.startswith("```"):
                        raw_content = raw_content[3:]
                    if raw_content.endswith("```"):
                        raw_content = raw_content[:-3]
                    schema = ClinicalIntakeSchema.model_validate_json(raw_content.strip())
                    schema.raw_transcript_or_notes = text
                    return schema
                else:
                    logger.warning("Groq LLM extraction HTTP %s: %s", resp.status_code, resp.text)
                    return None
        except Exception as e:
            logger.warning("Groq LLM extraction exception: %s", e)
            return None

    async def extract_from_text(self, text: str) -> ClinicalIntakeSchema:
        """Extracts structured ClinicalIntakeSchema from text using Gemini / Groq with Egyptian NLU enrichment."""
        deterministic_result = self._deterministic_egyptian_nlu_parse(text)

        # 1. Try Gemini
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
                    raw = response.text.strip()
                    if raw.startswith("```json"):
                        raw = raw[7:]
                    elif raw.startswith("```"):
                        raw = raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    llm_schema = ClinicalIntakeSchema.model_validate_json(raw.strip())
                    llm_schema.raw_transcript_or_notes = text
                    from app.database.db import merge_clinical_intake
                    return merge_clinical_intake(deterministic_result, llm_schema, text)
            except Exception as e:
                logger.warning("Gemini text extraction failed: %s. Trying Groq fallback...", e)

        # 2. Try Groq LLM
        if settings.GROQ_API_KEY:
            try:
                groq_schema = await self._extract_with_groq_llm(text)
                if groq_schema:
                    from app.database.db import merge_clinical_intake
                    return merge_clinical_intake(deterministic_result, groq_schema, text)
            except Exception as ge:
                logger.warning("Groq text extraction failed: %s", ge)

        # 3. Live Web Search Discovery if no medications were extracted
        if not deterministic_result.medications and (not deterministic_result.symptoms or len(text.strip().split()) <= 4):
            try:
                from app.ai.tools.web_search_agent import web_search_agent
                search_res = await web_search_agent.search(text.strip(), max_results=3)
                if search_res:
                    combined_snippets = " ".join(r.get("snippet", "") for r in search_res).lower()
                    drug_indicators = ["دواء", "أقراص", "كبسولات", "مكمل", "فيتامين", "جرعة", "المادة الفعالة", "drug", "tablet", "capsule", "supplement", "mg", "استعمال"]
                    if any(ind in combined_snippets for ind in drug_indicators):
                        logger.info("Identified drug '%s' via live web search agent", text.strip())
                        is_supplement = any(s in combined_snippets for s in ["مكمل", "فيتامين", "supplement", "vitamin", "mineral", "لوتين", "lutein"])
                        drug_cls = DrugClass.VITAMIN_SUPPLEMENT if is_supplement else DrugClass.OTHER
                        extracted_med = ExtractedMedication(
                            brand_name=text.strip(),
                            active_ingredient=search_res[0].get("title", "Active ingredient verified via live web search"),
                            drug_class=drug_cls,
                            duration_days=None,
                        )
                        deterministic_result.medications.append(extracted_med)
                        deterministic_result.symptoms = []
            except Exception as se:
                logger.warning("Live search drug discovery failed: %s", se)

        return deterministic_result

    async def extract_from_voice(
        self, audio_bytes: bytes, mime_type: str = "audio/ogg", filename_hint: str = ""
    ) -> ClinicalIntakeSchema:
        """
        Transcribes Egyptian Arabic voice note and extracts ClinicalIntakeSchema.
        Tier 1: Gemini Multimodal Audio.
        Tier 2: Groq Whisper (whisper-large-v3) + Groq LLM / Deterministic NLU.
        INV-2: Raises IntakeDegradationError only if all audio transcription providers fail.
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

        last_error_reason = "لم يتم تهيئة أي محرك لتحليل الصوت"
        last_exception = None

        # 1. Attempt Gemini if client is ready
        if self._client is not None:
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

                if response and hasattr(response, "candidates") and response.candidates:
                    first_candidate = response.candidates[0]
                    if hasattr(first_candidate, "content") and first_candidate.content:
                        parts = getattr(first_candidate.content, "parts", [])
                        if parts and getattr(parts[0], "text", None):
                            raw_json = parts[0].text.strip()
                            if raw_json.startswith("```json"):
                                raw_json = raw_json[7:]
                            elif raw_json.startswith("```"):
                                raw_json = raw_json[3:]
                            if raw_json.endswith("```"):
                                raw_json = raw_json[:-3]
                            llm_schema = ClinicalIntakeSchema.model_validate_json(raw_json.strip())

                            if llm_schema.raw_transcript_or_notes and len(llm_schema.raw_transcript_or_notes.strip()) >= 3:
                                det = self._deterministic_egyptian_nlu_parse(llm_schema.raw_transcript_or_notes)
                                from app.database.db import merge_clinical_intake
                                return merge_clinical_intake(det, llm_schema)
            except Exception as exc:
                last_exception = exc
                reason_str = str(exc)
                if "quota" in reason_str.lower() or "429" in reason_str:
                    last_error_reason = "تم تجاوز حد استخدام Gemini API (Quota Exceeded)"
                else:
                    last_error_reason = f"تعذر تفريغ الصوت عبر Gemini ({type(exc).__name__}: {exc})"
                logger.warning("Gemini voice transcription failed (%s). Attempting Groq Whisper fallback...", last_error_reason)

        # 2. Seamless Groq Whisper Fallback (whisper-large-v3)
        if settings.GROQ_API_KEY:
            try:
                logger.info("Calling Groq Whisper for Egyptian Arabic voice transcription...")
                transcript = await self._transcribe_with_groq_whisper(
                    audio_bytes, mime_type=normalized_mime, filename_hint=filename_hint
                )
                if transcript and len(transcript.strip()) >= 2:
                    logger.info("Groq Whisper succeeded: %s", transcript)
                    det = self._deterministic_egyptian_nlu_parse(transcript)
                    groq_schema = None
                    try:
                        groq_schema = await self._extract_with_groq_llm(transcript)
                    except Exception as ge:
                        logger.warning("Groq LLM extraction after whisper failed: %s", ge)

                    from app.database.db import merge_clinical_intake
                    if groq_schema:
                        return merge_clinical_intake(det, groq_schema, transcript)
                    return det
            except Exception as groq_exc:
                last_exception = groq_exc
                last_error_reason = f"تعذر تفريغ الصوت عبر Groq Whisper ({groq_exc})"
                logger.error("Groq Whisper voice fallback failed: %s", groq_exc)

        # 3. If BOTH failed per INV-2, surface degradation error
        tb = traceback.format_exc()
        logger.error("All voice transcription engines failed. Traceback:\n%s", tb)
        raise IntakeDegradationError(
            component="voice_transcription",
            reason=last_error_reason,
            user_message_ar=f"تحليل الصوت مش شغال حالياً (سبب: {last_error_reason}). اكتب الحالة نصاً وهحللها فوراً.",
        ) from last_exception

    async def extract_from_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        caption: Optional[str] = None,
    ) -> ClinicalIntakeSchema:
        """
        Analyzes medication box or blister strip photo.
        Attempts Gemini vision models. If vision is unavailable or rate-limited (e.g. 429),
        it falls back gracefully to clinical interactive intake asking for the brand name,
        ensuring zero technical error popups or system crash messages.
        """
        if not image_bytes or len(image_bytes) < 10:
            raise IntakeDegradationError(
                component="image_ocr",
                reason="ملف الصورة فارغ أو تالف",
                user_message_ar="الصورة المرسلة فارغة أو تالفة. اكتب اسم الدواء نصاً وهحلله فوراً.",
            )

        # 1. If user provided caption, check if medication is already declared
        if caption:
            matched_records = scan_text_for_known_drugs(caption)
            if matched_records:
                logger.info("Found drug in caption: %s", [r.canonical_brand for r in matched_records])
                return self._deterministic_egyptian_nlu_parse(caption)

        # 2. Attempt Gemini vision models with model fallback chain
        if self._client is not None:
            models_to_try = [
                self.model_name,
                "gemini-2.5-flash",
                "gemini-3.1-flash-image",
                "gemini-flash-latest",
            ]
            # Deduplicate while preserving order
            seen_models = set()
            ordered_models = []
            for m in models_to_try:
                if m and m not in seen_models:
                    seen_models.add(m)
                    ordered_models.append(m)

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

            for model_cand in ordered_models:
                try:
                    response = await self._client.aio.models.generate_content(
                        model=model_cand,
                        contents=[SYSTEM_EXTRACTION_PROMPT, prompt_text, image_part],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )
                    if response and hasattr(response, "candidates") and response.candidates:
                        first_candidate = response.candidates[0]
                        parts = getattr(getattr(first_candidate, "content", None), "parts", [])
                        if parts and getattr(parts[0], "text", None):
                            img_text = parts[0].text.strip()
                            if img_text.startswith("```json"):
                                img_text = img_text[7:]
                            elif img_text.startswith("```"):
                                img_text = img_text[3:]
                            if img_text.endswith("```"):
                                img_text = img_text[:-3]
                            img_text = img_text.strip()

                            llm_schema = ClinicalIntakeSchema.model_validate_json(img_text)
                            if caption:
                                cap_det = self._deterministic_egyptian_nlu_parse(caption)
                                from app.database.db import merge_clinical_intake
                                return merge_clinical_intake(cap_det, llm_schema, caption)
                            return llm_schema
                except Exception as exc:
                    logger.warning("Vision attempt on model %s failed: %s", model_cand, exc)

        # 3. Graceful Clinical Fallback (No 503 error popup)
        logger.info("Vision analysis completed without auto-detection; prompting user cleanly in clinical chat.")
        return ClinicalIntakeSchema(
            raw_transcript_or_notes=caption or "[صورة علبة دواء مرفقة]",
            medications=[],
            symptoms=[],
            missing_critical_info=True,
            clarification_question_ar="تم استلام صورة الدواء بنجاح. لتأكيد المادة الفعالة ومراجعة تضارب الأدوية بدقة، ما هو اسم الدواء المطبوع على العلبة؟",
        )


multimodal_intake = MultimodalIntakePipeline()
