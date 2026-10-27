"""
Module 3: The Dots Connector, Clinical Grounding & RAG Citations (`app/medical_engine/triage_orchestrator.py`)
Combines:
1. Deterministic Medical Rules Engine (INV-1: SOLE authority for safety alerts and ER escalation).
2. Trusted-Source RAG Grounding Engine (Task 2: WHO, EDA, KDIGO, NICE, MOH verbatim citations).
3. Caregiver Arabic Response with numbered citations.
4. Physician English Payload with EVIDENCE BASIS block.
5. Two Monochrome 1-Page PDF Reports (Arabic Hospital Edition & English ER Edition).
"""

import base64
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from app.ai.errors import IntakeDegradationError
from app.ai.multimodal_intake import multimodal_intake
from app.ai.rag.grounding import grounding_engine
from app.database.db import (
    get_patient_intake_session,
    merge_clinical_intake,
    reset_patient_session,
    save_patient_intake_session,
)
from app.medical_engine.rules_engine import MANDATORY_RENAL_LAB_PANEL, rules_engine
from app.reports.pdf_generator import pdf_generator
from app.schemas.clinical_schemas import (
    ClinicalIntakeSchema,
    ComponentDegradation,
    DrugClass,
    ExtractedMedication,
    ExtractedSymptom,
    GroundedAnswer,
    PhysicianEmergencyPayload,
    RuleEngineEvaluation,
    TriageOrchestratorResult,
    TriageSeverity,
)

logger = logging.getLogger(__name__)


class TriageOrchestrator:
    """
    Core Medical Safety & Triage Orchestrator ('The Dots Connector').
    Decoupled from messaging channels. Enforces INV-1, INV-2, and INV-3.
    """

    async def _build_caregiver_response_ar(
        self,
        intake: ClinicalIntakeSchema,
        rule_eval: RuleEngineEvaluation,
        grounded_evidence: Optional[GroundedAnswer] = None,
        raw_user_note: Optional[str] = None,
    ) -> str:
        """
        Generates an empathetic, clear, non-panicking, firm, emoji-free Egyptian Arabic response
        for the caregiver or patient, supporting Dual-Mode Clinical Intelligence:
        Mode A: Deterministic Drug/Renal Safety Alert (invoking mandatory stop orders & KDIGO labs).
        Mode B: Symptom-Driven Smart Triage with proactive Egyptian pharmacovigilance.
        """
        if rule_eval.halt_for_clarification and rule_eval.clarification_prompt_ar:
            return rule_eval.clarification_prompt_ar

        clean_meds = [m for m in rule_eval.normalized_medications if not m.brand_name.startswith("__")]

        def _format_med_label(m: ExtractedMedication) -> str:
            if m.active_ingredient:
                desc = m.active_ingredient
            elif m.drug_class == DrugClass.VITAMIN_SUPPLEMENT:
                desc = "فيتامينات ومكمل غذائي"
            elif m.drug_class == DrugClass.ANALGESIC:
                desc = "مسكن آمن وخافض حرارة"
            elif m.drug_class == DrugClass.ANTIBIOTIC:
                desc = "مضاد حيوي"
            elif m.drug_class == DrugClass.NSAID:
                desc = "مسكن ومضاد للالتهاب"
            else:
                desc = "مستحضر دوائي"
            dur = f" لمدة {m.duration_days} يوم" if m.duration_days else ""
            return f"{m.brand_name} ({desc}){dur}"

        meds_ar = (
            "، ".join(_format_med_label(m) for m in clean_meds)
            if clean_meds
            else "لا يوجد استخدام معلن لأدوية مزمنة"
        )

        is_drug_consultation = False
        if not rule_eval.triggered_alerts and clean_meds:
            if not intake.symptoms or len(intake.symptoms) == 0:
                is_drug_consultation = True
            elif any(m.drug_class in {DrugClass.VITAMIN_SUPPLEMENT, DrugClass.OTHER, DrugClass.ANALGESIC} for m in clean_meds) and len(intake.symptoms) <= 1:
                is_drug_consultation = True
            elif raw_user_note and any(m.brand_name.lower() in raw_user_note.lower() for m in clean_meds) and any(kw in raw_user_note.lower() for kw in ["دوا", "دواء", "فيتامين", "مكمل", "جرعة", "استخدام", "فوائد", "ايه ده", "عبارة عن ايه"]):
                is_drug_consultation = True

        symptoms_ar = (
            "استشارة دوائية ومراجعة سلامة المكمل / العلاج"
            if (is_drug_consultation and not intake.symptoms)
            else ("، ".join(s.symptom_Arabic for s in intake.symptoms) if intake.symptoms else "أعراض عامة / فحص وقائي")
        )

        lines: List[str] = []
        lines.append("**تقييم «صحتك AI» للسلامة الدوائية والفرز الطبي**")
        lines.append("--------------------------------------------------")
        lines.append("تمت مراجعة التاريخ الدوائي والأعراض الإكلينيكية المسجلة للمريض بدقة:")
        lines.append(f"- **الأعراض الحالية المسجلة:** {symptoms_ar}")
        lines.append(f"- **التاريخ الدوائي والأدوية المرصودة:** {meds_ar}")
        lines.append("")

        # ------------------------------------------------------------------
        # MODE A: Deterministic Drug/Renal Red-Flag Triggered
        # ------------------------------------------------------------------
        if rule_eval.triggered_alerts:
            lines.append("**التفسير الإكلينيكي وربط الأعراض بالأدوية (إنذار أحمر عاجل):**")
            for alert in rule_eval.triggered_alerts:
                lines.append(f"{alert.patient_warning_ar}")
                lines.append("")

            target_specialty = rule_eval.triggered_alerts[0].recommended_specialty_ar
            lines.append("**الإجراءات الطبية العاجلة المطلوبة:**")
            lines.append(
                "1. **إيقاف المسكن فوراً:** يجب إيقاف أي مسكنات من عائلة (كتافلام / فولتارين / كيتوفان / بروفين / باي-ألكوفان) فوراً تحت إشراف الطبيب، وعدم تناول أي مسكن إضافي لتسكين ألم الذراع أو الجسم."
            )
            lines.append(
                f"2. **التوجه الطبي المباشر:** التوجه إلى **{target_specialty}** وتقديم التقرير الطبي المرفق (نسخة المستشفى باللغة العربية أو النسخة الإنجليزية لطبيب الطوارئ)."
            )
            lines.append("3. **التحاليل والفحوصات المعملية العاجلة المطلوبة:**")
            for lab in rule_eval.urgent_labs_consolidated:
                lines.append(f"   - {lab}")

            lines.append("")
            if grounded_evidence and grounded_evidence.citations_formatted_ar:
                lines.append("**المصادر والأدلة الإكلينيكية الموثقة:**")
                for cite in grounded_evidence.citations_formatted_ar:
                    lines.append(f"- {cite}")
            else:
                lines.append("**المصادر والأدلة الإكلينيكية الموثقة:**")
                lines.append("- إرشادات KDIGO 2012 للوقاية من القصور الكلوي الحاد الناتج عن المسكنات.")
                lines.append("- النشرات الدورية لهيئة الدواء المصرية (EDA) ومنظمة الصحة العالمية حول أمان مضادات الالتهاب غير الستيرويدية.")

        # ------------------------------------------------------------------
        # MODE C: Medication & Dietary Supplement Consultation
        # ------------------------------------------------------------------
        elif is_drug_consultation:
            drug_consultation_body = await self._generate_medication_consultation_ar(intake, clean_meds, raw_user_note)
            lines.append(drug_consultation_body)
            lines.append("")
            lines.append("**المصادر والأدلة الإكلينيكية المعتمدة:**")
            lines.append("- هيئة الدواء المصرية (EDA) — الأدلة الإرشادية لسلامة وتسجيل المكملات الغذائية والأدوية المصرح بها.")
            lines.append("- المعهد الوطني الأمريكي للصحة (NIH) — مكتب المكملات الغذائية (Office of Dietary Supplements - ODS).")
            lines.append("- منظمة الصحة العالمية (WHO) — إرشادات المغذيات الدقيقة والمكملات المتوازنة.")
            lines.append("- الدليل الدوائي البريطاني (British National Formulary - BNF).")

        # ------------------------------------------------------------------
        # MODE B: Symptom-Driven Smart Triage (No chronic drug toxicity)
        # ------------------------------------------------------------------
        else:
            symptom_triage_body = await self._generate_symptom_driven_triage_ar(intake, raw_user_note)
            lines.append(symptom_triage_body)
            lines.append("")
            lines.append("**المصادر والأدلة الإكلينيكية المعتمدة:**")
            lines.append("- البروتوكول العلاجي للنزلات الفيروسية الحادة وإرشادات ترشيد الأدوية — وزارة الصحة والسكان المصرية (MOH).")
            lines.append("- الدليل الإرشادي لمنظمة الصحة العالمية (WHO) للتقييم الأولي للأعراض التنفسية والفرز المبكر.")
            lines.append("- تحذيرات هيئة الدواء المصرية (EDA) الرسمية بشأن مخاطر حقن البرد والخلطات العشوائية.")

        lines.append("")
        lines.append("**التقارير الطبية المرفقة:** تم إصدار نسختين من تقرير الفرز الطبي للطوارئ (نسخة باللغة العربية للمستشفى والأسرة، ونسخة باللغة الإنجليزية لطبيب الطوارئ).")
        lines.append("--------------------------------------------------")
        lines.append(
            "*إخلاء مسؤولية طبي: نظام «صحتك AI» هو نظام دعم قرار وفرز طبي مبكر (Clinical Triage & Safety Guardian) ولا يُعد بديلاً عن الفحص الطبي المباشر أو تشخيص الطبيب المعالج.*"
        )
        return "\n".join(lines)

    async def _generate_medication_consultation_ar(
        self,
        intake: ClinicalIntakeSchema,
        clean_meds: List[ExtractedMedication],
        raw_user_note: Optional[str] = None,
    ) -> str:
        """
        Mode C: Authoritative Clinical Pharmacological & Dietary Supplement Consultation
        Powered by Gemini / Groq with rich, deterministic Egyptian pharmacological fallback.
        """
        from app.config import settings

        med_names = "، ".join(m.brand_name for m in clean_meds)
        med_actives = "، ".join(m.active_ingredient or m.brand_name for m in clean_meds)

        prompt = (
            "أنت «صحتك AI» - خبير اليقظة الدوائية والصيدلة الإكلينيكية المعتمد في مصر.\n"
            f"المستحضر أو المكمل الغذائي المطلوب تحليله: {med_names}\n"
            f"المواد الفعالة أو الفئة: {med_actives}\n"
            f"استفسار المريض/المرافق: {raw_user_note or 'استفسار عام عن دواعي الاستخدام والجرعة والامان'}\n"
            f"العمر: {intake.patient_age or 'غير محدد'}\n\n"
            "المطلوب: تقديم دليل دوائي إكلينيكي رصين وموثق بأسلوب علمي ومطمئن باللهجة المصرية الطبية الراقية بدون أي إيموجيز نهائياً، وفق الأقسام التالية بالضبط:\n"
            "**1. التعريف بالمستحضر والمواد الفعالة:**\n"
            "(شرح علمي دقيق لماهية المستحضر وتصنيفه الطبي، والمواد الفعالة ومضادات الأكسدة مثل اللوتين أو الفيتامينات أو المعادن إن وجدت)\n\n"
            "**2. دواعي الاستخدام والفوائد الإكلينيكية:**\n"
            "(الأسباب الطبية لتناوله والفوائد الموثقة لصحة الجسم، العين، المناعة، أو التمثيل الغذائي)\n\n"
            "**3. إرشادات الجرعة وطريقة الاستخدام الصحيحة:**\n"
            "(الجرعة النموذجية للبالغين، وأفضل توقيت مع الوجبات لشرب الماء ولتحسين امتصاص الفيتامينات الذائبة في الدهون وتجنب تهيج المعدة)\n\n"
            "**4. موانع الاستخدام والتحذيرات والتعارضات الدوائية:**\n"
            "(موانع الاستخدام، التداخلات مع مكملات أخرى مثل فيتامين أ أو حديد، الفصل مع أدوية الحموضة، وأمانه على الكلى والمعدة)\n"
        )

        # Live Web Search Agent (retrieve real-time clinical evidence)
        live_search_snippets: List[str] = []
        try:
            from app.ai.tools.web_search_agent import web_search_agent
            search_query = f"{med_names} {med_actives}".strip()
            search_results = await web_search_agent.search(search_query, max_results=3)
            if search_results:
                live_evidence_text = web_search_agent.format_for_prompt(search_results)
                prompt += f"\n\n{live_evidence_text}\nاستند بدقة إلى بيانات البحث الطبي المباشر أعلاه في تعزيز وتحقيق الإجابة."
                for r in search_results:
                    if r.get("source") and r.get("url"):
                        live_search_snippets.append(f"{r['source']} ({r['url']})")
        except Exception as se:
            logger.warning("Live web search agent failed: %s", se)

        search_footer = ""
        if live_search_snippets:
            search_footer = "\n\n**توثيق البحث الحي عبر الإنترنت (Live Medical Search):**\n" + "\n".join(f"- تحقق حي: {s}" for s in live_search_snippets[:3])

        # 1. Try Gemini
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-google-gemini-api-key-here":
            try:
                from google import genai
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                res = await client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL_NAME,
                    contents=prompt,
                )
                if res and res.text and len(res.text.strip()) > 100:
                    return res.text.strip() + search_footer
            except Exception as e:
                logger.warning("Gemini drug consultation failed: %s. Trying Groq...", e)

        # 2. Try Groq LLM
        if settings.GROQ_API_KEY:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    groq_res = await http_client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                        json={
                            "model": settings.GROQ_LLM_MODEL or "qwen/qwen3.8-27b",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "أنت «صحتك AI» - استشاري الصيدلة الإكلينيكية واليقظة الدوائية في مصر. أجب بالعربية المصرية الطبية الرصينة بدون أي إيموجيز نهائياً.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 1200,
                        },
                    )
                    if groq_res.status_code == 200:
                        content = groq_res.json()["choices"][0]["message"]["content"].strip()
                        if len(content) > 100:
                            return content + search_footer
            except Exception as ge:
                logger.warning("Groq drug consultation failed: %s", ge)

        # 3. Deterministic Egyptian Pharmacological Knowledge Base Fallback
        search_target = f"{med_names} {med_actives} {raw_user_note or ''}".lower()

        if any(w in search_target for w in ["lutein", "لوتين", "centrum", "سنترم", "سنتروم"]):
            return (
                "**1. التعريف بالمستحضر والمواد الفعالة:**\n"
                "- **المستحضر:** سنترم لوتين (Centrum with Lutein) / سنترم سيلفر.\n"
                "- **التصنيف الطبي:** مكمل غذائي متكامل يحتوي على تركيبة متوازنة تضم 24 عنصراً من الفيتامينات الأساسية والمعادن النادرة، مدعمة بمركب **اللوتين (Lutein)**.\n"
                "- **المادة المميزة (اللوتين):** صبغة كاروتينويدية ومضاد أكسدة طبيعي عالي التركيز يتراكم بشكل خاص في بقعة شبكية العين (Macula)، ليعمل كمرشح واقٍ من أضرار الضوء الأزرق والأكسدة الخلوية.\n\n"
                "**2. دواعي الاستخدام والفوائد الإكلينيكية:**\n"
                "- **حماية صحة العين والإبصار:** المساهمة الفعالة في تقليل مخاطر الضمور البقعي المرتبط بالعمر (AMD) وحماية شبكية العين من الإجهاد التأكسدي.\n"
                "- **دعم الطاقة والحيوية اليومية:** بفضل احتوائه على فيتامينات مجموعة B المركبة (B1, B2, B6, B12, Niacin) التي تلعب دوراً محورياً في عمليات التمثيل الغذائي وإنتاج الطاقة من الغذاء.\n"
                "- **تعزيز كفاءة الجهاز المناعي:** لاحتوائه على جرعات متوازنة من فيتامين C، وفيتامين E، والزنك، والسيلينيوم كمضادات أكسدة قوية.\n"
                "- **الحفاظ على سلامة العظام والأسنان:** عبر تزويد الجسم بفيتامين D3 والكالسيوم والمغنيسيوم.\n\n"
                "**3. إرشادات الجرعة وطريقة الاستخدام الصحيحة:**\n"
                "- **الجرعة المعتادة للبالغين:** قرص واحد يومياً، ولا يجوز تجاوز الجرعة المقررة دون استشارة الطبيب.\n"
                "- **التوقيت الأمثل لتناول الدواء:** يُفضل تناول القرص مباشرة بعد وجبة الإفطار أو وجبة الغداء مع كوب كامل من الماء.\n"
                "- **نصيحة للامتصاص وتجنب اضطراب المعدة:** تناول المكمل مع وجبة تحتوي على قدر ضئيل من الدهون الصحية (مثل زيت الزيتون أو الزبادي) يرفع نسبة امتصاص الفيتامينات الذائبة في الدهون (A, D, E, K واللوتين)، ويقي من أي شعور بالغثيان أو انزعاج المعدة.\n\n"
                "**4. موانع الاستخدام والتحذيرات والتعارضات الدوائية:**\n"
                "- **تجنب الازدواج الدوائي:** يجب الامتناع عن أخذ مكملات أخرى تحتوي على فيتامين أ (Vitamin A) أو الحديد في نفس الوقت لتجنب التسمم بالتراكم الزائد (Hypervitaminosis).\n"
                "- **الفصل عن أدوية الحموضة:** يُنصح بترك فاصل زمني لا يقل عن ساعتين بين سنترم وأدوية الحموضة (Antacids) أو مشتقات الألومنيوم والمغنيسيوم لتفادي تثبيط امتصاص المعادن والحديد.\n"
                "- **مرضى القصور الكلوي:** يجب مراجعة الطبيب المعالج قبل الاستخدام لضبط معدلات المعادن الدقيقة وتفادي تراكمها.\n"
                "- **الأمان الكلوي والمعدي:** يعد سنترم لوتين آمناً تماماً على الكلى والمعدة عند الالتزام بالجرعة اليومية الواحدة."
            )
        elif any(w in search_target for w in ["omega", "اوميجا", "أوميجا", "زيت سمك", "fish oil"]):
            return (
                "**1. التعريف بالمستحضر والمواد الفعالة:**\n"
                "- **المستحضر:** أوميجا 3 بلس (Omega 3 Plus).\n"
                "- **التصنيف:** أحماض دهنية أساسية غير مشبعة مستخلصة من زيت السمك النقي وغنية بحمضي EPA و DHA، مدعمة بزيت جنين القمح كمصدر طبيعي لفيتامين E.\n\n"
                "**2. دواعي الاستخدام والفوائد الإكلينيكية:**\n"
                "- **دعم صحة القلب والشرايين:** تحسين مستويات الدهون الثلاثية بالدم، الحفاظ على مرونة الأوعية الدموية، ودعم ضغط الدم الصحي.\n"
                "- **تعزيز الوظائف الإدراكية والتركيز:** حمض DHA مكون بنيوي رئيسي لخلايا المخ والشبكية.\n"
                "- **مضاد طبيعي للالتهاب:** يساعد في تخفيف آلام وتيبس المفاصل والتهابات الجسم المزمنة.\n\n"
                "**3. إرشادات الجرعة وطريقة الاستخدام الصحيحة:**\n"
                "- **الجرعة المعتادة للبالغين:** كبسولة إلى كبسولتين يومياً بعد الوجبة الرئيسية مع كوب كبير من الماء.\n"
                "- يُفضل عدم تناولها على معدة خاوية لتقليل التجشؤ برائحة السمك وتحسين الامتصاص.\n\n"
                "**4. موانع الاستخدام والتحذيرات والتعارضات الدوائية:**\n"
                "- **الحذر مع مسيلات الدم:** الجرعات العالية (فوق 2000-3000 مجم يومياً) تمتلك تأثيراً خفيفاً مضاداً لتجمع الصفائح؛ لذا يجب استشارة الطبيب في حال تناول أدوية سيولة مثل وارفارين (ماريفان) أو بلافيكس.\n"
                "- التوقف قبل العمليات الجراحية المجدولة بأسبوع إلى أسبوعين بعد التنسيق مع الجراح."
            )
        elif any(w in search_target for w in ["vidrop", "فيدروب", "ديفارول", "devarol", "فيتامين د", "vitamin d"]):
            return (
                "**1. التعريف بالمستحضر والمواد الفعالة:**\n"
                "- **المستحضر:** فيدروب (Vidrop) / ديفارول (Devarol).\n"
                "- **المادة الفعالة:** كوليكالسيفيرول (Cholecalciferol - Vitamin D3).\n\n"
                "**2. دواعي الاستخدام والفوائد الإكلينيكية:**\n"
                "- علاج والوقاية من نقص فيتامين د ولين العظام والكساح.\n"
                "- تحفيز امتصاص الكالسيوم والفوسفور من الأمعاء لتقوية كثافة العظام والأسنان.\n"
                "- تنظيم وتدعيم الاستجابة المناعية ضد العدوى التنفسية.\n\n"
                "**3. إرشادات الجرعة وطريقة الاستخدام الصحيحة:**\n"
                "- تعتمد الجرعة بدقة على نتائج تحليل الدم (25-OH Vitamin D).\n"
                "- الجرعة الوقائية للبالغين: 600 - 1000 وحدة دولية يومياً (6 إلى 10 قطرات فيدروب)، أو وفق الخطة العلاجية المقررة من الطبيب.\n"
                "- تؤخذ بعد وجبة دسمة تحتوي على دهون صحية لضمان أعلى نسبة امتصاص.\n\n"
                "**4. موانع الاستخدام والتحذيرات والتعارضات الدوائية:**\n"
                "- يمنع في حالات فرط كالسيوم الدم (Hypercalcemia) أو وجود حصوات كلوية كلسية نشطة.\n"
                "- ضرورة فحص وظائف الكلى والكالسيوم دورياً في حال الاستخدام العلاجي بالجرعات العالية."
            )
        elif any(w in search_target for w in ["neurobion", "نيوروبيون", "milga", "ميلجا", "نيوروتون", "فيتامين ب"]):
            return (
                "**1. التعريف بالمستحضر والمواد الفعالة:**\n"
                "- **المستحضر:** نيوروبيون (Neurobion) / ميلجا (Milga Advance).\n"
                "- **المواد الفعالة:** فيتامين ب1 (ثيامين/بنفوتيامين) + فيتامين ب6 (بيريدوكسين) + فيتامين ب12 (سيانوكوبالامين).\n\n"
                "**2. دواعي الاستخدام والفوائد الإكلينيكية:**\n"
                "- علاج والوقاية من التهاب الأعصاب الطرفية (خصوصاً لمرضى السكري).\n"
                "- تخفيف آلام العصب والوخز والتنميل في الأطراف.\n"
                "- دعم تجدد غلاف المايلين المغذي للألياف العصبية وتصنيع كرات الدم الحمراء.\n\n"
                "**3. إرشادات الجرعة وطريقة الاستخدام الصحيحة:**\n"
                "- قرص واحد من مرة إلى مرتين يومياً بعد الأكل، أو حقنة عضلية كل يومين إلى ثلاثة أيام حسب توجيهات الطبيب.\n\n"
                "**4. موانع الاستخدام والتحذيرات والتعارضات الدوائية:**\n"
                "- مكملات فيتامينات ب الذائبة في الماء ذات أمان عالٍ جداً، ويتم التخلص من الفائض عبر البول دون إجهاد كلوي.\n"
                "- قد يلاحظ المريض تغير لون البول إلى الأصفر الفاقع أو المائل للبرتقالي وهو أمر طبيعي تماماً وغير مقلق."
            )
        elif any(w in search_target for w in ["panadol", "بنادول", "باراسيتامول", "paracetamol", "paramol", "بارامول"]):
            return (
                "**1. التعريف بالمستحضر والمواد الفعالة:**\n"
                "- **المستحضر:** بنادول (Panadol) / بارامول (Paramol).\n"
                "- **المادة الفعالة:** باراسيتامول (Paracetamol / Acetaminophen) 500 مجم.\n\n"
                "**2. دواعي الاستخدام والفوائد الإكلينيكية:**\n"
                "- خافض للحرارة ومسكن للآلام الخفيفة والمتوسطة (الصداع، آلام الأسنان، ألم العضلات، وأعراض البرد).\n"
                "- المسكن الأول الموصى به لمرضى الكلى وقرحة المعدة نظراً لعدم تثبيطه للبروستاغلاندينات الكلوية.\n\n"
                "**3. إرشادات الجرعة وطريقة الاستخدام الصحيحة:**\n"
                "- قرص إلى قرصين (500 - 1000 مجم) كل 6 إلى 8 ساعات عند اللزوم.\n"
                "- الحد الأقصى المطلق للبالغين ذوي الكبد السليم هو 4000 مجم (8 أقراص) خلال 24 ساعة.\n\n"
                "**4. موانع الاستخدام والتحذيرات والتعارضات الدوائية:**\n"
                "- يحظر تجاوز الجرعة القصوى لتجنب التسمم الكبدي الحاد.\n"
                "- يجب خفض الجرعة ومراجعة الطبيب لمرضى القصور الكبدي أو مدمني الكحول."
            )
        else:
            return (
                f"**1. التعريف بالمستحضر والمواد الفعالة:**\n"
                f"- **المستحضر:** {med_names}.\n"
                f"- **المكونات والتركيب:** {med_actives}.\n"
                "- مستحضر صيدلاني مصرح به للاستخدام وفق إرشادات هيئة الدواء المصرية والنشرات الطبية المعتمدة.\n\n"
                "**2. دواعي الاستخدام والفوائد الإكلينيكية:**\n"
                "- يُستخدم المستحضر لتحقيق الأثر العلاجي أو الوقائي المستهدف بحسب تركيبته الصيدلانية، مع مراعاة الحالة الصحية العامة والتاريخ المرضي للمريض.\n\n"
                "**3. إرشادات الجرعة وطريقة الاستخدام الصحيحة:**\n"
                "- الالتزام الدقيق بالجرعة المقررة من الطبيب أو الصيدلي.\n"
                "- يُنصح دائماً بتناول الأقراص والكبسولات مع كوب ماء كامل، والالتزام بمواعيد الجرعات دون مضاعفة الجرعة المنسية.\n\n"
                "**4. موانع الاستخدام والتحذيرات والتعارضات الدوائية:**\n"
                "- التأكد من عدم وجود حساسية سابقة تجاه أي من مكونات المستحضر.\n"
                "- إبلاغ الطبيب المعالج بكافة الأدوية المزمنة والمكملات المتناولة لتفادي أي تداخلات دوائية غير مرغوبة."
            )

    async def _generate_symptom_driven_triage_ar(
        self,
        intake: ClinicalIntakeSchema,
        raw_user_note: Optional[str] = None,
    ) -> str:
        """
        Mode B dynamic generation using Gemini 2.5 Flash, backed by an intelligent deterministic Egyptian clinical fallback.
        """
        symptoms_str = "، ".join(s.symptom_Arabic for s in intake.symptoms) if intake.symptoms else (raw_user_note or "أعراض عامة")
        from app.config import settings

        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-google-gemini-api-key-here":
            try:
                from google import genai
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                prompt = (
                    "أنت «صحتك AI» - نظام الفرز الطبي الذكي والسلامة الدوائية المعتمد في مصر.\n"
                    f"أعراض المريض: {symptoms_str}\n"
                    f"ملاحظات المريض/المرافق: {raw_user_note or 'غير محدد'}\n"
                    f"العمر: {intake.patient_age or 'غير محدد'}\n"
                    "لا يوجد استخدام معلن لمسكنات مزمنة تؤدي لقصور كلوي حتى الآن.\n\n"
                    "المطلوب: صياغة تقييم فرز إكلينيكي رصين ومطمئن باللهجة المصرية الطبية الراقية بدون أي إيموجيز نهائياً، وفق الأقسام التالية بالضبط:\n"
                    "**1. قراءة الحالة والتقييم المبدئي:**\n"
                    "(تحليل دقيق للأعراض المذكورة وتوضيح أكثر الاحتمالات شيوعاً كالنزلات الفيروسية التنفسية الحادة أو الإنفلونزا)\n\n"
                    "**2. أسئلة استبعاد علامات الخطر (Red-Flag Rule-Out Questions):**\n"
                    "(3 أسئلة محددة تستبعد الخطر فوراً، مثل: وجود سخونية عالية فوق 38.5، نهجان أو كتمة نفس، أو استمرار الأعراض لفترة طويلة)\n\n"
                    "**3. نصيحة «صحتك AI» للأمان الدوائي (حماية استباقية):**\n"
                    "(- تحذير قاطع وصريح من اللجوء لـ (حقنة البرد / الخلطة السحرية/الثلاثية) بالصيدليات لاحتوائها على كورتيزون ومسكن ومضاد حيوي عشوائي يدمر المناعة والكلى\n"
                    "- تحذير من أخذ مسكنات ثقيلة كالكتافلام والبروفين على معدة فارغة أو مع الجفاف\n"
                    "- التوصية بالباراسيتامول كمسكن آمن وخافض حرارة أولي مع الراحة والسوائل الدافئة)\n\n"
                    "**4. التوجيه الطبي والفحوصات المقترحة:**\n"
                    "(- التخصص المناسب كعيادة الصدرية أو الباطنة العامة\n"
                    "- الفحوصات المقترحة في حال استمرار الأعراض مثل صورة الدم CBC وتحليل CRP)\n"
                )
                res = await client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL_NAME,
                    contents=prompt,
                )
                if res and res.text and len(res.text.strip()) > 80:
                    return res.text.strip()
            except Exception as e:
                logger.warning("Gemini dynamic symptom triage failed, trying Groq fallback: %s", e)

        # Fallback to Groq LLM if Gemini failed and GROQ_API_KEY is available
        if settings.GROQ_API_KEY:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    groq_res = await http_client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                        json={
                            "model": settings.GROQ_LLM_MODEL or "qwen/qwen3.8-27b",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "أنت «صحتك AI» - نظام الفرز الطبي واليقظة الدوائية في مصر. أجب بالعربية المصرية الطبية الرصينة بدون أي إيموجيز نهائياً.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 1000,
                        },
                    )
                    if groq_res.status_code == 200:
                        content = groq_res.json()["choices"][0]["message"]["content"].strip()
                        if len(content) > 80:
                            return content
            except Exception as ge:
                logger.warning("Groq symptom triage fallback failed: %s", ge)

        # Fallback Deterministic Clinical Generation
        is_resp = any(
            any(w in (s.symptom_Arabic + s.clinical_term_English).lower() for w in ["كحة", "سعال", "برد", "احتقان", "رشح", "زور", "cough", "pharyngitis", "cold", "myalgia", "مكسر"])
            for s in intake.symptoms
        ) or any(w in (raw_user_note or "").lower() for w in ["كحة", "كحه", "سعال", "برد", "مكسر", "تكسير"])

        if is_resp:
            return (
                "**1. قراءة الحالة والتقييم المبدئي:**\n"
                f"بناءً على الأعراض المسجلة ({symptoms_str})، فإن الحالة تتماشى مبدئياً مع نزلة برد حادة أو اشتباه عدوى فيروسية في الجهاز التنفسي العلوي (مثل الإنفلونزا الموسمية أو التهاب الشعب الهوائية الحاد).\n\n"
                "**2. أسئلة استبعاد علامات الخطر (Red-Flag Rule-Out Questions):**\n"
                "1. هل تعاني من ارتفاع شديد في درجة الحرارة (سخونية فوق 38.5 درجة) لا يستجيب للمخفضات العادية؟\n"
                "2. هل هناك نهجان، صعوبة في أخذ النفس، أو كتمة حادة في الصدر؟\n"
                "3. الأعراض دي مستمرة بقالها كام يوم؟ وهل الكحة مصحوبة ببلغم مدمم أو ألم حاد في أحد الجانبين؟\n\n"
                "**3. نصيحة «صحتك AI» للأمان الدوائي (حماية استباقية):**\n"
                "- **تحذير حاسم من حقنة البرد:** نُحذر وبشدة من اللجوء لما يُعرف شعبياً بـ (حقنة البرد أو الخلطة السحرية الثلاثية) في الصيدليات؛ لاحتوائها على كورتيزون ومضاد حيوي ومسكن قوي معاً، وهو خلط عشوائي يسبب هبوطاً مفاجئاً في المناعة وقرحة المعدة وإجهاداً كلوياً شديداً دون مبرر طبي.\n"
                "- **تجنب المسكنات غير الستيرويدية القوية:** تجنب تناول مسكنات مثل كتافلام، فولتارين، كيتوفان، أو بروفين على معدة فارغة أو مع قلة السوائل.\n"
                "- **المسكن الآمن:** المسكن الأكثر أماناً مبدئياً لتسكين تكسير الجسم والصداع هو **الباراسيتامول (مثل بنادول / بارامول)** بالجرعات النظامية، مع الإكثار من شرب السوائل الدافئة والراحة التامة.\n\n"
                "**4. التوجيه الطبي والفحوصات المقترحة:**\n"
                "- **التخصص الطبي:** يُنصح بمراجعة **طبيب أمراض صدرية أو باطنة عامة** في حال استمرار الأعراض لأكثر من 3 إلى 5 أيام أو في حال تفاقمها.\n"
                "- **التحاليل المبدئية المقترحة:** قد يطلب الطبيب صورة دم كاملة (CBC) ومعدل الالتهاب (CRP) للتأكد من عدم وجود عدوى بكتيرية ثانوية."
            )
        else:
            return (
                "**1. قراءة الحالة والتقييم المبدئي:**\n"
                f"تم فحص الأعراض المذكورة ({symptoms_str})، ولم يتم رصد تداخلات دوائية حرجة أو استخدام مسكنات مزمنة، وتتطلب الحالة فحصاً سريرياً مباشراً لتحديد السبب بدقة.\n\n"
                "**2. أسئلة استبعاد علامات الخطر (Red-Flag Rule-Out Questions):**\n"
                "1. هل تشعر بدوخة حادة، إغماء، أو هبوط مفاجئ في ضغط الدم؟\n"
                "2. هل هناك قيء متكرر أو آلام حادة غير محتملة؟\n"
                "3. هل لاحظت أي تغير في كمية البول اليومية أو تورم في القدمين؟\n\n"
                "**3. نصيحة «صحتك AI» للأمان الدوائي (حماية استباقية):**\n"
                "- تجنب تناول أي مسكنات قوية بدون استشارة الطبيب لتجنب إخفاء الأعراض أو إجهاد الكلى والمعدة.\n"
                "- الخيار الأكثر أماناً لتسكين الألم العام لحين العرض على الطبيب هو الباراسيتامول مع شرب كميات كافية من الماء.\n\n"
                "**4. التوجيه الطبي والفحوصات المقترحة:**\n"
                "- مراجعة **عيادة الباطنة العامة** للفحص الإكلينيكي المباشر وقياس العلامات الحيوية (الضغط، النبض، الحرارة)."
            )

    def _build_physician_payload_en(
        self,
        intake: ClinicalIntakeSchema,
        rule_eval: RuleEngineEvaluation,
        grounded_evidence: Optional[GroundedAnswer] = None,
    ) -> PhysicianEmergencyPayload:
        report_id = f"SHK-{uuid.uuid4().hex[:6].upper()}"

        age_str = f"{intake.patient_age}yo" if intake.patient_age else "Age unconfirmed"
        sex_str = (intake.patient_sex or "Sex unconfirmed").capitalize()
        comorbid_str = (
            ", ".join(intake.chronic_conditions)
            if intake.chronic_conditions
            else "No documented chronic comorbidities"
        )
        demographics = f"{age_str} {sex_str} | PMH: {comorbid_str}"

        if intake.symptoms:
            cc_list = [
                f"{s.clinical_term_English} ({s.severity.value.upper()})"
                + (f" [{s.duration_days}d]" if s.duration_days else "")
                for s in intake.symptoms
            ]
            chief_complaint = "; ".join(cc_list)
        else:
            chief_complaint = "Asymptomatic / Medication safety review requested by caregiver."

        med_exposures: List[str] = []
        clean_meds = [m for m in rule_eval.normalized_medications if not m.brand_name.startswith("__")]
        for m in clean_meds:
            dur_txt = f"Duration: {m.duration_days} days" if m.duration_days is not None else "Duration: Unknown"
            freq_txt = f" | Freq: {m.frequency}" if m.frequency else ""
            med_exposures.append(
                f"{m.brand_name} ({m.active_ingredient or 'Active ingredient N/A'}) [{m.drug_class.value}] — {dur_txt}{freq_txt}"
            )

        is_renal_alert = bool(rule_eval.triggered_alerts)
        is_drug_consultation = not is_renal_alert and bool(clean_meds) and (
            not intake.symptoms or any(m.drug_class in {DrugClass.VITAMIN_SUPPLEMENT, DrugClass.OTHER, DrugClass.ANALGESIC} for m in clean_meds)
        )

        if is_drug_consultation and not intake.symptoms:
            chief_complaint = f"Pharmacological & Dietary Supplement Review: {', '.join(m.brand_name for m in clean_meds)}"

        if is_renal_alert:
            correlations = " || ".join(a.clinical_finding_en for a in rule_eval.triggered_alerts)
            badge = "[URGENT TRIAGE ALERT - RENAL/DRUG SAFETY]"
            workup = list(rule_eval.urgent_labs_consolidated) if rule_eval.urgent_labs_consolidated else list(MANDATORY_RENAL_LAB_PANEL)
            workup.append("Immediate discontinuation of offending NSAIDs / nephrotoxic agents (KDIGO Stage 1-3 AKI Prevention)")
            workup.append("Clinical volume status assessment, strict urine output charting, and BP monitoring")
            guideline_ref = (
                "KDIGO Clinical Practice Guideline for Acute Kidney Injury (AKI); "
                "NICE Guideline NG148 (Acute Kidney Injury: Prevention, Detection and Management); "
                "Egyptian Drug Authority (EDA) NSAID Warnings."
            )
            evidence_basis = []
            if grounded_evidence and grounded_evidence.claims:
                for c in grounded_evidence.claims:
                    evidence_basis.append(f"{c.source_org} ({c.doc_title}, {c.section_or_page}): \"{c.evidence_quote}\"")
            else:
                evidence_basis = [
                    "KDIGO AKI Guideline 2012: Chronic or high-dose NSAID usage induces severe renal hemodynamic impairment via COX-mediated prostaglandin inhibition.",
                    "WHO Safety Alert: Cumulative NSAID exposure without renal function monitoring poses high risk of silent tubular necrosis."
                ]
        elif is_drug_consultation:
            correlations = (
                f"Pharmacological and dietary supplement consultation for: {', '.join(m.brand_name for m in clean_meds)}. "
                "Comprehensive review of active ingredients, antioxidant/micronutrient profile, "
                "recommended dietary allowance (RDA), safe administration timing, and drug-drug interaction screen. "
                "No critical organ toxicity or nephrotoxic flags identified."
            )
            badge = "[PHARMACOLOGICAL CONSULTATION - DRUG & SUPPLEMENT SAFETY]"
            workup = [
                "Medication reconciliation: ensure no duplicate multivitamin or concurrent high-dose fat-soluble vitamins (A, D, E, K)",
                "Advise post-prandial administration with adequate fluids to enhance bioavailability of fat-soluble fractions and prevent GI upset",
                "Ensure a 2-hour interval between multivitamin/mineral intake and oral antacids, iron supplements, or tetracyclines",
                "Baseline Complete Blood Count (CBC) and 25-OH Vitamin D only if constitutional symptoms or fatigue are reported",
                "Routine outpatient counseling; no acute emergency intervention required",
            ]
            guideline_ref = (
                "Egyptian Drug Authority (EDA) Guidelines for Dietary Supplements; "
                "NIH Office of Dietary Supplements (ODS) Clinical Fact Sheets; "
                "WHO Guidelines on Micronutrients and Essential Nutrition Actions."
            )
            evidence_basis = [
                "NIH ODS Fact Sheet: Balanced multivitamin-mineral formulations are safe when consumed within Established Upper Tolerable Intake Levels (UL).",
                "EDA Pharmacovigilance Guidance: Post-prandial intake optimizes absorption of carotenoids and fat-soluble vitamins while minimizing gastric irritation."
            ]
        else:
            correlations = (
                "No deterministic chronic nephrotoxicity or drug-drug interaction alert triggered. "
                "Presenting symptoms evaluated for acute clinical presentation (e.g., upper respiratory / acute viral prodrome)."
            )
            badge = "[CLINICAL TRIAGE SUMMARY - SYMPTOM EVALUATION]"

            is_resp = any(
                any(w in (s.clinical_term_English + s.symptom_Arabic).lower() for w in ["cough", "fever", "respiratory", "cold", "myalgia", "body aches", "كحة", "برد", "مكسر"])
                for s in intake.symptoms
            )
            if is_resp:
                workup = [
                    "Complete Blood Count (CBC) with differential (assess leukocytosis vs viral lymphopenia)",
                    "C-Reactive Protein (CRP) / Inflammatory markers if symptoms persist > 3 days",
                    "Chest Auscultation & Clinical Vitals (Temperature, SpO2, Heart Rate, Blood Pressure)",
                    "Supportive oral hydration and safe antipyretic therapy (Paracetamol); strictly avoid empiric cold cocktails",
                    "Chest X-Ray (CXR) if persistent high fever, tachypnea, or focal chest signs develop",
                ]
            else:
                workup = [
                    "Comprehensive outpatient physical examination and vital signs monitoring",
                    "Complete Blood Count (CBC) and basic metabolic screening",
                    "Targeted diagnostic workup based on primary organ system complaint",
                    "Avoid unprescribed NSAIDs or polypharmacy cocktails; maintain adequate oral hydration",
                ]

            guideline_ref = (
                "Egyptian MOH Outpatient Clinical Protocols; "
                "WHO Acute Respiratory & General Triage Guidelines; "
                "Egyptian Drug Authority (EDA) Safe Analgesic Stewardship."
            )
            evidence_basis = [
                "Egyptian MOH Protocol: Empiric combination injections ('Cold Cocktails') are strictly contraindicated due to corticosteroid and NSAID risks.",
                "WHO Clinical Guidance: Acute viral syndromes require supportive hydration and monitoring for secondary bacterial superinfection."
            ]

        return PhysicianEmergencyPayload(
            report_id=report_id,
            severity_badge=badge,
            patient_demographics=demographics,
            chief_complaint_and_timeline=chief_complaint,
            high_risk_medication_exposure=med_exposures,
            red_flag_clinical_correlation=correlations,
            recommended_immediate_workup=workup,
            clinical_guidelines_reference=guideline_ref,
            evidence_basis=evidence_basis,
        )

    async def process_Session_and_orchestrate(
        self,
        session_id: str,
        incoming_intake: ClinicalIntakeSchema,
        channel: str = "telegram",
        raw_user_note: Optional[str] = None,
        user_id: Optional[int] = None,
        degradation: Optional[ComponentDegradation] = None,
    ) -> TriageOrchestratorResult:
        """
        Processes session intake, runs deterministic rules (INV-1), retrieves trusted RAG grounding (Task 2),
        and emits dual Arabic/English reports.
        """
        existing_intake = await get_patient_intake_session(session_id)
        merged_intake = merge_clinical_intake(existing_intake, incoming_intake, raw_user_note)

        real_meds = [m for m in merged_intake.medications if m.brand_name and not m.brand_name.startswith("__")]
        merged_intake.medications = real_meds

        # INV-1: Deterministic rules engine is the SOLE authority for safety alerts
        rule_eval = rules_engine.evaluate(merged_intake)

        if rule_eval.halt_for_clarification:
            await save_patient_intake_session(
                session_id=session_id,
                intake=merged_intake,
                channel=channel,
                raw_user_message=raw_user_note,
                status="clarification_needed",
                user_id=user_id,
            )
            caregiver_msg = await self._build_caregiver_response_ar(merged_intake, rule_eval, raw_user_note=raw_user_note)
            return TriageOrchestratorResult(
                session_id=session_id,
                status="clarification_needed",
                severity=TriageSeverity.NEEDS_CLARIFICATION,
                caregiver_response_ar=caregiver_msg,
                physician_payload_en=None,
                pdf_report_path=None,
                pdf_report_path_ar=None,
                pdf_base64=None,
                pdf_base64_ar=None,
                intake_snapshot=merged_intake,
                rule_evaluation=rule_eval,
                grounded_evidence=None,
                degradation=degradation,
            )

        # Task 2: Grounding with trusted medical corpus
        grounded_evidence = None
        try:
            grounded_evidence = await grounding_engine.ground_triage(merged_intake, rule_eval)
        except Exception as e:
            logger.warning("RAG Grounding failed: %s", e)
            if not degradation:
                degradation = ComponentDegradation(component="rag_grounding", reason=str(e))

        caregiver_msg = await self._build_caregiver_response_ar(merged_intake, rule_eval, grounded_evidence, raw_user_note=raw_user_note)
        physician_payload = self._build_physician_payload_en(merged_intake, rule_eval, grounded_evidence)

        if rule_eval.triggered_alerts:
            arabic_pdf_lines = [
                "1. إيقاف المسكنات فوراً (مثل كتافلام، فولتارين، كيتوفان، بروفين، باي-ألكوفان) تحت إشراف الطبيب.",
                "2. التوجه العاجل إلى قسم طوارئ الباطنة والكلى بالمستشفى لتقييم وظائف الكلى وضغط الدم.",
                "3. إجراء التحاليل المذكورة أعلاه بشكل عاجل وعرض هذا التقرير على الطبيب المعالج فور الوصول.",
            ]
        else:
            arabic_pdf_lines = [
                "1. مراجعة العيادة التخصصية المختصة (باطنة عامة أو صدرية) للفحص السريري الدقيق وتقييم العلامات الحيوية.",
                "2. تجنب الخلطات الدوائية الشعبية وحقن البرد العشوائية، واستخدام الباراسيتامول كخيار آمن مبدئياً عند اللزوم.",
                "3. إجراء الفحوصات المقترحة (مثل CBC و CRP) في حال استمرار الأعراض لأكثر من 3 أيام أو ظهور حرارة.",
            ]

        # 1. Generate English Physician PDF (with EVIDENCE BASIS section)
        pdf_path_en = pdf_generator.generate_triage_pdf(
            physician_payload=physician_payload,
            arabic_family_instructions=arabic_pdf_lines,
            rule_eval=rule_eval,
        )
        # 2. Generate Arabic Hospital PDF (with citations)
        pdf_path_ar = pdf_generator.generate_arabic_hospital_pdf(
            physician_payload=physician_payload,
            intake=merged_intake,
            rule_eval=rule_eval,
            arabic_family_instructions=arabic_pdf_lines,
        )

        pdf_b64_en = None
        pdf_b64_ar = None
        try:
            pdf_b64_en = base64.b64encode(Path(pdf_path_en).read_bytes()).decode("ascii")
        except Exception:
            pass

        try:
            pdf_b64_ar = base64.b64encode(Path(pdf_path_ar).read_bytes()).decode("ascii")
        except Exception:
            pass

        await save_patient_intake_session(
            session_id=session_id,
            intake=merged_intake,
            channel=channel,
            raw_user_message=raw_user_note,
            status="completed",
            user_id=user_id,
            pdf_report_path=pdf_path_en,
            pdf_report_path_ar=pdf_path_ar,
        )

        return TriageOrchestratorResult(
            session_id=session_id,
            status="completed",
            severity=rule_eval.overall_severity,
            caregiver_response_ar=caregiver_msg,
            physician_payload_en=physician_payload,
            pdf_report_path=pdf_path_en,
            pdf_report_path_ar=pdf_path_ar,
            pdf_base64=pdf_b64_en,
            pdf_base64_ar=pdf_b64_ar,
            intake_snapshot=merged_intake,
            rule_evaluation=rule_eval,
            grounded_evidence=grounded_evidence,
            degradation=degradation,
        )

    async def handle_text_message(
        self,
        session_id: str,
        text: str,
        channel: str = "telegram",
        reset: bool = False,
        user_id: Optional[int] = None,
    ) -> TriageOrchestratorResult:
        if reset:
            await reset_patient_session(session_id)
        extracted = await multimodal_intake.extract_from_text(text)
        return await self.process_Session_and_orchestrate(
            session_id=session_id,
            incoming_intake=extracted,
            channel=channel,
            raw_user_note=text,
            user_id=user_id,
        )

    async def handle_voice_message(
        self,
        session_id: str,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        channel: str = "telegram",
        user_id: Optional[int] = None,
    ) -> TriageOrchestratorResult:
        """
        Voice note intake handler.
        INV-2: If extraction raises IntakeDegradationError, re-raises so the caller can return
        an explicit degradation response WITHOUT performing triage.
        """
        extracted = await multimodal_intake.extract_from_voice(audio_bytes, mime_type=mime_type)
        return await self.process_Session_and_orchestrate(
            session_id=session_id,
            incoming_intake=extracted,
            channel=channel,
            raw_user_note=extracted.raw_transcript_or_notes or "[Voice Note]",
            user_id=user_id,
        )

    async def handle_image_message(
        self,
        session_id: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        caption: Optional[str] = None,
        channel: str = "telegram",
        user_id: Optional[int] = None,
    ) -> TriageOrchestratorResult:
        extracted = await multimodal_intake.extract_from_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            caption=caption,
        )
        if not extracted.medications and not extracted.symptoms:
            return TriageOrchestratorResult(
                session_id=session_id,
                status="needs_input",
                severity=TriageSeverity.MODERATE_CAUTION,
                caregiver_response_ar=(
                    "**تم استلام صورة علبة الدواء بنجاح.**\n"
                    "--------------------------------------------------\n"
                    "لتأكيد المادة الفعالة والجرعة ومراجعة تضارب الأدوية بدقة تامة، يرجى كتابة **اسم الدواء المطبوع على العلبة**، وهل المريض يعاني من أي أمراض مزمنة كأمراض الكلى أو الضغط؟"
                ),
                caregiver_response_en="Medication packaging photo received. Please confirm the brand name written on the box to cross-reference drug safety.",
                hospital_report_ar="[تم استلام صورة مستحضر دوائي قيد تأكيد الاسم التجاري والجرعة]",
                emergency_report_en="[Medication packaging image uploaded; pending brand name confirmation]",
                pdf_report_path=None,
                pdf_report_path_ar=None,
                intake_snapshot=extracted,
                rule_evaluation=RuleEngineEvaluation(
                    overall_severity=TriageSeverity.MODERATE_CAUTION,
                    halt_for_clarification=False,
                    clarification_prompt_ar="ما هو اسم الدواء المكتوب على العلبة؟",
                    triggered_alerts=[],
                    normalized_medications=[],
                    urgent_labs_consolidated=[],
                ),
                grounded_evidence=None,
                degradation=None,
            )

        return await self.process_Session_and_orchestrate(
            session_id=session_id,
            incoming_intake=extracted,
            channel=channel,
            raw_user_note=caption or "[Medication Strip Photo]",
            user_id=user_id,
        )


triage_orchestrator = TriageOrchestrator()
