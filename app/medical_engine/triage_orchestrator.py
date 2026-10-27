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
        meds_ar = (
            "، ".join(
                f"{m.brand_name} ({m.active_ingredient or 'مسكن'})"
                + (f" لمدة {m.duration_days} يوم" if m.duration_days else "")
                for m in clean_meds
            )
            if clean_meds
            else "لا يوجد استخدام معلن لأدوية مزمنة"
        )
        symptoms_ar = (
            "، ".join(s.symptom_Arabic for s in intake.symptoms)
            if intake.symptoms
            else "أعراض عامة / فحص وقائي"
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
                logger.warning("Gemini dynamic symptom triage failed, using clinical fallback: %s", e)

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
        return await self.process_Session_and_orchestrate(
            session_id=session_id,
            incoming_intake=extracted,
            channel=channel,
            raw_user_note=caption or "[Medication Strip Photo]",
            user_id=user_id,
        )


triage_orchestrator = TriageOrchestrator()
