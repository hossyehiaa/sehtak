"""
Module 3: The Dots Connector & Clinical Grounding (`app/medical_engine/triage_orchestrator.py`)
Combines the extracted patient timeline + Deterministic Rule Engine outputs + Evidence-Based Clinical Guidelines
(USPSTF / NICE NG148 / KDIGO Acute Kidney Injury guidelines) to produce two distinct outputs:
1. Caregiver Response (Egyptian Arabic)
2. Physician Emergency Triage Payload (English 5-second ER format) + Monochrome 1-Page PDF Report
"""

import base64
import uuid
from pathlib import Path
from typing import List, Optional

from app.ai.multimodal_intake import multimodal_intake
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
    PhysicianEmergencyPayload,
    RuleEngineEvaluation,
    TriageOrchestratorResult,
    TriageSeverity,
)


class TriageOrchestrator:
    """
    Core Medical Safety & Triage Orchestrator ('The Dots Connector').
    Completely decoupled from messaging channels (Telegram, WhatsApp, Web REST API).
    """

    def _build_caregiver_response_ar(
        self,
        intake: ClinicalIntakeSchema,
        rule_eval: RuleEngineEvaluation,
    ) -> str:
        """
        Generates an empathetic, clear, non-panicking but firm actionable Egyptian Arabic response
        for the caregiver (son/daughter) or patient.
        """
        if rule_eval.halt_for_clarification and rule_eval.clarification_prompt_ar:
            return rule_eval.clarification_prompt_ar

        meds_ar = (
            "، ".join(
                f"{m.brand_name} ({m.active_ingredient or 'مسكن'})"
                + (f" لمدة {m.duration_days} يوم" if m.duration_days else "")
                for m in rule_eval.normalized_medications
            )
            if rule_eval.normalized_medications
            else "غير محدد"
        )
        symptoms_ar = (
            "، ".join(s.symptom_Arabic for s in intake.symptoms)
            if intake.symptoms
            else "أعراض عامة / فحص وقائي"
        )

        lines: List[str] = []
        lines.append("🩺 **تقييم «صحتك AI» للسلامة الدوائية والفرز الطبي الذكي**")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("ألف سلامة على المريض، ربنا يطمنكم عليه ويبارك في صحته. احنا راجعنا التاريخ الدوائي والأعراض بدقة:")
        lines.append(f"• **الأدوية المرصودة:** {meds_ar}")
        lines.append(f"• **الأعراض الحالية:** {symptoms_ar}")
        lines.append("")

        if rule_eval.triggered_alerts:
            lines.append("🔍 **ربط الخيوط الطبية (ليه الأعراض دي مهمة مع الأدوية؟):**")
            for alert in rule_eval.triggered_alerts:
                lines.append(f"{alert.patient_warning_ar}")
                lines.append("")

            target_specialty = rule_eval.triggered_alerts[0].recommended_specialty_ar
            lines.append("⚡ **الخطوات العملية المطلوبة فوراً (بدون قلق بس بحسم):**")
            lines.append(
                "1️⃣ **إيقاف المسكن فوراً:** وقفوا أي مسكنات من عيلة (كتافلام / فولتارين / كيتوفان / بروفين / باي-ألكوفان) فوراً تحت إشراف الطبيب، وممنوع أخذ مسكن تاني لتسكين وجع الدراع أو الجسم."
            )
            lines.append(
                f"2️⃣ **التخصص الطبي المطلوب:** التوجه في أقرب وقت إلى **{target_specialty}** (ومعكم تقرير الطوارئ الـ PDF المرفق لتسليمه للدكتور)."
            )
            lines.append("3️⃣ **التحاليل والفحوصات العاجلة المطلوب عرضها على الطبيب:**")
            for lab in rule_eval.urgent_labs_consolidated:
                lines.append(f"   • {lab}")
        else:
            lines.append("✅ **التقييم المبدئي:**")
            lines.append(
                "لم نرصد تناول مسكنات عالية الخطورة لفترات طويلة، ولكن ننصح بمراجعة طبيب باطنة مختص لتقييم الأعراض بدقة وعدم تناول أي مسكنات بدون وصفة طبية."
            )

        lines.append("")
        lines.append("📄 **مرفق مع الرسالة:** ملف `Emergency_Triage_Report.pdf` جاهز للطباعة أو العرض من الموبايل مباشرة لدكتور الطوارئ أو الباطنة عشان يشوف ملخص الحالة في 5 ثواني.")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(
            "⚖️ *إخلاء مسؤولية طبي: نظام «صحتك AI» هو مساعد ذكي للفرز الطبي والتنبيه الدوائي المبكر (Clinical Triage & Safety Guardian) ولا يُعد بديلاً عن الكشف الطبي المباشر أو تشخيص الطبيب المعالج.*"
        )
        return "\n".join(lines)

    def _build_physician_payload_en(
        self,
        intake: ClinicalIntakeSchema,
        rule_eval: RuleEngineEvaluation,
    ) -> PhysicianEmergencyPayload:
        """
        Builds the structured English Emergency Triage Summary designed for an overworked
        Egyptian ER physician to read in 5 seconds, grounded in KDIGO / NICE AKI guidelines.
        """
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
        for m in rule_eval.normalized_medications:
            dur_txt = f"Duration: {m.duration_days} days" if m.duration_days is not None else "Duration: Unknown"
            freq_txt = f" | Freq: {m.frequency}" if m.frequency else ""
            med_exposures.append(
                f"{m.brand_name} ({m.active_ingredient or 'Active ingredient N/A'}) [{m.drug_class.value}] — {dur_txt}{freq_txt}"
            )

        if rule_eval.triggered_alerts:
            correlations = " || ".join(a.clinical_finding_en for a in rule_eval.triggered_alerts)
        else:
            correlations = (
                "No deterministic high-risk NSAID nephrotoxicity or Triple-Whammy interaction triggered. "
                "Evaluate presenting symptoms clinically."
            )

        workup = list(rule_eval.urgent_labs_consolidated) if rule_eval.urgent_labs_consolidated else list(MANDATORY_RENAL_LAB_PANEL)
        workup.append("Immediate discontinuation of offending NSAIDs / nephrotoxic agents (KDIGO Stage 1-3 AKI Prevention)")
        workup.append("Clinical volume status assessment, strict urine output charting, and BP monitoring")

        badge = (
            "[URGENT TRIAGE ALERT - RENAL/DRUG SAFETY]"
            if rule_eval.overall_severity in {TriageSeverity.CRITICAL_RED_FLAG, TriageSeverity.HIGH_RISK_WARNING}
            else "[CLINICAL TRIAGE SUMMARY - MEDICATION SAFETY]"
        )

        return PhysicianEmergencyPayload(
            report_id=report_id,
            severity_badge=badge,
            patient_demographics=demographics,
            chief_complaint_and_timeline=chief_complaint,
            high_risk_medication_exposure=med_exposures,
            red_flag_clinical_correlation=correlations,
            recommended_immediate_workup=workup,
            clinical_guidelines_reference=(
                "KDIGO Clinical Practice Guideline for Acute Kidney Injury (AKI); "
                "NICE Guideline NG148 (Acute Kidney Injury: Prevention, Detection and Management); "
                "USPSTF Medication Safety Principles."
            ),
        )

    async def process_Session_and_orchestrate(
        self,
        session_id: str,
        incoming_intake: ClinicalIntakeSchema,
        channel: str = "telegram",
        raw_user_note: Optional[str] = None,
    ) -> TriageOrchestratorResult:
        """
        Merges new multimodal intake into the patient's session, runs the deterministic
        rules engine, and either halts for clarification (Rule 2) or generates the complete
        Arabic caregiver alert + English ER physician payload + monochrome PDF report.
        """
        existing_intake = await get_patient_intake_session(session_id)
        merged_intake = merge_clinical_intake(existing_intake, incoming_intake, raw_user_note)

        # Clean up internal duration placeholder if actual medications already exist
        real_meds = [m for m in merged_intake.medications if m.brand_name != "__DURATION_UPDATE__"]
        if real_meds:
            merged_intake.medications = real_meds

        # Run Deterministic Medical Rules Engine BEFORE finalizing output
        rule_eval = rules_engine.evaluate(merged_intake)

        if rule_eval.halt_for_clarification:
            await save_patient_intake_session(
                session_id=session_id,
                intake=merged_intake,
                channel=channel,
                raw_user_message=raw_user_note,
                status="clarification_needed",
            )
            caregiver_msg = self._build_caregiver_response_ar(merged_intake, rule_eval)
            return TriageOrchestratorResult(
                session_id=session_id,
                status="clarification_needed",
                severity=TriageSeverity.NEEDS_CLARIFICATION,
                caregiver_response_ar=caregiver_msg,
                physician_payload_en=None,
                pdf_report_path=None,
                intake_snapshot=merged_intake,
                rule_evaluation=rule_eval,
            )

        # Build both caregiver Arabic output and physician English ER summary
        caregiver_msg = self._build_caregiver_response_ar(merged_intake, rule_eval)
        physician_payload = self._build_physician_payload_en(merged_intake, rule_eval)

        arabic_pdf_lines = [
            f"1. إيقاف المسكنات فوراً (مثل كتافلام، فولتارين، كيتوفان، بروفين) تحت إشراف الطبيب.",
            f"2. التوجه إلى طوارئ الباطنة والكلى لتقييم وظائف الكلى والاطمئنان على المريض.",
            f"3. التحاليل المطلوبة بشكل عاجل: كرياتينين (Serum Creatinine)، بولينا (BUN)، معدل الفلترة (eGFR)، بوتاسيوم وصوديوم بالدم، وتحليل بول كامل.",
            f"4. اعرض هذا التقرير الطبي مباشرة على طبيب الطوارئ المختص فور الوصول.",
        ]

        pdf_path = pdf_generator.generate_triage_pdf(
            physician_payload=physician_payload,
            arabic_family_instructions=arabic_pdf_lines,
            rule_eval=rule_eval,
        )
        pdf_b64: Optional[str] = None
        try:
            pdf_bytes = Path(pdf_path).read_bytes()
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        except Exception:
            pdf_b64 = None

        await save_patient_intake_session(
            session_id=session_id,
            intake=merged_intake,
            channel=channel,
            raw_user_message=raw_user_note,
            status="completed",
        )

        return TriageOrchestratorResult(
            session_id=session_id,
            status="completed",
            severity=rule_eval.overall_severity,
            caregiver_response_ar=caregiver_msg,
            physician_payload_en=physician_payload,
            pdf_report_path=pdf_path,
            pdf_base64=pdf_b64,
            intake_snapshot=merged_intake,
            rule_evaluation=rule_eval,
        )

    async def handle_text_message(
        self,
        session_id: str,
        text: str,
        channel: str = "telegram",
        reset: bool = False,
    ) -> TriageOrchestratorResult:
        """End-to-end entry point for text input from any channel."""
        if reset:
            await reset_patient_session(session_id)
        extracted = await multimodal_intake.extract_from_text(text)
        return await self.process_Session_and_orchestrate(
            session_id=session_id,
            incoming_intake=extracted,
            channel=channel,
            raw_user_note=text,
        )

    async def handle_voice_message(
        self,
        session_id: str,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        channel: str = "telegram",
    ) -> TriageOrchestratorResult:
        """End-to-end entry point for Egyptian Arabic voice note input."""
        extracted = await multimodal_intake.extract_from_voice(audio_bytes, mime_type=mime_type)
        return await self.process_Session_and_orchestrate(
            session_id=session_id,
            incoming_intake=extracted,
            channel=channel,
            raw_user_note=extracted.raw_transcript_or_notes or "[Voice Note]",
        )

    async def handle_image_message(
        self,
        session_id: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        caption: Optional[str] = None,
        channel: str = "telegram",
    ) -> TriageOrchestratorResult:
        """End-to-end entry point for medication box / blister strip photo input."""
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
        )


triage_orchestrator = TriageOrchestrator()
