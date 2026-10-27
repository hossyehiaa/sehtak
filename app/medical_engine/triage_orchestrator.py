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

    def _build_caregiver_response_ar(
        self,
        intake: ClinicalIntakeSchema,
        rule_eval: RuleEngineEvaluation,
        grounded_evidence: Optional[GroundedAnswer] = None,
    ) -> str:
        """
        Generates an empathetic, clear, non-panicking, firm, emoji-free Egyptian Arabic response
        for the caregiver or patient, appending numbered trusted-source citations (Task 2.12).
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
        lines.append("**تقييم «صحتك AI» للسلامة الدوائية والفرز الطبي**")
        lines.append("--------------------------------------------------")
        lines.append("تمت مراجعة التاريخ الدوائي والأعراض الإكلينيكية المسجلة للمريض بدقة:")
        lines.append(f"- **الأدوية المرصودة:** {meds_ar}")
        lines.append(f"- **الأعراض الحالية:** {symptoms_ar}")
        lines.append("")

        if rule_eval.triggered_alerts:
            lines.append("**التفسير الإكلينيكي وربط الأعراض بالأدوية:**")
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
        else:
            lines.append("**التقييم المبدئي:**")
            lines.append(
                "لم يتم رصد استخدام مسكنات عالية الخطورة لفترات مطولة، ولكن يُنصح بمراجعة طبيب باطنة مختص لتقييم الأعراض بدقة وتجنب تناول أي مسكنات دون وصفة طبية."
            )

        # Append Grounded Citations per Task 2.12 & Task 2.13
        lines.append("")
        if grounded_evidence and grounded_evidence.citations_formatted_ar:
            lines.append("**المصادر والأدلة الإكلينيكية الموثقة:**")
            for cite in grounded_evidence.citations_formatted_ar:
                lines.append(f"- {cite}")
        elif grounded_evidence and grounded_evidence.refusal:
            lines.append(f"*{grounded_evidence.refusal}*")
        else:
            lines.append("**المصادر:** المصادر الموثقة غير متاحة حالياً — الإنذارات صادرة من محرك القواعد الطبي المعتمد.")

        lines.append("")
        lines.append("**التقارير الطبية المرفقة:** تم إصدار نسختين من تقرير الفرز الطبي للطوارئ (نسخة باللغة العربية للمستشفى والأسرة، ونسخة باللغة الإنجليزية لطبيب الطوارئ).")
        lines.append("--------------------------------------------------")
        lines.append(
            "*إخلاء مسؤولية طبي: نظام «صحتك AI» هو نظام دعم قرار وفرز طبي مبكر (Clinical Triage & Safety Guardian) ولا يُعد بديلاً عن الفحص الطبي المباشر أو تشخيص الطبيب المعالج.*"
        )
        return "\n".join(lines)

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

        evidence_basis: List[str] = []
        if grounded_evidence and grounded_evidence.claims:
            for c in grounded_evidence.claims:
                evidence_basis.append(f"{c.source_org} ({c.doc_title}, {c.section_or_page}): \"{c.evidence_quote}\"")

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

        real_meds = [m for m in merged_intake.medications if m.brand_name != "__DURATION_UPDATE__"]
        if real_meds:
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
            caregiver_msg = self._build_caregiver_response_ar(merged_intake, rule_eval)
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

        caregiver_msg = self._build_caregiver_response_ar(merged_intake, rule_eval, grounded_evidence)
        physician_payload = self._build_physician_payload_en(merged_intake, rule_eval, grounded_evidence)

        arabic_pdf_lines = [
            "1. إيقاف المسكنات فوراً (مثل كتافلام، فولتارين، كيتوفان، بروفين، باي-ألكوفان) تحت إشراف الطبيب.",
            "2. التوجه العاجل إلى قسم طوارئ الباطنة والكلى بالمستشفى لتقييم وظائف الكلى وضغط الدم.",
            "3. إجراء التحاليل المذكورة أعلاه بشكل عاجل وعرض هذا التقرير على الطبيب المعالج فور الوصول.",
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
