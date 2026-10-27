"""Pydantic v2 clinical schemas for Sehtak AI."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DrugClass(str, Enum):
    NSAID = "NSAID"
    CORTICOSTEROID = "CORTICOSTEROID"
    ANTICOAGULANT = "ANTICOAGULANT"
    ANTIPLATELET = "ANTIPLATELET"
    ACE_INHIBITOR = "ACE_INHIBITOR"
    ARB = "ARB"
    DIURETIC = "DIURETIC"
    PPI = "PPI"
    OTHER = "OTHER"


class TriageSeverity(str, Enum):
    CRITICAL_RED_FLAG = "CRITICAL_RED_FLAG"
    HIGH_RISK_WARNING = "HIGH_RISK_WARNING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    MODERATE_CAUTION = "MODERATE_CAUTION"
    LOW_RISK = "LOW_RISK"


class SymptomSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ExtractedMedication(BaseModel):
    """Represents a medication extracted from text, voice note, or strip image."""

    brand_name: str = Field(..., description="Egyptian brand name or generic name as stated/seen, e.g., Cataflam, Ketofan")
    active_ingredient: Optional[str] = Field(default=None, description="Standard active ingredient, e.g., Diclofenac Potassium")
    drug_class: DrugClass = Field(default=DrugClass.OTHER, description="Pharmacological class of the medication")
    dosage: Optional[str] = Field(default=None, description="Strength or dosage, e.g., 50mg, 150mg")
    duration_days: Optional[int] = Field(
        default=None,
        description="Number of days the patient has been taking this medication. Null if not mentioned.",
    )
    frequency: Optional[str] = Field(default=None, description="How often the patient takes it, e.g., twice daily, PRN, chronic daily")


class ExtractedSymptom(BaseModel):
    """Represents a patient symptom in both Egyptian Arabic and standard English clinical terminology."""

    symptom_Arabic: str = Field(..., description="Symptom in Egyptian Arabic as expressed by patient/caregiver, e.g., وجع في الدراع، همدان")
    clinical_term_English: str = Field(
        ...,
        description="Standard clinical English equivalent, e.g., Referred upper limb pain, Generalized fatigue, Oliguria, Nausea",
    )
    duration_days: Optional[int] = Field(default=None, description="Duration of the symptom in days")
    severity: SymptomSeverity = Field(default=SymptomSeverity.MODERATE, description="Estimated clinical severity")
    is_systemic_or_vague: bool = Field(
        default=True,
        description="True if systemic/vague symptom (fatigue, generalized pain, arm/back pain, nausea, reduced urine, edema, dyspnea)",
    )


class ClinicalIntakeSchema(BaseModel):
    """Strict structured output from Multimodal AI Intake Pipeline."""

    patient_age: Optional[int] = Field(default=None, description="Patient age in years if mentioned")
    patient_sex: Optional[str] = Field(default=None, description="Patient biological sex ('male', 'female', or None)")
    chronic_conditions: List[str] = Field(
        default_factory=list,
        description="Chronic illnesses such as Hypertension (ضغط), Diabetes (سكر), CKD, Gastritis, Heart Failure",
    )
    medications: List[ExtractedMedication] = Field(
        default_factory=list,
        description="List of medications detected from voice, text, or drug strip photo",
    )
    symptoms: List[ExtractedSymptom] = Field(
        default_factory=list,
        description="List of symptoms experienced by the patient",
    )
    missing_critical_info: bool = Field(
        default=False,
        description="True if a high-risk drug (e.g., NSAID) is detected without duration_days or critical clinical context",
    )
    clarification_question_ar: Optional[str] = Field(
        default=None,
        description="Clarifying question in warm Egyptian Arabic if missing_critical_info is True",
    )
    raw_transcript_or_notes: Optional[str] = Field(
        default=None,
        description="Transcribed Egyptian Arabic voice note or OCR text from image",
    )


class TriggeredRuleAlert(BaseModel):
    """Deterministic safety rule alert emitted by the Rules Engine."""

    rule_id: str = Field(..., description="Unique identifier of the deterministic rule, e.g., RULE_01_SILENT_RENAL_DANGER")
    rule_name: str = Field(..., description="Human-readable name of the rule")
    severity: TriageSeverity = Field(..., description="Severity level assigned by the rule")
    clinical_finding_en: str = Field(..., description="Clinical explanation in English for the ER physician")
    patient_warning_ar: str = Field(..., description="Clear warning in Egyptian Arabic for the caregiver/patient")
    mandatory_action_en: str = Field(..., description="Mandatory clinical action & immediate drug cessation advice")
    urgent_lab_panel: List[str] = Field(default_factory=list, description="Required urgent laboratory workup")
    recommended_specialty_ar: str = Field(default="طوارئ باطنة وكلى (Internal Medicine / Nephrology ER)", description="Target specialty in Arabic")


class RuleEngineEvaluation(BaseModel):
    """Aggregated deterministic evaluation output."""

    overall_severity: TriageSeverity = Field(default=TriageSeverity.LOW_RISK)
    halt_for_clarification: bool = Field(default=False, description="True if Rule 2 (Missing Duration Guardrail) halted final triage")
    clarification_prompt_ar: Optional[str] = Field(default=None)
    triggered_alerts: List[TriggeredRuleAlert] = Field(default_factory=list)
    normalized_medications: List[ExtractedMedication] = Field(default_factory=list)
    urgent_labs_consolidated: List[str] = Field(default_factory=list)


class PhysicianEmergencyPayload(BaseModel):
    """Structured English clinical summary designed for an overworked ER physician to read in 5 seconds."""

    report_id: str = Field(..., description="Unique triage report identifier")
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    severity_badge: str = Field(default="[URGENT TRIAGE ALERT - RENAL/DRUG SAFETY]")
    patient_demographics: str = Field(..., description="Concise age/sex/comorbidities string")
    chief_complaint_and_timeline: str = Field(..., description="Bullet-speed summary of presenting symptoms and duration")
    high_risk_medication_exposure: List[str] = Field(
        default_factory=list,
        description="Highlighted high-risk drugs with active ingredients and exposure duration",
    )
    red_flag_clinical_correlation: str = Field(
        ...,
        description="Connecting the dots between chronic drug exposure and systemic symptoms (e.g., NSAID-Induced AKI / Uremia)",
    )
    recommended_immediate_workup: List[str] = Field(
        default_factory=list,
        description="Immediate laboratory and clinical investigations based on KDIGO/NICE guidelines",
    )
    clinical_guidelines_reference: str = Field(
        default="KDIGO Clinical Practice Guideline for Acute Kidney Injury (AKI) & NICE Acute Kidney Injury Guideline (NG148)"
    )


class TriageOrchestratorResult(BaseModel):
    """Complete output returned by the Triage Orchestrator to any channel (Telegram, WhatsApp, Web API)."""

    session_id: str
    status: str = Field(
        ...,
        description="'clarification_needed' if halted by Rule 2 guardrail, or 'completed' when full triage is ready",
    )
    severity: TriageSeverity
    caregiver_response_ar: str = Field(
        ...,
        description="Empathetic, firm Egyptian Arabic message for the family/patient (or clarification question if halted)",
    )
    physician_payload_en: Optional[PhysicianEmergencyPayload] = Field(
        default=None,
        description="Structured English ER physician summary (populated when status == 'completed')",
    )
    pdf_report_path: Optional[str] = Field(
        default=None,
        description="File system path to the generated monochrome 1-page English Emergency_Triage_Report_EN.pdf",
    )
    pdf_report_path_ar: Optional[str] = Field(
        default=None,
        description="File system path to the generated monochrome 1-page Arabic Emergency_Triage_Report_AR.pdf",
    )
    pdf_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded English PDF binary payload for instant browser viewing/downloading",
    )
    pdf_base64_ar: Optional[str] = Field(
        default=None,
        description="Base64-encoded Arabic PDF binary payload for instant hospital/family downloading",
    )
    intake_snapshot: ClinicalIntakeSchema
    rule_evaluation: RuleEngineEvaluation


class TriageAnalyzeRequest(BaseModel):
    """REST API request body for POST /api/v1/triage/analyze."""

    session_id: str = Field(default="web-default-session", description="Unique user or patient session identifier")
    text_input: Optional[str] = Field(default=None, description="Patient story or medication duration answer in Arabic or English")
    reset_session: bool = Field(default=False, description="Whether to clear previous session state before analyzing")
