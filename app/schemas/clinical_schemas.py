"""Pydantic v2 clinical, authentication, and RAG grounding schemas for Sehtak AI v2."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
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


# ---------------------------------------------------------------------------
# Auth & User Schemas (Task 1)
# ---------------------------------------------------------------------------
class UserRole(str, Enum):
    CAREGIVER = "caregiver"
    PHYSICIAN = "physician"
    ADMIN = "admin"


class UserSignupRequest(BaseModel):
    email: str = Field(..., description="Unique email address")
    password: str = Field(..., min_length=6, description="User password")
    full_name: str = Field(..., description="Full name of user")
    phone: Optional[str] = Field(default=None, description="Optional phone number")
    role: UserRole = Field(default=UserRole.CAREGIVER, description="Role")


class UserLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserProfileResponse(BaseModel):
    id: int
    email: str
    full_name: str
    phone: Optional[str] = None
    role: str
    created_at: str


# ---------------------------------------------------------------------------
# RAG Grounding & Trusted Citation Schemas (Task 2)
# ---------------------------------------------------------------------------
class GroundedClaim(BaseModel):
    claim: str = Field(..., description="The clinical claim statement")
    evidence_quote: str = Field(..., description="Verbatim quote from trusted source")
    source_org: str = Field(..., description="WHO, EDA, KDIGO, NICE, or MOH")
    doc_title: str = Field(..., description="Document title")
    section_or_page: str = Field(..., description="Section name or page number")
    caveats: Optional[str] = Field(default=None, description="Relevant caveats or contraindications")
    is_verbatim_verified: bool = Field(default=True, description="True if verified against raw chunk")


class GroundedAnswer(BaseModel):
    claims: List[GroundedClaim] = Field(default_factory=list)
    overall_confidence: float = Field(default=1.0)
    refusal: Optional[str] = Field(default=None, description="Grounded refusal message if confidence is low")
    citations_formatted_ar: List[str] = Field(default_factory=list, description="Numbered Arabic citations [WHO - Doc, p.X]")
    citations_formatted_en: List[str] = Field(default_factory=list, description="Numbered English citations")


# ---------------------------------------------------------------------------
# Clinical Intake Schemas
# ---------------------------------------------------------------------------
class ExtractedMedication(BaseModel):
    brand_name: str
    active_ingredient: Optional[str] = None
    drug_class: DrugClass = DrugClass.OTHER
    dosage: Optional[str] = None
    duration_days: Optional[int] = None
    frequency: Optional[str] = None


class ExtractedSymptom(BaseModel):
    symptom_Arabic: str
    clinical_term_English: str
    duration_days: Optional[int] = None
    severity: SymptomSeverity = SymptomSeverity.MODERATE
    is_systemic_or_vague: bool = True


class ClinicalIntakeSchema(BaseModel):
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    chronic_conditions: List[str] = Field(default_factory=list)
    medications: List[ExtractedMedication] = Field(default_factory=list)
    symptoms: List[ExtractedSymptom] = Field(default_factory=list)
    missing_critical_info: bool = False
    clarification_question_ar: Optional[str] = None
    raw_transcript_or_notes: Optional[str] = None


class TriggeredRuleAlert(BaseModel):
    rule_id: str
    rule_name: str
    severity: TriageSeverity
    clinical_finding_en: str
    patient_warning_ar: str
    mandatory_action_en: str
    urgent_lab_panel: List[str] = Field(default_factory=list)
    recommended_specialty_ar: str = "طوارئ باطنة وكلى (Internal Medicine / Nephrology ER)"


class RuleEngineEvaluation(BaseModel):
    overall_severity: TriageSeverity = TriageSeverity.LOW_RISK
    halt_for_clarification: bool = False
    clarification_prompt_ar: Optional[str] = None
    triggered_alerts: List[TriggeredRuleAlert] = Field(default_factory=list)
    normalized_medications: List[ExtractedMedication] = Field(default_factory=list)
    urgent_labs_consolidated: List[str] = Field(default_factory=list)


class PhysicianEmergencyPayload(BaseModel):
    report_id: str
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    severity_badge: str = "[URGENT TRIAGE ALERT - RENAL/DRUG SAFETY]"
    patient_demographics: str
    chief_complaint_and_timeline: str
    high_risk_medication_exposure: List[str] = Field(default_factory=list)
    red_flag_clinical_correlation: str
    recommended_immediate_workup: List[str] = Field(default_factory=list)
    clinical_guidelines_reference: str = (
        "KDIGO Clinical Practice Guideline for Acute Kidney Injury (AKI) & NICE Acute Kidney Injury Guideline (NG148)"
    )
    evidence_basis: Optional[List[str]] = Field(default_factory=list, description="Verified citations for ER physician")


class ComponentDegradation(BaseModel):
    component: str = Field(..., description="e.g. voice_transcription, image_ocr, rag_grounding")
    reason: str = Field(..., description="Technical or operational reason for degradation")
    user_message_ar: Optional[str] = Field(default=None, description="Explicit Arabic degradation notice")


class TriageOrchestratorResult(BaseModel):
    session_id: str
    status: str
    severity: TriageSeverity
    caregiver_response_ar: str
    physician_payload_en: Optional[PhysicianEmergencyPayload] = None
    pdf_report_path: Optional[str] = None
    pdf_report_path_ar: Optional[str] = None
    pdf_base64: Optional[str] = None
    pdf_base64_ar: Optional[str] = None
    intake_snapshot: ClinicalIntakeSchema
    rule_evaluation: RuleEngineEvaluation
    grounded_evidence: Optional[GroundedAnswer] = None
    degradation: Optional[ComponentDegradation] = None


class TriageAnalyzeRequest(BaseModel):
    session_id: str = Field(default="web-default-session")
    text_input: Optional[str] = None
    reset_session: bool = False
