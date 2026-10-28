"""Pydantic v2 clinical, authentication, and RAG grounding schemas for Sehtak AI v2."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DrugClass(str, Enum):
    NSAID = "NSAID"
    CORTICOSTEROID = "CORTICOSTEROID"
    ANTICOAGULANT = "ANTICOAGULANT"
    ANTIPLATELET = "ANTIPLATELET"
    ACE_INHIBITOR = "ACE_INHIBITOR"
    ARB = "ARB"
    DIURETIC = "DIURETIC"
    PPI = "PPI"
    VITAMIN_SUPPLEMENT = "VITAMIN_SUPPLEMENT"
    ANTIBIOTIC = "ANTIBIOTIC"
    ANALGESIC = "ANALGESIC"
    ANTIHISTAMINE = "ANTIHISTAMINE"
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


class GoogleAuthRequest(BaseModel):
    credential: Optional[str] = Field(default=None, description="Google ID Token JWT")
    email: Optional[str] = Field(default=None, description="Direct email if verified")
    name: Optional[str] = Field(default=None, description="User full name from Google")
    google_id: Optional[str] = Field(default=None, description="Google user unique ID")
    role: UserRole = Field(default=UserRole.CAREGIVER, description="Role")


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

    @field_validator("drug_class", mode="before")
    @classmethod
    def normalize_drug_class(cls, v):
        if not v:
            return DrugClass.OTHER
        if isinstance(v, DrugClass):
            return v
        if isinstance(v, str):
            clean = v.strip().upper()
            if any(term in clean for term in ["SUPPLEMENT", "VITAMIN", "MINERAL", "DIETARY", "فيتامين", "مكمل", "MULTIVITAMIN"]):
                return DrugClass.VITAMIN_SUPPLEMENT
            if any(term in clean for term in ["NSAID", "NONSTEROIDAL", "NON-STEROIDAL", "مسكن", "ANALGESIC_ANTI_INFLAMMATORY"]):
                return DrugClass.NSAID
            if any(term in clean for term in ["CORTICO", "STEROID", "كورتيزون"]):
                return DrugClass.CORTICOSTEROID
            if any(term in clean for term in ["ANTIBIOTIC", "ANTIBACTERIAL", "مضاد حيوي", "ANTIMICROBIAL"]):
                return DrugClass.ANTIBIOTIC
            if any(term in clean for term in ["ANTICOAG", "مضاد تجلط"]):
                return DrugClass.ANTICOAGULANT
            if any(term in clean for term in ["ANTIPLATELET", "صفائح"]):
                return DrugClass.ANTIPLATELET
            if any(term in clean for term in ["ACE_INHIBITOR", "ACEI", "ACE INHIBITOR"]):
                return DrugClass.ACE_INHIBITOR
            if any(term in clean for term in ["ARB", "ANGIOTENSIN"]):
                return DrugClass.ARB
            if any(term in clean for term in ["DIURETIC", "مدر"]):
                return DrugClass.DIURETIC
            if any(term in clean for term in ["PPI", "PROTON", "معدة", "حموضة"]):
                return DrugClass.PPI
            if any(term in clean for term in ["ANALGESIC", "PARACETAMOL", "باراسيتامول", "خافض حرارة"]):
                return DrugClass.ANALGESIC
            if any(term in clean for term in ["ANTIHISTAMINE", "حساسية"]):
                return DrugClass.ANTIHISTAMINE
            for dc in DrugClass:
                if dc.value == clean:
                    return dc
        return DrugClass.OTHER


class ExtractedSymptom(BaseModel):
    symptom_Arabic: str = "عرض غير محدد"
    clinical_term_English: str = "Unspecified symptom"
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

    @field_validator("patient_age", mode="before")
    @classmethod
    def normalize_patient_age(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            import re
            digits = re.findall(r"\d+", v)
            return int(digits[0]) if digits else None
        return None

    @field_validator("symptoms", mode="before")
    @classmethod
    def normalize_symptoms(cls, v):
        if not v:
            return []
        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append({
                    "symptom_Arabic": item,
                    "clinical_term_English": item,
                    "severity": "moderate",
                    "is_systemic_or_vague": True,
                })
            elif isinstance(item, dict):
                ar = item.get("symptom_Arabic") or item.get("clinical_term_English") or item.get("name") or "عرض غير محدد"
                en = item.get("clinical_term_English") or item.get("symptom_Arabic") or item.get("name") or "Unspecified symptom"
                item_copy = dict(item)
                item_copy["symptom_Arabic"] = ar
                item_copy["clinical_term_English"] = en
                normalized.append(item_copy)
            else:
                normalized.append(item)
        return normalized

    @field_validator("medications", mode="before")
    @classmethod
    def normalize_medications(cls, v):
        if not v:
            return []
        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append({
                    "brand_name": item,
                    "drug_class": "OTHER",
                })
            elif isinstance(item, dict):
                item_copy = dict(item)
                brand = item_copy.get("brand_name") or item_copy.get("name") or item_copy.get("medication") or item_copy.get("active_ingredient") or "دواء غير محدد"
                item_copy["brand_name"] = brand
                dur = item_copy.get("duration_days")
                if isinstance(dur, str):
                    import re
                    digits = re.findall(r"\d+", dur)
                    item_copy["duration_days"] = int(digits[0]) if digits else None
                normalized.append(item_copy)
            else:
                normalized.append(item)
        return normalized

    @field_validator("chronic_conditions", mode="before")
    @classmethod
    def normalize_chronic_conditions(cls, v):
        if not v:
            return []
        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(item.get("name") or item.get("condition") or str(item))
            else:
                normalized.append(str(item))
        return normalized


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
