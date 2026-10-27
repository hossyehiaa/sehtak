"""
Module 1: Egyptian Drug Database & Deterministic Red-Flag Engine (`app/medical_engine/rules_engine.py`)
CRITICAL: Runs BEFORE and ALONGSIDE the LLM to enforce hard-coded medical safety guardrails.
"""

from typing import List
from app.database.egyptian_drugs_seed import (
    lookup_egyptian_drug,
    scan_text_for_known_drugs,
)
from app.schemas.clinical_schemas import (
    ClinicalIntakeSchema,
    DrugClass,
    ExtractedMedication,
    RuleEngineEvaluation,
    TriageSeverity,
    TriggeredRuleAlert,
)

# Keywords indicating systemic/vague symptoms often linked to silent uremia / AKI / GI bleeding / cardiac referral
SYSTEMIC_VAGUE_SYMPTOM_KEYWORDS = [
    # English clinical terms
    "fatigue", "lethargy", "malaise", "weakness", "generalized pain", "body ache",
    "arm pain", "upper limb pain", "shoulder pain", "back pain", "flank pain", "loin pain",
    "nausea", "vomiting", "anorexia", "loss of appetite", "oliguria", "reduced urine",
    "decreased urine", "anuria", "edema", "swelling", "lower limb swelling", "puffiness",
    "dyspnea", "shortness of breath", "breathlessness", "dizziness", "confusion", "pruritus",
    "epigastric pain", "pallor",
    # Egyptian Arabic expressions
    "همدان", "تعبان", "تعب عام", "إرهاق", "ارهاق", "دوخة", "وجع في الجسم", "تكسير في الجسم",
    "وجع في الدراع", "وجع دراع", "ذراع", "دراع", "كتف", "وجع في الضهر", "وجع ضهر", "اجناب", "جنب",
    "غممان نفس", "غثيان", "ترجيع", "قيء", "قلة البول", "بول قليل", "مبيجيبش بول", "حبس مية",
    "ورم", "رجل مورمة", "تورم", "نهجان", "كرشة نفس", "ضيق تنفس", "معدته", "فم المعدة",
]

MANDATORY_RENAL_LAB_PANEL = [
    "Serum Creatinine (تحليل كرياتينين في الدم)",
    "BUN - Blood Urea Nitrogen (بولينا في الدم)",
    "Estimated GFR - eGFR (معدل الفلترة الكلوية)",
    "Serum Electrolytes: Potassium K+ & Sodium Na+ (أملاح الدم: بوتاسيوم وصوديوم)",
    "Complete Urine Analysis & ACR (تحليل بول كامل ونسبة الزلال)",
]


class DeterministicRulesEngine:
    """
    Deterministic clinical safety rule engine tailored for the Egyptian healthcare context.
    Never relies solely on LLM probabilities for life-threatening drug complications.
    """

    CLARIFICATION_QUESTION_AR = (
        "مهم جداً تقولي: المريض بياخد المسكن ده بقاله كام يوم أو كام أسبوع؟"
    )

    def normalize_and_enrich_medications(
        self, intake: ClinicalIntakeSchema
    ) -> List[ExtractedMedication]:
        """
        Cross-references LLM-extracted medications and raw text against the deterministic
        Egyptian Drug Knowledge Base to ensure no NSAID, Corticosteroid, or Anticoagulant is missed.
        """
        normalized: List[ExtractedMedication] = []
        seen_brands: set[str] = set()

        for med in intake.medications:
            matched_record = lookup_egyptian_drug(med.brand_name) or (
                lookup_egyptian_drug(med.active_ingredient) if med.active_ingredient else None
            )
            if matched_record:
                enriched = ExtractedMedication(
                    brand_name=matched_record.canonical_brand,
                    active_ingredient=matched_record.active_ingredient,
                    drug_class=matched_record.drug_class,
                    dosage=med.dosage or (matched_record.common_dosages[0] if matched_record.common_dosages else None),
                    duration_days=med.duration_days,
                    frequency=med.frequency,
                )
                normalized.append(enriched)
                seen_brands.add(matched_record.canonical_brand.lower())
            else:
                normalized.append(med)
                seen_brands.add(med.brand_name.lower())

        # Also scan raw transcript/notes deterministically in case the LLM missed an Egyptian brand name
        if intake.raw_transcript_or_notes:
            text_matches = scan_text_for_known_drugs(intake.raw_transcript_or_notes)
            for rec in text_matches:
                if rec.canonical_brand.lower() not in seen_brands:
                    # Check if any existing normalized medication has a duration we can inherit
                    known_duration = next(
                        (m.duration_days for m in normalized if m.duration_days is not None),
                        None,
                    )
                    normalized.append(
                        ExtractedMedication(
                            brand_name=rec.canonical_brand,
                            active_ingredient=rec.active_ingredient,
                            drug_class=rec.drug_class,
                            dosage=rec.common_dosages[0] if rec.common_dosages else None,
                            duration_days=known_duration,
                            frequency=None,
                        )
                    )
                    seen_brands.add(rec.canonical_brand.lower())

        return normalized

    def _has_systemic_or_vague_symptoms(self, intake: ClinicalIntakeSchema) -> bool:
        """Checks if the patient presents with any systemic or vague symptom."""
        for sym in intake.symptoms:
            if sym.is_systemic_or_vague:
                return True
            combined = f"{sym.symptom_Arabic} {sym.clinical_term_English}".lower()
            if any(kw in combined for kw in SYSTEMIC_VAGUE_SYMPTOM_KEYWORDS):
                return True

        if intake.raw_transcript_or_notes:
            raw_low = intake.raw_transcript_or_notes.lower()
            if any(kw in raw_low for kw in SYSTEMIC_VAGUE_SYMPTOM_KEYWORDS):
                return True

        return False

    def evaluate(self, intake: ClinicalIntakeSchema) -> RuleEngineEvaluation:
        """
        Executes all deterministic medical safety rules on the patient intake state.
        Enforces:
        - Rule 2 (Missing Duration Guardrail)
        - Rule 1 (Silent Renal Danger: NSAID > 7 days + systemic/vague symptoms -> AKI Alert)
        - Rule 3 (Triple Whammy Nephrotoxicity: ACEi/ARB + Diuretic + NSAID)
        - Rule 4 (Anticoagulant/Antiplatelet + NSAID or Corticosteroid + NSAID GI Hemorrhage Alert)
        """
        normalized_meds = self.normalize_and_enrich_medications(intake)
        intake.medications = normalized_meds

        nsaid_meds = [m for m in normalized_meds if m.drug_class == DrugClass.NSAID]
        high_risk_meds = [
            m
            for m in normalized_meds
            if m.drug_class
            in {
                DrugClass.NSAID,
                DrugClass.CORTICOSTEROID,
                DrugClass.ANTICOAGULANT,
                DrugClass.ANTIPLATELET,
            }
        ]

        # =========================================================================
        # RULE 2: Missing Duration Guardrail
        # If an NSAID or high-risk drug is detected and duration_days is unknown/null
        # -> System MUST halt final triage and ask the clarifying question in Egyptian Arabic.
        # =========================================================================
        meds_missing_duration = [m for m in (nsaid_meds or high_risk_meds) if m.duration_days is None]
        if meds_missing_duration:
            drug_names_str = "، ".join(m.brand_name for m in meds_missing_duration)
            intake.missing_critical_info = True
            intake.clarification_question_ar = self.CLARIFICATION_QUESTION_AR
            return RuleEngineEvaluation(
                overall_severity=TriageSeverity.NEEDS_CLARIFICATION,
                halt_for_clarification=True,
                clarification_prompt_ar=(
                    f"⚠️ رصدنا إن المريض بياخد دواء عالي الخطورة على الكلى والمعدة ({drug_names_str}).\n\n"
                    f"🛑 **{self.CLARIFICATION_QUESTION_AR}**\n"
                    f"(وممكن تقولي كمان بياخد كام قرص أو كيس في اليوم، وهل عنده أي أعراض زي همدان، غممان نفس، أو وجع في الجسم؟)"
                ),
                triggered_alerts=[],
                normalized_medications=normalized_meds,
                urgent_labs_consolidated=[],
            )

        triggered_alerts: List[TriggeredRuleAlert] = []
        has_systemic_symptom = self._has_systemic_or_vague_symptoms(intake)

        # =========================================================================
        # RULE 1: Silent Renal Danger (NSAID > 7 days + Systemic/Vague Symptoms)
        # =========================================================================
        chronic_nsaids = [
            m for m in nsaid_meds if (m.duration_days is not None and m.duration_days > 7)
        ]
        if chronic_nsaids and (has_systemic_symptom or len(intake.symptoms) > 0):
            nsaid_summary_en = ", ".join(
                f"{m.brand_name} ({m.active_ingredient}) for {m.duration_days} days"
                for m in chronic_nsaids
            )
            nsaid_summary_ar = " و ".join(
                f"{m.brand_name} ({m.active_ingredient}) لمدة {m.duration_days} يوم"
                for m in chronic_nsaids
            )
            symptoms_summary_en = (
                ", ".join(s.clinical_term_English for s in intake.symptoms)
                if intake.symptoms
                else "Systemic/vague constitutional symptoms (fatigue, generalized/referred pain)"
            )

            # Check if arm/shoulder/chest pain is present to also flag cardiac/uremic differential
            has_upper_limb_pain = any(
                k in f"{s.symptom_Arabic} {s.clinical_term_English}".lower()
                for s in intake.symptoms
                for k in ["arm", "shoulder", "chest", "دراع", "ذراع", "كتف", "صدر"]
            )
            cardiac_addendum_en = (
                " Note: Unexplained arm/upper-body pain with NSAID use also warrants immediate 12-lead ECG & Troponin to rule out NSAID-associated acute coronary syndrome alongside uremic neuropathy/pericarditis."
                if has_upper_limb_pain
                else ""
            )
            labs_for_rule1 = list(MANDATORY_RENAL_LAB_PANEL)
            if has_upper_limb_pain:
                labs_for_rule1.append("12-Lead ECG & High-Sensitivity Troponin (رسم قلب وإنزيمات قلب لاستبعاد قصور الشريان التاجي)")

            triggered_alerts.append(
                TriggeredRuleAlert(
                    rule_id="RULE_01_SILENT_RENAL_DANGER",
                    rule_name="High Suspicion of Drug-Induced Nephrotoxicity / Acute Kidney Injury (AKI)",
                    severity=TriageSeverity.CRITICAL_RED_FLAG,
                    clinical_finding_en=(
                        f"CRITICAL RED FLAG: Prolonged NSAID exposure ({nsaid_summary_en}) combined with systemic/vague presentation "
                        f"({symptoms_summary_en}). High suspicion of NSAID-Induced Nephrotoxicity, Hemodynamic Acute Kidney Injury (AKI), "
                        f"Acute Interstitial Nephritis, or Silent Uremia.{cardiac_addendum_en}"
                    ),
                    patient_warning_ar=(
                        f"🚨 **تحذير طبي عاجل (خطر صامت على الكلى):**\n"
                        f"استخدام المسكن ({nsaid_summary_ar}) لفترة طويلة (أكتر من 7 أيام) مع ظهور أعراض عامة زي الهمدان أو وجع الدراع/الجسم أو غممان النفس "
                        f"هو علامة إنذار مبكرة لاحتمال وجود **إجهاد حاد أو قصور صامت في وظائف الكلى (Drug-Induced AKI)** بسبب المسكنات."
                    ),
                    mandatory_action_en=(
                        "1. IMMEDIATE CESSATION of all NSAIDs (Cataflam/Ketofan/Brufen/Voltaren) under medical supervision.\n"
                        "2. Urgent stat laboratory workup: Serum Creatinine, BUN, eGFR, Serum Electrolytes (K+, Na+), and Urine Analysis.\n"
                        "3. Strict avoidance of further nephrotoxins or IV contrast; assess volume status and blood pressure immediately."
                    ),
                    urgent_lab_panel=labs_for_rule1,
                    recommended_specialty_ar="طوارئ باطنة وكلى فوراً (Internal Medicine & Nephrology ER)",
                )
            )
        elif chronic_nsaids:
            # Chronic NSAID > 7 days even without explicit symptoms reported yet
            nsaid_summary_en = ", ".join(
                f"{m.brand_name} ({m.active_ingredient}) for {m.duration_days} days"
                for m in chronic_nsaids
            )
            triggered_alerts.append(
                TriggeredRuleAlert(
                    rule_id="RULE_01B_CHRONIC_NSAID_EXPOSURE",
                    rule_name="Prolonged NSAID Exposure (>7 Days) - Silent Renal & GI Hazard",
                    severity=TriageSeverity.HIGH_RISK_WARNING,
                    clinical_finding_en=(
                        f"Patient has been taking NSAIDs ({nsaid_summary_en}) beyond the safe 7-day threshold. "
                        f"High risk of subclinical renal function decline (AKI/CKD progression) and silent peptic ulceration."
                    ),
                    patient_warning_ar=(
                        "⚠️ **تنبيه هام بخصوص المسكنات:** المريض بياخد المسكن بقاله أكتر من أسبوع، وده بيحط الكلى وجدار المعدة في خطر صامت حتى لو لسه مظهرش أعراض شديدة."
                    ),
                    mandatory_action_en=(
                        "Discontinue chronic NSAID regimen under physician guidance and perform baseline renal function screening (Serum Creatinine, BUN, eGFR, Electrolytes, Urine Analysis)."
                    ),
                    urgent_lab_panel=list(MANDATORY_RENAL_LAB_PANEL),
                    recommended_specialty_ar="طبيب باطنة أو كلى (Internal Medicine / Nephrology)",
                )
            )

        # =========================================================================
        # RULE 3: "Triple Whammy" Acute Kidney Injury Risk (ACEi/ARB + Diuretic + NSAID)
        # =========================================================================
        ace_or_arb = [
            m for m in normalized_meds if m.drug_class in {DrugClass.ACE_INHIBITOR, DrugClass.ARB}
        ]
        diuretics = [m for m in normalized_meds if m.drug_class == DrugClass.DIURETIC]
        if nsaid_meds and (ace_or_arb or diuretics):
            combo_meds = ", ".join(m.brand_name for m in (nsaid_meds + ace_or_arb + diuretics))
            is_full_triple = bool(nsaid_meds and ace_or_arb and diuretics)
            rule_title = (
                "Lethal 'Triple Whammy' Nephrotoxic Combination (ACEi/ARB + Diuretic + NSAID)"
                if is_full_triple
                else "High-Risk Hemodynamic Renal Interaction (Antihypertensive/Diuretic + NSAID)"
            )
            triggered_alerts.append(
                TriggeredRuleAlert(
                    rule_id="RULE_03_TRIPLE_WHAMMY_AKI",
                    rule_name=rule_title,
                    severity=TriageSeverity.CRITICAL_RED_FLAG,
                    clinical_finding_en=(
                        f"Concurrent use of {combo_meds}. NSAIDs constrict the afferent renal arteriole while ACEi/ARBs dilate the efferent arteriole "
                        f"and diuretics reduce intravascular volume—precipitating rapid hemodynamic Acute Kidney Injury and life-threatening Hyperkalemia."
                    ),
                    patient_warning_ar=(
                        f"🚨 **تفاعل دوائي خطير جداً على الكلى ({combo_meds}):**\n"
                        f"الجمع بين أدوية الضغط/مدرات البول والمسكنات بيقفل ضغط الفلترة جوه الكلى تماماً (المعروف طبياً بـ Triple Whammy) وبيسبب قصور كلوي حاد وارتفاع خطير في بوتاسيوم الدم."
                    ),
                    mandatory_action_en=(
                        "STOP NSAID immediately. Check stat Serum Creatinine, eGFR, BUN, and Serum Potassium (K+). Monitor blood pressure and urine output closely."
                    ),
                    urgent_lab_panel=list(MANDATORY_RENAL_LAB_PANEL),
                    recommended_specialty_ar="طوارئ باطنة وكلى وقلب (Internal Medicine / Nephrology ER)",
                )
            )

        # =========================================================================
        # RULE 4: Severe GI Bleeding / Hemorrhage Risk (NSAID + Anticoagulant/Antiplatelet/Steroid)
        # =========================================================================
        blood_thinners = [
            m
            for m in normalized_meds
            if m.drug_class in {DrugClass.ANTICOAGULANT, DrugClass.ANTIPLATELET}
        ]
        steroids = [m for m in normalized_meds if m.drug_class == DrugClass.CORTICOSTEROID]
        if nsaid_meds and (blood_thinners or steroids):
            interacting = ", ".join(m.brand_name for m in (nsaid_meds + blood_thinners + steroids))
            triggered_alerts.append(
                TriggeredRuleAlert(
                    rule_id="RULE_04_HEMORRHAGE_GI_BLEED_RISK",
                    rule_name="High Risk of Major Gastrointestinal Hemorrhage & Ulcer Perforation",
                    severity=TriageSeverity.CRITICAL_RED_FLAG,
                    clinical_finding_en=(
                        f"Co-administration of NSAID with Anticoagulant/Antiplatelet/Corticosteroid ({interacting}). "
                        f"Synergistic mucosal injury and platelet/coagulation inhibition drastically elevate the risk of silent upper GI bleeding, anemia, and hemorrhagic shock."
                    ),
                    patient_warning_ar=(
                        f"🚨 **خطر نزيف معوي وتقرحات حادة ({interacting}):**\n"
                        f"أخد المسكنات مع أدوية السيولة أو الكورتيزون بيضاعف خطر نزيف المعدة الصامت والأنيميا الحادة."
                    ),
                    mandatory_action_en=(
                        "Discontinue NSAID immediately. Evaluate for occult/overt GI bleeding (CBC for Hemoglobin drop, Stool Occult Blood, Coagulation profile / INR if on Warfarin)."
                    ),
                    urgent_lab_panel=[
                        "Complete Blood Count - CBC (صورة دم كاملة لمتابعة الهيموجلوبين)",
                        "INR / Coagulation Profile (سيولة الدم)",
                        *MANDATORY_RENAL_LAB_PANEL[:3],
                    ],
                    recommended_specialty_ar="طوارئ باطنة وجهاز هضمي (Internal Medicine / Gastroenterology)",
                )
            )

        # Consolidate labs and overall severity
        consolidated_labs: List[str] = []
        for alert in triggered_alerts:
            for lab in alert.urgent_lab_panel:
                if lab not in consolidated_labs:
                    consolidated_labs.append(lab)

        overall_severity = TriageSeverity.LOW_RISK
        if any(a.severity == TriageSeverity.CRITICAL_RED_FLAG for a in triggered_alerts):
            overall_severity = TriageSeverity.CRITICAL_RED_FLAG
        elif any(a.severity == TriageSeverity.HIGH_RISK_WARNING for a in triggered_alerts):
            overall_severity = TriageSeverity.HIGH_RISK_WARNING
        elif normalized_meds or intake.symptoms:
            overall_severity = TriageSeverity.MODERATE_CAUTION

        intake.missing_critical_info = False
        intake.clarification_question_ar = None

        return RuleEngineEvaluation(
            overall_severity=overall_severity,
            halt_for_clarification=False,
            clarification_prompt_ar=None,
            triggered_alerts=triggered_alerts,
            normalized_medications=normalized_meds,
            urgent_labs_consolidated=consolidated_labs,
        )


rules_engine = DeterministicRulesEngine()
