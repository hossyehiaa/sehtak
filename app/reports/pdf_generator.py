"""
Module 4: Monochrome Emergency Triage Sheet Generator (`app/reports/pdf_generator.py`)
Generates two distinct, high-contrast Monochrome (Black & White) 1-page PDF reports:
1. Arabic Hospital Triage Report (`Emergency_Triage_Report_AR_<ID>.pdf`) with trusted evidence citations.
2. English Physician Triage Report (`Emergency_Triage_Report_EN_<ID>.pdf`) with EVIDENCE BASIS block.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from fpdf import FPDF

from app.config import settings
from app.schemas.clinical_schemas import (
    ClinicalIntakeSchema,
    PhysicianEmergencyPayload,
    RuleEngineEvaluation,
)

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_SHAPING = True
except ImportError:
    HAS_ARABIC_SHAPING = False


BUNDLED_FONT_DIR = Path(__file__).resolve().parent / "fonts"

CANDIDATE_REGULAR_FONTS = [
    str(BUNDLED_FONT_DIR / "Tahoma.ttf"),
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

CANDIDATE_BOLD_FONTS = [
    str(BUNDLED_FONT_DIR / "Tahoma-Bold.ttf"),
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    str(BUNDLED_FONT_DIR / "Tahoma.ttf"),
]


def shape_arabic_line(text: str) -> str:
    if not text:
        return ""
    if HAS_ARABIC_SHAPING:
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text
    return text


def wrap_and_shape_arabic_paragraph(text: str, max_chars_per_line: int = 82) -> List[str]:
    if not text:
        return []
    visual_lines: List[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.strip().split()
        if not words:
            continue
        current_words: List[str] = []
        current_len = 0
        for w in words:
            if current_len + len(w) + 1 > max_chars_per_line and current_words:
                visual_lines.append(shape_arabic_line(" ".join(current_words)))
                current_words = [w]
                current_len = len(w)
            else:
                current_words.append(w)
                current_len += len(w) + 1
        if current_words:
            visual_lines.append(shape_arabic_line(" ".join(current_words)))
    return visual_lines


def find_unicode_fonts() -> Tuple[Optional[str], Optional[str]]:
    reg = next((p for p in CANDIDATE_REGULAR_FONTS if os.path.exists(p)), None)
    bold = next((p for p in CANDIDATE_BOLD_FONTS if os.path.exists(p)), reg)
    return reg, bold


class MonochromeTriagePDF(FPDF):
    banner_title: str = "SEHTAK AI  |  EMERGENCY MEDICAL TRIAGE & DRUG SAFETY SUMMARY"

    def header(self) -> None:
        self.set_fill_color(0, 0, 0)
        self.set_text_color(255, 255, 255)
        self.set_font(getattr(self, "_active_font_family", "Helvetica"), "B", 11.5)
        self.cell(
            0,
            10,
            self.banner_title,
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
            fill=True,
        )
        self.ln(2)


class EmergencyPDFGenerator:
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir or settings.REPORTS_OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def _configure_pdf_fonts(self, pdf: MonochromeTriagePDF) -> Tuple[str, bool]:
        reg_path, bold_path = find_unicode_fonts()
        if reg_path:
            try:
                pdf.add_font("UniFont", "", reg_path)
                pdf.add_font("UniFont", "B", bold_path or reg_path)
                pdf._active_font_family = "UniFont"
                return "UniFont", True
            except Exception:
                pass
        pdf._active_font_family = "Helvetica"
        return "Helvetica", False

    def generate_triage_pdf(
        self,
        physician_payload: PhysicianEmergencyPayload,
        arabic_family_instructions: Optional[List[str]] = None,
        rule_eval: Optional[RuleEngineEvaluation] = None,
        filename: Optional[str] = None,
    ) -> str:
        pdf = MonochromeTriagePDF(orientation="P", unit="mm", format="A4")
        pdf.banner_title = "SEHTAK AI  |  EMERGENCY MEDICAL TRIAGE & DRUG SAFETY SUMMARY (ENGLISH)"
        pdf.set_auto_page_break(auto=True, margin=12)
        font_family, has_unicode_font = self._configure_pdf_fonts(pdf)
        pdf.add_page()

        pdf.set_text_color(0, 0, 0)
        pdf.set_font(font_family, "B", 11)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(
            0,
            8,
            physician_payload.severity_badge,
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
            fill=True,
        )

        pdf.set_font(font_family, "", 8.5)
        pdf.cell(95, 6, f"Report ID: {physician_payload.report_id}", border="B", new_x="RIGHT", new_y="TOP")
        pdf.cell(95, 6, f"Timestamp: {physician_payload.generated_at_utc}", border="B", new_x="LMARGIN", new_y="NEXT", align="R")
        pdf.ln(3)

        self._render_section_header(pdf, font_family, "1. PATIENT DEMOGRAPHICS & CHIEF COMPLAINT", align="L")
        pdf.set_font(font_family, "B", 9.5)
        pdf.multi_cell(0, 5.5, f"Patient Profile: {self._english_only(physician_payload.patient_demographics)}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_family, "", 9.5)
        pdf.multi_cell(0, 5.5, f"Chief Complaint & Timeline: {self._english_only(physician_payload.chief_complaint_and_timeline)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2.5)

        self._render_section_header(pdf, font_family, "2. HIGH-RISK MEDICATION EXPOSURE (PHARMACOVIGILANCE ALERT)", align="L")
        pdf.set_font(font_family, "B", 9.5)
        if physician_payload.high_risk_medication_exposure:
            for med_line in physician_payload.high_risk_medication_exposure:
                pdf.multi_cell(0, 5.5, f"  [ALERT] {self._english_only(med_line)}", border="L", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.multi_cell(0, 5.5, "  No high-risk OTC/NSAID exposure explicitly confirmed.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2.5)

        self._render_section_header(pdf, font_family, "3. RED-FLAG CLINICAL CORRELATION & PATHOPHYSIOLOGY", align="L")
        pdf.set_font(font_family, "", 9.5)
        pdf.multi_cell(
            0,
            5.5,
            self._english_only(physician_payload.red_flag_clinical_correlation),
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2.5)

        is_renal_alert = bool(rule_eval and rule_eval.triggered_alerts)
        workup_header = "4. RECOMMENDED STAT WORKUP (KDIGO / NICE AKI PROTOCOL)" if is_renal_alert else "4. RECOMMENDED CLINICAL WORKUP & INITIAL TRIAGE"
        self._render_section_header(pdf, font_family, workup_header, align="L")
        pdf.set_font(font_family, "B", 9)
        for idx, item in enumerate(physician_payload.recommended_immediate_workup, start=1):
            clean_item = self._english_only(item)
            pdf.multi_cell(0, 5.2, f"  [ ] {idx}. {clean_item}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Task 2.12: EVIDENCE BASIS Block
        if physician_payload.evidence_basis:
            self._render_section_header(pdf, font_family, "5. EVIDENCE BASIS (TRUSTED CLINICAL SOURCES)", align="L")
            pdf.set_font(font_family, "", 8)
            for ev in physician_payload.evidence_basis:
                pdf.multi_cell(0, 4.5, f"  - {self._english_only(ev)}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        pdf.set_font(font_family, "", 8)
        pdf.multi_cell(0, 4.5, f"Guideline Reference: {physician_payload.clinical_guidelines_reference}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_font(font_family, "B", 8)
        pdf.multi_cell(
            0,
            4.8,
            "CLINICAL DISCLAIMER: AI-assisted clinical decision-support summary generated by Sehtak AI to support emergency evaluation. NOT a formal diagnosis or prescription. Requires verification by an attending human clinician.",
            border="T",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )

        out_name = filename or f"Emergency_Triage_Report_EN_{physician_payload.report_id}.pdf"
        out_path = self.output_dir / out_name
        pdf.output(str(out_path))
        return str(out_path.resolve())

    def generate_arabic_hospital_pdf(
        self,
        physician_payload: PhysicianEmergencyPayload,
        intake: ClinicalIntakeSchema,
        rule_eval: RuleEngineEvaluation,
        arabic_family_instructions: Optional[List[str]] = None,
        filename: Optional[str] = None,
    ) -> str:
        pdf = MonochromeTriagePDF(orientation="P", unit="mm", format="A4")
        font_family, has_unicode_font = self._configure_pdf_fonts(pdf)
        if has_unicode_font:
            pdf.banner_title = shape_arabic_line("صحتك AI  |  تقرير الفرز الطبي واليقظة الدوائية العاجل للمستشفى والطوارئ")
        else:
            pdf.banner_title = "SEHTAK AI  |  HOSPITAL EMERGENCY TRIAGE REPORT (ARABIC EDITION)"

        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()

        if not has_unicode_font:
            return self.generate_triage_pdf(physician_payload, arabic_family_instructions, rule_eval, filename)

        severity_ar = (
            "تنبيه طوارئ عاجل: اشتباه مضاعفات دوائية حادة على الكلى / المعدة"
            if rule_eval.overall_severity.value in {"CRITICAL_RED_FLAG", "HIGH_RISK_WARNING"}
            else "تقرير فرز طبي ومراجعة سلامة دوائية"
        )
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(230, 230, 230)
        pdf.set_font(font_family, "B", 11)
        pdf.cell(0, 8, shape_arabic_line(severity_ar), border=1, new_x="LMARGIN", new_y="NEXT", align="C", fill=True)

        pdf.set_font(font_family, "", 8.5)
        pdf.cell(95, 6, f"Timestamp: {physician_payload.generated_at_utc}", border="B", new_x="RIGHT", new_y="TOP", align="L")
        pdf.cell(
            95,
            6,
            shape_arabic_line(f"رقم التقرير الطبي: {physician_payload.report_id}"),
            border="B",
            new_x="LMARGIN",
            new_y="NEXT",
            align="R",
        )
        pdf.ln(3)

        # 1. بيانات المريض
        self._render_arabic_section_header(pdf, font_family, "أولاً: بيانات المريض والشكوى الطبية الحالية")
        age_txt = f"{intake.patient_age} سنة" if intake.patient_age else "غير محدد"
        sex_txt = "أنثى" if intake.patient_sex == "female" else ("ذكر" if intake.patient_sex == "male" else "غير محدد")
        chronic_txt = "، ".join(intake.chronic_conditions) if intake.chronic_conditions else "لا توجد أمراض مزمنة مسجلة"
        symptoms_txt = "، ".join(s.symptom_Arabic for s in intake.symptoms) if intake.symptoms else "فحص سلامة دوائية"

        self._write_arabic_paragraph(pdf, font_family, f"• السن والنوع: {age_txt} ({sex_txt})   |   التاريخ المرضي المزمن: {chronic_txt}", bold=True)
        self._write_arabic_paragraph(pdf, font_family, f"• الأعراض الإكلينيكية المرصودة: {symptoms_txt}", bold=False)
        pdf.ln(2)

        # 2. الأدوية
        self._render_arabic_section_header(pdf, font_family, "ثانياً: الأدوية المرصودة عالية الخطورة ومدة التعرض")
        if rule_eval.normalized_medications:
            for med in rule_eval.normalized_medications:
                dur_ar = f"لمدة {med.duration_days} يوم" if med.duration_days else "المدة غير محددة"
                med_line = f"• الدواء: {med.brand_name} ({med.active_ingredient or ''}) — التصنيف: {med.drug_class.value} — {dur_ar}"
                self._write_arabic_paragraph(pdf, font_family, med_line, bold=True)
        else:
            self._write_arabic_paragraph(pdf, font_family, "• لم يتم رصد مسكنات أو أدوية عالية الخطورة.", bold=False)
        pdf.ln(2)

        # 3. التفسير الإكلينيكي
        self._render_arabic_section_header(pdf, font_family, "ثالثاً: التفسير الإكلينيكي والربط الطبي (اليقظة الدوائية)")
        if rule_eval.triggered_alerts:
            for alert in rule_eval.triggered_alerts:
                clean_warn = alert.patient_warning_ar.replace("**", "")
                self._write_arabic_paragraph(pdf, font_family, f"• {clean_warn}", bold=False)
        else:
            self._write_arabic_paragraph(pdf, font_family, "• لا توجد تفاعلات دوائية حرجة مرصودة حالياً، يُنصح بالفحص الإكلينيكي المعتاد.", bold=False)
        pdf.ln(2)

        # 4. الفحوصات والتحاليل
        self._render_arabic_section_header(pdf, font_family, "رابعاً: التحاليل والفحوصات الطبية العاجلة المطلوبة فوراً")
        labs_to_render = rule_eval.urgent_labs_consolidated or [
            "Serum Creatinine (تحليل كرياتينين في الدم)",
            "BUN - Blood Urea Nitrogen (بولينا في الدم)",
            "Estimated GFR - eGFR (معدل الفلترة الكلوية)",
            "Serum Electrolytes: K+ & Na+ (بوتاسيوم وصوديوم بالدم)",
            "Complete Urine Analysis (تحليل بول كامل)",
        ]
        for idx, lab in enumerate(labs_to_render, start=1):
            self._write_arabic_paragraph(pdf, font_family, f"{idx}. {lab}", bold=True)
        pdf.ln(2)

        # 5. التوصيات والمصادر الموثقة
        self._render_arabic_section_header(pdf, font_family, "خامساً: التوصيات الطبية والمصادر الموثقة")
        instructions = arabic_family_instructions or [
            "1. الإيقاف الفوري لأي مسكنات من مجموعة (NSAIDs مثل كتافلام، فولتارين، كيتوفان، بروفين) تحت إشراف الطبيب.",
            "2. العرض العاجل على طبيب طوارئ الباطنة والكلى بالمستشفى.",
            "3. المصادر الإكلينيكية المعتمدة: منظمة الصحة العالمية (WHO) وهيئة الدواء المصرية (EDA) وإرشادات KDIGO 2012.",
        ]
        for line in instructions:
            self._write_arabic_paragraph(pdf, font_family, line, bold=False)

        pdf.ln(3)
        pdf.set_font(font_family, "B", 8)
        disclaimer = shape_arabic_line(
            "إخلاء مسؤولية طبي: ملخص فرز طبي ويقظة دوائية مدعوم بالذكاء الاصطناعي (AI-Assisted) لمعاونة أطقم الطوارئ والباطنة — لا يُعد تشخيصاً نهائياً أو وصفة علاجية، ويستلزم تقييم الطبيب البشري المعالج."
        )
        pdf.cell(0, 6, disclaimer, border="T", new_x="LMARGIN", new_y="NEXT", align="C")

        out_name = filename or f"Emergency_Triage_Report_AR_{physician_payload.report_id}.pdf"
        out_path = self.output_dir / out_name
        pdf.output(str(out_path))
        return str(out_path.resolve())

    @staticmethod
    def _render_section_header(pdf: FPDF, font_family: str, title: str, align: str = "L") -> None:
        pdf.set_fill_color(238, 238, 238)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(font_family, "B", 9.5)
        pdf.cell(0, 6.5, f" {title}", border=1, new_x="LMARGIN", new_y="NEXT", align=align, fill=True)

    @staticmethod
    def _render_arabic_section_header(pdf: FPDF, font_family: str, title_ar: str) -> None:
        pdf.set_fill_color(238, 238, 238)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(font_family, "B", 10)
        pdf.cell(0, 6.8, shape_arabic_line(title_ar), border=1, new_x="LMARGIN", new_y="NEXT", align="R", fill=True)

    @staticmethod
    def _write_arabic_paragraph(pdf: FPDF, font_family: str, text_ar: str, bold: bool = False) -> None:
        pdf.set_font(font_family, "B" if bold else "", 9.2)
        for shaped_line in wrap_and_shape_arabic_paragraph(text_ar, max_chars_per_line=84):
            pdf.cell(0, 5.5, shaped_line, new_x="LMARGIN", new_y="NEXT", align="R")

    @staticmethod
    def _english_only(text: str) -> str:
        if "(" in text and any("\u0600" <= ch <= "\u06FF" for ch in text):
            cleaned_parts = []
            for segment in text.split("("):
                if ")" in segment:
                    inside, rest = segment.split(")", 1)
                    if any("\u0600" <= ch <= "\u06FF" for ch in inside):
                        cleaned_parts.append(rest.strip())
                    else:
                        cleaned_parts.append(f"({inside}){rest}")
                else:
                    cleaned_parts.append(segment)
            result = " ".join(p for p in cleaned_parts if p).strip()
            return result if result else text.encode("latin-1", errors="ignore").decode("latin-1")
        return text.encode("latin-1", errors="ignore").decode("latin-1")


pdf_generator = EmergencyPDFGenerator()
