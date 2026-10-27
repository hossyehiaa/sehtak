"""
Module 4: Monochrome Emergency Triage Sheet Generator (`app/reports/pdf_generator.py`)
Generates a clean, ultra-minimalist Black & White 1-page PDF (`Emergency_Triage_Report.pdf`)
containing the English Physician Triage Summary on the top half, and key Arabic instructions
for the family on the bottom half, with a clear timestamp and severity badge:
`[URGENT TRIAGE ALERT - RENAL/DRUG SAFETY]`.
"""

import os
from pathlib import Path
from typing import List, Optional
from fpdf import FPDF

from app.config import settings
from app.schemas.clinical_schemas import PhysicianEmergencyPayload, RuleEngineEvaluation

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_SHAPING = True
except ImportError:
    HAS_ARABIC_SHAPING = False


# Candidate system Unicode fonts that support both English and Arabic glyphs on macOS, Linux, and Windows
CANDIDATE_UNICODE_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/GeezaPro.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
]


def shape_arabic_text(text: str) -> str:
    """Shapes and reorders Arabic RTL strings for PDF rendering while keeping ASCII fallback safe."""
    if not text:
        return ""
    if HAS_ARABIC_SHAPING:
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text
    return text


def find_unicode_font() -> Optional[str]:
    """Finds an installed TrueType font capable of rendering Arabic and Latin text."""
    for font_path in CANDIDATE_UNICODE_FONTS:
        if font_path.endswith(".ttf") and os.path.exists(font_path):
            return font_path
    return None


class MonochromeTriagePDF(FPDF):
    """Strict Black & White high-contrast ER Triage Summary sheet."""

    def header(self) -> None:
        # Top monochrome black banner
        self.set_fill_color(0, 0, 0)
        self.set_text_color(255, 255, 255)
        self.set_font(self._active_font_family, "B", 12)
        self.cell(
            0,
            10,
            "SEHTAK AI  |  EMERGENCY MEDICAL TRIAGE & DRUG SAFETY SUMMARY",
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
            fill=True,
        )
        self.ln(2)


class EmergencyPDFGenerator:
    """Generates a 1-page monochrome Emergency Triage PDF for ER physicians and patient caregivers."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir or settings.REPORTS_OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_triage_pdf(
        self,
        physician_payload: PhysicianEmergencyPayload,
        arabic_family_instructions: List[str],
        rule_eval: Optional[RuleEngineEvaluation] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Builds the 1-page Black & White Emergency Triage PDF and returns its absolute file path.
        """
        pdf = MonochromeTriagePDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=10)

        unicode_font_path = find_unicode_font()
        has_unicode_font = False
        if unicode_font_path:
            try:
                pdf.add_font("UniFont", "", unicode_font_path)
                pdf.add_font("UniFont", "B", unicode_font_path)
                pdf._active_font_family = "UniFont"
                has_unicode_font = True
            except Exception:
                pdf._active_font_family = "Helvetica"
        else:
            pdf._active_font_family = "Helvetica"

        pdf.add_page()
        font_family = pdf._active_font_family

        # ------------------------------------------------------------------
        # Metadata & High-Contrast Severity Badge
        # ------------------------------------------------------------------
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(font_family, "B", 11)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(
            0,
            8,
            f"{physician_payload.severity_badge}",
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
            fill=True,
        )

        pdf.set_font(font_family, "", 8.5)
        pdf.cell(
            95,
            6,
            f"Report ID: {physician_payload.report_id}",
            border="B",
            new_x="RIGHT",
            new_y="TOP",
        )
        pdf.cell(
            95,
            6,
            f"Timestamp: {physician_payload.generated_at_utc}",
            border="B",
            new_x="LMARGIN",
            new_y="NEXT",
            align="R",
        )
        pdf.ln(2)

        # ------------------------------------------------------------------
        # TOP HALF: PHYSICIAN EMERGENCY TRIAGE PAYLOAD (ENGLISH - 5-SEC READ)
        # ------------------------------------------------------------------
        self._render_section_header(pdf, font_family, "1. PATIENT DEMOGRAPHICS & CHIEF COMPLAINT (ER PHYSICIAN VIEW)")
        pdf.set_font(font_family, "B", 9)
        pdf.multi_cell(0, 5, f"Patient Profile: {self._safe_str(physician_payload.patient_demographics, has_unicode_font)}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_family, "", 9)
        pdf.multi_cell(0, 5, f"Chief Complaint & Timeline: {self._safe_str(physician_payload.chief_complaint_and_timeline, has_unicode_font)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)

        # Highlighted High-Risk Medication Exposure Box
        self._render_section_header(pdf, font_family, "2. HIGH-RISK MEDICATION EXPOSURE (PHARMACOVIGILANCE ALERT)")
        pdf.set_font(font_family, "B", 9)
        if physician_payload.high_risk_medication_exposure:
            for med_line in physician_payload.high_risk_medication_exposure:
                pdf.multi_cell(0, 5, f"  [!] {self._safe_str(med_line, has_unicode_font)}", border="L", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.multi_cell(0, 5, "  No high-risk OTC/NSAID exposure explicitly confirmed.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)

        # Red-Flag Clinical Correlation
        self._render_section_header(pdf, font_family, "3. RED-FLAG CLINICAL CORRELATION (THE DOTS CONNECTOR)")
        pdf.set_font(font_family, "", 9)
        pdf.multi_cell(
            0,
            5,
            self._safe_str(physician_payload.red_flag_clinical_correlation, has_unicode_font),
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(1.5)

        # Recommended Immediate Workup (KDIGO / NICE Guidelines)
        self._render_section_header(pdf, font_family, "4. RECOMMENDED IMMEDIATE WORKUP (STAT LABS & EVALUATION)")
        pdf.set_font(font_family, "B", 8.5)
        for idx, item in enumerate(physician_payload.recommended_immediate_workup, start=1):
            clean_item = self._strip_arabic_if_needed(item, has_unicode_font)
            pdf.multi_cell(0, 4.8, f"  [ ] {idx}. {clean_item}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font(font_family, "", 7.5)
        pdf.multi_cell(
            0,
            4.5,
            f"Clinical Guideline Grounding: {physician_payload.clinical_guidelines_reference}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)

        # ------------------------------------------------------------------
        # DIVIDER LINE BETWEEN ENGLISH ER VIEW & ARABIC CAREGIVER SUMMARY
        # ------------------------------------------------------------------
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.6)
        y_pos = pdf.get_y()
        pdf.line(10, y_pos, 200, y_pos)
        pdf.ln(2)

        # ------------------------------------------------------------------
        # BOTTOM HALF: KEY ARABIC INSTRUCTIONS FOR PATIENT & FAMILY
        # ------------------------------------------------------------------
        if has_unicode_font:
            ar_header = shape_arabic_text("إرشادات عاجلة لأسرة المريض (ملخص التوجيه الطبي - صحتك AI)")
            pdf.set_fill_color(0, 0, 0)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font(font_family, "B", 10)
            pdf.cell(0, 7, ar_header, border=1, new_x="LMARGIN", new_y="NEXT", align="R", fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font(font_family, "", 9.5)
            for line in arabic_family_instructions:
                shaped_line = shape_arabic_text(line)
                pdf.multi_cell(0, 5.5, shaped_line, align="R", new_x="LMARGIN", new_y="NEXT")

            disclaimer_ar = shape_arabic_text(
                "إخلاء مسؤولية طبي: هذا التقرير صادر عن نظام فرز ذكي لمساعدة الطبيب وأسرة المريض، ولا يغني عن الفحص الطبي المباشر في الطوارئ."
            )
            pdf.ln(1)
            pdf.set_font(font_family, "B", 8)
            pdf.multi_cell(0, 4.5, disclaimer_ar, border="T", align="C", new_x="LMARGIN", new_y="NEXT")
        else:
            # Fallback if running in a bare container without Arabic TTF fonts installed
            self._render_section_header(pdf, font_family, "5. CAREGIVER & FAMILY ACTIONABLE SUMMARY (BILINGUAL FALLBACK)")
            pdf.set_font(font_family, "", 8.5)
            for line in arabic_family_instructions:
                safe_line = self._safe_str(line, has_unicode_font=False)
                if safe_line.strip():
                    pdf.multi_cell(0, 4.8, f"- {safe_line}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_family, "B", 7.5)
            pdf.multi_cell(
                0,
                4.5,
                "MEDICAL DISCLAIMER: Clinical decision-support & triage summary only. Present immediately to the attending ER physician.",
                border="T",
                align="C",
                new_x="LMARGIN",
                new_y="NEXT",
            )

        out_name = filename or f"Emergency_Triage_Report_{physician_payload.report_id}.pdf"
        out_path = self.output_dir / out_name
        pdf.output(str(out_path))
        return str(out_path.resolve())

    @staticmethod
    def _render_section_header(pdf: FPDF, font_family: str, title: str) -> None:
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(font_family, "B", 9.5)
        pdf.cell(0, 6, f" {title}", border=1, new_x="LMARGIN", new_y="NEXT", fill=True)

    @staticmethod
    def _strip_arabic_if_needed(text: str, has_unicode_font: bool) -> str:
        """If no Unicode font is available, strips parenthetical Arabic so core Helvetica never crashes."""
        if not has_unicode_font:
            return text.encode("latin-1", errors="ignore").decode("latin-1").replace("()", "").strip()
        # Even with Unicode font, English workup lines look cleaner with ASCII or shaped Arabic
        if "(" in text and any("\u0600" <= ch <= "\u06FF" for ch in text):
            en_part = text.split("(")[0].strip()
            return en_part
        return text

    @staticmethod
    def _safe_str(text: str, has_unicode_font: bool) -> str:
        if has_unicode_font:
            if any("\u0600" <= ch <= "\u06FF" for ch in text):
                return shape_arabic_text(text)
            return text
        return text.encode("latin-1", errors="ignore").decode("latin-1")


pdf_generator = EmergencyPDFGenerator()
