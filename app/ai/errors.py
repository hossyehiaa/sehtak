"""Standardized error hierarchy for Sehtak AI intake and multimodal pipelines."""

from typing import Optional


class IntakeDegradationError(Exception):
    """
    Raised when an intake component (voice transcription, image OCR/vision, text extraction)
    degrades or fails. ZERO silent fallbacks allowed per INV-2.
    """

    def __init__(self, component: str, reason: str, user_message_ar: Optional[str] = None):
        self.component = component
        self.reason = reason
        comp_name_ar = "الصوت" if "voice" in component.lower() else ("الصورة" if "image" in component.lower() else component)
        self.user_message_ar = user_message_ar or (
            f"تحليل {comp_name_ar} مش شغال حالياً (سبب: {reason}). اكتب الحالة نصاً وهحللها فوراً."
        )
        super().__init__(f"Intake degradation in '{component}': {reason}")
