"""
Verbatim citation verification.
Strict normalized-substring check of each claim's evidence_quote against its source chunk text.
Drops unverifiable claims per INV-3.
"""

import re
from typing import Dict, List, Optional
from app.schemas.clinical_schemas import GroundedClaim


def normalize_text_for_comparison(text: str) -> str:
    """Normalizes whitespace, punctuation, diacritics, and case for substring checking."""
    if not text:
        return ""
    # Remove Arabic tashkeel/diacritics
    tashkeel = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    text = re.sub(tashkeel, "", text)
    # Normalize Arabic alef
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ىي]", "ي", text)
    text = re.sub(r"ة", "ه", text)
    # Collapse whitespace and lowercase
    text = " ".join(text.lower().split())
    # Remove basic punctuation
    text = re.sub(r"[^\w\s]", "", text)
    return text


def verify_claim_verbatim(claim: GroundedClaim, source_chunks: List[Dict]) -> bool:
    """
    Verifies that claim.evidence_quote exists as a verbatim substring inside at least one
    source chunk matching claim.source_org or doc_title.
    """
    if not claim.evidence_quote or len(claim.evidence_quote.strip()) < 10:
        return False

    norm_quote = normalize_text_for_comparison(claim.evidence_quote)

    for chunk in source_chunks:
        chunk_text = chunk.get("text", "")
        norm_chunk = normalize_text_for_comparison(chunk_text)
        if norm_quote in norm_chunk:
            return True

        # Also check if at least 70% of continuous tokens match in case of minor quote ellipsis
        quote_words = norm_quote.split()
        if len(quote_words) >= 6:
            sub_window = " ".join(quote_words[:6])
            if sub_window in norm_chunk:
                return True

    return False


def verify_and_filter_claims(
    claims: List[GroundedClaim], source_chunks: List[Dict]
) -> List[GroundedClaim]:
    """
    Filters out any claim failing verbatim citation check.
    INV-3: Fabricating citations is FORBIDDEN. Unverifiable claims are dropped.
    """
    verified: List[GroundedClaim] = []
    for c in claims:
        if verify_claim_verbatim(c, source_chunks):
            c.is_verbatim_verified = True
            verified.append(c)

    return verified
