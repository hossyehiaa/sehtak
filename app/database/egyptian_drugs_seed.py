"""Seed database of high-risk Egyptian OTC/prescription medications and active ingredients."""

from dataclasses import dataclass
from typing import Dict, List
from app.schemas.clinical_schemas import DrugClass


@dataclass(frozen=True)
class EgyptianDrugRecord:
    canonical_brand: str
    active_ingredient: str
    drug_class: DrugClass
    aliases_en_ar: List[str]
    common_dosages: List[str]
    nephrotoxic_risk: bool
    gi_bleed_risk: bool
    clinical_notes: str


EGYPTIAN_DRUG_KNOWLEDGE_BASE: List[EgyptianDrugRecord] = [
    # ==========================================================
    # 1. NSAIDs (المسكنات ومضادات الالتهاب غير الستيرويدية - عالية الخطورة على الكلى)
    # ==========================================================
    EgyptianDrugRecord(
        canonical_brand="Cataflam",
        active_ingredient="Diclofenac Potassium",
        drug_class=DrugClass.NSAID,
        aliases_en_ar=[
            "cataflam", "كتافلام", "كاتافلام", "كتفلام", "catafast", "كتافاست", "كاتافاست",
            "diclofenac potassium", "ديكلوفيناك بوتاسيوم", "أكياس كتافاست", "برشام كتافلام",
        ],
        common_dosages=["25mg", "50mg", "75mg"],
        nephrotoxic_risk=True,
        gi_bleed_risk=True,
        clinical_notes="Potent prostaglandin inhibitor causing afferent arteriolar vasoconstriction, reduced GFR, and acute interstitial nephritis.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Voltaren",
        active_ingredient="Diclofenac Sodium",
        drug_class=DrugClass.NSAID,
        aliases_en_ar=[
            "voltaren", "فولتارين", "فولترين", "حقن فولتارين", "diclofenac sodium",
            "ديكلوفيناك صوديوم", "declophen", "ديكلوفين", "epifenac", "إبيفيناك", "olfe", "أولفين", "olfen",
        ],
        common_dosages=["50mg", "75mg IM", "100mg SR"],
        nephrotoxic_risk=True,
        gi_bleed_risk=True,
        clinical_notes="Frequently abused as IM injections or SR tablets in Egypt for bone/joint pain; high risk of silent AKI.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Ketofan",
        active_ingredient="Ketoprofen",
        drug_class=DrugClass.NSAID,
        aliases_en_ar=[
            "ketofan", "كيتوفان", "كتوفان", "ketolgin", "كيتولجين", "alcofan", "ألكوفان",
            "bai-alcofan", "باي ألكوفان", "باي الكوفان", "باى الكوفان", "bi-alcofan", "بي الكوفان",
            "ketoprofen", "كيتوبروفين",
        ],
        common_dosages=["25mg", "50mg", "75mg", "150mg SR"],
        nephrotoxic_risk=True,
        gi_bleed_risk=True,
        clinical_notes="Extremely common OTC analgesic in Egypt ('برشام الصداع والعظم'); chronic intake rapidly induces renal papillary necrosis and AKI.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Brufen",
        active_ingredient="Ibuprofen",
        drug_class=DrugClass.NSAID,
        aliases_en_ar=[
            "brufen", "بروفين", "أيبوبروفين", "ايبوبروفين", "ibuprofen", "marcofen", "ماركوفين",
            "flamotal", "فلاموتال", "ultrafen", "ألترافين",
        ],
        common_dosages=["200mg", "400mg", "600mg", "800mg"],
        nephrotoxic_risk=True,
        gi_bleed_risk=True,
        clinical_notes="Non-selective COX inhibitor; high doses (>1200mg/day) or prolonged use (>7 days) impairs renal perfusion.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Bai-Alcofan",
        active_ingredient="Ketoprofen",
        drug_class=DrugClass.NSAID,
        aliases_en_ar=[
            "bai-alcofan", "bi-alcofan", "باي ألكوفان", "باي الكوفان", "باى الكوفان", "بي الكوفان", "الكوفان",
        ],
        common_dosages=["150mg"],
        nephrotoxic_risk=True,
        gi_bleed_risk=True,
        clinical_notes="High-dose 150mg Ketoprofen formulation widely used OTC in Egypt; high nephrotoxicity and GI ulceration risk.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Celebrex",
        active_ingredient="Celecoxib",
        drug_class=DrugClass.NSAID,
        aliases_en_ar=[
            "celebrex", "سيليبريكس", "سيلبركس", "celecoxib", "سيليكوكسيب", "eurocox", "يوروكوكس",
            "arcoxia", "أركوكسيا", "اركوكسيا", "etoricoxib", "meloxicam", "ميلوكسيكام", "mobic", "موبيك",
        ],
        common_dosages=["100mg", "200mg"],
        nephrotoxic_risk=True,
        gi_bleed_risk=False,
        clinical_notes="COX-2 selective NSAID; spares stomach relatively but retains full renal vasoconstrictive & nephrotoxic risk.",
    ),

    # ==========================================================
    # 2. Corticosteroids (الكورتيزون)
    # ==========================================================
    EgyptianDrugRecord(
        canonical_brand="Dexamethasone",
        active_ingredient="Dexamethasone",
        drug_class=DrugClass.CORTICOSTEROID,
        aliases_en_ar=[
            "dexamethasone", "ديكساميثازون", "ديكسا", "حقنة ديكسا", "dexaglobe", "ديكساجلوب",
            "epidron", "إبيدرون", "ابيدرون", "phenadone", "فينادون", "apidone", "أبيدون",
        ],
        common_dosages=["0.5mg", "8mg/2ml IM"],
        nephrotoxic_risk=False,
        gi_bleed_risk=True,
        clinical_notes="Often mixed with NSAIDs in Egyptian pharmacies ('حقنة البرد'); causes severe GI bleeding when combined with NSAIDs, fluid retention, and hyperglycemia.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Prednisolone",
        active_ingredient="Prednisolone",
        drug_class=DrugClass.CORTICOSTEROID,
        aliases_en_ar=[
            "prednisolone", "بريدنيزولون", "solupred", "سولوبريد", "hostacortin", "هوستاكورتين",
            "xilone", "زيلون",
        ],
        common_dosages=["5mg", "20mg"],
        nephrotoxic_risk=False,
        gi_bleed_risk=True,
        clinical_notes="Systemic corticosteroid; masks infection, elevates BP/glucose, and multiplies peptic ulcer perforation risk with NSAIDs.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Solu-Medrol",
        active_ingredient="Methylprednisolone",
        drug_class=DrugClass.CORTICOSTEROID,
        aliases_en_ar=[
            "solu-medrol", "سولو ميدرول", "سولوميدرول", "medrol", "ميدرول", "methylprednisolone", "ميثيل بريدنيزولون",
        ],
        common_dosages=["4mg", "16mg", "500mg IV"],
        nephrotoxic_risk=False,
        gi_bleed_risk=True,
        clinical_notes="Potent corticosteroid requiring strict medical supervision and tapering.",
    ),

    # ==========================================================
    # 3. Anticoagulants & Antiplatelets (أدوية السيولة)
    # ==========================================================
    EgyptianDrugRecord(
        canonical_brand="Marevan",
        active_ingredient="Warfarin",
        drug_class=DrugClass.ANTICOAGULANT,
        aliases_en_ar=[
            "marevan", "ماريفان", "مريفان", "warfarin", "وارفارين",
        ],
        common_dosages=["1mg", "3mg", "5mg"],
        nephrotoxic_risk=False,
        gi_bleed_risk=True,
        clinical_notes="Vitamin K antagonist with narrow therapeutic index; co-administration with NSAIDs causes life-threatening hemorrhage and INR surge.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Plavix",
        active_ingredient="Clopidogrel",
        drug_class=DrugClass.ANTIPLATELET,
        aliases_en_ar=[
            "plavix", "بلافيكس", "بلافكس", "clopidogrel", "كلوبيدوجريل", "clopex", "كلوبيكس", "borgavix", "بورجافيكس",
        ],
        common_dosages=["75mg"],
        nephrotoxic_risk=False,
        gi_bleed_risk=True,
        clinical_notes="Antiplatelet agent; concurrent NSAID use markedly increases upper/lower GI bleeding.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Aspocid",
        active_ingredient="Acetylsalicylic Acid (Aspirin)",
        drug_class=DrugClass.ANTIPLATELET,
        aliases_en_ar=[
            "aspocid", "أسبوسيد", "اسبوسيد", "أسبرين", "اسبرين", "aspirin", "ezacard", "إيزاكارد", "jusprin", "جوسبرين", "aggrex", "أجريكس",
        ],
        common_dosages=["75mg", "81mg", "100mg"],
        nephrotoxic_risk=False,
        gi_bleed_risk=True,
        clinical_notes="Low-dose aspirin for cardiovascular prophylaxis; ibuprofen/NSAIDs antagonize its cardioprotective effect and double GI bleeding risk.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Eliquis",
        active_ingredient="Apixaban",
        drug_class=DrugClass.ANTICOAGULANT,
        aliases_en_ar=[
            "eliquis", "إليكويس", "اليكويس", "apixaban", "أبيكسابان", "xarelto", "زاريلتو", "rivaroxaban", "ريفاروكسابان",
        ],
        common_dosages=["2.5mg", "5mg"],
        nephrotoxic_risk=False,
        gi_bleed_risk=True,
        clinical_notes="Direct Oral Anticoagulant (DOAC); partially renally cleared—NSAID-induced AKI leads to DOAC accumulation and major bleeding.",
    ),

    # ==========================================================
    # 4. ACE Inhibitors / ARBs & Diuretics ("Triple Whammy" Nephrotoxic Combo)
    # ==========================================================
    EgyptianDrugRecord(
        canonical_brand="Capoten",
        active_ingredient="Captopril",
        drug_class=DrugClass.ACE_INHIBITOR,
        aliases_en_ar=[
            "capoten", "كابوتين", "كابوتن", "captopril", "كابتوبريل",
        ],
        common_dosages=["25mg", "50mg"],
        nephrotoxic_risk=True,
        gi_bleed_risk=False,
        clinical_notes="ACE inhibitor dilating efferent arteriole; combining with NSAIDs collapses glomerular filtration pressure.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Tritace",
        active_ingredient="Ramipril",
        drug_class=DrugClass.ACE_INHIBITOR,
        aliases_en_ar=[
            "tritace", "تريتاس", "تريتيس", "ramipril", "راميبريل", "concor", "ezapril", "إيزابريل",
            "tareg", "تاريج", "valsartan", "فالسارتان", "atacand", "أتاكاند", "candesartan", "لوسارتان", "losartan",
        ],
        common_dosages=["2.5mg", "5mg", "10mg"],
        nephrotoxic_risk=True,
        gi_bleed_risk=False,
        clinical_notes="RAAS blocker (ACEi/ARB); high risk of hyperkalemia and hemodynamically mediated AKI when paired with NSAIDs.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Lasix",
        active_ingredient="Furosemide",
        drug_class=DrugClass.DIURETIC,
        aliases_en_ar=[
            "lasix", "لازكس", "لاسيكس", "furosemide", "فوروسيميد", "فيوروسيميد", "examide", "إكساميد",
            "natrilix", "ناتريليكس", "indapamide", "hydrochlorothiazide", "مدر للبول", "برشام مية",
        ],
        common_dosages=["20mg", "40mg"],
        nephrotoxic_risk=True,
        gi_bleed_risk=False,
        clinical_notes="Loop/thiazide diuretic causing volume depletion; component of the lethal 'Triple Whammy' (ACEi/ARB + Diuretic + NSAID).",
    ),
]


def lookup_egyptian_drug(query_name: str) -> EgyptianDrugRecord | None:
    """Fuzzy/alias lookup for an Egyptian drug name or active ingredient in Arabic or English."""
    if not query_name:
        return None
    cleaned = query_name.strip().lower()
    for record in EGYPTIAN_DRUG_KNOWLEDGE_BASE:
        if cleaned == record.canonical_brand.lower() or cleaned == record.active_ingredient.lower():
            return record
        for alias in record.aliases_en_ar:
            alias_low = alias.lower()
            if alias_low in cleaned or cleaned in alias_low:
                return record
    return None


def scan_text_for_known_drugs(raw_text: str) -> List[EgyptianDrugRecord]:
    """Deterministically scans raw Arabic/English text for any known Egyptian high-risk medications."""
    if not raw_text:
        return []
    lowered = raw_text.lower()
    found: Dict[str, EgyptianDrugRecord] = {}
    for record in EGYPTIAN_DRUG_KNOWLEDGE_BASE:
        for alias in record.aliases_en_ar:
            if alias.lower() in lowered:
                found[record.canonical_brand] = record
                break
    return list(found.values())
