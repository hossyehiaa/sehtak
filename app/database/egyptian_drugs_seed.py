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

    # ==========================================================
    # 5. Vitamins & Dietary Supplements (المكملات الغذائية والفيتامينات والمعادن)
    # ==========================================================
    EgyptianDrugRecord(
        canonical_brand="Centrum Lutein",
        active_ingredient="Multivitamins, Minerals & Lutein",
        drug_class=DrugClass.VITAMIN_SUPPLEMENT,
        aliases_en_ar=[
            "centrum lutein", "سنترم لوتين", "سنتروم لوتين", "سنترم باللوتين", "سنتروم باللوتين",
            "centrum with lutein", "لوتين", "lutein",
        ],
        common_dosages=["1 tablet daily with food (قرص واحد يومياً مع الطعام)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Complete multivitamin formulation enriched with Lutein for macular & retinal eye health, cellular antioxidant defense, and energy metabolism.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Centrum",
        active_ingredient="Multivitamins & Essential Minerals",
        drug_class=DrugClass.VITAMIN_SUPPLEMENT,
        aliases_en_ar=[
            "centrum", "سنترم", "سنتروم", "centrum silver", "سنترم سيلفر", "سنتروم سيلفر",
            "centrum adults", "فيتامين سنترم",
        ],
        common_dosages=["1 tablet daily with food (قرص واحد يومياً مع الطعام)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Comprehensive daily multivitamin addressing micronutrient deficiencies, cellular vitality, and immune maintenance.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Omega 3 Plus",
        active_ingredient="Fish Oil (EPA/DHA) & Wheat Germ Oil",
        drug_class=DrugClass.VITAMIN_SUPPLEMENT,
        aliases_en_ar=[
            "omega 3", "omega 3 plus", "اوميجا 3", "أوميجا 3", "اوميجا ٣", "أوميجا ٣",
            "زيت سمك", "fish oil", "اوميجا", "أوميجا",
        ],
        common_dosages=["1000mg 1-2 capsules daily after meals (كبسولة إلى كبسولتين يومياً)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Essential omega-3 polyunsaturated fatty acids supporting lipid profile, endothelial function, and cognitive health. Mild antiplatelet action at high doses.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Vidrop",
        active_ingredient="Cholecalciferol (Vitamin D3)",
        drug_class=DrugClass.VITAMIN_SUPPLEMENT,
        aliases_en_ar=[
            "vidrop", "فيدروب", "ديفارول", "devarol", "فيتامين د", "فيتامين d", "vitamin d",
            "vitamin d3", "cholecalciferol", "كوليكالسيفيرول", "نقط فيدروب",
        ],
        common_dosages=["1 bottle weekly or 1 drop = 100 IU (100,000 IU monthly or pediatric drops)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Regulates calcium-phosphate homeostasis, bone mineralization, and neuromuscular/immune competence. Caution in renal impairment to avoid hypercalcemia.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Neurobion",
        active_ingredient="Vitamin B1 (Thiamine) + B6 (Pyridoxine) + B12 (Cyanocobalamin)",
        drug_class=DrugClass.VITAMIN_SUPPLEMENT,
        aliases_en_ar=[
            "neurobion", "نيوروبيون", "neuroton", "نيوروتون", "ميلجا", "milga", "milga advance",
            "ميلجا ادفانس", "فيتامين ب", "vitamin b", "vitamin b complex", "حقن نيوروبيون",
        ],
        common_dosages=["1 tablet 1-3 times daily or 1 ampoule IM twice weekly"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Neurotropic B-vitamins indicated for diabetic/peripheral neuropathy, neuritis, and metabolic nerve cell repair. High safety profile.",
    ),

    # ==========================================================
    # 6. Safe Analgesics & Antipyretics (المسكنات الآمنة وخوافض الحرارة)
    # ==========================================================
    EgyptianDrugRecord(
        canonical_brand="Panadol",
        active_ingredient="Paracetamol (Acetaminophen)",
        drug_class=DrugClass.ANALGESIC,
        aliases_en_ar=[
            "panadol", "بنادول", "باراسيتامول", "paracetamol", "adol", "أدول", "pyral", "بيرال",
            "paramol", "بارامول", "cetamol", "سيتامول", "panadol extra", "بنادول ازرق", "بنادول اكسترا",
            "panadol advance", "بنادول ادفانس", "ابيمول", "abimol",
        ],
        common_dosages=["500mg - 1000mg every 6-8 hours (Max 4000mg/24h)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="First-line antipyretic & analgesic. Spares renal hemodynamics and gastric mucosa. Caution in acute or chronic hepatic impairment.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Congestal",
        active_ingredient="Paracetamol + Pseudoephedrine HCl + Chlorpheniramine Maleate",
        drug_class=DrugClass.ANALGESIC,
        aliases_en_ar=[
            "congestal", "كونجستال", "123", "وان تو ثري", "comtrex", "كومتركس", "flustop", "فلستوب",
            "day & night", "دواء برد", "برشام برد",
        ],
        common_dosages=["1 tablet every 8 hours (قرص كل 8 ساعات بعد الأكل)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Combination cold and flu symptomatic relief. Safe for kidneys, but sympathomimetic pseudoephedrine warrants caution in severe hypertension and glaucoma.",
    ),

    # ==========================================================
    # 7. Common Egyptian Antibiotics & Antiseptics
    # ==========================================================
    EgyptianDrugRecord(
        canonical_brand="Antinal",
        active_ingredient="Nifuroxazide",
        drug_class=DrugClass.ANTIBIOTIC,
        aliases_en_ar=[
            "antinal", "أنتينال", "انتينال", "nifuroxazide", "نيفوروكزازيد", "مطهر معوي",
        ],
        common_dosages=["200mg 4 times daily (كبسولة كل 6 ساعات)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Locally active broad-spectrum intestinal antiseptic for acute infectious diarrhea. Negligible systemic absorption.",
    ),
    EgyptianDrugRecord(
        canonical_brand="Augmentin",
        active_ingredient="Amoxicillin + Clavulanic Acid",
        drug_class=DrugClass.ANTIBIOTIC,
        aliases_en_ar=[
            "augmentin", "اوجمنتين", "أوجمنتين", "hibiotic", "هاي بيوتك", "curam", "كيورام",
            "megamox", "ميجاموكس", "amoxicillin", "اموكسيسيللين",
        ],
        common_dosages=["1g (1000mg) every 12 hours (قرص كل 12 ساعة)"],
        nephrotoxic_risk=False,
        gi_bleed_risk=False,
        clinical_notes="Broad-spectrum beta-lactamase inhibitor combination. Requires dose reduction in severe renal impairment (eGFR < 30 mL/min).",
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
    """Deterministically scans raw Arabic/English text for any known Egyptian medications (longest alias wins)."""
    if not raw_text:
        return []
    lowered = raw_text.lower()
    matches: List[tuple[int, str, EgyptianDrugRecord]] = []
    for record in EGYPTIAN_DRUG_KNOWLEDGE_BASE:
        best_alias = ""
        for alias in record.aliases_en_ar:
            if alias.lower() in lowered:
                if len(alias) > len(best_alias):
                    best_alias = alias.lower()
        if best_alias:
            matches.append((len(best_alias), best_alias, record))

    # Sort descending by alias length so more specific matches (e.g. "centrum lutein") come first
    matches.sort(key=lambda x: x[0], reverse=True)

    final_records: Dict[str, EgyptianDrugRecord] = {}
    chosen_aliases: List[str] = []

    for _, alias, rec in matches:
        if any(alias in chosen for chosen in chosen_aliases):
            continue
        if rec.canonical_brand not in final_records:
            final_records[rec.canonical_brand] = rec
            chosen_aliases.append(alias)

    return list(final_records.values())
