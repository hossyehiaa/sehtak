"""SQLAlchemy Async database and stateful session store for Sehtak AI."""

import json
from datetime import datetime, timezone
from typing import Dict, Optional
from sqlalchemy import Column, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import settings
from app.schemas.clinical_schemas import (
    ClinicalIntakeSchema,
    ExtractedMedication,
    ExtractedSymptom,
)

Base = declarative_base()


class PatientSessionModel(Base):
    """Persistent patient triage session stored in SQLite/PostgreSQL."""

    __tablename__ = "patient_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), unique=True, index=True, nullable=False)
    channel = Column(String(32), default="telegram", nullable=False)
    intake_json = Column(Text, nullable=False, default="{}")
    conversation_history_json = Column(Text, nullable=False, default="[]")
    last_status = Column(String(64), default="active", nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DrugCatalogModel(Base):
    """Seeded Egyptian Drug Catalog table in SQLite/PostgreSQL."""

    __tablename__ = "egyptian_drug_catalog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_brand = Column(String(128), unique=True, index=True, nullable=False)
    active_ingredient = Column(String(128), nullable=False)
    drug_class = Column(String(64), nullable=False)
    aliases_json = Column(Text, nullable=False)
    nephrotoxic_risk = Column(Integer, default=0)
    gi_bleed_risk = Column(Integer, default=0)
    clinical_notes = Column(Text, nullable=True)


engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Fast in-memory fallback cache synced with SQLite for instant stateful access
_MEMORY_SESSION_CACHE: Dict[str, ClinicalIntakeSchema] = {}
_MEMORY_HISTORY_CACHE: Dict[str, list[str]] = {}


async def init_db() -> None:
    """Initializes database schema and seeds the Egyptian Drug Knowledge Base."""
    from app.database.egyptian_drugs_seed import EGYPTIAN_DRUG_KNOWLEDGE_BASE

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DrugCatalogModel).limit(1))
        existing = result.scalar_one_or_none()
        if not existing:
            for rec in EGYPTIAN_DRUG_KNOWLEDGE_BASE:
                session.add(
                    DrugCatalogModel(
                        canonical_brand=rec.canonical_brand,
                        active_ingredient=rec.active_ingredient,
                        drug_class=rec.drug_class.value,
                        aliases_json=json.dumps(rec.aliases_en_ar, ensure_ascii=False),
                        nephrotoxic_risk=1 if rec.nephrotoxic_risk else 0,
                        gi_bleed_risk=1 if rec.gi_bleed_risk else 0,
                        clinical_notes=rec.clinical_notes,
                    )
                )
            await session.commit()


async def get_patient_intake_session(session_id: str) -> ClinicalIntakeSchema:
    """Retrieves the current ClinicalIntakeSchema for a given user/patient session."""
    if session_id in _MEMORY_SESSION_CACHE:
        return _MEMORY_SESSION_CACHE[session_id]

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PatientSessionModel).where(PatientSessionModel.session_id == session_id)
        )
        row = result.scalar_one_or_none()
        if row and row.intake_json:
            intake = ClinicalIntakeSchema.model_validate_json(row.intake_json)
            _MEMORY_SESSION_CACHE[session_id] = intake
            _MEMORY_HISTORY_CACHE[session_id] = json.loads(row.conversation_history_json or "[]")
            return intake

    fresh_intake = ClinicalIntakeSchema()
    _MEMORY_SESSION_CACHE[session_id] = fresh_intake
    _MEMORY_HISTORY_CACHE[session_id] = []
    return fresh_intake


async def get_conversation_history(session_id: str) -> list[str]:
    """Returns accumulated raw user messages for context continuity."""
    await get_patient_intake_session(session_id)
    return _MEMORY_HISTORY_CACHE.get(session_id, [])


async def reset_patient_session(session_id: str) -> ClinicalIntakeSchema:
    """Clears the patient session state to start a fresh medical triage."""
    fresh = ClinicalIntakeSchema()
    _MEMORY_SESSION_CACHE[session_id] = fresh
    _MEMORY_HISTORY_CACHE[session_id] = []

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PatientSessionModel).where(PatientSessionModel.session_id == session_id)
        )
        row = result.scalar_one_or_none()
        if row:
            row.intake_json = fresh.model_dump_json()
            row.conversation_history_json = "[]"
            row.last_status = "reset"
            row.updated_at = datetime.now(timezone.utc)
            await db.commit()
    return fresh


def merge_clinical_intake(
    existing: ClinicalIntakeSchema,
    incoming: ClinicalIntakeSchema,
    new_raw_note: Optional[str] = None,
) -> ClinicalIntakeSchema:
    """Intelligently merges newly extracted medications, durations, and symptoms into the active session."""
    patient_age = incoming.patient_age or existing.patient_age
    patient_sex = incoming.patient_sex or existing.patient_sex

    chronic_set = list(dict.fromkeys(existing.chronic_conditions + incoming.chronic_conditions))

    # Merge medications by brand_name or active_ingredient
    meds_by_key: Dict[str, ExtractedMedication] = {}
    for med in existing.medications:
        key = (med.brand_name or "").strip().lower()
        meds_by_key[key] = med.model_copy(deep=True)

    for inc_med in incoming.medications:
        key = (inc_med.brand_name or "").strip().lower()
        # Check if incoming medication updates an existing one
        matched_key = None
        for existing_key, existing_med in meds_by_key.items():
            if (
                key == existing_key
                or (
                    inc_med.active_ingredient
                    and existing_med.active_ingredient
                    and inc_med.active_ingredient.lower() == existing_med.active_ingredient.lower()
                )
            ):
                matched_key = existing_key
                break

        if matched_key:
            target = meds_by_key[matched_key]
            if inc_med.duration_days is not None:
                target.duration_days = inc_med.duration_days
            if inc_med.frequency:
                target.frequency = inc_med.frequency
            if inc_med.dosage:
                target.dosage = inc_med.dosage
            if inc_med.active_ingredient:
                target.active_ingredient = inc_med.active_ingredient
            if inc_med.drug_class and inc_med.drug_class.value != "OTHER":
                target.drug_class = inc_med.drug_class
        else:
            meds_by_key[key] = inc_med.model_copy(deep=True)

    # If the user just replied with a duration (e.g., "بقاله شهرين" / 60 days) and incoming had a generic duration
    # or we have an existing medication missing duration_days, propagate discovered duration if unambiguous
    incoming_durations = [m.duration_days for m in incoming.medications if m.duration_days is not None]
    if incoming_durations:
        for med in meds_by_key.values():
            if med.duration_days is None:
                med.duration_days = incoming_durations[0]

    # Merge symptoms
    symptoms_by_term: Dict[str, ExtractedSymptom] = {}
    for sym in existing.symptoms:
        symptoms_by_term[sym.clinical_term_English.lower()] = sym.model_copy(deep=True)
    for inc_sym in incoming.symptoms:
        term_key = inc_sym.clinical_term_English.lower()
        if term_key in symptoms_by_term:
            if inc_sym.duration_days is not None:
                symptoms_by_term[term_key].duration_days = inc_sym.duration_days
            symptoms_by_term[term_key].severity = inc_sym.severity
        else:
            symptoms_by_term[term_key] = inc_sym.model_copy(deep=True)

    combined_notes = " | ".join(
        filter(None, [existing.raw_transcript_or_notes, incoming.raw_transcript_or_notes, new_raw_note])
    )

    return ClinicalIntakeSchema(
        patient_age=patient_age,
        patient_sex=patient_sex,
        chronic_conditions=chronic_set,
        medications=list(meds_by_key.values()),
        symptoms=list(symptoms_by_term.values()),
        missing_critical_info=incoming.missing_critical_info,
        clarification_question_ar=incoming.clarification_question_ar,
        raw_transcript_or_notes=combined_notes or None,
    )


async def save_patient_intake_session(
    session_id: str,
    intake: ClinicalIntakeSchema,
    channel: str = "telegram",
    raw_user_message: Optional[str] = None,
    status: str = "active",
) -> None:
    """Persists updated ClinicalIntakeSchema to both memory cache and SQLite/PostgreSQL."""
    _MEMORY_SESSION_CACHE[session_id] = intake
    history = _MEMORY_HISTORY_CACHE.setdefault(session_id, [])
    if raw_user_message:
        history.append(raw_user_message)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PatientSessionModel).where(PatientSessionModel.session_id == session_id)
        )
        row = result.scalar_one_or_none()
        if not row:
            row = PatientSessionModel(
                session_id=session_id,
                channel=channel,
                intake_json=intake.model_dump_json(),
                conversation_history_json=json.dumps(history, ensure_ascii=False),
                last_status=status,
                updated_at=datetime.now(timezone.utc),
            )
            db.add(row)
        else:
            row.intake_json = intake.model_dump_json()
            row.conversation_history_json = json.dumps(history, ensure_ascii=False)
            row.last_status = status
            row.updated_at = datetime.now(timezone.utc)
        await db.commit()
