"""
db.models
=========
طبقة تخزين للتاريخ/التدقيق (export / audit / history). مصمَّمة على
PostgreSQL في الإنتاج، وتعمل محليًا على SQLite بلا أي تعديل في الكود —
فقط بتغيير DATABASE_URL. هذا هو "adapter pattern" المطلوب في المعمارية:
منطق التشخيص (core/) لا يعرف شيئًا عن قواعد البيانات إطلاقًا.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# الإنتاج: postgresql+psycopg2://user:pass@host:5432/agent_diagnostics
# محليًا/تطوير: sqlite (بلا تغيير كود، فقط متغير بيئة)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./agent_diagnostics.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DiagnosticRun(Base):
    """سجل تشخيص واحد — يخدم متطلب export/audit/history في طبقة المنتج."""
    __tablename__ = "diagnostic_runs"

    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String, index=True, nullable=False)
    task_description = Column(String, default="")
    diverged = Column(Boolean, nullable=False)
    first_divergence_step_id = Column(String, nullable=True)
    category = Column(String, nullable=False)
    full_report_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # حقول جاهزة لطبقة الصلاحيات (permissions) — غير مفعّلة الآن، موجودة
    # كي لا تحتاج هجرة مخطط (schema migration) لاحقًا عند إضافتها فعليًا
    organization_id = Column(String, nullable=True, index=True)
    created_by = Column(String, nullable=True)


class ApprovedReference(Base):
    """
    مرجع مُعتمَد للتحقق (golden run) — يربط بوابة الموافقة البشرية
    (groundtruth/human_gate.py) بتخزين دائم بدل أن تكون في الذاكرة فقط.
    """
    __tablename__ = "approved_references"

    id = Column(Integer, primary_key=True, index=True)
    reference_trace_id = Column(String, nullable=False, unique=True)
    approved_by = Column(String, nullable=False)
    approved_at = Column(String, nullable=False)
    flagged_for_review = Column(Boolean, default=False)
    flag_reason = Column(String, nullable=True)
    reference_trace_json = Column(JSON, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
