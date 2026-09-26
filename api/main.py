"""
api.main
========
واجهة API الحقيقية بالمكدّس المطلوب (FastAPI). هذا الملف **كود إنتاج
كامل**، لكن لا يمكن تشغيله داخل هذه البيئة تحديدًا لأن لا اتصال شبكة
لتثبيت fastapi/uvicorn (جُرِّب فعليًا: pip install فشل بلا نتائج).

الكود صحيح نحويًا (مُتحقَّق عبر py_compile) وجاهز للتشغيل فور توفر بيئة
حقيقية بـ `pip install fastapi uvicorn sqlalchemy`. للتحقق من صحة نفس
العقد (request/response contract) والمنطق فعليًا الآن، انظر
`api/main_stdlib_demo.py` — نسخة مطابقة بمكتبة Python القياسية فقط،
مُشغَّلة ومُختبَرة فعليًا في هذا التقرير.

لا هذا الملف ولا db/models.py يستوردان core.engine بطريقة تُعدِّل
سلوكه — كلاهما يستهلك diagnose() كدالة جاهزة كما هي.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.engine import diagnose
from core.tracing import verify_reconstructable
from nlp.upgraded_comparator import StructuralHybridComparator
from adapters.trace_json import trace_from_json, report_to_json

from db.models import get_db, init_db, DiagnosticRun


# ---------------------------------------------------------------------------
# نماذج الطلب/الاستجابة (pydantic) — العقد الخارجي للـ API
# ---------------------------------------------------------------------------

class TraceStepIn(BaseModel):
    step_id: Optional[str] = None
    step_type: str
    parent_step_id: Optional[str] = None
    start_time: Optional[float] = None
    state_before: dict = Field(default_factory=dict)
    state_after: dict = Field(default_factory=dict)
    input: Optional[str] = None
    decision: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[object] = None
    tool_succeeded: bool = True
    tool_error: Optional[str] = None
    expected_outcome: Optional[object] = None
    observed_outcome: Optional[object] = None


class TraceIn(BaseModel):
    trace_id: str
    task_description: str = ""
    final_status: str = "unknown"
    steps: list[TraceStepIn]


class DiagnosticRunOut(BaseModel):
    id: int
    trace_id: str
    diverged: bool
    first_divergence_step_id: Optional[str]
    category: str
    created_at: str


# ---------------------------------------------------------------------------
# دورة حياة التطبيق
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Agent Diagnostics API",
    description="تشخيص أول نقطة انحراف في تنفيذ AI Agent وربط السبب بالأثر.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يُضيَّق في الإنتاج الفعلي لنطاقات الواجهة المعروفة
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# المسارات
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/diagnose", response_model=DiagnosticRunOut)
def diagnose_endpoint(payload: TraceIn, db: Session = Depends(get_db)):
    """
    يستقبل trace بصيغة JSON، يشغّل diagnose() كما هو (بلا أي تعديل على
    core)، يحفظ النتيجة للتاريخ/التدقيق، ويرجّع ملخصًا.
    """
    trace = trace_from_json(payload.model_dump())

    ok, problems = verify_reconstructable(trace)
    if not ok:
        raise HTTPException(status_code=422, detail={"reconstruction_errors": problems})

    report = diagnose(trace, comparator=StructuralHybridComparator())

    run = DiagnosticRun(
        trace_id=trace.trace_id,
        task_description=trace.task_description,
        diverged=report.diverged,
        first_divergence_step_id=report.first_divergence_step_id,
        category=report.category.value,
        full_report_json=report_to_json(report),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return DiagnosticRunOut(
        id=run.id, trace_id=run.trace_id, diverged=run.diverged,
        first_divergence_step_id=run.first_divergence_step_id,
        category=run.category, created_at=run.created_at.isoformat(),
    )


@app.get("/v1/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(DiagnosticRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run.full_report_json


@app.get("/v1/runs")
def list_runs(limit: int = 50, db: Session = Depends(get_db)):
    """لطبقة الـ Dashboard: قائمة آخر التشخيصات — export/audit/history."""
    runs = db.query(DiagnosticRun).order_by(DiagnosticRun.id.desc()).limit(limit).all()
    return [
        {"id": r.id, "trace_id": r.trace_id, "diverged": r.diverged,
         "category": r.category, "created_at": r.created_at.isoformat()}
        for r in runs
    ]
