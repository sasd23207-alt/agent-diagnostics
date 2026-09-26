"""
core.engine
===========
نقطة الدخول الوحيدة للنظام. كل واجهة (CLI / API / Test Runner) تستدعي:

    from core.engine import diagnose
    report = diagnose(trace)

هذا الملف لا يعرف شيئًا عن أي واجهة — هذا ما يجعل المحرك مستقلًا.
"""

from __future__ import annotations

from typing import Optional

from .models import (
    ExecutionTrace, DivergenceReport, DivergenceCategory, StepStatus,
)
from .divergence import DivergenceDetector, OutcomeComparator
from .impact import ImpactTracker
from .tracing import verify_reconstructable


class DiagnosticEngine:

    def __init__(self, comparator: Optional[OutcomeComparator] = None):
        self.detector = DivergenceDetector(comparator)
        self.tracker = ImpactTracker()

    def diagnose(self, trace: ExecutionTrace) -> DivergenceReport:
        # 0) التحقق من سلامة السجل قبل أي تحليل — تشخيص مبني على سجل ناقص
        #    أسوأ من عدم التشخيص، لذلك نفشل بصوت عالٍ بدل التخمين.
        ok, problems = verify_reconstructable(trace)
        if not ok:
            raise ValueError(
                "Trace is not reconstructable; refusing to diagnose. Problems: "
                + "; ".join(problems)
            )

        # 1) كشف أول انحراف
        root, steps = self.detector.find_first_divergence(trace)
        statuses = {s.step_id: s.status for s in steps}

        if root is None:
            return DivergenceReport(
                trace_id=trace.trace_id,
                task_description=trace.task_description,
                total_steps=len(steps),
                diverged=False,
                first_divergence_step_id=None,
                category=DivergenceCategory.UNCLASSIFIED,
                why_divergent="No step deviated from its expected outcome.",
                why_not_later_steps="",
                step_statuses=statuses,
            )

        # 2) تفسير + تبرير + أثر
        root.divergence_reason = self.detector.explain_why_divergent(root)

        return DivergenceReport(
            trace_id=trace.trace_id,
            task_description=trace.task_description,
            total_steps=len(steps),
            diverged=True,
            first_divergence_step_id=root.step_id,
            category=self.detector.categorize(root),
            expected=root.expected_outcome,
            observed=root.observed_outcome,
            why_divergent=root.divergence_reason,
            why_not_later_steps=self.detector.explain_why_not_later(root, steps),
            impact_chain=self.tracker.build(trace, steps, root),
            step_statuses=statuses,
        )


def diagnose(trace: ExecutionTrace, comparator: Optional[OutcomeComparator] = None) -> DivergenceReport:
    return DiagnosticEngine(comparator).diagnose(trace)
