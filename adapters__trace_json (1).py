"""
adapters.trace_json
=====================
محوّل واحد بين عقد JSON الخارجي ونماذج core.models — يُستخدم من أي
واجهة (FastAPI، خادم stdlib للتجربة، CLI) بلا تكرار منطق التحويل.

هذا الملف لا يستورد fastapi ولا أي إطار عمل — مستقل تمامًا، وهذا بالضبط
ما تعنيه "لا تخلطوا المنطق التشخيصي بالواجهة" في المعمارية المطلوبة.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from core.models import ExecutionTrace, StepType, DivergenceReport
from core.tracing import TraceLogger


def trace_from_json(raw: dict) -> ExecutionTrace:
    lg = TraceLogger(raw.get("trace_id", "unknown"), raw.get("task_description", ""))
    for s in raw.get("steps", []):
        lg.log_step(
            StepType(s["step_type"]),
            step_id=s.get("step_id"),
            state_before=s.get("state_before", {}),
            state_after=s.get("state_after", {}),
            input=s.get("input"),
            decision=s.get("decision"),
            tool_name=s.get("tool_name"),
            tool_input=s.get("tool_input"),
            tool_output=s.get("tool_output"),
            tool_succeeded=s.get("tool_succeeded", True),
            tool_error=s.get("tool_error"),
            expected_outcome=s.get("expected_outcome"),
            observed_outcome=s.get("observed_outcome"),
            parent_step_id=s.get("parent_step_id"),
        )
    return lg.finish(raw.get("final_status", "unknown"))


def report_to_json(report: DivergenceReport) -> dict[str, Any]:
    return report.to_dict()
