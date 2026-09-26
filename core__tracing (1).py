"""
core.tracing
============
طبقة التسجيل. مسؤوليتها الوحيدة: ضمان أن كل خطوة تُسجَّل كاملة،
وأن التسلسل قابل لإعادة البناء من السجلات وحدها.

لا تحتوي أي منطق تشخيص — المحرك (core.engine) هو من يحلّل.
"""

from __future__ import annotations

from typing import Any, Optional
import itertools

from .models import (
    ExecutionTrace, Step, StepType, StateSnapshot,
    ToolCall, ToolResult,
)


class TraceLogger:
    """
    مسجّل تسلسل تنفيذ واحد.

    الاستخدام:
        logger = TraceLogger("trace_1", "book a flight")
        logger.log_step(StepType.PLANNING, state_before={...}, state_after={...},
                        expected_outcome="...", observed_outcome="...")
        trace = logger.finish(final_status="failed")
    """

    def __init__(self, trace_id: str, task_description: str = ""):
        self.trace = ExecutionTrace(trace_id=trace_id, task_description=task_description)
        self._clock = itertools.count()   # ترتيب زمني حتمي (deterministic) للاختبارات
        self._last_step_id: Optional[str] = None

    def log_step(
        self,
        step_type: StepType,
        *,
        step_id: Optional[str] = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        input: Any = None,
        decision: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: Any = None,
        tool_succeeded: bool = True,
        tool_error: Optional[str] = None,
        expected_outcome: Any = None,
        observed_outcome: Any = None,
        parent_step_id: Optional[str] = None,
    ) -> Step:
        ts = float(next(self._clock))
        sid = step_id or f"s{int(ts) + 1}"

        call = result = None
        if tool_name is not None:
            call = ToolCall(tool_name=tool_name, tool_input=tool_input or {}, timestamp=ts)
            result = ToolResult(
                call_id=call.call_id,
                tool_output=tool_output,
                succeeded=tool_succeeded,
                error_message=tool_error,
                timestamp=ts,
            )

        step = Step(
            step_id=sid,
            step_type=step_type,
            timestamp=ts,
            # إذا لم يُحدَّد الأب صراحة، فهو الخطوة السابقة — هذا ما يجعل
            # إعادة بناء السلسلة ممكنة من السجلات وحدها.
            parent_step_id=parent_step_id if parent_step_id is not None else self._last_step_id,
            input=input,
            decision=decision,
            tool_call=call,
            tool_result=result,
            state_before=StateSnapshot(state_before or {}),
            state_after=StateSnapshot(state_after or {}),
            expected_outcome=expected_outcome,
            observed_outcome=observed_outcome,
        )

        self.trace.add_step(step)
        self._last_step_id = sid
        return step

    def finish(self, final_status: str = "unknown") -> ExecutionTrace:
        self.trace.final_status = final_status
        return self.trace


# ---------------------------------------------------------------------------
# التحقق من قابلية إعادة البناء (معيار القبول رقم 1)
# ---------------------------------------------------------------------------

def verify_reconstructable(trace: ExecutionTrace) -> tuple[bool, list[str]]:
    """
    يتأكد أن التسلسل الكامل قابل لإعادة البناء من السجلات وحدها:
      - كل step_id فريد
      - كل parent_step_id يشير لخطوة موجودة فعلًا
      - كل خطوة (عدا الأولى) لها أب  → أي لا توجد خطوات يتيمة منفصلة
      - الترتيب الزمني بلا تكرار في الطوابع
    يرجّع (هل هي سليمة، قائمة المشاكل).
    """
    problems: list[str] = []
    steps = trace.ordered_steps()
    ids = [s.step_id for s in steps]

    if len(ids) != len(set(ids)):
        problems.append("duplicate step_id found")

    timestamps = [s.timestamp for s in steps]
    if len(timestamps) != len(set(timestamps)):
        problems.append("duplicate timestamps — temporal order is ambiguous")

    id_set = set(ids)
    for i, s in enumerate(steps):
        if i == 0:
            if s.parent_step_id is not None:
                problems.append(f"{s.step_id}: first step must have no parent")
            continue
        if s.parent_step_id is None:
            problems.append(f"{s.step_id}: orphan step (no parent)")
        elif s.parent_step_id not in id_set:
            problems.append(f"{s.step_id}: parent '{s.parent_step_id}' does not exist")

    return (len(problems) == 0, problems)
