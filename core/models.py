"""
core.models
===========
نماذج البيانات السبعة المطلوبة.

مبدأ التصميم: كل نموذج يحمل معرفًا فريدًا وترتيبًا زمنيًا، وكل الحقول مجتمعة
كافية لإعادة بناء المسار كاملًا من السجلات وحدها — بدون أي معلومة خارجية.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import uuid


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# التصنيفات
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_CALL = "tool_call"
    RESULT_INTERPRETATION = "result_interpretation"
    DECISION = "decision"


class StepStatus(str, Enum):
    OK = "ok"
    DIVERGENT = "divergent"      # خرجت عن المسار الصحيح
    UNVERIFIABLE = "unverifiable"  # لا يوجد expected_outcome نقارن به


class DivergenceCategory(str, Enum):
    PLANNING_ERROR = "Planning Error"
    TOOL_SELECTION_ERROR = "Tool Selection Error"
    TOOL_INVOCATION_ERROR = "Tool Invocation Error"
    RESULT_INTERPRETATION_ERROR = "Result Interpretation Error"
    DECISION_ERROR = "Decision Error"
    UNCLASSIFIED = "Unclassified"


# الخريطة من نوع الخطوة إلى تصنيف الانحراف.
# مقصودة أن تكون صريحة وليست ضمنية، لأن التصنيف جزء من مخرج التشخيص.
STEP_TYPE_TO_CATEGORY: dict[StepType, DivergenceCategory] = {
    StepType.PLANNING: DivergenceCategory.PLANNING_ERROR,
    StepType.TOOL_SELECTION: DivergenceCategory.TOOL_SELECTION_ERROR,
    StepType.TOOL_CALL: DivergenceCategory.TOOL_INVOCATION_ERROR,
    StepType.RESULT_INTERPRETATION: DivergenceCategory.RESULT_INTERPRETATION_ERROR,
    StepType.DECISION: DivergenceCategory.DECISION_ERROR,
}


# ---------------------------------------------------------------------------
# 1. StateSnapshot
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    """
    لقطة من حالة الوكيل في لحظة محددة.
    `data` حرة الشكل (key-value) لأن كل وكيل يحمل حالة مختلفة،
    لكن المقارنة بينها آلية وليست يدوية (انظر .diff و .path_diff).
    """
    data: dict[str, Any] = field(default_factory=dict)
    snapshot_id: str = field(default_factory=lambda: _new_id("state"))

    def diff(self, other: "StateSnapshot") -> dict[str, dict[str, Any]]:
        """
        مقارنة على مستوى المفتاح الأعلى (سلوك أصلي، محفوظ كما هو لأن
        divergence.py و impact.py يعتمدان عليه):
            {key: {"before": ..., "after": ...}}
        للتحديد الدقيق داخل البنى المتداخلة استخدم .path_diff
        """
        changed: dict[str, dict[str, Any]] = {}
        for k in set(self.data) | set(other.data):
            before = self.data.get(k, "<absent>")
            after = other.data.get(k, "<absent>")
            if not _deep_equal(before, after):
                changed[k] = {"before": before, "after": after}
        return changed

    def path_diff(self, other: "StateSnapshot") -> list["StateChange"]:
        """
        مقارنة على مستوى المسار الكامل داخل البنى المتداخلة:
            state.context.routing.department
            state.tools[2].name

        لماذا؟ القول إن 'context تغيّر' عديم الفائدة حين يكون context كائنًا
        بعشرين مفتاحًا. تحديد الحقل الذي تغيّر فعلًا هو ما يجعل نقطة الانحراف
        قابلة للإصلاح لا مجرد قابلة للرصد.
        """
        changes: list[StateChange] = []
        _walk_diff(self.data, other.data, "state", changes)
        return sorted(changes, key=lambda c: c.path)


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    TYPE_CHANGED = "type_changed"


@dataclass
class StateChange:
    """تغيّر واحد محدد بمساره الكامل داخل الحالة."""
    path: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None

    def __str__(self) -> str:
        if self.change_type is ChangeType.ADDED:
            return f"{self.path} += {self.new_value!r}"
        if self.change_type is ChangeType.REMOVED:
            return f"{self.path} -= {self.old_value!r}"
        if self.change_type is ChangeType.TYPE_CHANGED:
            return (f"{self.path}: {type(self.old_value).__name__} → "
                    f"{type(self.new_value).__name__} ({self.old_value!r} → {self.new_value!r})")
        return f"{self.path}: {self.old_value!r} → {self.new_value!r}"


_ABSENT = object()


def _walk_diff(before: Any, after: Any, path: str, out: list[StateChange]) -> None:
    """يمشي في البنيتين بالتوازي ويسجّل كل اختلاف بمساره الكامل."""
    if before is _ABSENT:
        out.append(StateChange(path, ChangeType.ADDED, None, after))
        return
    if after is _ABSENT:
        out.append(StateChange(path, ChangeType.REMOVED, before, None))
        return

    if type(before) is not type(after) and not (
        isinstance(before, (int, float)) and isinstance(after, (int, float))
    ):
        out.append(StateChange(path, ChangeType.TYPE_CHANGED, before, after))
        return

    if isinstance(before, dict):
        for k in sorted(set(before) | set(after), key=str):
            _walk_diff(before.get(k, _ABSENT), after.get(k, _ABSENT), f"{path}.{k}", out)
        return

    if isinstance(before, list):
        for i in range(max(len(before), len(after))):
            b = before[i] if i < len(before) else _ABSENT
            a = after[i] if i < len(after) else _ABSENT
            _walk_diff(b, a, f"{path}[{i}]", out)
        return

    if before != after:
        out.append(StateChange(path, ChangeType.MODIFIED, before, after))


def _deep_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# 2. ToolCall  /  3. ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """استدعاء أداة واحد. منفصل عن Step لأن الخطوة قد تحمل استدعاءً أو لا."""
    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: _new_id("call"))
    timestamp: float = 0.0


@dataclass
class ToolResult:
    """
    نتيجة استدعاء أداة. مرتبطة بـ call_id لإمكانية إعادة الربط،
    و`succeeded` تفرّق بين "الأداة فشلت تقنيًا" و"الأداة نجحت لكن النتيجة خاطئة"
    — وهذا تمييز جوهري في التشخيص.
    """
    call_id: str
    tool_output: Any = None
    succeeded: bool = True
    error_message: Optional[str] = None
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# 4. Step
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """خطوة واحدة داخل تنفيذ الوكيل. الحقول مطابقة لمواصفة Trace Logging."""
    step_id: str
    step_type: StepType
    timestamp: float

    parent_step_id: Optional[str] = None

    input: Any = None
    decision: Optional[str] = None

    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None

    state_before: StateSnapshot = field(default_factory=StateSnapshot)
    state_after: StateSnapshot = field(default_factory=StateSnapshot)

    expected_outcome: Any = None
    observed_outcome: Any = None

    # يُملأ من محرك التشخيص لا عند التسجيل
    status: StepStatus = StepStatus.UNVERIFIABLE
    divergence_reason: Optional[str] = None

    # اختصارات للوصول المسطّح كما في المواصفة (tool_name / tool_input / tool_output)
    @property
    def tool_name(self) -> Optional[str]:
        return self.tool_call.tool_name if self.tool_call else None

    @property
    def tool_input(self) -> Optional[dict]:
        return self.tool_call.tool_input if self.tool_call else None

    @property
    def tool_output(self) -> Any:
        return self.tool_result.tool_output if self.tool_result else None

    def state_change(self) -> dict[str, dict[str, Any]]:
        """تغيّرات على مستوى المفتاح الأعلى (سلوك أصلي)."""
        return self.state_before.diff(self.state_after)

    def state_path_changes(self) -> list["StateChange"]:
        """تغيّرات محددة بمسارها الكامل داخل البنية المتداخلة."""
        return self.state_before.path_diff(self.state_after)


# ---------------------------------------------------------------------------
# 5. ExecutionTrace
# ---------------------------------------------------------------------------

@dataclass
class ExecutionTrace:
    trace_id: str
    task_description: str = ""
    steps: list[Step] = field(default_factory=list)
    final_status: str = "unknown"  # success | failed | unknown

    def add_step(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def ordered_steps(self) -> list[Step]:
        """
        إعادة بناء التسلسل الزمني. `timestamp` هو مصدر الحقيقة وليس ترتيب
        الإدخال في القائمة، لأن الخطوات قد تُسجَّل بترتيب غير مضمون.
        """
        return sorted(self.steps, key=lambda s: s.timestamp)

    def get(self, step_id: str) -> Optional[Step]:
        return next((s for s in self.steps if s.step_id == step_id), None)


# ---------------------------------------------------------------------------
# 6. ImpactChain  /  7. DivergenceReport
# ---------------------------------------------------------------------------

@dataclass
class ImpactLink:
    """حلقة واحدة في سلسلة التأثير: كيف انتقل أثر خطوة إلى التي تليها."""
    from_step_id: str
    to_step_id: str
    mechanism: str          # ماذا انتقل بالضبط (أي مفتاح حالة أو أي مخرج)
    downstream_status: StepStatus


@dataclass
class ImpactChain:
    root_step_id: str
    links: list[ImpactLink] = field(default_factory=list)
    final_outcome: str = "unknown"

    def as_arrow_text(self) -> str:
        if not self.links:
            return f"{self.root_step_id} → (no downstream steps)"
        parts = [self.root_step_id]
        for l in self.links:
            parts.append(f"{l.to_step_id} [{l.downstream_status.value}]")
        return " → ".join(parts) + f" → {self.final_outcome}"


@dataclass
class DivergenceReport:
    """
    التقرير النهائي. يجيب على الأسئلة الأربعة المطلوبة صراحة:
    أين بدأ الخطأ / ما نوعه / لماذا هو السبب الجذري / ما أثره.
    """
    trace_id: str
    task_description: str
    total_steps: int

    diverged: bool
    first_divergence_step_id: Optional[str]
    category: DivergenceCategory

    expected: Any = None
    observed: Any = None

    why_divergent: str = ""
    why_not_later_steps: str = ""

    impact_chain: Optional[ImpactChain] = None
    step_statuses: dict[str, StepStatus] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["step_statuses"] = {k: v.value for k, v in self.step_statuses.items()}
        if self.impact_chain:
            d["impact_chain"]["links"] = [
                {**asdict(l), "downstream_status": l.downstream_status.value}
                for l in self.impact_chain.links
            ]
        return d

    def to_text(self) -> str:
        if not self.diverged:
            return (
                f"Trace: {self.trace_id}\n"
                f"No divergence detected across {self.total_steps} steps.\n"
                f"(Steps without expected_outcome are unverifiable, not passing.)"
            )

        lines = [
            f"Trace: {self.trace_id}",
            f"Task:  {self.task_description}",
            "",
            "═══ ROOT CAUSE REPORT ═══",
            "",
            "WHERE did the error begin?",
            f"  → Step {self.first_divergence_step_id}",
            "",
            "WHAT type of error?",
            f"  → {self.category.value}",
            f"  Expected: {self.expected}",
            f"  Observed: {self.observed}",
            "",
            "WHY is this the root cause?",
            f"  {self.why_divergent}",
            f"  {self.why_not_later_steps}",
            "",
            "WHAT was its impact on the rest of the chain?",
        ]
        if self.impact_chain:
            lines.append(f"  {self.impact_chain.as_arrow_text()}")
            lines.append("")
            for l in self.impact_chain.links:
                lines.append(f"  {l.from_step_id} → {l.to_step_id}: {l.mechanism}")
        else:
            lines.append("  (no downstream steps affected)")

        return "\n".join(lines)
