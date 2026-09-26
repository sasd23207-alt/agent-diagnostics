"""
core.divergence
===============
محرك كشف أول نقطة انحراف.

القرار التصميمي الأهم هنا: المقارنة بين expected و observed تتم عبر
مُقارِن قابل للاستبدال (Comparator)، وليست مدمجة في المحرك.

لماذا؟ لأن المقارنة النصية الساذجة (مثل difflib) تقيس *شكل* الكلمات لا *معناها*:
"route to billing department" و"route to technical support" متشابهتان شكليًا
ومتناقضتان معنى. لذلك الافتراضي هنا مقارنة صارمة (strict) — تعطي false positives
أقل خطورة من false negatives في أداة تشخيص، ويمكن استبدالها لاحقًا بـ
embedding/LLM-judge بدون لمس منطق المحرك.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from .models import (
    ExecutionTrace, Step, StepStatus, StepType,
    DivergenceCategory, STEP_TYPE_TO_CATEGORY,
)


# ---------------------------------------------------------------------------
# المقارنة
# ---------------------------------------------------------------------------

class OutcomeComparator(Protocol):
    def matches(self, expected: Any, observed: Any) -> bool: ...
    def describe(self) -> str: ...


class StrictComparator:
    """
    المقارن الافتراضي: تطابق بعد تطبيع بسيط (حذف مسافات + توحيد حالة الأحرف).
    لا يحاول فهم المعنى — وهذا مقصود: أي اختلاف معنوي يجب أن يُبلَّغ عنه،
    وأي تطابق معنوي بصياغة مختلفة يجب أن يُحلّ بمقارن دلالي لاحقًا، لا بتخمين.
    """
    def matches(self, expected: Any, observed: Any) -> bool:
        if expected is None:
            return True
        if isinstance(expected, str) and isinstance(observed, str):
            return expected.strip().lower() == observed.strip().lower()
        return expected == observed

    def describe(self) -> str:
        return "strict (normalized exact match)"


class PredicateComparator:
    """
    مقارن يسمح بتمرير دالة تحقق مخصّصة لكل خطوة — مفيد عندما يكون
    "النجاح" شرطًا منطقيًا (مثل: النتيجة ضمن الميزانية) لا قيمة ثابتة.
    """
    def __init__(self, predicate: Callable[[Any, Any], bool], label: str = "custom predicate"):
        self._p = predicate
        self._label = label

    def matches(self, expected: Any, observed: Any) -> bool:
        return self._p(expected, observed)

    def describe(self) -> str:
        return self._label


# ---------------------------------------------------------------------------
# كشف الانحراف
# ---------------------------------------------------------------------------

class DivergenceDetector:

    def __init__(self, comparator: Optional[OutcomeComparator] = None):
        self.comparator = comparator or StrictComparator()

    # ---- تصنيف خطوة واحدة ------------------------------------------------
    def classify(self, step: Step) -> StepStatus:
        """
        ثلاث حالات فقط، بلا تدرّج مزيف:
          UNVERIFIABLE : لا يوجد expected_outcome → لا يمكن الحكم (وليس "ناجحة")
          OK           : المخرج يطابق المتوقع
          DIVERGENT    : المخرج لا يطابق المتوقع
        فشل الأداة تقنيًا (tool_result.succeeded == False) يُعتبر انحرافًا
        حتى لو لم يُحدَّد expected_outcome، لأنه فشل موضوعي لا يحتاج مرجعًا.
        """
        if step.tool_result is not None and not step.tool_result.succeeded:
            return StepStatus.DIVERGENT

        if step.expected_outcome is None:
            return StepStatus.UNVERIFIABLE

        return (StepStatus.OK if self.comparator.matches(step.expected_outcome, step.observed_outcome)
                else StepStatus.DIVERGENT)

    # ---- أول انحراف ------------------------------------------------------
    def find_first_divergence(self, trace: ExecutionTrace) -> tuple[Optional[Step], list[Step]]:
        """
        يمر بالترتيب الزمني ويرجّع أول خطوة DIVERGENT.
        هذا تحديدًا ما يمنع الوقوع في فخ "آخر خطأ = السبب": الفشل النهائي
        يظهر عادة في آخر الخطوات، لكن أول خروج عن المسار هو ما يجب إصلاحه.
        """
        steps = trace.ordered_steps()
        for s in steps:
            s.status = self.classify(s)

        first = next((s for s in steps if s.status == StepStatus.DIVERGENT), None)
        return first, steps

    # ---- سبب اعتبارها انحرافًا -------------------------------------------
    def explain_why_divergent(self, step: Step) -> str:
        if step.tool_result is not None and not step.tool_result.succeeded:
            return (
                f"Tool '{step.tool_name}' failed at invocation "
                f"(error: {step.tool_result.error_message})."
            )

        path_changes = step.state_path_changes()
        change_desc = (
            "; ".join(str(c) for c in path_changes[:5])
            + (f" (+{len(path_changes) - 5} more)" if len(path_changes) > 5 else "")
            if path_changes else "no state change recorded"
        )
        return (
            f"Step type '{step.step_type.value}' produced {step.observed_outcome!r} "
            f"where {step.expected_outcome!r} was required "
            f"[comparator: {self.comparator.describe()}]. "
            f"State transition at this step: {change_desc}."
        )

    # ---- لماذا الخطوات التالية ليست السبب الجذري -------------------------
    def explain_why_not_later(self, root: Step, steps: list[Step]) -> str:
        """
        يبني تبريرًا صريحًا مبنيًا على شرطين، لا على الترتيب وحده:

        (أ) شرط الأسبقية: كل خطوة قبل الجذر كانت OK أو UNVERIFIABLE — فلا يوجد
            انحراف أسبق منه.
        (ب) شرط الاعتمادية السببية: الخطوات المنحرفة اللاحقة تستهلك في
            state_before قيمًا أنتجها الجذر في state_after. أي أنها نتيجة
            لا سبب — إصلاحها لن يمنع الانحراف، وإصلاح الجذر يمنعها.
        """
        idx = steps.index(root)
        before = steps[:idx]
        after = steps[idx + 1:]

        # التمييز ضروري: خطوة UNVERIFIABLE لم تُفحص، وليست خطوة نجحت.
        # الخلط بينهما ينتج تبريرًا يبدو قاطعًا بينما هو مبني على فراغ.
        checked_before = [s for s in before if s.status == StepStatus.OK]
        unchecked_before = [s for s in before if s.status == StepStatus.UNVERIFIABLE]
        has_earlier_divergence = any(s.status == StepStatus.DIVERGENT for s in before)

        if has_earlier_divergence:
            part_a = "WARNING: an earlier divergent step exists — root selection is inconsistent."
        elif unchecked_before:
            part_a = (
                f"{len(checked_before)} preceding step(s) verified non-divergent, but "
                f"{len(unchecked_before)} ({', '.join(s.step_id for s in unchecked_before)}) "
                f"had no ground truth and were NOT checked. This root is the earliest "
                f"*detectable* divergence, not necessarily the earliest one."
            )
        else:
            part_a = (
                f"All {len(before)} preceding step(s) were verified non-divergent, "
                f"so no earlier deviation exists."
            )

        later_divergent = [s for s in after if s.status == StepStatus.DIVERGENT]
        if not later_divergent:
            return part_a + " No later divergent steps to rule out."

        produced = set(root.state_after.data.keys())
        dependent = []
        for s in later_divergent:
            consumed = set(s.state_before.data.keys())
            shared = produced & consumed
            # اعتماد سببي حقيقي: الخطوة اللاحقة تقرأ مفتاحًا كتبه الجذر بقيمة منحرفة
            if shared and any(
                s.state_before.data.get(k) == root.state_after.data.get(k) for k in shared
            ):
                dependent.append((s.step_id, sorted(shared)))

        if dependent:
            detail = "; ".join(f"{sid} consumes {keys}" for sid, keys in dependent)
            part_b = (
                f"The {len(later_divergent)} later divergent step(s) are downstream effects: "
                f"{detail} — state written by the root step. They are consequences, not causes; "
                f"fixing them would not prevent the divergence, fixing the root would."
            )
        else:
            part_b = (
                f"{len(later_divergent)} later divergent step(s) exist but show no direct state "
                f"dependency on the root step. They may represent an independent second fault "
                f"— worth separate investigation after fixing the root."
            )

        return f"{part_a} {part_b}"

    # ---- التصنيف ----------------------------------------------------------
    def categorize(self, step: Step) -> DivergenceCategory:
        # فشل تقني في الاستدعاء يُصنَّف كخطأ استدعاء بغض النظر عن نوع الخطوة
        if step.tool_result is not None and not step.tool_result.succeeded:
            return DivergenceCategory.TOOL_INVOCATION_ERROR
        return STEP_TYPE_TO_CATEGORY.get(step.step_type, DivergenceCategory.UNCLASSIFIED)
