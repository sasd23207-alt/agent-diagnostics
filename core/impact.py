"""
core.impact
===========
بناء سلسلة التأثير: كيف انتقل أثر أول انحراف عبر بقية الخطوات وصولًا
للنتيجة النهائية.

الفرق الجوهري عن "قائمة أخطاء": كل حلقة هنا تحدد *الآلية* (mechanism) —
أي ما الذي انتقل بالضبط من خطوة لأخرى (مفتاح حالة، مخرج أداة، قرار).
"""

from __future__ import annotations

from .models import (
    ExecutionTrace, Step, StepStatus,
    ImpactChain, ImpactLink,
)


class ImpactTracker:

    def build(self, trace: ExecutionTrace, steps: list[Step], root: Step) -> ImpactChain:
        idx = steps.index(root)
        downstream = steps[idx:]

        links: list[ImpactLink] = []
        for current, nxt in zip(downstream, downstream[1:]):
            links.append(ImpactLink(
                from_step_id=current.step_id,
                to_step_id=nxt.step_id,
                mechanism=self._mechanism(current, nxt),
                downstream_status=nxt.status,
            ))

        return ImpactChain(
            root_step_id=root.step_id,
            links=links,
            final_outcome=f"task {trace.final_status}",
        )

    def _mechanism(self, current: Step, nxt: Step) -> str:
        """
        يحدد ما الذي انتقل فعليًا من current إلى nxt.
        الترتيب مقصود: نفضّل الاعتماد على الحالة (أقوى دليل سببي)،
        ثم مخرج الأداة، ثم القرار، وأخيرًا التتابع الزمني المجرد (أضعف دليل).
        """
        produced = current.state_after.data
        consumed = nxt.state_before.data

        carried = [
            k for k in set(produced) & set(consumed)
            if produced[k] == consumed[k]
        ]
        if carried:
            vals = ", ".join(f"{k}={produced[k]!r}" for k in sorted(carried)[:3])
            return f"carried state [{vals}] into {nxt.step_type.value}"

        if current.tool_output is not None and nxt.input == current.tool_output:
            return f"tool output of '{current.tool_name}' became input of next step"

        if current.decision:
            return f"decision {current.decision!r} determined the next step"

        return "temporal succession only (no explicit data dependency recorded)"
