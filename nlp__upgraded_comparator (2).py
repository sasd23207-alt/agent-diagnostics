"""
nlp.upgraded_comparator
========================
الموصل الفعلي: يلفّ nlp.structural_comparator.free_text_similarity داخل
core.comparators.CallableComparator (الموجودة أصلًا منذ المرحلة الثانية،
غير مُعدَّلة الآن)، ثم يوجّه بين strict والنص الحر بنفس منطق التوجيه
المستخدم أصلًا (إعادة استخدام عبر استيراد، لا نسخ منطق جديد).

هذا هو الكائن الذي يُمرَّر لـ core.engine.diagnose(trace, comparator=...).
لم يُلمس core/engine.py ولا core/comparators.py إطلاقًا.
"""

from __future__ import annotations

from typing import Any

from core.comparators import CallableComparator, _looks_like_identifier
from core.divergence import StrictComparator
from nlp.structural_comparator import free_text_similarity


class StructuralHybridComparator:
    """
    قابل للتبديل بالكامل: بديل مباشر لأي OutcomeComparator آخر.

        diagnose(trace, comparator=StructuralHybridComparator())

    التوجيه: نفس قاعدة core.comparators.HybridComparator بالضبط (قيمة
    غير نصية أو معرّف بلا مسافات → strict؛ نص طبيعي → الطبقة الجديدة).
    الفرق الوحيد: النص الطبيعي يمر عبر مقارن بنيوي يحفظ الترتيب والدور،
    لا حقيبة كلمات، مع احتياطي معجمي عند عدم انطباق أي نمط بنيوي.
    """

    def __init__(self, threshold: float = 0.75):
        self.strict = StrictComparator()
        # هذا هو "الموصل الفعلي": CallableComparator حقيقية من core،
        # متصلة بدالة بنيوية حقيقية، لا دالة وهمية.
        self.free_text = CallableComparator(
            free_text_similarity, threshold=threshold, label="structural (order+role aware)"
        )
        self.last_mode: str = "strict"

    def _route(self, expected: Any, observed: Any) -> str:
        if not isinstance(expected, str) or not isinstance(observed, str):
            return "strict"
        if _looks_like_identifier(expected) or _looks_like_identifier(observed):
            return "strict"
        return "structural"

    def matches(self, expected: Any, observed: Any) -> bool:
        mode = self._route(expected, observed)
        self.last_mode = mode
        return (self.strict if mode == "strict" else self.free_text).matches(expected, observed)

    def describe(self) -> str:
        return "structural-hybrid (strict for structured/identifiers, order+role-aware for free text)"
