"""
core.comparators
================
طبقة مضافة فوق النواة. لا تعدّل models / tracing / engine / divergence.

تُستخدم عبر نقطة الامتداد الموجودة أصلًا:

    from core.engine import diagnose
    from core.comparators import HybridComparator
    report = diagnose(trace, comparator=HybridComparator())

قيد تصميمي مهم:
بروتوكول OutcomeComparator الحالي هو matches(expected, observed) — بلا سياق
الخطوة. تغييره يعني تعديل النواة، وهذا ممنوع في هذه المرحلة. لذلك التوجيه
بين strict و semantic يتم من **شكل القيمة** لا من step_type:

    dict / list / رقم / bool        → strict  (قيم منظمة)
    نص يشبه معرّفًا (snake_case)     → strict  (أسماء أدوات، أكواد)
    نص طبيعي (فيه مسافات)           → semantic

هذا التقريب يصيب في كل حالات الاختبار الحالية، لكنه تقريب — موثّق في القيود.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .divergence import StrictComparator


# ---------------------------------------------------------------------------
# معجم التطبيع الدلالي (قابل للتوسعة/الاستبدال)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "for", "with", "as", "by", "from", "that", "this",
    "it", "its", "has", "have", "had", "and", "or", "but", "into", "out",
}

# مجموعات ترادف: كل مجموعة تُطبَّع إلى ممثّل واحد.
SYNONYM_GROUPS: list[set[str]] = [
    {"issue", "problem", "trouble", "fault"},
    {"error", "failure", "failed", "fail"},
    {"customer", "client", "user"},
    {"eligible", "qualified", "entitled"},
    {"approve", "accept", "grant", "authorize"},
    {"reject", "deny", "decline", "refuse"},
    {"escalate", "elevate", "forward"},
    {"summary", "summarization", "digest"},
    {"required", "needed", "need", "necessary"},
    {"result", "results", "outcome", "output"},
    {"select", "selected", "choose", "chose", "pick", "picked"},
    {"call", "invoke", "invocation", "invoked"},
    {"retry", "reattempt"},
    {"data", "info", "information"},
    {"plan", "planning"},
    {"unavailable", "missing", "absent"},
]

_SYN_CANON: dict[str, str] = {}
for _group in SYNONYM_GROUPS:
    _rep = sorted(_group)[0]
    for _w in _group:
        _SYN_CANON[_w] = _rep

NEGATIONS = {"not", "no", "never", "without", "non", "un", "cannot", "cant", "isnt", "wasnt"}

# أزواج متضادة صريحة. ضرورية لأن قاعدة "المحتوى المميّز على الجانبين" وحدها
# لا تكفي حين يكون الفرق كلمة واحدة فقط: "booking confirmed" مقابل
# "booking cancelled" — الجانبان يحملان كلمة مميّزة واحدة لكل منهما، وهو
# نمط تحمله أيضًا إعادات الصياغة المشروعة. التضاد الصريح يحسم الحالة.
ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("increase", "decrease"), ("increase", "reduce"), ("raise", "lower"),
    ("confirm", "cancel"), ("confirm", "abort"),
    ("accept", "decline"), ("enable", "disable"),
    ("open", "close"), ("start", "stop"), ("add", "remove"),
    ("grant", "revoke"), ("success", "failure"), ("valid", "invalid"),
    ("read", "write"), ("authenticate", "authorize"),
    ("before", "after"), ("include", "exclude"), ("allow", "block"),
]

_ANTONYMS: dict[str, set[str]] = {}
for _a, _b in ANTONYM_PAIRS:
    _ANTONYMS.setdefault(_a, set()).add(_b)
    _ANTONYMS.setdefault(_b, set()).add(_a)


def _antonym_conflict(ew: set[str], ow: set[str]) -> bool:
    """هل يحمل الطرفان كلمتين متضادتين صراحة؟"""
    for w in ew - ow:
        base = _stem(w)
        for opp in _ANTONYMS.get(w, set()) | _ANTONYMS.get(base, set()):
            if opp in ow or _normalize_word(opp) in ow:
                return True
    return False


# أنماط قيم منظمة داخل النص: أكواد عملات، معرّفات، تواريخ، أكواد حالة.
# هذه تُقارن بصرامة حتى لو وردت داخل جملة طبيعية.
_STRUCTURED_TOKEN = re.compile(
    r"^(?:[A-Z]{3}|[0-9a-f]{8,}|\d{4}-\d{2}-\d{2}|[A-Z_]{3,}\d*|\d+(?:\.\d+)?)$"
)


def _structured_tokens(text: str) -> set[str]:
    """
    يستخرج الرموز التي يجب أن تتطابق حرفيًا حتى داخل نص طبيعي:
    'refund 100 EUR' و'refund 100 USD' ليستا إعادة صياغة.
    """
    return {t for t in re.findall(r"\b[\w-]+\b", text) if _STRUCTURED_TOKEN.match(t)}

# لواحق صرفية بسيطة تُقطَع للتطبيع (تجنّب lemmatizer خارجي)
_SUFFIXES = ("ing", "ed", "es", "s")

# لواحق اشتقاقية تحوّل الفعل إلى اسم. غيابها عطب بنيوي لا نقص معجم:
# 'escalation' و'escalate' نفس المفهوم، وأي معجم ترادف لن ينفع ما لم
# تُردّ الصيغة الاسمية إلى فعلها أولًا.
_DERIVATIONAL = (
    ("ations", "ate"), ("ation", "ate"), ("ments", "ment"),
    ("ities", "ity"), ("ions", "ion"),
)


def _stem(w: str) -> str:
    for long_suf, replacement in _DERIVATIONAL:
        if len(w) > len(long_suf) + 2 and w.endswith(long_suf):
            return w[: -len(long_suf)] + replacement
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _normalize_word(w: str) -> str:
    """
    التطبيع: ترادف ← ثم تجذير ← ثم ترادف على الجذر.

    ترتيب هذه الخطوات ليس تفصيلًا. النسخة السابقة جذّرت ثم بحثت في جدول
    الترادف مرة واحدة فقط، فكانت 'escalated' تصبح 'escalat' ولا تطابق أبدًا
    'escalate' → 'elevate'. ظهر ذلك كـ"فشل في المبني للمجهول" وُثّق خطأً
    على أنه قصور دلالي يحتاج تحليلًا نحويًا — بينما كان خطأ ترتيب بسيطًا.
    """
    w = w.lower()
    if w in _SYN_CANON:
        return _SYN_CANON[w]

    stem = _stem(w)
    if stem in _SYN_CANON:
        return _SYN_CANON[stem]

    # قد يكون الجذر أقصر من المدخل في الجدول (approve/approv)
    for form in (stem + "e", stem):
        if form in _SYN_CANON:
            return _SYN_CANON[form]

    return stem


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_words(text: str) -> set[str]:
    return {
        _normalize_word(t) for t in _tokenize(text)
        if t not in STOPWORDS and t not in NEGATIONS and not t.isdigit()
    }


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _has_negation(text: str) -> bool:
    return bool(NEGATIONS & set(_tokenize(text)))


# حروف تحدد أدوارًا اتجاهية. عكسها يقلب المعنى تمامًا رغم تطابق الكلمات.
_ROLE_MARKERS = ("from", "to", "into", "for")


def _directional_roles(text: str) -> dict[str, str]:
    """
    يستخرج أدوارًا اتجاهية: "transfer from savings to checking"
      → {"from": "savings", "to": "checking"}

    لماذا هذا ضروري؟ المقارنة القائمة على مجموعة الكلمات تعطي تطابقًا تامًا
    لـ "from savings to checking" و"from checking to savings" — وهو خطأ من
    نوع false positive، أي أنه **يخفي انحرافًا حقيقيًا بصمت**. وهذا أخطر
    أنواع الفشل في أداة تشخيص.
    """
    tokens = _tokenize(text)
    roles: dict[str, str] = {}
    for i, t in enumerate(tokens):
        if t in _ROLE_MARKERS and i + 1 < len(tokens):
            nxt = next((w for w in tokens[i + 1:] if w not in STOPWORDS), None)
            if nxt:
                roles.setdefault(t, _normalize_word(nxt))
    return roles


def _roles_conflict(a: str, b: str) -> bool:
    ra, rb = _directional_roles(a), _directional_roles(b)
    shared = set(ra) & set(rb)
    # نشترط دورين على الأقل: دور واحد مختلف قد يكون مجرد صياغة مختلفة،
    # لكن اختلاف زوج كامل (from/to) يعني انقلاب الاتجاه.
    if len(shared) < 2:
        return False
    return any(ra[k] != rb[k] for k in shared)


def _looks_like_identifier(s: str) -> bool:
    """
    معرّف تقني (اسم أداة، كود حالة): بلا مسافات، أو snake/camel/kebab case.
    هذه القيم تُقارَن بصرامة دائمًا — الترادف لا معنى له في أسماء الأدوات.
    """
    s = s.strip()
    return bool(s) and " " not in s


# ---------------------------------------------------------------------------
# المقارن الدلالي
# ---------------------------------------------------------------------------

class SemanticComparator:
    """
    مقارنة دلالية محلية (بلا شبكة/نموذج خارجي).

    تعتبر النصّين متطابقين دلاليًا فقط إذا:
      1. تطابقا في النفي (أحدهما منفي والآخر لا → اختلاف فوري)
      2. تطابقا في الأرقام (100 EUR ≠ 250 EUR)
      3. لم يحمل كل طرف كلمةَ محتوى مميّزة لا يقابلها شيء في الطرف الآخر
         (هذا ما يمسك "billing department" مقابل "technical support")
      4. تجاوزت نسبة تداخل كلمات المحتوى العتبة

    الشرط (3) هو جوهر الفكرة: الفرق بين إعادة صياغة وتناقض هو أن التناقض
    يحمل محتوى مميّزًا على **الجانبين** معًا، بينما إعادة الصياغة تحمل
    مرادفات تُطبَّع لنفس الممثّل.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self._strict = StrictComparator()

    def explain(self, expected: Any, observed: Any) -> tuple[float, str]:
        """
        يرجّع (الدرجة، سبب القرار). وجود السبب ليس ترفًا: مقارن يقول
        'مختلفان' بلا تبرير يجعل تشخيص السبب الجذري غير قابل للمراجعة.
        """
        if not isinstance(expected, str) or not isinstance(observed, str):
            ok = self._strict.matches(expected, observed)
            return (1.0 if ok else 0.0), "non-text → strict comparison"

        e, o = expected.strip().lower(), observed.strip().lower()
        if e == o:
            return 1.0, "exact match"

        # 1) النفي: يقلب المعنى كليًا
        if _has_negation(e) != _has_negation(o):
            return 0.0, "negation mismatch"

        # 2) الرموز المنظمة داخل النص (أكواد، أرقام، عملات، تواريخ)
        se, so = _structured_tokens(expected), _structured_tokens(observed)
        if se != so:
            return 0.0, f"structured token mismatch: {sorted(se)} vs {sorted(so)}"

        # 3) الأرقام
        if _numbers(e) != _numbers(o):
            return 0.0, "numeric mismatch"

        # 4) الأدوار الاتجاهية (from/to معكوسة)
        if _roles_conflict(e, o):
            return 0.0, "directional roles reversed"

        ew, ow = _content_words(e), _content_words(o)
        if not ew and not ow:
            return 1.0, "no content words on either side"
        if not ew or not ow:
            return 0.0, "content present on one side only"

        # 5) التضاد الصريح
        if _antonym_conflict(ew, ow):
            return 0.0, "explicit antonym conflict"

        shared = ew & ow
        only_e, only_o = ew - ow, ow - ew

        # 6) محتوى مميّز على الجانبين = موضوعان مختلفان لا إعادة صياغة
        if only_e and only_o:
            return 0.0, f"distinct content on both sides: {sorted(only_e)} vs {sorted(only_o)}"

        score = len(shared) / max(len(ew), len(ow))
        return score, f"lexical overlap {len(shared)}/{max(len(ew), len(ow))}"

    def score(self, expected: Any, observed: Any) -> float:
        """درجة تشابه دلالي [0..1] — مفيدة للـ benchmark والمعايرة."""
        return self.explain(expected, observed)[0]

    def matches(self, expected: Any, observed: Any) -> bool:
        if expected is None:
            return True
        return self.score(expected, observed) >= self.threshold

    def describe(self) -> str:
        return f"semantic (local lexical-semantic, threshold={self.threshold})"


# ---------------------------------------------------------------------------
# المقارن الهجين — هو الافتراضي الموصى به
# ---------------------------------------------------------------------------

class HybridComparator:
    """
    يوجّه كل مقارنة إلى المقارن المناسب حسب شكل القيمة:
      - غير نصية (dict/list/int/bool)  → strict
      - نص يشبه معرّفًا (بلا مسافات)    → strict
      - نص طبيعي                        → semantic
    """

    def __init__(self, semantic: Optional[SemanticComparator] = None):
        self.strict = StrictComparator()
        self.semantic = semantic or SemanticComparator()
        self.last_mode: str = "strict"

    def route(self, expected: Any, observed: Any) -> str:
        if not isinstance(expected, str) or not isinstance(observed, str):
            return "strict"
        if _looks_like_identifier(expected) or _looks_like_identifier(observed):
            return "strict"
        return "semantic"

    def explain_route(self, expected: Any, observed: Any) -> str:
        mode = self.route(expected, observed)
        if mode == "strict":
            if not isinstance(expected, str) or not isinstance(observed, str):
                return "strict: non-text value (dict/list/number/bool)"
            return "strict: identifier-like token (tool name / code)"
        emb = _structured_tokens(str(expected)) | _structured_tokens(str(observed))
        extra = f"; structured tokens held strict: {sorted(emb)}" if emb else ""
        return f"semantic: natural-language text{extra}"

    def matches(self, expected: Any, observed: Any) -> bool:
        mode = self.route(expected, observed)
        self.last_mode = mode
        return (self.strict if mode == "strict" else self.semantic).matches(expected, observed)

    def describe(self) -> str:
        return "hybrid (strict for structured/identifiers, semantic for natural text)"


# ---------------------------------------------------------------------------
# محوّلات للمقارنات الخارجية (غير مفعّلة هنا — لا شبكة في هذه البيئة)
# ---------------------------------------------------------------------------

class CallableComparator:
    """
    يلفّ أي دالة خارجية بصيغة (expected, observed) -> bool أو float.
    هذه هي نقطة الوصل الجاهزة لـ embeddings أو LLM-as-judge لاحقًا،
    بدون أي تعديل على النواة:

        cmp = CallableComparator(my_embedding_fn, threshold=0.85, label="openai-embed")
        diagnose(trace, comparator=cmp)
    """

    def __init__(self, fn: Callable[[Any, Any], Any], threshold: float = 0.85,
                 label: str = "external"):
        self.fn = fn
        self.threshold = threshold
        self.label = label

    def matches(self, expected: Any, observed: Any) -> bool:
        if expected is None:
            return True
        out = self.fn(expected, observed)
        return bool(out) if isinstance(out, bool) else float(out) >= self.threshold

    def describe(self) -> str:
        return f"external:{self.label} (threshold={self.threshold})"


COMPARATORS = {
    "strict": StrictComparator,
    "semantic": SemanticComparator,
    "hybrid": HybridComparator,
}


def get_comparator(mode: str):
    if mode not in COMPARATORS:
        raise ValueError(f"unknown comparator mode '{mode}'; choose from {sorted(COMPARATORS)}")
    return COMPARATORS[mode]()
