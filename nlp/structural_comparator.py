"""
nlp.structural_comparator
==========================
مقارن نص حر **يحفظ الترتيب والدور**، لا يعامل الجملة كحقيبة كلمات.

هذا ليس نموذجًا عصبيًا — لا يوجد اتصال شبكة في بيئة التنفيذ، فلا يمكن
وصل embeddings حقيقية. هذا مستخرِج قواعد صريح (rule-based) يستهدف تحديدًا
الأنماط الستة التي كسرت المقارن السابق في المرحلة الرابعة:

  ترتيب الكلمات → لا يُستخدم إطلاقًا هنا؛ كل استخراج يحافظ على من سبق من
  الدور النحوي   → فاعل/مفعول عبر موقع الكلمة قبل/بعد الفعل
  النفي          → يُتتبَّع كرمز في تسلسله، لا يُطرح جانبًا
  السبب/النتيجة  → يُستخرَج كزوج (سبب, نتيجة) مُطبَّع بغض النظر عن صياغة السطح
  الاتجاه        → أزواج from/to/before/after تُقارَن بالترتيب لا كمجموعة
  الفاعل/المفعول → موقع الاسم قبل الفعل مقابل بعده، لفعل معروف

هذا الملف مستقل تمامًا عن core/. لا يستورد engine ولا divergence ولا
tracing. يُستهلَك فقط عبر core.comparators.CallableComparator من خارج
هذا الملف — وهو الموصل الفعلي المطلوب، بلا أي تعديل على core/comparators.py.
"""

from __future__ import annotations

import re
from typing import Optional

# نعيد استخدام أدوات التطبيع الموجودة فعليًا (ترادف، تجذير، كلمات محتوى)
# عبر استيراد -- لا تعديل -- من core.comparators. هذا استهلاك لواجهة
# موجودة، وليس لمسًا للملف.
from core.comparators import (
    _content_words, _normalize_word, _tokenize, _numbers,
    _has_negation, _antonym_conflict, _structured_tokens,
)

# نستخدم _content_words للفحوصات العامة، لكن الفحوصات القائمة على عبارات
# قصيرة (سبب/نتيجة/مقارنة) تحتاج استخراجًا أكثر تحفّظًا: _content_words
# يحذف "a" كأداة تعريف، وهذا يتلف معرّفات أحادية الحرف حقيقية مثل
# "user A" مقابل "user B" — عيب حقيقي اكتُشف بالاختبار على tier4، لا حد
# نظري. الحل: مجموعة أدوات وظيفية أضيق، تُبقي أي رمز وحيد الحرف لأنه
# غالبًا معرّف (A/B/X/Y) لا أداة تعريف حين يظهر بعد اسم مثل "user".
_MINIMAL_FUNCTION_WORDS = {"the", "an", "is", "was", "are", "were", "to", "of", "that", "this"}


def _phrase_words(text: str) -> frozenset:
    """استخراج محافظ لكلمات عبارة: يحذف أدوات وظيفية أساسية فقط، ويُبقي
    المعرّفات وحيدة الحرف كما هي (بخلاف _content_words العام)."""
    toks = _tokenize(text)
    return frozenset(_normalize_word(t) for t in toks if t not in _MINIMAL_FUNCTION_WORDS)


# ---------------------------------------------------------------------------
# 1) استخراج النطاق المنطقي (نفي × كمّية) — يحل محل قاعدة الأمان السابقة
# ---------------------------------------------------------------------------

_LOGICAL_MARKERS = ("not", "no", "never", "all", "every", "some", "any", "none")


def _logical_skeleton(text: str) -> tuple[str, ...]:
    """
    تسلسل رموز النفي/الكمّية بترتيب ورودها في الجملة، محتفظًا بالترتيب.
    "not all users" → ("not", "all")
    "all users not" → ("all", "not")
    هذان تسلسلان مختلفان رغم أن كل الكلمات الأخرى متطابقة — وهذا بالضبط
    الفرق بين "بعض المستخدمين لم يتأثروا" و"لا أحد تأثر".
    """
    toks = _tokenize(text)
    return tuple(t for t in toks if t in _LOGICAL_MARKERS)


def check_quantifier_scope(a: str, b: str) -> Optional[bool]:
    skel_a, skel_b = _logical_skeleton(a), _logical_skeleton(b)
    if len(skel_a) < 2 or len(skel_b) < 2:
        return None  # لا يوجد نطاق منطقي كافٍ للحكم، اترك القرار لغيره
    if set(skel_a) != set(skel_b):
        return None  # رموز مختلفة أصلًا — ليس هذا الفحص المختص
    if skel_a == skel_b:
        return None  # نفس الترتيب — غير حاسم بمفرده، لا يمنع فحوصات أخرى
    return False  # نفس الرموز، ترتيب مختلف = انعكاس نطاق منطقي حقيقي


# ---------------------------------------------------------------------------
# 2) استخراج السبب/النتيجة — يُطبَّع بغض النظر عن صياغة السطح
# ---------------------------------------------------------------------------

_CAUSAL_PATTERNS = [
    # (regex, cause_group, effect_group) — ترتيب المحاولة مهم: الأكثر تحديدًا أولًا
    (re.compile(r"^(.*?)\bwas caused by\b(.*)$"), 2, 1),
    (re.compile(r"^(.*?)\bwere caused by\b(.*)$"), 2, 1),
    (re.compile(r"^(.*?)\bcaused by\b(.*)$"), 2, 1),
    (re.compile(r"^(.*?)\bled to\b(.*)$"), 1, 2),
    (re.compile(r"^(.*?)\bresulted in\b(.*)$"), 1, 2),
    (re.compile(r"^(.*?)\bcaused\b(.*)$"), 1, 2),
    (re.compile(r"^(.*?),?\s*\bbecause\b(.*)$"), 1, 2),
    (re.compile(r"^(.*?)\bdue to\b(.*)$"), 2, 1),
]


def extract_causal(text: str) -> Optional[tuple[frozenset, frozenset]]:
    t = text.strip().lower()
    for pattern, cause_g, effect_g in _CAUSAL_PATTERNS:
        m = pattern.search(t)
        if m:
            cause = _phrase_words(m.group(cause_g))
            effect = _phrase_words(m.group(effect_g))
            if cause and effect:
                return cause, effect
    return None


def _phrase_overlap(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def check_causal_direction(a: str, b: str) -> Optional[bool]:
    ca = extract_causal(a)
    cb = extract_causal(b)
    if ca is None or cb is None:
        return None
    (cause_a, effect_a), (cause_b, effect_b) = ca, cb

    same_direction = _phrase_overlap(cause_a, cause_b) >= 0.5 and _phrase_overlap(effect_a, effect_b) >= 0.5
    swapped = _phrase_overlap(cause_a, effect_b) >= 0.5 and _phrase_overlap(effect_a, cause_b) >= 0.5

    if same_direction and not swapped:
        return True
    if swapped and not same_direction:
        return False
    return None  # غامض (كلاهما أو لا شيء) — لا نجازف بحكم غير مؤكد


# ---------------------------------------------------------------------------
# 3) استخراج اتجاه المقارنة ("A أسرع من B" مقابل "B أسرع من A")
# ---------------------------------------------------------------------------

_COMPARISON_RE = re.compile(
    r"^(.*?)\b(?:is|was|are|were)?\s*"
    r"(faster|slower|greater|less|more|higher|lower|earlier|later|bigger|smaller|older|newer|"
    r"cheaper|more expensive|stronger|weaker)\s+than\b(.*)$"
)


def extract_comparison(text: str) -> Optional[tuple[frozenset, str, frozenset]]:
    t = text.strip().lower()
    m = _COMPARISON_RE.search(t)
    if not m:
        return None
    subject = _phrase_words(m.group(1))
    # الفعل الوصفي المشترك (مثل "responded"/"completed") يتكرر عادة في
    # الجملتين على جانب الفاعل فقط، فيُخفي الفرق الحقيقي خلف كلمة حشو
    # مشتركة لا تُزال بمقارنة الجانبين الأربعة. نزيله بحذر: كلمة تنتهي
    # بـ"ed" وطولها > 4 غالبًا فعل ماضٍ لا معرّفًا مميّزًا.
    subject = frozenset(w for w in subject if not (w.endswith("ed") and len(w) > 4))
    comparator_word = m.group(2)
    obj = _phrase_words(m.group(3))
    if not subject or not obj:
        return None
    return subject, comparator_word, obj


def check_comparison_direction(a: str, b: str) -> Optional[bool]:
    ca, cb = extract_comparison(a), extract_comparison(b)
    if ca is None or cb is None:
        return None
    (subj_a, cmp_a, obj_a), (subj_b, cmp_b, obj_b) = ca, cb
    if cmp_a != cmp_b:
        return None

    # كلمات مشتركة بين كل الأطراف الأربعة (مثل "server" أو الفعل نفسه
    # المكرر في كلا الشقين) لا تميّز شيئًا وتُخفي الفرق الحقيقي خلف عتبة
    # تشابه فضفاضة — نفس العطب المُكتشَف سابقًا في check_subject_object،
    # هنا يظهر عبر أربع مجموعات لا مجموعتين فقط.
    boilerplate = subj_a & obj_a & subj_b & obj_b
    da, oa = subj_a - boilerplate, obj_a - boilerplate
    db, ob = subj_b - boilerplate, obj_b - boilerplate

    if not da or not oa or not db or not ob:
        return None  # لا يوجد جزء مميّز كافٍ بعد إزالة الحشو — لا نجازف

    if da == db and oa == ob:
        return True
    if da == ob and oa == db:
        return False
    return None


# ---------------------------------------------------------------------------
# 4) استخراج الاتجاه (from/to/before/after) — بالترتيب لا كمجموعة
# ---------------------------------------------------------------------------

_ROLE_MARKERS = ("from", "to", "into", "onto", "before", "after", "above", "below")


def extract_directional_roles(text: str) -> tuple[dict[str, frozenset], frozenset]:
    """
    يرجّع (roles, residual): roles كما في السابق، وresidual هو كل محتوى
    الجملة **خارج** العبارات المُلتقَطة بعد أي علامة اتجاهية — أي الجزء
    الذي كان يُتجاهَل بالكامل سابقًا. هذا اكتُشف كخطأ حقيقي: "refund 100
    eur to customer" و"refund 250 eur to customer" لهما نفس عبارة "to"
    تمامًا (customer)، فكان الفحص يُعلن تطابقًا بينما الفرق الحقيقي (المبلغ)
    خارج نطاق ما يقارنه إطلاقًا.
    """
    toks = _tokenize(text)
    roles: dict[str, list[str]] = {}
    consumed = set()
    i = 0
    while i < len(toks):
        if toks[i] in _ROLE_MARKERS:
            consumed.add(i)  # العلامة نفسها
            marker = toks[i]
            j = i + 1
            phrase = []
            while j < len(toks) and toks[j] not in _ROLE_MARKERS:
                phrase.append(toks[j])
                consumed.add(j)
                j += 1
            if marker not in roles:
                roles[marker] = phrase
            i = j
        else:
            i += 1

    role_dict = {k: _phrase_words(" ".join(v)) for k, v in roles.items()}
    residual_tokens = [t for idx, t in enumerate(toks) if idx not in consumed]
    residual = _phrase_words(" ".join(residual_tokens))
    return role_dict, residual


def check_directional_roles(a: str, b: str) -> Optional[bool]:
    (ra, residual_a), (rb, residual_b) = extract_directional_roles(a), extract_directional_roles(b)
    shared = set(ra) & set(rb)
    if not shared:
        return None

    for marker in shared:
        va, vb = ra[marker], rb[marker]
        if va and vb and _phrase_overlap(va, vb) < 0.4:
            return False  # نفس العلامة، عبارة مختلفة تمامًا = انعكاس/استبدال حقيقي

    # العلامات المشتركة متطابقة — الآن نفحص ما تبقى من الجملة (كان يُتجاهَل
    # سابقًا بالكامل، وهذا بالضبط ما أخفى فرق الأرقام وأسماء القنوات).
    # نشترط أن تكون البقيتان متعددتَي الكلمات على الجانبين قبل اعتبار
    # الاختلاف حاسمًا: البقية أحادية الكلمة شائعة جدًا عند إعادة الصياغة
    # بين المبني للمعلوم والمبني للمجهول (فعل وحيد متبقٍّ في جانب، عبارة
    # اسمية كاملة في الآخر) — اكتُشف هذا فعليًا كمصدر FP جديد عبر اختبار
    # المرجع، وليس خللًا نظريًا.
    if not residual_a and not residual_b:
        return True
    if residual_a == residual_b:
        return True
    if len(residual_a) >= 2 and len(residual_b) >= 2:
        return False if _phrase_overlap(residual_a, residual_b) < 0.3 else None
    return None


# ---------------------------------------------------------------------------
# 5) الفاعل/المفعول حول فعل معروف — يمسك انعكاسًا لا تكشفه الأدوار وحدها
# ---------------------------------------------------------------------------

_VERB_LEXICON = {
    "escalate", "escalated", "transfer", "transferred", "migrate", "migrated",
    "approve", "approved", "reject", "rejected", "notify", "notified",
    "deploy", "deployed", "review", "reviewed", "complete", "completed",
    "finish", "finished", "deliver", "delivered", "provision", "provisioned",
    "scale", "scaled", "cancel", "cancelled", "confirm", "confirmed",
    "grant", "granted", "revoke", "revoked", "send", "sent", "assign", "assigned",
}

_STOPWORDS_LOCAL = {"the", "an", "to", "of", "in", "on", "at", "for", "with", "by"}


def extract_subject_object(text: str) -> Optional[tuple[str, frozenset, frozenset]]:
    toks = _tokenize(text)
    verb_idx = next((i for i, t in enumerate(toks) if t in _VERB_LEXICON), None)
    if verb_idx is None:
        return None

    verb = _normalize_word(toks[verb_idx])
    before = [t for t in toks[:verb_idx] if t not in _STOPWORDS_LOCAL]
    after = toks[verb_idx + 1:]
    obj_tokens = []
    for t in after:
        if t in _ROLE_MARKERS:
            break
        if t not in _STOPWORDS_LOCAL:
            obj_tokens.append(t)

    if not before:
        return None
    subject = frozenset(_normalize_word(w) for w in before)
    obj = frozenset(_normalize_word(w) for w in obj_tokens)
    return verb, subject, obj


def check_subject_object(a: str, b: str) -> Optional[bool]:
    sa, sb = extract_subject_object(a), extract_subject_object(b)
    if sa is None or sb is None:
        return None
    verb_a, subj_a, obj_a = sa
    verb_b, subj_b, obj_b = sb

    # شرط أساسي مفقود سابقًا: هذا الفحص يقارن *أدوار حول فعل*، فإن كان
    # الفعل نفسه مختلفًا (مثل "approved" مقابل "rejected") فلا معنى
    # لمقارنة الفاعل/المفعول إطلاقًا — الاختلاف في الفعل نفسه، لا الدور.
    # اكتُشف هذا كخطأ حقيقي: فاعل فارغ متطابق على الجانبين خدع الفحص
    # ليُعلن "تطابق" رغم تضاد الفعلين تمامًا.
    if verb_a != verb_b:
        return None  # ليس هذا الفحص المختص بمعنى الفعل، اترك القرار لغيره

    if not subj_a or not subj_b:
        return None

    distinct_a = subj_a - obj_a
    distinct_b = subj_b - obj_b

    if not distinct_a or not distinct_b:
        return True if _phrase_overlap(subj_a, subj_b) >= 0.8 else None

    if distinct_a == distinct_b:
        return True
    if distinct_a.isdisjoint(distinct_b):
        return False
    return None


# ---------------------------------------------------------------------------
# النقطة المُوحَّدة: أول فحص حاسم يفوز؛ الرفض (False) أقوى من القبول دائمًا
# ---------------------------------------------------------------------------

_CHECKS = [
    check_quantifier_scope,
    check_causal_direction,
    check_comparison_direction,
    check_directional_roles,
    check_subject_object,
]


def structural_verdict(expected: str, observed: str) -> Optional[bool]:
    """
    يرجّع True (متطابقان بنيويًا)، False (تناقض بنيوي مؤكد)، أو None
    (لا فحص بنيوي ينطبق — القرار يُترك للمقارن الاحتياطي).

    قاعدة الأمان: أي False من أي فحص يحسم النتيجة فورًا — حتى لو رجّح
    فحص آخر True. رفض خاطئ (FN) مزعج لكنه مرئي؛ قبول خاطئ (FP) يُخفي
    انحرافًا حقيقيًا. نرجّح دائمًا جانب الرفض عند التعارض.
    """
    results = [check(expected, observed) for check in _CHECKS]
    if False in results:
        return False
    if True in results:
        return True

    # حارس أمان أخير: الاحتياطي المعجمي (حقيبة كلمات) لا يفهم الترتيب
    # إطلاقًا. إن كانت الجملتان تحملان نمطًا حسّاسًا للترتيب أصلًا (مقارنة
    # أو سبب/نتيجة أو فاعل/مفعول اكتُشف على الجانبين) لكن كل الفحوصات
    # الخمسة عجزت عن الحسم، ومع ذلك حقيبة الكلمات متطابقة تمامًا — تمرير
    # هذا للاحتياطي الأعمى يعني إعلان تطابق زائف بثقة كاملة (اكتُشف هذا
    # فعليًا: "server X ... than server Y" مقابل عكسها، كلمات متطابقة
    # 100%). الشك هنا يجب أن يُحسَم لصالح الرفض، لا القبول.
    pattern_on_both = (
        (extract_comparison(expected) is not None and extract_comparison(observed) is not None)
        or (extract_causal(expected) is not None and extract_causal(observed) is not None)
        or (extract_subject_object(expected) is not None and extract_subject_object(observed) is not None)
    )
    if pattern_on_both:
        from core.comparators import _content_words
        if _content_words(expected) == _content_words(observed) and expected.strip().lower() != observed.strip().lower():
            return False

    return None


def _safe_lexical_fallback(expected: str, observed: str) -> float:
    """
    احتياطي معجمي محلي، مبني عمدًا بدون استدعاء
    core.comparators.SemanticComparator.explain() كاملة: تلك الدالة تملك
    فحص أدوار اتجاهية خاصًا بها (_roles_conflict) يحمل نفس عطب المبني
    للمجهول المكتشف هنا في هذا الملف — استدعاؤها كان يُنتج حكمًا قاطعًا
    خاطئًا (False) على إعادة صياغة مشروعة بالمبني للمجهول، عبر مسار لا
    نملك حق تعديله (core/comparators.py غير مسموح لمسه في هذه المرحلة).

    هذه النسخة تعيد استخدام النفي والتضاد والأرقام والرموز المنظمة (عبر
    استيراد فقط، لا نسخ منطق) لكنها تتجاهل فحص الأدوار القديم عمدًا،
    وتعتمد بدلًا منه على فحوصات الاتجاه/الفاعل الأحدث في هذا الملف
    (المُشغَّلة مسبقًا في structural_verdict قبل الوصول لهذا الاحتياطي).
    """
    e, o = expected.strip().lower(), observed.strip().lower()
    if e == o:
        return 1.0
    if _has_negation(e) != _has_negation(o):
        return 0.0
    se, so = _structured_tokens(expected), _structured_tokens(observed)
    if se != so:
        return 0.0
    if _numbers(e) != _numbers(o):
        return 0.0

    ew, ow = _content_words(e), _content_words(o)
    if not ew and not ow:
        return 1.0
    if not ew or not ow:
        return 0.0
    if _antonym_conflict(ew, ow):
        return 0.0

    shared = ew & ow
    only_e, only_o = ew - ow, ow - ew
    if only_e and only_o:
        return 0.0
    return len(shared) / max(len(ew), len(ow))


def free_text_similarity(expected, observed) -> float:
    """
    الدالة التي تُوصَل فعليًا عبر core.comparators.CallableComparator.
    تُستدعى فقط على قيم نصية (التوجيه بين strict/free-text يبقى خارج هذا
    الملف، في الراوتر في nlp/upgraded_comparator.py).
    """
    if not isinstance(expected, str) or not isinstance(observed, str):
        return 1.0 if expected == observed else 0.0

    e, o = expected.strip(), observed.strip()
    if e.lower() == o.lower():
        return 1.0

    verdict = structural_verdict(e, o)
    if verdict is True:
        return 1.0
    if verdict is False:
        return 0.0

    # لا فحص بنيوي ينطبق: احتياطي معجمي آمن لتغطية إعادة الصياغة المفتوحة
    # المفردات — بدون إعادة استخدام فحص الأدوار المعطوب من core.comparators.
    return _safe_lexical_fallback(e, o)
