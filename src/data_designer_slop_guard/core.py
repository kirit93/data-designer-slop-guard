# Prose linter for AI slop patterns — extracted from https://github.com/eric-tramel/slop-guard
# Original license: MIT — Copyright (c) eric-tramel
#
# Runs ~80 compiled regex patterns against text and returns a numeric score (0-100),
# a list of specific violations with surrounding context, and actionable advice.

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hyperparameters:
    """Tunable thresholds, caps, and penalties used by the analyzer."""

    concentration_alpha: float = 2.5
    decay_lambda: float = 0.04
    claude_categories: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"contrast_pairs", "pithy_fragment", "setup_resolution"}
        )
    )

    context_window_chars: int = 60
    short_text_word_count: int = 10

    repeated_ngram_min_n: int = 4
    repeated_ngram_max_n: int = 8
    repeated_ngram_min_count: int = 3

    slop_word_penalty: int = -2
    slop_phrase_penalty: int = -3
    structural_bold_header_min: int = 3
    structural_bold_header_penalty: int = -5
    structural_bullet_run_min: int = 6
    structural_bullet_run_penalty: int = -3
    triadic_record_cap: int = 5
    triadic_penalty: int = -1
    triadic_advice_min: int = 3
    tone_penalty: int = -3
    sentence_opener_penalty: int = -2
    weasel_penalty: int = -2
    ai_disclosure_penalty: int = -10
    placeholder_penalty: int = -5
    rhythm_min_sentences: int = 5
    rhythm_cv_threshold: float = 0.3
    rhythm_penalty: int = -5
    em_dash_words_basis: float = 150.0
    em_dash_density_threshold: float = 1.0
    em_dash_penalty: int = -3
    contrast_record_cap: int = 5
    contrast_penalty: int = -1
    contrast_advice_min: int = 2
    setup_resolution_record_cap: int = 5
    setup_resolution_penalty: int = -3
    colon_words_basis: float = 150.0
    colon_density_threshold: float = 1.5
    colon_density_penalty: int = -3
    pithy_max_sentence_words: int = 6
    pithy_record_cap: int = 3
    pithy_penalty: int = -2
    bullet_density_threshold: float = 0.40
    bullet_density_penalty: int = -8
    blockquote_min_lines: int = 3
    blockquote_free_lines: int = 2
    blockquote_cap: int = 4
    blockquote_penalty_step: int = -3
    bold_bullet_run_min: int = 3
    bold_bullet_run_penalty: int = -5
    horizontal_rule_min: int = 4
    horizontal_rule_penalty: int = -3
    phrase_reuse_record_cap: int = 5
    phrase_reuse_penalty: int = -1

    density_words_basis: float = 1000.0
    score_min: int = 0
    score_max: int = 100
    band_clean_min: int = 80
    band_light_min: int = 60
    band_moderate_min: int = 40
    band_heavy_min: int = 20


DEFAULT_HYPERPARAMETERS = Hyperparameters()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    rule: str
    match: str
    context: str
    penalty: int

    def to_payload(self) -> dict[str, object]:
        return {
            "type": "Violation",
            "rule": self.rule,
            "match": self.match,
            "context": self.context,
            "penalty": self.penalty,
        }


@dataclass
class _RuleContext:
    text: str
    word_count: int
    sentences: list[str]
    advice: list[str]
    counts: dict[str, int]
    hp: Hyperparameters


@dataclass(frozen=True)
class _AnalysisContext:
    text: str
    word_count: int
    sentences: list[str]
    hp: Hyperparameters


@dataclass
class _RuleResult:
    violations: list[Violation]
    advice: list[str]
    count_deltas: dict[str, int]


@dataclass(frozen=True)
class _AnalysisState:
    violations: tuple[Violation, ...]
    advice: tuple[str, ...]
    counts: dict[str, int]

    @classmethod
    def initial(cls, counts: dict[str, int]) -> _AnalysisState:
        return cls(violations=(), advice=(), counts=dict(counts))

    def merge(self, violations: list[Violation], advice: list[str], count_deltas: dict[str, int]) -> _AnalysisState:
        merged_counts = dict(self.counts)
        for key, delta in count_deltas.items():
            if delta:
                merged_counts[key] = merged_counts.get(key, 0) + delta
        return _AnalysisState(
            violations=self.violations + tuple(violations),
            advice=self.advice + tuple(advice),
            counts=merged_counts,
        )


_LegacyRule = Callable[[list[str], list[Violation], _RuleContext], None]
_FunctionalRule = Callable[[list[str], _AnalysisContext], _RuleResult]

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_SLOP_ADJECTIVES = [
    "crucial", "groundbreaking", "pivotal", "paramount", "seamless", "holistic",
    "multifaceted", "meticulous", "profound", "comprehensive", "invaluable",
    "notable", "noteworthy", "game-changing", "revolutionary", "pioneering",
    "visionary", "formidable", "quintessential", "unparalleled",
    "stunning", "breathtaking", "captivating", "nestled", "robust",
    "innovative", "cutting-edge", "impactful",
]
_SLOP_VERBS = [
    "delve", "delves", "delved", "delving", "embark", "embrace", "elevate",
    "foster", "harness", "unleash", "unlock", "orchestrate", "streamline",
    "transcend", "navigate", "underscore", "showcase", "leverage",
    "ensuring", "highlighting", "emphasizing", "reflecting",
]
_SLOP_NOUNS = [
    "landscape", "tapestry", "journey", "paradigm", "testament", "trajectory",
    "nexus", "symphony", "spectrum", "odyssey", "pinnacle", "realm", "intricacies",
]
_SLOP_HEDGE = [
    "notably", "importantly", "furthermore", "additionally", "particularly",
    "significantly", "interestingly", "remarkably", "surprisingly", "fascinatingly",
    "moreover", "however", "overall",
]
_ALL_SLOP_WORDS = _SLOP_ADJECTIVES + _SLOP_VERBS + _SLOP_NOUNS + _SLOP_HEDGE
_SLOP_WORD_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _ALL_SLOP_WORDS) + r")\b", re.IGNORECASE)

_SLOP_PHRASES = [
    "it's worth noting", "it's important to note", "this is where things get interesting",
    "here's the thing", "at the end of the day", "in today's fast-paced",
    "as technology continues to", "something shifted", "everything changed",
    "the answer? it's simpler than you think", "what makes this work is",
    "this is exactly", "let's break this down", "let's dive in",
    "in this post, we'll explore", "in this article, we'll",
    "let me know if", "would you like me to", "i hope this helps",
    "as mentioned earlier", "as i mentioned", "without further ado",
    "on the other hand", "in addition", "in summary", "in conclusion",
    "you might be wondering", "the obvious question is",
    "no discussion would be complete", "great question", "that's a great",
    "if you want, i can", "i can adapt this", "i can make this",
    "here are some options", "here are a few options", "would you prefer",
    "shall i", "if you'd like, i can", "i can also",
    "in other words", "put differently", "that is to say",
    "to put it simply", "to put it another way", "what this means is",
    "the takeaway is", "the bottom line is", "the key takeaway", "the key insight",
]
_SLOP_PHRASE_RES = [re.compile(re.escape(p), re.IGNORECASE) for p in _SLOP_PHRASES]
_NOT_JUST_BUT_RE = re.compile(r"not (just|only) .{1,40}, but (also )?", re.IGNORECASE)

_BOLD_HEADER_RE = re.compile(r"\*\*[^*]+[.:]\*\*\s+\S")
_BULLET_LINE_RE = re.compile(r"^(\s*[-*]\s|\s*\d+\.\s)")
_TRIADIC_RE = re.compile(r"\w+, \w+, and \w+", re.IGNORECASE)

_META_COMM_PATTERNS = [
    re.compile(r"would you like", re.IGNORECASE),
    re.compile(r"let me know if", re.IGNORECASE),
    re.compile(r"as mentioned", re.IGNORECASE),
    re.compile(r"i hope this", re.IGNORECASE),
    re.compile(r"feel free to", re.IGNORECASE),
    re.compile(r"don't hesitate to", re.IGNORECASE),
]
_FALSE_NARRATIVITY_PATTERNS = [
    re.compile(r"then something interesting happened", re.IGNORECASE),
    re.compile(r"this is where things get interesting", re.IGNORECASE),
    re.compile(r"that's when everything changed", re.IGNORECASE),
]
_SENTENCE_OPENER_PATTERNS = [
    re.compile(r"(?:^|[.!?]\s+)(certainly[,! ])", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|[.!?]\s+)(absolutely[,! ])", re.IGNORECASE | re.MULTILINE),
]
_WEASEL_PATTERNS = [
    re.compile(r"\bsome critics argue\b", re.IGNORECASE),
    re.compile(r"\bmany believe\b", re.IGNORECASE),
    re.compile(r"\bexperts suggest\b", re.IGNORECASE),
    re.compile(r"\bstudies show\b", re.IGNORECASE),
    re.compile(r"\bsome argue\b", re.IGNORECASE),
    re.compile(r"\bit is widely believed\b", re.IGNORECASE),
    re.compile(r"\bresearch suggests\b", re.IGNORECASE),
]
_AI_DISCLOSURE_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bas a language model\b", re.IGNORECASE),
    re.compile(r"\bi don't have personal\b", re.IGNORECASE),
    re.compile(r"\bi cannot browse\b", re.IGNORECASE),
    re.compile(r"\bup to my last training\b", re.IGNORECASE),
    re.compile(r"\bas of my (last |knowledge )?cutoff\b", re.IGNORECASE),
    re.compile(r"\bi'm just an? ai\b", re.IGNORECASE),
]
_PLACEHOLDER_RE = re.compile(
    r"\[insert [^\]]*\]|\[describe [^\]]*\]|\[url [^\]]*\]|\[your [^\]]*\]|\[todo[^\]]*\]",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?][\"'\u201D\u2019)\]]*(?:\s|$)")
_EM_DASH_RE = re.compile(r"\u2014| -- ")
_CONTRAST_PAIR_RE = re.compile(r"\b(\w+), not (\w+)\b")
_SETUP_RESOLUTION_A_RE = re.compile(
    r"\b(this|that|these|those|it|they|we)\s+"
    r"(isn't|aren't|wasn't|weren't|doesn't|don't|didn't|hasn't|haven't|won't|can't|couldn't|shouldn't"
    r"|is\s+not|are\s+not|was\s+not|were\s+not|does\s+not|do\s+not|did\s+not"
    r"|has\s+not|have\s+not|will\s+not|cannot|could\s+not|should\s+not)\b"
    r".{0,80}[.;:,]\s*"
    r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is"
    r"|these\s+are|those\s+are|he\s+is|she\s+is|we\s+are|what's|what\s+is"
    r"|the\s+real|the\s+actual|instead|rather)",
    re.IGNORECASE,
)
_SETUP_RESOLUTION_B_RE = re.compile(
    r"\b(it's|that's|this\s+is|they're|he's|she's|we're)\s+not\b"
    r".{0,80}[.;:,]\s*"
    r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is"
    r"|these\s+are|those\s+are|what's|what\s+is|the\s+real|the\s+actual|instead|rather)",
    re.IGNORECASE,
)
_ELABORATION_COLON_RE = re.compile(r": [a-z]")
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_URL_COLON_RE = re.compile(r"https?:")
_MD_HEADER_LINE_RE = re.compile(r"^\s*#", re.MULTILINE)
_JSON_COLON_RE = re.compile(r': ["{\[\d]|: true|: false|: null')
_PITHY_PIVOT_RE = re.compile(r",\s+(?:but|yet|and|not|or)\b", re.IGNORECASE)
_BULLET_DENSITY_RE = re.compile(r"^\s*[-*]\s|^\s*\d+[.)]\s", re.MULTILINE)
_BLOCKQUOTE_LINE_RE = re.compile(r"^>", re.MULTILINE)
_BOLD_TERM_BULLET_RE = re.compile(r"^\s*[-*]\s+\*\*|^\s*\d+[.)]\s+\*\*")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:---+|\*\*\*+|___+)\s*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "it", "that", "this", "with", "as", "by", "from", "was", "were", "are",
    "be", "been", "has", "have", "had", "not", "no", "do", "does", "did", "will",
    "would", "could", "should", "can", "may", "might", "if", "then", "than", "so",
    "up", "out", "about", "into", "over", "after", "before", "between", "through",
    "just", "also", "very", "more", "most", "some", "any", "each", "every", "all",
    "both", "few", "other", "such", "only", "own", "same", "too", "how", "what",
    "which", "who", "when", "where", "why",
})
_PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _ctx(text: str, start: int, end: int, hp: Hyperparameters) -> str:
    width = hp.context_window_chars
    mid = (start + end) // 2
    half = width // 2
    s = max(0, mid - half)
    e = min(len(text), mid + half)
    snippet = text[s:e].replace("\n", " ")
    return ("..." if s > 0 else "") + snippet + ("..." if e < len(text) else "")


def _word_count(text: str) -> int:
    return len(text.split())


def _strip_code_blocks(text: str) -> str:
    return _FENCED_CODE_BLOCK_RE.sub("", text)


def _find_repeated_ngrams(text: str, hp: Hyperparameters) -> list[dict]:
    raw_tokens = text.split()
    tokens = [_PUNCT_STRIP_RE.sub("", t).lower() for t in raw_tokens]
    tokens = [t for t in tokens if t]
    if len(tokens) < hp.repeated_ngram_min_n:
        return []

    ngram_counts: dict[tuple[str, ...], int] = {}
    for n in range(hp.repeated_ngram_min_n, hp.repeated_ngram_max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

    repeated = {
        gram: count for gram, count in ngram_counts.items()
        if count >= hp.repeated_ngram_min_count and not all(w in _STOPWORDS for w in gram)
    }
    if not repeated:
        return []

    to_remove: set[tuple[str, ...]] = set()
    sorted_grams = sorted(repeated.keys(), key=len, reverse=True)
    for i, longer in enumerate(sorted_grams):
        longer_str = " ".join(longer)
        for shorter in sorted_grams[i + 1 :]:
            if shorter in to_remove:
                continue
            if " ".join(shorter) in longer_str and repeated[longer] >= repeated[shorter]:
                to_remove.add(shorter)

    return [
        {"phrase": " ".join(gram), "count": repeated[gram], "n": len(gram)}
        for gram in sorted(repeated.keys(), key=lambda g: (-len(g), -repeated[g]))
        if gram not in to_remove
    ]


# ---------------------------------------------------------------------------
# Rule counter categories
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "slop_words", "slop_phrases", "structural", "tone", "weasel", "ai_disclosure",
    "placeholder", "rhythm", "em_dash", "contrast_pairs", "colon_density",
    "pithy_fragment", "setup_resolution", "bullet_density", "blockquote_density",
    "bold_bullet_list", "horizontal_rules", "phrase_reuse",
]


def _zero_counts() -> dict[str, int]:
    return {k: 0 for k in _CATEGORIES}


# ---------------------------------------------------------------------------
# Rules — each writes violations, advice, and count deltas into _RuleContext
# ---------------------------------------------------------------------------


def _rule_slop_words(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    for m in _SLOP_WORD_RE.finditer(ctx.text):
        word = m.group(0)
        violations.append(Violation("slop_word", word.lower(), _ctx(ctx.text, m.start(), m.end(), ctx.hp), ctx.hp.slop_word_penalty))
        ctx.advice.append(f"Replace '{word.lower()}' \u2014 what specifically do you mean?")
        ctx.counts["slop_words"] += 1


def _rule_slop_phrases(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    for pat in _SLOP_PHRASE_RES:
        for m in pat.finditer(ctx.text):
            phrase = m.group(0)
            violations.append(Violation("slop_phrase", phrase.lower(), _ctx(ctx.text, m.start(), m.end(), ctx.hp), ctx.hp.slop_phrase_penalty))
            ctx.advice.append(f"Cut '{phrase.lower()}' \u2014 just state the point directly.")
            ctx.counts["slop_phrases"] += 1
    for m in _NOT_JUST_BUT_RE.finditer(ctx.text):
        phrase = m.group(0).strip()
        violations.append(Violation("slop_phrase", phrase.lower(), _ctx(ctx.text, m.start(), m.end(), ctx.hp), ctx.hp.slop_phrase_penalty))
        ctx.advice.append(f"Cut '{phrase.lower()}' \u2014 just state the point directly.")
        ctx.counts["slop_phrases"] += 1


def _rule_structural(lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    bold_matches = list(_BOLD_HEADER_RE.finditer(ctx.text))
    if len(bold_matches) >= hp.structural_bold_header_min:
        violations.append(Violation("structural", "bold_header_explanation", f"Found {len(bold_matches)} instances of **Bold.** pattern", hp.structural_bold_header_penalty))
        ctx.advice.append(f"Vary paragraph structure \u2014 {len(bold_matches)} bold-header-explanation blocks in a row reads as LLM listicle.")
        ctx.counts["structural"] += 1

    run_length = 0
    for line in lines:
        if _BULLET_LINE_RE.match(line):
            run_length += 1
        else:
            if run_length >= hp.structural_bullet_run_min:
                violations.append(Violation("structural", "excessive_bullets", f"Run of {run_length} consecutive bullet lines", hp.structural_bullet_run_penalty))
                ctx.advice.append(f"Consider prose instead of this {run_length}-item bullet list.")
                ctx.counts["structural"] += 1
            run_length = 0
    if run_length >= hp.structural_bullet_run_min:
        violations.append(Violation("structural", "excessive_bullets", f"Run of {run_length} consecutive bullet lines", hp.structural_bullet_run_penalty))
        ctx.advice.append(f"Consider prose instead of this {run_length}-item bullet list.")
        ctx.counts["structural"] += 1

    triadic_matches = list(_TRIADIC_RE.finditer(ctx.text))
    for m in triadic_matches[: hp.triadic_record_cap]:
        violations.append(Violation("structural", "triadic", _ctx(ctx.text, m.start(), m.end(), hp), hp.triadic_penalty))
        ctx.counts["structural"] += 1
    if len(triadic_matches) >= hp.triadic_advice_min:
        ctx.advice.append(f"{len(triadic_matches)} triadic structures ('X, Y, and Z') \u2014 vary your list cadence.")


def _rule_tone(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    for pat in _META_COMM_PATTERNS:
        for m in pat.finditer(ctx.text):
            phrase = m.group(0)
            violations.append(Violation("tone", phrase.lower(), _ctx(ctx.text, m.start(), m.end(), hp), hp.tone_penalty))
            ctx.advice.append(f"Remove '{phrase.lower()}' \u2014 this is a direct AI tell.")
            ctx.counts["tone"] += 1
    for pat in _FALSE_NARRATIVITY_PATTERNS:
        for m in pat.finditer(ctx.text):
            phrase = m.group(0)
            violations.append(Violation("tone", phrase.lower(), _ctx(ctx.text, m.start(), m.end(), hp), hp.tone_penalty))
            ctx.advice.append(f"Cut '{phrase.lower()}' \u2014 announce less, show more.")
            ctx.counts["tone"] += 1
    for pat in _SENTENCE_OPENER_PATTERNS:
        for m in pat.finditer(ctx.text):
            word = m.group(1).strip(" ,!")
            violations.append(Violation("tone", word.lower(), _ctx(ctx.text, m.start(), m.end(), hp), hp.sentence_opener_penalty))
            ctx.advice.append(f"'{word.lower()}' as a sentence opener is an AI tell \u2014 just make the point.")
            ctx.counts["tone"] += 1


def _rule_weasel(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    for pat in _WEASEL_PATTERNS:
        for m in pat.finditer(ctx.text):
            phrase = m.group(0)
            violations.append(Violation("weasel", phrase.lower(), _ctx(ctx.text, m.start(), m.end(), ctx.hp), ctx.hp.weasel_penalty))
            ctx.advice.append(f"Cut '{phrase.lower()}' \u2014 either cite a source or own the claim.")
            ctx.counts["weasel"] += 1


def _rule_ai_disclosure(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    for pat in _AI_DISCLOSURE_PATTERNS:
        for m in pat.finditer(ctx.text):
            phrase = m.group(0)
            violations.append(Violation("ai_disclosure", phrase.lower(), _ctx(ctx.text, m.start(), m.end(), ctx.hp), ctx.hp.ai_disclosure_penalty))
            ctx.advice.append(f"Remove '{phrase.lower()}' \u2014 AI self-disclosure in authored prose is a critical tell.")
            ctx.counts["ai_disclosure"] += 1


def _rule_placeholder(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    for m in _PLACEHOLDER_RE.finditer(ctx.text):
        match_text = m.group(0)
        violations.append(Violation("placeholder", match_text.lower(), _ctx(ctx.text, m.start(), m.end(), ctx.hp), ctx.hp.placeholder_penalty))
        ctx.advice.append(f"Remove placeholder '{match_text.lower()}' \u2014 this is unfinished template text.")
        ctx.counts["placeholder"] += 1


def _rule_rhythm(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    if len(ctx.sentences) < hp.rhythm_min_sentences:
        return
    lengths = [len(s.split()) for s in ctx.sentences]
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    cv = math.sqrt(variance) / mean
    if cv < hp.rhythm_cv_threshold:
        violations.append(Violation("rhythm", "monotonous_rhythm", f"CV={cv:.2f} across {len(ctx.sentences)} sentences (mean {mean:.1f} words)", hp.rhythm_penalty))
        ctx.advice.append(f"Sentence lengths are too uniform (CV={cv:.2f}) \u2014 vary short and long.")
        ctx.counts["rhythm"] += 1


def _rule_em_dash(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    if ctx.word_count <= 0:
        return
    count = len(list(_EM_DASH_RE.finditer(ctx.text)))
    ratio = (count / ctx.word_count) * hp.em_dash_words_basis
    if ratio > hp.em_dash_density_threshold:
        violations.append(Violation("em_dash", "em_dash_density", f"{count} em dashes in {ctx.word_count} words ({ratio:.1f} per 150 words)", hp.em_dash_penalty))
        ctx.advice.append(f"Too many em dashes ({count} in {ctx.word_count} words) \u2014 use other punctuation.")
        ctx.counts["em_dash"] += 1


def _rule_contrast_pair(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    matches = list(_CONTRAST_PAIR_RE.finditer(ctx.text))
    for m in matches[: hp.contrast_record_cap]:
        matched = m.group(0)
        violations.append(Violation("contrast_pair", matched, _ctx(ctx.text, m.start(), m.end(), hp), hp.contrast_penalty))
        ctx.advice.append(f"'{matched}' \u2014 'X, not Y' contrast \u2014 consider rephrasing to avoid the Claude pattern.")
        ctx.counts["contrast_pairs"] += 1
    if len(matches) >= hp.contrast_advice_min:
        ctx.advice.append(f"{len(matches)} 'X, not Y' contrasts \u2014 this is a Claude rhetorical tic. Vary your phrasing.")


def _rule_setup_resolution(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    recorded = 0
    for pat in (_SETUP_RESOLUTION_A_RE, _SETUP_RESOLUTION_B_RE):
        for m in pat.finditer(ctx.text):
            if recorded < hp.setup_resolution_record_cap:
                matched = m.group(0)
                violations.append(Violation("setup_resolution", matched, _ctx(ctx.text, m.start(), m.end(), hp), hp.setup_resolution_penalty))
                ctx.advice.append(f"'{matched}' \u2014 setup-and-resolution is a Claude rhetorical tic. Just state the point directly.")
                recorded += 1
            ctx.counts["setup_resolution"] += 1


def _rule_colon_density(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    stripped = _strip_code_blocks(ctx.text)
    colon_count = 0
    for line in stripped.split("\n"):
        if _MD_HEADER_LINE_RE.match(line):
            continue
        for cm in _ELABORATION_COLON_RE.finditer(line):
            col_pos = cm.start()
            before = line[: col_pos + 1]
            if before.endswith("http:") or before.endswith("https:"):
                continue
            snippet = line[col_pos : col_pos + 10]
            if _JSON_COLON_RE.match(snippet):
                continue
            colon_count += 1
    wc = _word_count(stripped)
    if wc <= 0:
        return
    ratio = (colon_count / wc) * hp.colon_words_basis
    if ratio > hp.colon_density_threshold:
        violations.append(Violation("colon_density", "colon_density", f"{colon_count} elaboration colons in {wc} words ({ratio:.1f} per 150 words)", hp.colon_density_penalty))
        ctx.advice.append(f"Too many elaboration colons ({colon_count} in {wc} words) \u2014 use periods or restructure sentences.")
        ctx.counts["colon_density"] += 1


def _rule_pithy_fragment(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    pithy_count = 0
    for sent in ctx.sentences:
        s = sent.strip()
        if not s or len(s.split()) > hp.pithy_max_sentence_words:
            continue
        if _PITHY_PIVOT_RE.search(s):
            if pithy_count < hp.pithy_record_cap:
                violations.append(Violation("pithy_fragment", s, s, hp.pithy_penalty))
                ctx.advice.append(f"'{s}' \u2014 pithy evaluative fragments are a Claude tell. Expand or cut.")
            pithy_count += 1
            ctx.counts["pithy_fragment"] += 1


def _rule_bullet_density(lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    non_empty = [line for line in lines if line.strip()]
    total = len(non_empty)
    if total <= 0:
        return
    bullet_count = sum(1 for line in non_empty if _BULLET_DENSITY_RE.match(line))
    ratio = bullet_count / total
    if ratio > hp.bullet_density_threshold:
        violations.append(Violation("structural", "bullet_density", f"{bullet_count} of {total} non-empty lines are bullets ({ratio:.0%})", hp.bullet_density_penalty))
        ctx.advice.append(f"Over {ratio:.0%} of lines are bullets \u2014 write prose instead of lists.")
        ctx.counts["bullet_density"] += 1


def _rule_blockquote(lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    in_code = False
    bq_count = 0
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code and line.startswith(">"):
            bq_count += 1
    if bq_count >= hp.blockquote_min_lines:
        excess = bq_count - hp.blockquote_free_lines
        capped = min(excess, hp.blockquote_cap)
        violations.append(Violation("structural", "blockquote_density", f"{bq_count} blockquote lines \u2014 Claude uses these as thesis statements", hp.blockquote_penalty_step * capped))
        ctx.advice.append(f"{bq_count} blockquotes \u2014 integrate key claims into prose instead of pulling them out as blockquotes.")
        ctx.counts["blockquote_density"] += 1


def _rule_bold_bullet_run(lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    run = 0
    for line in lines:
        if _BOLD_TERM_BULLET_RE.match(line):
            run += 1
            continue
        if run >= hp.bold_bullet_run_min:
            violations.append(Violation("structural", "bold_bullet_list", f"Run of {run} bold-term bullets", hp.bold_bullet_run_penalty))
            ctx.advice.append(f"Run of {run} bold-term bullets \u2014 this is an LLM listicle pattern. Use varied paragraph structure.")
            ctx.counts["bold_bullet_list"] += 1
        run = 0
    if run >= hp.bold_bullet_run_min:
        violations.append(Violation("structural", "bold_bullet_list", f"Run of {run} bold-term bullets", hp.bold_bullet_run_penalty))
        ctx.advice.append(f"Run of {run} bold-term bullets \u2014 this is an LLM listicle pattern. Use varied paragraph structure.")
        ctx.counts["bold_bullet_list"] += 1


def _rule_horizontal_rules(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    hr_count = len(_HORIZONTAL_RULE_RE.findall(ctx.text))
    if hr_count >= hp.horizontal_rule_min:
        violations.append(Violation("structural", "horizontal_rules", f"{hr_count} horizontal rules \u2014 excessive section dividers", hp.horizontal_rule_penalty))
        ctx.advice.append(f"{hr_count} horizontal rules \u2014 section headers alone are sufficient, dividers are a crutch.")
        ctx.counts["horizontal_rules"] += 1


def _rule_phrase_reuse(_lines: list[str], violations: list[Violation], ctx: _RuleContext) -> None:
    hp = ctx.hp
    recorded = 0
    for ng in _find_repeated_ngrams(ctx.text, hp):
        if recorded >= hp.phrase_reuse_record_cap:
            break
        phrase, count, n = ng["phrase"], ng["count"], ng["n"]
        violations.append(Violation("phrase_reuse", phrase, f"'{phrase}' ({n}-word phrase) appears {count} times", hp.phrase_reuse_penalty))
        ctx.advice.append(f"'{phrase}' appears {count} times \u2014 vary your phrasing to avoid repetition.")
        ctx.counts["phrase_reuse"] += 1
        recorded += 1


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def _wrap(legacy_rule: _LegacyRule) -> _FunctionalRule:
    def _rule(lines: list[str], context: _AnalysisContext) -> _RuleResult:
        scratch_violations: list[Violation] = []
        scratch = _RuleContext(
            text=context.text, word_count=context.word_count,
            sentences=context.sentences, advice=[], counts=_zero_counts(), hp=context.hp,
        )
        legacy_rule(lines, scratch_violations, scratch)
        return _RuleResult(violations=scratch_violations, advice=scratch.advice, count_deltas=scratch.counts)
    return _rule


_PIPELINE: list[_FunctionalRule] = [
    _wrap(_rule_slop_words),
    _wrap(_rule_slop_phrases),
    _wrap(_rule_structural),
    _wrap(_rule_tone),
    _wrap(_rule_weasel),
    _wrap(_rule_ai_disclosure),
    _wrap(_rule_placeholder),
    _wrap(_rule_rhythm),
    _wrap(_rule_em_dash),
    _wrap(_rule_contrast_pair),
    _wrap(_rule_setup_resolution),
    _wrap(_rule_colon_density),
    _wrap(_rule_pithy_fragment),
    _wrap(_rule_bullet_density),
    _wrap(_rule_blockquote),
    _wrap(_rule_bold_bullet_run),
    _wrap(_rule_horizontal_rules),
    _wrap(_rule_phrase_reuse),
]


def _run_pipeline(lines: list[str], context: _AnalysisContext, pipeline: list[_FunctionalRule]) -> _AnalysisState:
    initial = _AnalysisState.initial(_zero_counts())

    def _merge(state: _AnalysisState, rule_fn: Callable[[], _RuleResult]) -> _AnalysisState:
        result = rule_fn()
        return state.merge(result.violations, result.advice, result.count_deltas)

    return reduce(_merge, [partial(rule, lines, context) for rule in pipeline], initial)


def _weighted_sum(violations: list[Violation], counts: dict[str, int], hp: Hyperparameters) -> float:
    total = 0.0
    for v in violations:
        penalty = abs(v.penalty)
        cat_count = counts.get(v.rule, 0) or counts.get(v.rule + "s", 0)
        count_key = v.rule if v.rule in hp.claude_categories else ((v.rule + "s") if (v.rule + "s") in hp.claude_categories else None)
        if count_key and count_key in hp.claude_categories and cat_count > 1:
            total += penalty * (1 + hp.concentration_alpha * (cat_count - 1))
        else:
            total += penalty
    return total


def _band(score: int, hp: Hyperparameters) -> str:
    if score >= hp.band_clean_min:
        return "clean"
    if score >= hp.band_light_min:
        return "light"
    if score >= hp.band_moderate_min:
        return "moderate"
    if score >= hp.band_heavy_min:
        return "heavy"
    return "saturated"


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_text(text: str, hyperparameters: Hyperparameters | None = None) -> dict:
    """Score text for AI slop patterns.

    Args:
        text: The prose to analyze.
        hyperparameters: Optional tuning overrides. Uses sensible defaults if omitted.

    Returns:
        Dict with keys: score (0-100), band, word_count, violations, counts,
        total_penalty, weighted_sum, density, advice.
    """
    hp = hyperparameters or DEFAULT_HYPERPARAMETERS
    wc = _word_count(text)
    counts = _zero_counts()

    if wc < hp.short_text_word_count:
        return {
            "score": hp.score_max, "band": "clean", "word_count": wc,
            "violations": [], "counts": counts, "total_penalty": 0,
            "weighted_sum": 0.0, "density": 0.0, "advice": [],
        }

    lines = text.split("\n")
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    context = _AnalysisContext(text=text, word_count=wc, sentences=sentences, hp=hp)
    state = _run_pipeline(lines, context, _PIPELINE)

    total_penalty = sum(v.penalty for v in state.violations)
    ws = _weighted_sum(list(state.violations), state.counts, hp)
    density = (ws / (wc / hp.density_words_basis)) if wc > 0 else 0.0
    raw_score = hp.score_max * math.exp(-hp.decay_lambda * density)
    score = max(hp.score_min, min(hp.score_max, round(raw_score)))

    return {
        "score": score,
        "band": _band(score, hp),
        "word_count": wc,
        "violations": [v.to_payload() for v in state.violations],
        "counts": state.counts,
        "total_penalty": total_penalty,
        "weighted_sum": round(ws, 2),
        "density": round(density, 2),
        "advice": _deduplicate(list(state.advice)),
    }
