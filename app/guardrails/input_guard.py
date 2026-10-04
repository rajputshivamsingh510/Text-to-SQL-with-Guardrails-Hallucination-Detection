"""Input guardrail. Runs before anything reaches the LLM.

Checks: empty / too long, prompt-injection phrases, destructive "write intent", and PII (masked or blocked).
These are fast heuristics, not a complete defence: the SQL guardrail and the read-only DB role are the
real safety net. This layer exists to stop obvious abuse early and cheaply, and to keep PII out of the LLM.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier|your) (instructions|rules|prompt|guidelines)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your)",
    r"forget (all |everything |your )?(previous|prior|above|instructions|rules)",
    r"(reveal|show|print|repeat|display|leak) (me )?(your |the )?(system|hidden|initial|original) (prompt|instructions|message)",
    r"\bsystem prompt\b",
    r"you are (now|no longer)\b",
    r"\b(developer|debug|admin|god|jailbreak) mode\b",
    r"\bdo anything now\b|(?-i:\bDAN\b)",
    r"pretend (to be|you are|that you)",
    r"act as (if|though)? ?(you (are|were)|an? unrestricted)",
    r"override (the |your )?(safety|security|rules|guardrails|restrictions)",
    r"bypass (the |your )?(safety|security|rules|guardrails|filters?|restrictions)",
    r"</?(system|assistant|instructions?)>",
    r"\bnew instructions?\s*:",
]

_WRITE_INTENT = [
    r"\b(drop|truncate)\s+(table|database|schema|index|view)\b",
    r"\bdelete\s+(from|all|every|the)\b",
    r"\binsert\s+into\b",
    r"\bupdate\s+\w+\s+set\b",
    r"\balter\s+(table|database|role|user)\b",
    r"\b(grant|revoke)\s+\w+",
    r"\bcreate\s+(table|database|user|role|index)\b",
    r";\s*(drop|delete|update|insert|alter|truncate|create)\b",
    r"\bunion\s+(all\s+)?select\b",
    r"'\s*(or|and)\s+'?1'?\s*=\s*'?1",
    r"--\s*$",
    r"\bxp_cmdshell\b|\bpg_sleep\b|\bload_extension\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION]
_WRITE_RE = [re.compile(p, re.IGNORECASE) for p in _WRITE_INTENT]


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _mask_cards(text: str) -> tuple[str, bool]:
    found = False

    def repl(m: re.Match) -> str:
        nonlocal found
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found = True
            return "[CARD]"
        return m.group()

    return re.sub(r"\b(?:\d[ -]?){13,19}\b", repl, text), found


# Order matters: more specific patterns first.
_PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    ("aadhaar", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[AADHAAR]"),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    ("phone", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:[6-9]\d{9}|\(?\d{3}\)?[ -]\d{3}[ -]\d{4})\b"), "[PHONE]"),
]


@dataclass
class InputCheck:
    allowed: bool
    sanitized: str
    category: str | None = None  # empty | too_long | prompt_injection | write_intent | pii
    reason: str | None = None
    pii_found: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def check_input(question: str, *, max_chars: int = 500, pii_mode: str = "mask") -> InputCheck:
    text = normalize(question or "")
    if not text:
        return InputCheck(False, "", "empty", "Please enter a question about your data.")
    if len(text) > max_chars:
        return InputCheck(False, text[:max_chars], "too_long", f"Questions are limited to {max_chars} characters.")

    for rx in _INJECTION_RE:
        if rx.search(text):
            return InputCheck(False, text, "prompt_injection",
                              "This looks like an attempt to change my instructions, so I can't run it.")
    for rx in _WRITE_RE:
        if rx.search(text):
            return InputCheck(False, text, "write_intent",
                              "I can only read data. Requests that modify data or the database are blocked.")

    pii: list[str] = []
    sanitized = text
    if pii_mode != "off":
        sanitized, card = _mask_cards(sanitized)
        if card:
            pii.append("card")
        for name, rx, token in _PII_PATTERNS:
            sanitized, n = rx.subn(token, sanitized)
            if n:
                pii.append(name)
        if pii and pii_mode == "block":
            return InputCheck(False, text, "pii",
                              f"Your question contains personal data ({', '.join(pii)}). Please remove it and try again.",
                              pii_found=pii)
    return InputCheck(True, sanitized, pii_found=pii)
