"""Mask credentials in session text before it leaves the machine (HydraDB sync, `ccs ask`, MCP).

Each match becomes `[redacted:<kind>]`. The patterns are for things that are secrets on their
own (API keys, tokens, private keys) or values assigned to a secret-sounding name; they are
deliberately narrow, so ordinary code, hashes and ids stay searchable.
"""

import re
from collections import Counter

# (kind, pattern, group to mask: 0 = whole match)
PATTERNS = [
    ("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:.*?-----END [A-Z ]*PRIVATE KEY-----|.*\Z)", 0),
    ("stripe-style", r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{12,}", 0),  # also HydraDB keys
    ("sk-key", r"\bsk-(?:ant-|proj-|or-v1-)?[A-Za-z0-9_-]{20,}", 0),  # OpenAI, Anthropic, OpenRouter
    ("github-token", r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})", 0),
    ("aws-key-id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 0),
    ("google-api-key", r"\bAIza[0-9A-Za-z_-]{35}\b", 0),
    ("slack-token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}", 0),
    ("hf-token", r"\bhf_[A-Za-z0-9]{30,}", 0),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", 0),
    ("url-password", r"\b[a-z][a-z0-9+.-]*://[^:/\s@]+:([^@\s/]{3,})@", 1),
    ("bearer", r"(?i)\bBearer\s+([A-Za-z0-9._~+/=-]{16,})", 1),
    ("assigned-secret",
     # the name must end in the secret word: API_KEY, client_secret, GITHUB_TOKEN, not tokenizer_path
     r"(?i)\b[A-Z0-9_]*(?:api[_-]?key|secret|token|passw(?:or)?d|access[_-]?key)"
     r"[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-./+=]{16,})", 1),
]
_COMPILED = [(kind, re.compile(p, re.S if kind == "private-key" else 0), g) for kind, p, g in PATTERNS]


def redact(text, counts=None):
    """`text` with credentials masked; adds one per masked value to `counts` (a Counter)."""
    if not text:
        return text
    for kind, rx, group in _COMPILED:
        def mask(m, kind=kind, group=group):
            if "[redacted:" in m.group(group):
                return m.group(0)
            if counts is not None:
                counts[kind] += 1
            start, end = m.start(group) - m.start(0), m.end(group) - m.start(0)
            return m.group(0)[:start] + f"[redacted:{kind}]" + m.group(0)[end:]
        text = rx.sub(mask, text)
    return text


def scan(texts):
    """Counter of credential kinds found across `texts` (nothing is changed)."""
    counts = Counter()
    for t in texts:
        redact(t, counts)
    return counts
