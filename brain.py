"""
brain.py - How OpsMate understands questions and writes answers.

  Layer 1  UNDERSTAND  spelling fix + Hinglish (offline), topic + answer format;
                       AI rephrases unclear questions (fast model, only when needed)
  Layer 2  SEARCH      in knowledge.py - word search + Gemini semantic search over
                       EVERY document, adjusted by what staff marked 👍 / 👎
  Layer 3  ANSWER      AI compares the passages and returns: answered / not found /
                       conflict, with a confidence level. Without AI, a short answer is
                       cut from the best passage and the confidence is measured offline.
"""
import difflib
import io
import json
import re

import config
import llm
import prompts


def _json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def image_size(img_bytes):
    from PIL import Image
    try:
        return Image.open(io.BytesIO(img_bytes)).size
    except Exception:
        return (0, 0)


# ===========================================================================
# LAYER 1 - UNDERSTAND THE QUESTION
# ===========================================================================

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
STEP_WORDS = re.compile(r"\b(step|steps|process|procedure|how (do|to|can|should)|kaise|"
                        r"walk me|guide|flow|sequence|checklist|what (should|do|must) i check|what (to|do i|should i) do|"
                        r"kya kar|handle|steps to)\b", re.I)
TABLE_WORDS = re.compile(r"\b(table|list|all (the )?commands|matrix|compare|comparison|difference)\b", re.I)


# Words that say HOW to answer, not WHAT to find - removed before searching.
FORMAT_WORDS = re.compile(r"\b(step[ -]?by[ -]?step|in steps|steps?|briefly|in short|short|"
                          r"please|pls|plz|explain|tell me|list of|give me|table of|table)\b", re.I)

# Common Hinglish / chat words staff type, mapped to SOP language (offline).
HINGLISH = {
    "kaise": "how", "kese": "how", "kaisa": "how", "kya": "what", "kab": "when",
    "kitne": "how many", "kitna": "how much", "din": "days", "dino": "days", "time": "time",
    "paisa": "money refund", "paise": "money refund", "wapas": "back refund", "aata": "come",
    "aayega": "come", "milega": "get", "karna": "do", "karein": "do", "karu": "do",
    "kare": "do", "hai": "", "hota": "", "me": "", "mein": "", "ka": "", "ki": "", "ke": "",
    "ko": "", "se": "", "pe": "", "par": "", "agar": "if", "nahi": "not", "galat": "wrong",
    "naam": "name",
    "badalna": "change", "badle": "change", "cancel": "cancel", "kal": "next day",
    # chat short forms
    "wat": "what", "wht": "what", "hw": "how", "abt": "about", "u": "you", "r": "are",
    "pls": "please", "plz": "please", "tkt": "ticket", "tix": "ticket", "pax": "passenger",
    "hr": "hour", "hrs": "hours", "dt": "date",
}
# Everyday words -> the words SOPs usually use (added to the search, not replaced).
SYNONYMS = {
    "wrong": "mistake correction", "mistake": "correction", "error": "mistake issue",
    "misspelt": "spelling correction", "misspelled": "spelling correction",
    "money": "refund", "cancelled": "cancel", "late": "delay", "delayed": "delay",
    "ticket image": "e-ticket record TWD", "boss": "manager approval", "urgent": "escalate",
    "usa": "united states", "us": "united states", "uk": "united kingdom",
    "dollar": "usd", "dollars": "usd", "euro": "eur", "euros": "eur", "rupees": "inr", "rs": "inr",
    "pounds": "gbp", "approves": "approval approve", "approve": "approval",
}
TIME_WORDS = re.compile(r"\b(how long|how many days|how much time|days|kitne din)\b", re.I)


def core_text(question):
    """The agent's own words (Hinglish translated, format words removed) - used for the
    confidence check. search_text() adds synonyms on top, only to FIND passages."""
    q = " ".join(HINGLISH.get(w.lower(), w) for w in question.split())
    return re.sub(r"\s+", " ", FORMAT_WORDS.sub(" ", q)).strip() or question


def search_text(question):
    """The words to SEARCH with: Hinglish translated, format words removed."""
    q = " ".join(HINGLISH.get(w.lower(), w) for w in question.split())
    q = FORMAT_WORDS.sub(" ", q)
    # generic words ('looks', 'image', 'Europe', 'command') would pull in sections that merely
    # contain them - leave them out when the question has 2+ specific words
    words = re.findall(r"[A-Za-z0-9]+", q)
    hard = [w for w in words if len(w) > 2 and w.lower() not in _STOP and w.lower() not in _SOFT]
    if len(hard) >= 2:
        q = " ".join(w for w in q.split() if re.sub(r"[^a-z0-9]", "", w.lower()) not in _SOFT)
    extra = [v for k, v in SYNONYMS.items() if re.search(rf"\b{k}\b", q, re.I)]
    q += " " + " ".join(extra)
    if TIME_WORDS.search(question):
        q += " time timeline take"
    return re.sub(r"\s+", " ", q).strip() or question


def detect_format(question):
    if STEP_WORDS.search(question):
        return "steps"
    if TABLE_WORDS.search(question):
        return "table"
    return "short"


def detect_topic(question):
    """Put a question into one of config.TOPICS by its words (offline, free)."""
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    best, best_n = "Other", 0
    for topic, keys in config.TOPICS.items():
        n = len(words & set(keys.split()))
        if n > best_n:
            best, best_n = topic, n
    return best


def build_vocab(texts):
    vocab = set()
    for t in texts:
        vocab.update(w.lower() for w in _WORD.findall(t) if len(w) >= 3)
    return vocab


# everyday words that are spelled right but may not appear in the documents
_PLAIN = set("""give gives share shared shows tell send sent need want help explain kindly please thanks thank
could would should where there their them they which other also every sure know known able like just only
does doing done what when while about after before again still need needs okay fine good best""".split())


def fix_spelling(question, vocab):
    """Correct words that are not in the documents to the closest word that is.
    'refnd tiket' -> 'refund ticket'. Works offline, no AI needed."""
    if not vocab:
        return question, []
    vocab_list = list(vocab)
    changes = []

    def repl(m):
        w = m.group(0)
        lw = w.lower()
        if (len(lw) < 4 or lw in vocab or lw in HINGLISH or not lw.isalpha()
                or lw in _STOP or lw in _SOFT or lw in _PLAIN):   # everyday words are never 'corrected'
            return w
        cut = 0.75 if len(lw) >= 6 else 0.8
        best = difflib.get_close_matches(lw, vocab_list, n=1, cutoff=cut)
        if best:
            changes.append((w, best[0]))
            return best[0]
        return w

    fixed = _WORD.sub(repl, question)
    return fixed, changes


def ai_understand(question, screenshot_text=""):
    """Fast-model rewrite: corrected question + 2-3 search phrases + answer format."""
    block = f'\nText read from the screenshot they pasted:\n"""{screenshot_text[:1500]}"""\n' if screenshot_text else ""
    out = _json(llm.generate(prompts.QUERY_PROMPT.format(question=question, screenshot_block=block),
                             fast=True, json_mode=True, max_tokens=300))
    queries = [q for q in out.get("queries", []) if isinstance(q, str) and q.strip()][:3]
    fmt = out.get("format", "short")
    return {"corrected": out.get("corrected") or question, "queries": queries,
            "format": fmt if fmt in ("steps", "table", "short") else "short"}


# ===========================================================================
# SCREENSHOTS AND IMAGES
# ===========================================================================

def read_screenshot(img_bytes, question=""):
    """Staff pasted a screenshot: read its text and guess what they need."""
    out = _json(llm.generate(prompts.SCREENSHOT_PROMPT.format(question=question or "(no text)"),
                             images=[img_bytes], json_mode=True, max_tokens=900))
    return {"text": out.get("text", ""), "issue": out.get("issue", "")}


def describe_doc_image(img_bytes, title, section):
    """Image inside an SOP (screenshot / table / flowchart) -> searchable text."""
    w, h = image_size(img_bytes)
    if w < config.MIN_IMAGE_PX or h < config.MIN_IMAGE_PX:
        return ""                                   # icons, logos, bullets
    text = llm.generate(prompts.DOC_IMAGE_PROMPT.format(title=title, section=section),
                        images=[img_bytes], max_tokens=700)
    return "" if text.strip().upper().startswith("SKIP") else text


# ===========================================================================
# LAYER 3 - ANSWER WITH AI
# ===========================================================================

def ai_answer(question, region, hits, fmt, screenshot_text="", lob=""):
    """Returns dict: status (answered|not_found|conflict), answer, important,
    confidence (high|medium|low), used (source numbers), conflict (text)."""
    sources = "\n\n".join(
        f"[S{i}] {'VERIFIED ANSWER' if h['kind'] == 'learned' else h['title']} | {h['type']} | "
        f"{h['region']} | {h['location']}" + (f" | LOB: {h['lob']}" if h.get("lob") else "")
        + (f" | v{h['version']}" if h.get("version") else "")
        + f"\n{h['text']}"
        for i, h in enumerate(hits, 1))
    block = f'\nThe staff member also pasted a screenshot. Its text:\n"""{screenshot_text[:1500]}"""\n' if screenshot_text else ""
    raw = llm.generate(prompts.ANSWER_PROMPT.format(
        region=region, question=question, sources=sources, format=fmt, lob=lob or "all LOBs",
        max_words=config.MAX_ANSWER_WORDS, screenshot_block=block), json_mode=True, max_tokens=900)
    out = _json(raw)
    status = out.get("status", "answered" if raw and "NOT_FOUND" not in raw else "not_found")
    if status not in ("answered", "not_found", "conflict"):
        status = "answered"
    conf = str(out.get("confidence", "medium")).lower()
    used = [int(n) for n in out.get("sources_used", []) if str(n).isdigit()]
    return {"status": status, "answer": (out.get("answer") or "").strip(),
            "important": (out.get("important") or "").strip(),
            "confidence": conf if conf in ("high", "medium", "low") else "medium",
            "used": used, "conflict": (out.get("conflict") or "").strip()}


# ===========================================================================
# WITHOUT AI - confidence and conflict checks done offline
# ===========================================================================

_STOP = set("""a an the is are was were be been am do does did i me my we our you your it its of to in
on for at by with from and or as this that these those can could should would will how what when where
which who why please tell know need want get about any there step steps process
long take takes time timeline many much come does happen kya hai have has had give share send provide some""".split())


# Generic words staff add to a question ("how to CHECK ... in the GDS", "COMMAND for ...") that a
# good passage often does not repeat. They count only when the question has nothing more specific.
_SOFT = set("""check checking find see show use using make way gds system systems command commands cmd entry
entries format formats tool tools screen look looks looking like image images picture pictures pic sample
samples example examples screenshot refer reference europe noram apac qbay global region lob""".split())


def coverage(question, text):
    """Share of the question's important words that appear in the passage."""
    norm = lambda x: re.sub(r"(\d),(\d{3})", r"\1\2", x.lower())
    q = {w for w in re.findall(r"[a-z0-9]+", norm(question)) if len(w) > 2 and w not in _STOP}
    if len(q - _SOFT) >= 2:              # ignore generic words only when 2+ specific words remain
        q = q - _SOFT
    if not q:
        return 1.0
    t = set(re.findall(r"[a-z0-9]+", norm(text)))
    loose = len(q) > 1                   # one-word questions must match the whole word, not 5 letters

    def present(w, alts=True):
        if w in t or w + "s" in t or w.rstrip("s") in t:
            return True
        if loose and len(w) >= 5 and any(x.startswith(w[:5]) for x in t):
            return True
        # 'usa' -> 'united states', 'wrong' -> 'mistake correction'
        return alts and any(present(a, False) for a in SYNONYMS.get(w, "").split())
    return sum(1 for w in q if present(w)) / len(q)


def content_words(question):
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2 and w not in _STOP]
    hard = [w for w in words if w not in _SOFT]
    return hard if len(hard) >= 2 else words


def offline_confidence(score, cover, semantic=None, nwords=3):
    """high / medium / low. 'low' means OpsMate gives the supervisor message instead
    of guessing. Tested: real questions match 70%+ of their key words, out-of-scope
    questions (leave, canteen, salary...) match half or fewer."""
    if semantic is not None:
        if semantic >= config.SEM_HIGH and cover >= 0.5:
            return "high"
        if semantic >= config.SEM_MEDIUM and cover >= 0.34:
            return "medium"
        if score >= config.CONF_HIGH and cover >= 0.8:     # strong word match still counts
            return "medium"
        return "low"
    if score >= config.CONF_HIGH and cover >= 0.8:
        return "high"
    if score >= config.CONF_MEDIUM and cover >= 0.7:
        return "medium"
    if score >= config.CONF_MEDIUM * 0.5 and cover >= 0.99 and nwords <= 2:   # "what is TTP"
        return "medium"
    return "low"


_FACT = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*(?:to|-|–)\s*(\d+(?:\.\d+)?))?\s*"
    r"(business days|working days|calendar days|days|day|hours|hour|hrs|weeks|week|months|month|"
    r"minutes|mins|%|percent)", re.I)
_UNIT = {"business days": "days", "working days": "days", "calendar days": "days", "day": "days",
         "hour": "hours", "hrs": "hours", "week": "weeks", "month": "months", "mins": "minutes",
         "percent": "%"}


def _facts(question, text):
    """Numbers with units from the sentences that answer the question."""
    sents = _sentences(_clean(text))
    best = sorted(sents, key=lambda x: -_overlap(question, x))[:2]
    found = {}
    for sent in best:
        for a, b, unit in _FACT.findall(sent):
            u = _UNIT.get(unit.lower(), unit.lower())
            found.setdefault(u, set()).add((a, b))
    times = set(re.findall(r"\b\d{1,2}:\d{2}\b", " ".join(best)))
    if times:
        found["time"] = {(t, "") for t in times}
    return found


def offline_conflict(question, hits):
    """Two DIFFERENT documents, both strong matches, giving different numbers for the
    same thing (e.g. '7 to 10 days' vs '14 days') -> returns the two hits, else None."""
    docs = [h for h in hits if h["kind"] == "doc"]
    if len(docs) < 2:
        return None
    top = docs[0]
    f1 = _facts(question, top["text"])
    if not f1:
        return None
    for other in docs[1:]:
        if other["doc"] == top["doc"] or other["score"] < top["score"] * 0.4:
            continue
        if coverage(question, other.get("search", other["text"])) < 0.6:
            continue
        f2 = _facts(question, other["text"])
        for unit in set(f1) & set(f2):
            if not (f1[unit] & f2[unit]):
                return top, other
    return None


# ---- Without AI: cut a crisp answer out of the best passage ---------------

_STEP_LINE = re.compile(r"^\s*(\d+[.)]|[-•*▪]|step\s*\d+)\s*", re.I)


def _clean(text, heading=""):
    """Drop the section heading line and the FAQ 'Q:' line; keep the answer."""
    lines = text.splitlines()
    if lines and heading and lines[0].strip() == heading.strip():
        lines = lines[1:]
    text = "\n".join(l for l in lines if not l.strip().lower().startswith(("also asked as", "keywords:")))
    text = re.sub(r"^\s*Q:.*?\n\s*A:\s*", "", text, flags=re.S)
    return re.sub(r"^\s*A:\s*", "", text)


def _sentences(text):
    # join lines that were only wrapped by the PDF (not new list items)
    text = re.sub(r"\n(?!\s*(\d+[.)]|[-•*▪]))", " ", text)
    parts = [p.strip(" ;") for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text) if len(p.strip()) > 3]
    full = [p for p in parts if len(p.split()) >= 5]      # drop stray headings
    return full or parts


def _inline_steps(text):
    """'Steps: 1. Do this. 2. Do that.' -> ['Do this', 'Do that']"""
    found = re.split(r"(?:^|\s)(\d{1,2})[.)]\s+", " " + text)
    if len(found) < 5:
        return []
    return [found[i + 1].strip(" ;.") for i in range(1, len(found) - 1, 2) if found[i + 1].strip()]


def snippet(question, text, location="", n=2):
    """The 1-2 sentences of a passage that answer the question (for conflict display)."""
    sents = _sentences(_clean(text, location))
    best = sorted(range(len(sents)), key=lambda i: -_overlap(question, sents[i]))[:n]
    return " ".join(sents[i] for i in sorted(best)) or text[:200]


def _overlap(a, b):
    wa = {w.lower() for w in _WORD.findall(a) if len(w) > 2}
    wb = {w.lower() for w in _WORD.findall(b) if len(w) > 2}
    return len(wa & wb) / (len(wa) or 1)


def _as_table(text):
    rows = [r for r in text.splitlines() if "|" in r]
    if len(rows) < 2:
        return None
    cells = [[c.strip() for c in r.split("|")] for r in rows]
    n = max(len(c) for c in cells)
    cells = [c + [""] * (n - len(c)) for c in cells]
    md = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * n]
    md += ["| " + " | ".join(c) + " |" for c in cells[1:]]
    return "\n".join(md)


def crisp_answer(question, hits, fmt):
    """No AI key: build a short answer from the best passage(s)."""
    top = hits[0]
    if top["kind"] == "learned":
        return top["text"]

    text = _clean(top["text"], top["location"])
    body = None
    if fmt == "table" or ("|" in text and text.count("|") > 4):
        table = _as_table(text)
        if table:
            lines = table.splitlines()
            if fmt == "table":            # "list of ..." -> whole table, original order
                body = "\n".join(lines[:22])
            else:                         # specific question -> only the matching rows
                rows = sorted(lines[2:], key=lambda r: -_overlap(question, r))
                rows = [r for r in rows[:3] if _overlap(question, r) > 0] or rows[:3]
                body = "\n".join(lines[:2] + rows)
    if body is None:
        steps = [_STEP_LINE.sub("", l).strip() for l in text.splitlines() if _STEP_LINE.match(l)]
        numbered = sum(1 for l in text.splitlines() if re.match(r"^\s*\d+[.)]\s", l)) >= 3
        if len(steps) < 2:   # steps written inline: "1. ... 2. ..."
            steps = _inline_steps(re.sub(r"\s+", " ", text))
            numbered = len(steps) >= 3
        if len(steps) >= 2 and (fmt == "steps" or numbered):
            intro = re.split(r"(?:^|\s)1[.)]\s", re.sub(r"\s+", " ", text))[0].strip()
            intro = intro if 0 < len(intro.split()) <= 25 and not _STEP_LINE.match(intro) else ""
            body = (f"{intro}\n\n" if intro else "") + "\n".join(
                f"{i}. {s}" for i, s in enumerate(steps[:10], 1))
    if body is None:
        sents = _sentences(text)
        ranked = sorted(range(len(sents)), key=lambda i: -_overlap(question, sents[i]))[:3]
        picked, words = [], 0
        for i in sorted(ranked):
            w = len(sents[i].split())
            if words + w > config.MAX_ANSWER_WORDS and picked:
                break
            picked.append(sents[i]); words += w
        body = "\n".join(f"- {s}" for s in picked) if len(picked) > 1 else (picked[0] if picked else text[:300])

    return body


# ---------------------------------------------------------------------------
# Confidence %  (shown with every answer)
# ---------------------------------------------------------------------------
def supporting_documents(hits, ratio=0.6):
    """How many DIFFERENT documents (or verified answers) strongly match the question."""
    if not hits:
        return 0
    top = hits[0]["score"] or 0
    docs = {h["doc"] for h in hits if top and h["score"] >= top * ratio}
    return len(docs)


def confidence_percent(level, best, cover, supporting=1):
    """Turns the confidence level into a %.
    The level (high/medium/low) comes from the tested rules; the % places the answer
    inside that level's band (config.CONF_BANDS) from three things:
      * match strength of the best passage (meaning match with Gemini, word match without)
      * coverage: share of the question's key words found in that passage
      * agreement: +5 points of evidence for each extra document saying the same (max +10)
    A verified answer written by the process team counts as full strength."""
    lo, hi = config.CONF_BANDS.get(level, config.CONF_BANDS["low"])
    if best is None:
        return lo
    if best.get("sem") is not None:
        strength = (best["sem"] - 0.45) / (0.85 - 0.45)
    else:
        strength = best.get("lex", 0) / 0.60
    if best.get("kind") == "learned" and not best.get("auto"):
        strength = 1.0
    strength = max(0.0, min(1.0, strength))
    cover = max(0.0, min(1.0, cover or 0))
    agree = min(0.10, 0.05 * max(0, (supporting or 1) - 1))
    raw = min(1.0, 0.55 * strength + 0.45 * cover + agree)
    return int(round(lo + (hi - lo) * raw))


# ---------------------------------------------------------------------------
# GDS commands: show them like a terminal screen (no AI needed)
# ---------------------------------------------------------------------------
GDS_WORDS = ("amadeus", "sabre", "galileo", "worldspan", "apollo", "gds", "pnr", "1a", "1s")
# entry codes: a token is a command when it IS one of these, or starts with one and then has a
# digit or symbol (RTABC123, NM1SHARMA/RAHUL, WETR*1, PQD-ALL). Plain words (APAC, ADM) never match.
GDS_CODES = sorted(set("""RT RH RF ET ER EF IG IR XE XI NM AP APE APM TK TKOK TKTL SS SR OS RM RC RX ST SM QE QP QU
    NU SP FXP FXX FXB FXA FXQ FXR TQT TQN TQM TTP TTE TWD TWX TRF TRFU TRFP TRFIG FQD FQN FP FM FT FA FO DAC DAN DNA
    DD AN SN TN MD MU MT MB HE GP WP WPRQ WPNC WPNCB PQ PQD WETR WV WFR FQ RD WC QR DC W""".split()),
                   key=len, reverse=True)
_TOKEN = re.compile(r"(?<![\w`])([*\-]?[A-Z0-9][A-Z0-9*¥/\-.+]{0,40}[A-Z0-9*¥]|\*[A-Z])(?![\w`])")


def gds_context(question, answer, sources=None):
    """'Amadeus' / 'Sabre' / 'GDS' when the answer is about GDS entries, else ''."""
    locs = " ".join((s.get("location", "") + " " + s.get("title", "")) for s in (sources or [])[:2]).lower()
    blob = f"{question} {answer[:400]} {locs}".lower()
    for name in ("Amadeus", "Sabre", "Galileo", "Worldspan", "Apollo"):
        if name.lower() in locs or re.search(rf"\b{name.lower()}\b", question.lower()):
            return name
    return "GDS" if any(re.search(rf"\b{w}\b", blob) for w in GDS_WORDS) else ""


def _is_command(tok):
    t = tok.lstrip("-")
    if tok.startswith("*") and len(tok) >= 2:
        return True
    if not re.search(r"[A-Z]", t) or t.isdigit():
        return False
    for code in GDS_CODES:
        if t == code:
            return len(code) >= 2
        if t.startswith(code) and re.search(r"[0-9*¥/\-+]", t[len(code):]):
            return True
    return "¥" in t


def gds_commands(text, limit=12):
    """GDS entries found in an answer, in order, without repeats."""
    out = []
    for part in re.split(r"`[^`]*`", text or ""):
        for m in _TOKEN.finditer(part):
            tok = m.group(1).rstrip(".")
            if _is_command(tok) and tok not in out:
                out.append(tok)
    for m in re.finditer(r"`([^`]+)`", text or ""):        # commands the AI already marked
        tok = m.group(1).strip()
        if tok and tok not in out and len(tok) <= 40:
            out.append(tok)
    return out[:limit]


def highlight_commands(text):
    """Wrap GDS entries in `code` so they show as green-on-dark terminal text."""
    pieces = re.split(r"(`[^`]*`)", text or "")
    for k, p in enumerate(pieces):
        if p.startswith("`"):
            continue
        pieces[k] = _TOKEN.sub(lambda m: f"`{m.group(1)}`" if _is_command(m.group(1).rstrip(".")) else m.group(0), p)
    return "".join(pieces)


# ---------------------------------------------------------------------------
# FOLLOW-UP QUESTIONS: "how about Sabre?" after "How to reissue a ticket in Amadeus?"
# ---------------------------------------------------------------------------
_FOLLOW = re.compile(r"^\s*(?:and\s+)?(?:how\s+about|what\s+about|wat\s+about|same\s+for|and\s+for|and\s+in|"
                     r"and\s+with|for|in|with|also\s+for|and)\s+(.+?)\s*\??\s*$", re.I)
SWAP_GROUPS = [["Amadeus", "Sabre", "Galileo", "Worldspan", "Apollo"],
               ["NORAM", "EUROPE", "APAC", "QBAY"],
               ["refund", "exchange", "reissue", "void", "issue", "cancel", "cancellation"]]


def merge_followup(previous, question, extra_names=()):
    """Short follow-up -> full question, using the previous question. Returns the question unchanged
    when it is not a follow-up. 'how about Sabre' + 'How to reissue a ticket in Amadeus?'
    -> 'How to reissue a ticket in Sabre?'"""
    q = (question or "").strip()
    if not previous or len(q.split()) > 6:
        return q
    m = _FOLLOW.match(q)
    if not m:
        return q
    tail = m.group(1).strip(" ?")
    prev = previous.strip().rstrip("?").strip()
    groups = SWAP_GROUPS + ([list(extra_names)] if extra_names else [])
    for group in groups:
        new = next((g for g in group if re.search(rf"\b{re.escape(g)}\b", tail, re.I)), None)
        old = next((g for g in group if re.search(rf"\b{re.escape(g)}\b", prev, re.I)), None)
        if new and old and new.lower() != old.lower():
            return re.sub(rf"\b{re.escape(old)}\b", new, prev, count=1, flags=re.I) + "?"
    if tail.lower() in prev.lower():
        return prev + "?"
    return f"{prev} - {tail}?"
