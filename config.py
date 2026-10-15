"""
config.py - All OpsMate settings in one place.
Change values here (or in Streamlit Secrets / environment variables).
"""
import os
from pathlib import Path


# What happened when OpsMate read Streamlit Secrets - shown to admins so a format mistake is obvious.
# state: "ok" | "missing" (no Secrets saved) | "error" (Secrets saved but could not be read)
SECRETS_STATUS = {"state": "ok", "detail": ""}
SECRET_SOURCE = {}            # setting name -> "environment" / "secrets" / "default" (never the value)


def _secrets_problem(err):
    """Short, safe description of a Secrets problem (no file paths, no secret values)."""
    import re
    msg = str(err)
    if "No secrets found" in msg:
        return "missing", "No Secrets are saved for this app."
    m = re.search(r"\(line (\d+) column (\d+)", msg)
    reason = msg.split(":")[-1].split("(")[0].strip() if ":" in msg else "format problem"
    where = f" (line {m.group(1)})" if m else ""
    return "error", f"{reason}{where}"


def _secret(name, default=""):
    """Read a setting from an environment variable, or from Streamlit Secrets."""
    value = os.getenv(name)
    if value:
        SECRET_SOURCE[name] = "environment"
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            SECRET_SOURCE[name] = "secrets"
            return str(st.secrets[name])
    except Exception as e:                  # Secrets could not be read at all
        state, detail = _secrets_problem(e)
        SECRETS_STATUS.update(state=state, detail=detail)
    SECRET_SOURCE[name] = "default"
    return default


# ---- App identity ---------------------------------------------------------
APP_NAME = "OpsMate"
TAGLINE = "Your AI Operations Companion"
COMPANY_NAME = "Your Company"

# ---- The two standard replies (edit the wording here) -----------------------
FALLBACK_TEXT = "I do not have answer to this question, please consult with your team supervisor."
CONFLICT_TEXT = ("I found different answers to this question in more than one document, so I cannot "
                 "confirm which one is correct. " + FALLBACK_TEXT)

# ---- Lines of business (LOB) ----------------------------------------------
# Each LOB is a tab above the chat; its sub-LOBs are the chips under the tab.
# Questions search the selected LOB's documents (plus GLOBAL documents) first,
# ranking the selected sub-LOB's documents highest. If nothing is found there,
# OpsMate searches the other LOBs and says clearly where the answer came from.
# Sub-LOBs can be changed by admins in Settings (saved in the database).
# Adding a NEW top-level LOB needs a change here (REGIONS and DEFAULT_LOBS).
REGIONS = ["NORAM", "EUROPE", "APAC", "QBAY"]
GLOBAL_REGION = "GLOBAL"            # GLOBAL documents are visible in every LOB
REGION_LABELS = {"NORAM": "🌎 NORAM", "EUROPE": "🌍 EUROPE", "APAC": "🌏 APAC", "QBAY": "📘 QBAY"}
DEFAULT_LOBS = {
    "NORAM": ["OBT Cancellation", "Schedule Change", "Ticketing", "Refund", "Profile Management", "Others"],
    "EUROPE": ["Ticketing", "Refund", "UETTR", "AGM", "BRIM ERM", "BRIM", "Matching", "Conferma", "SAP",
               "OTSMP", "DQS"],
    "APAC": ["AU TED", "AU Queues", "AU Ticketing", "SG Ticketing", "SG Refund", "SG Fares and Commission",
             "SG Support Service", "HK Ticketing", "HK Refund", "General Documents"],
    "QBAY": ["User Guide"],
}
SUB_LOB_BOOST = 2.20          # documents of the selected sub-LOB rank first...
OTHER_SUB_FACTOR = 0.80       # ...other sub-LOBs of the same LOB are still searched, slightly lower
GENERAL_DOC_FACTOR = 0.75     # documents with no sub-LOB (and GLOBAL ones) when a sub-LOB is chosen
CROSS_LOB_FALLBACK = True     # nothing found in this LOB -> look in the other LOBs (clearly labelled)

# ---- Topics (home-screen buttons + dashboard grouping) --------------------
# name: words that put a question into that topic
TOPICS = {
    "Ticketing": "ticket issue issuance ttp tst pnr booking book void e-ticket eticket itinerary "
                 "name correction spelling passenger",
    "Refunds": "refund refunds refundable money back cancellation fee refnd",
    "Exchanges": "exchange reissue revalidation date change change fee atc rebook",
    "OBT": "obt online booking tool self booking self-booking concur cytric",
    "Invoicing": "invoice invoicing billing credit note supplier gst vat",
    "Fare Rules": "fare rule rules category cat 16 31 33 penalty fare basis",
    "GDS (Amadeus/Sabre)": "amadeus sabre galileo gds command cryptic entry twd fxp tqt",
    "Profiles": "profile profiles traveller profile frequent flyer ffp passport details",
    "Travel Policy": "policy approval business class economy hotel preferred out of policy",
    "Quality / Errors": "error errors adm quality debit memo mistake issue failed invalid",
}
TOPIC_LABELS = {"GDS (Amadeus/Sabre)": "GDS", "Quality / Errors": "Quality"}   # short button text
TOPIC_ICONS = {"Ticketing": "✈️", "Refunds": "💸", "Exchanges": "🔄", "OBT": "🧳", "Invoicing": "🧾",
               "Fare Rules": "📜", "GDS (Amadeus/Sabre)": "🖥️", "Profiles": "👤",
               "Travel Policy": "📋", "Quality / Errors": "⚠️"}

# ---- Document categories --------------------------------------------------
DOC_TYPES = ["SOP", "FAQ", "Quick Guide", "Process Knowledge", "Policy", "Other"]
ALLOWED_EXTENSIONS = ["pdf", "docx", "pptx", "xlsx", "txt", "md", "csv", "png", "jpg", "jpeg"]

# ---- Admin access ---------------------------------------------------------
# Set ADMIN_PASSWORD in Streamlit Secrets. "admin123" is only a demo fallback and
# the app shows a warning to admins while it is in use.
ADMIN_PASSWORD = _secret("ADMIN_PASSWORD", "admin123")
DEFAULT_ADMIN_PASSWORD_IN_USE = ADMIN_PASSWORD == "admin123"


def password_warning():
    """None when the admin password is set properly, else the exact reason and how to fix it."""
    if not DEFAULT_ADMIN_PASSWORD_IN_USE:
        return None
    fix = ("Fix: in Streamlit → ⋮ → Settings → Secrets, put each entry on ONE line, e.g. "
           "ADMIN_PASSWORD = \"your-strong-password\" and GEMINI_API_KEY = \"your-key\" - "
           "then Save and ⋮ → Reboot app.")
    if SECRETS_STATUS["state"] == "error":
        return (f"Streamlit Secrets could not be read - check the format: {SECRETS_STATUS['detail']}. "
                "OpsMate is using the demo password admin123 until this is fixed. "
                "A value split over two lines is the usual cause. " + fix)
    if SECRET_SOURCE.get("ADMIN_PASSWORD") in ("secrets", "environment"):
        return "ADMIN_PASSWORD in Secrets is still admin123. Change it to your own password, Save, then ⋮ → Reboot app."
    if SECRETS_STATUS["state"] == "missing":
        return "No Secrets are saved for this app, so the demo password admin123 is in use. " + fix
    return ("ADMIN_PASSWORD is not in Streamlit Secrets, so the demo password admin123 is in use. "
            "If you just added it, ⋮ → Reboot app so OpsMate reads it. " + fix)

# ---- Demo mode ------------------------------------------------------------
# Load sample_docs/ automatically when the knowledge base is empty (free hosting
# wipes data on restart). File names start with the region: NORAM_..., GLOBAL_...
AUTO_LOAD_SAMPLES = os.getenv("AUTO_LOAD_SAMPLES", "true").lower() == "true"

# ---- Storage --------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "knowledge.db"

# ---- Search tuning --------------------------------------------------------
CHUNK_SIZE = 700          # characters per knowledge piece
CHUNK_OVERLAP = 150
CANDIDATES = 8            # passages compared for each answer (from all documents)
PER_DOC = 2               # max passages from one document -> answers mix documents
LEARNED_BOOST = 1.25      # verified answers (written by the process team)
AUTO_LEARNED_BOOST = 1.10 # answers OpsMate learned by itself from 👍 votes

# Confidence without AI (word-match score of the best passage)
CONF_HIGH = 0.22
CONF_MEDIUM = 0.11        # below this OpsMate gives FALLBACK_TEXT instead of guessing
# Confidence with Gemini semantic search (cosine similarity of the best passage)
SEM_HIGH = 0.72
SEM_MEDIUM = 0.60

MAX_ANSWER_WORDS = 60     # "short" answers stay under this many words

# Confidence % shown with every answer. The level (High / Medium) is decided by the
# rules above; the % places the answer inside that level's band using how strongly
# the best passage matches, how much of the question it covers and how many
# documents support it.
CONF_BANDS = {"high": (80, 98), "medium": (60, 79), "low": (20, 59)}

# ---- Self-learning --------------------------------------------------------
AUTO_LEARN = True         # save answers that staff mark 👍 as "auto-learned"
AUTO_LEARN_MIN_VOTES = 2  # 👍 from this many DIFFERENT people before an answer is auto-learned
                          # (an admin's 👍 counts as enough on its own)
FEEDBACK_SIMILARITY = 0.55  # how similar a new question must be to reuse past 👍/👎

# ---- AI providers ----------------------------------------------------------
# OpsMate works with NO key (free, offline). Add ONE key to turn on AI:
#   Gemini (free tier available): GEMINI_API_KEY  -> also gives semantic search
#   Claude (paid):                ANTHROPIC_API_KEY
# Streamlit Cloud: App -> Settings -> Secrets. Laptop: setx GEMINI_API_KEY "..."
GEMINI_API_KEY = _secret("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY", "")
AI_PROVIDER = _secret("AI_PROVIDER", "auto")   # auto | gemini | claude | off

GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-3.8-flash")            # answers, screenshots
GEMINI_FAST_MODEL = _secret("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")  # rephrasing
GEMINI_EMBED_MODEL = _secret("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768

CLAUDE_MODEL = _secret("CLAUDE_MODEL", "claude-sonnet-5-5")
CLAUDE_FAST_MODEL = _secret("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")

# ---- Images inside documents ----------------------------------------------
READ_DOC_IMAGES = True    # read screenshots/tables/scans inside uploaded files (AI mode)
MAX_IMAGES_PER_DOC = 25
# Ask a short question first when the answer depends on something the agent did not say
ASK_CLARIFY = True
CLARIFY_GROUPS = {"GDS": ["Amadeus", "Sabre", "Galileo", "Worldspan", "Apollo"]}
CLARIFY_ALIASES = {"Amadeus": ["1A", "ama"], "Sabre": ["1S"], "Galileo": ["1G", "Travelport"],
                   "Worldspan": ["1P"], "Apollo": ["1V"]}
CLARIFY_CLOSE = 0.70          # other variants count when they score at least 70% of the best one
CLARIFY_MIN_COVER = 0.5       # ...and match at least half of the question's key words
# Showing document pictures with answers (works without the AI key)
SHOW_ANSWER_IMAGES = True
MAX_ANSWER_IMAGES = 2         # pictures shown under one answer
IMAGE_SOURCE_PASSAGES = 1     # pictures only from the passage the answer came from
MAX_STORED_IMAGES_PER_DOC = 60
MIN_IMAGE_BYTES = 3000        # smaller files are icons / logos / lines
MIN_IMAGE_SIDE = 80           # pixels
MAX_IMAGE_WIDTH = 1400        # bigger pictures are shrunk to save space
MIN_IMAGE_PX = 120
