"""
ui.py - OpsMate look and feel: colours, header bar, cards and small display helpers.
"""
import html

import streamlit as st

import config

NAVY, BLUE, PURPLE, TEAL, ORANGE = "#0f1b3d", "#1F5FBF", "#6A3FB5", "#0F8A7E", "#E8743B"
REGION_COLORS = {"NORAM": BLUE, "EUROPE": PURPLE, "APAC": TEAL, "QBAY": ORANGE, "GLOBAL": "#5b6475"}
RESULT_BADGE = {"answered": ("Answered", "#e7f6ee", "#16794c"),
                "not_found": ("Not found", "#fdecec", "#b42318"),
                "conflict": ("Conflict", "#fff4e5", "#b54708")}

CSS = f"""
<style>
  /* The header holds the top menu bar - it must stay visible */
  header[data-testid="stHeader"] {{background: #ffffff; border-bottom: 1px solid #e3e8f2;}}
  .block-container {{padding-top: 5.6rem; padding-bottom: 6rem; max-width: 1400px;}}
  .om-conf {{background:#e7f6ee; color:#16794c; border-radius:999px; padding:1px 10px; font-size:.78rem;
             font-weight:700; margin-left:6px;}}
  [data-testid="stSegmentedControl"] button p {{font-weight:600;}}
  [data-testid="stSidebar"] {{background: #f6f8fc; border-right: 1px solid #e3e8f2;}}
  [data-testid="stSidebar"] .stRadio label p {{font-size: 0.95rem;}}
  .om-top {{display:flex; align-items:center; gap:18px; padding:12px 22px; border-radius:14px;
           background: linear-gradient(100deg, {NAVY} 0%, #1d3a8a 55%, #3b5bd6 100%); color:#fff;
           margin-bottom: 16px;}}
  .om-logo {{display:flex; align-items:center; gap:10px; font-size:1.6rem; font-weight:800;}}
  .om-logo .ring {{width:38px; height:38px; border-radius:50%; background:#fff; color:{NAVY};
                   display:flex; align-items:center; justify-content:center; font-size:1.2rem;}}
  .om-tag {{opacity:.85; border-left:1px solid rgba(255,255,255,.35); padding-left:14px;}}
  .om-sp {{flex:1}}
  .om-pill {{background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
             padding:6px 14px; border-radius:999px; font-weight:600; font-size:.9rem;}}
  .om-user {{display:flex; align-items:center; gap:10px; font-size:.85rem; line-height:1.15;}}
  .om-av {{width:34px; height:34px; border-radius:50%; background:#fff; color:{BLUE};
           display:flex; align-items:center; justify-content:center; font-weight:700;}}
  .om-motto {{font-weight:700; letter-spacing:.02em; opacity:.95}}
  .om-hero {{border-radius:16px; padding:18px 28px 16px; color:#fff; position:relative; overflow:hidden;
            background: linear-gradient(115deg, #0e1a3c 0%, #23367a 40%, #c9643f 82%, #f6b26b 100%);}}
  .om-hero h1 {{font-size:1.5rem !important; line-height:1.2; margin:0 0 8px; color:#fff !important; padding:0 !important;}}
  .om-hero h1 span {{color:#8fb8ff !important;}}
  .om-hero p {{margin:0; opacity:.92; font-size:.95rem; color:#fff !important;}}
  .om-hero svg {{position:absolute; right:24px; top:8px; height:78px; width:150px; opacity:.9}}
  .om-card {{background:#fff; border:1px solid #e3e8f2; border-radius:14px; padding:16px 18px; margin-bottom:12px;}}
  .om-card h4 {{margin:0 0 10px; font-size:1rem;}}
  .om-badge {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:700;}}
  .om-region {{display:inline-block; padding:3px 12px; border-radius:999px; color:#fff; font-weight:700; font-size:.78rem;}}
  .om-kpi {{background:#fff; border:1px solid #e3e8f2; border-radius:14px; padding:14px 16px;}}
  .om-kpi .v {{font-size:1.7rem; font-weight:800; color:{NAVY};}}
  .om-kpi .l {{color:#5b6475; font-size:.85rem;}}
  .om-kpi .i {{font-size:1.1rem; margin-right:6px}}
  .om-answer-head {{display:flex; align-items:center; gap:8px; font-weight:700; color:{BLUE}; margin-bottom:4px;}}
  .om-important {{background:#fff7e8; border:1px solid #f6d58e; border-radius:10px; padding:10px 14px; margin:10px 0;}}
  .om-conflict {{background:#fff4e5; border:1px solid #f5c58a; border-radius:10px; padding:10px 14px; margin:8px 0;}}
  .om-fallback {{background:#fdecec; border:1px solid #f4b4b0; border-radius:10px; padding:10px 14px;}}
  .om-src {{border-left:4px solid #ccd5e6; background:#f7f9fc; border-radius:6px; padding:8px 12px; margin:6px 0; font-size:.88rem;}}
  .om-meta {{color:#5b6475; font-size:.82rem;}}
  .om-help {{background:#eef3fd; border-radius:12px; padding:12px 14px; font-size:.86rem; color:#33415c;}}
  div[data-testid="stButton"] button[kind="secondary"] {{border-radius:10px;}}
  .om-clarify {{background:#eef3fd; border:1px solid #bcd0f5; border-radius:10px; padding:10px 14px; margin:4px 0 8px;
                font-weight:600; color:#1d3a8a;}}
  [data-testid="stChatMessage"] code {{background:#0f2238; color:#7CFC9A; border-radius:5px; padding:1px 6px;
                font-family: Consolas, "DejaVu Sans Mono", monospace; font-size:.88em;}}
  .om-term {{background:#0f2238; border-radius:10px; margin:10px 0; overflow:hidden; max-width:640px;}}
  .om-term .bar {{background:#1e3a5f; color:#e2e8f0; font-size:.78rem; padding:5px 12px; font-weight:600;}}
  .om-term .body {{padding:8px 14px 10px;}}
  .om-term .ln {{color:#7CFC9A !important; font-family: Consolas, "DejaVu Sans Mono", monospace !important;
                 font-size:.9rem; line-height:1.6; white-space:pre-wrap; word-break:break-all;}}
  .om-term .ln span {{color:#94a3b8 !important;}}
  .om-section {{font-weight:800; color:{NAVY}; font-size:1.05rem; margin:6px 0 8px;}}
</style>
"""

PLANE_SVG = """<svg width="230" height="120" viewBox="0 0 230 120" xmlns="http://www.w3.org/2000/svg">
<path d="M10 92 L220 92" stroke="rgba(255,255,255,.35)" stroke-width="2" stroke-dasharray="10 8"/>
<g transform="translate(118 40) rotate(-12)" fill="#fff">
<path d="M-70 4 L40 -4 C52 -5 62 0 62 4 C62 8 52 13 40 12 L-70 6 Z"/>
<path d="M-5 0 L-35 -38 L-22 -38 L18 -2 Z"/><path d="M-5 8 L-35 46 L-22 46 L18 10 Z"/>
<path d="M-62 3 L-74 -16 L-66 -16 L-50 3 Z"/></g>
<circle cx="200" cy="24" r="10" fill="rgba(255,214,140,.9)"/></svg>"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    return html.escape("" if text is None else str(text))


def initials(name):
    parts = [p for p in str(name).replace("(", " ").split() if p[:1].isalpha()]
    return ("".join(p[0] for p in parts[:2]) or "U").upper()


def topbar(user, role, region):
    st.markdown(
        f"<div class='om-top'><div class='om-logo'><div class='ring'>✈</div>{config.APP_NAME}</div>"
        f"<div class='om-tag'>{esc(config.TAGLINE)}</div><div class='om-sp'></div>"
        f"<div class='om-pill'>🌐 {esc(region)}</div>"
        f"<div class='om-user'><div class='om-av'>{esc(initials(user))}</div>"
        f"<div><b>{esc(user)}</b><br>{esc(role)}</div></div>"
        f"<div class='om-motto'>Ask · Resolve · Learn</div></div>", unsafe_allow_html=True)


def hero(first_name):
    st.markdown(
        f"<div class='om-hero'>{PLANE_SVG}<h1>Hi {esc(first_name)}, how can <span>{config.APP_NAME}</span> "
        f"help you today?</h1><p>Ask anything about products, processes, systems or policies.</p></div>",
        unsafe_allow_html=True)


def region_pill(region):
    return (f"<span class='om-region' style='background:{REGION_COLORS.get(region, '#5b6475')}'>"
            f"{esc(region)}</span>")


def result_badge(result):
    label, bg, fg = RESULT_BADGE.get(result or "answered", RESULT_BADGE["answered"])
    return f"<span class='om-badge' style='background:{bg};color:{fg}'>{label}</span>"


def kpi(col, icon, value, label, sub=""):
    col.markdown(f"<div class='om-kpi'><div class='v'><span class='i'>{icon}</span>{esc(value)}</div>"
                 f"<div class='l'>{esc(label)}</div><div class='om-meta'>{esc(sub)}</div></div>",
                 unsafe_allow_html=True)


def terminal(commands, system="GDS"):
    """GDS-style box listing the commands used in an answer, in order."""
    lines = "".join(f"<div class='ln'><span>&gt;</span> {esc(c)}</div>" for c in commands)
    return (f"<div class='om-term'><div class='bar'>🖥 {esc(system)} entries in this answer</div>"
            f"<div class='body'>{lines}</div></div>")


def card(title, body_html):
    st.markdown(f"<div class='om-card'><h4>{title}</h4>{body_html}</div>", unsafe_allow_html=True)
