"""
app.py - OpsMate (NORAM / EUROPE / APAC). Staff screens + navigation.
Admin screens are in admin_pages.py.   Run:  streamlit run app.py
"""
import io

import streamlit as st

import admin_pages
import brain
import config
import knowledge as kb
import llm
import ui

st.set_page_config(page_title=config.APP_NAME, page_icon="✈️", layout="wide")
ui.setup()
kb.init_db()
if config.AUTO_LOAD_SAMPLES:
    kb.load_samples_if_empty()   # demo hosting: refill sample SOPs after a restart


@st.cache_resource(show_spinner=False)
def _pictures_ready():
    """Once per server start: keep the pictures of documents uploaded before this version."""
    try:
        return kb.backfill_images()
    except Exception:
        return 0


_pictures_ready()

try:                                    # "Paste screenshot" button (optional add-on)
    from streamlit_paste_button import paste_image_button
except ImportError:
    paste_image_button = None

# Shown under 'Try asking' only when no document or history suggests anything yet
DEFAULT_TRY = ["How to reissue a ticket in Amadeus?", "Refund process step by step",
               "Can I void a ticket the next day?", "What is the escalation matrix?",
               "List of Amadeus commands"]

ss = st.session_state
for k, v in {"user": None, "is_admin": False, "region": config.REGIONS[0], "pasted": None,
             "paste_n": 0, "topic": None, "pending_q": None,
             "pending_clarify": None}.items():
    ss.setdefault(k, v)
ss.setdefault("chats", {r: [] for r in config.REGIONS})

# ---------------------------------------------------------------------------
# SIGN IN
# ---------------------------------------------------------------------------
if not ss.user:
    ui.topbar("Guest", "Not signed in", "—")
    left, right = st.columns([1.1, 1])
    with left:
        ui.hero("there")
        st.write("")
        with st.form("login"):
            name = st.text_input("Your name / employee ID")
            c1, c2 = st.columns(2)
            role = c1.radio("I am", ["Agent", "Admin / Process Owner"], horizontal=False)
            region = c2.radio("My LOB", config.REGIONS, horizontal=False,
                              format_func=lambda r: config.REGION_LABELS.get(r, r))
            pwd = st.text_input("Admin password", type="password", help="Only for Admin / Process Owner")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                if not name.strip():
                    st.error("Please enter your name.")
                elif role.startswith("Admin") and pwd != config.ADMIN_PASSWORD:
                    st.error("Wrong admin password.")
                else:
                    ss.user, ss.is_admin, ss.region = name.strip(), role.startswith("Admin"), region
                    st.rerun()
    with right:
        ui.card("How OpsMate works", "<ol style='margin:0;padding-left:18px;line-height:1.9'>"
                "<li>Pick your region and a topic, or just type your question</li>"
                "<li>Spelling mistakes and Hindi / Kannada / Telugu in English letters are fine</li>"
                "<li>Paste a screenshot of an error to get help with it</li>"
                "<li>Get one short answer with its SOP source and a confidence level</li>"
                "<li>Rate it 👍 / 👎 - OpsMate learns from every rating</li></ol>")
        ui.card("When OpsMate is not sure", f"<p style='margin:0'>It never guesses. You will see: "
                f"<i>“{ui.esc(config.FALLBACK_TEXT)}”</i> and the question goes to the process team.</p>")
    st.stop()

# ---------------------------------------------------------------------------
# TOP BAR: who is signed in, region, sign out  (never hidden - no sidebar needed)
# ---------------------------------------------------------------------------
LOBS = kb.get_lobs()

bar, ctl = st.columns([5.2, 1.3], vertical_alignment="center")
with bar:
    ui.topbar(ss.user, "Admin" if ss.is_admin else "Agent", ss.region)
with ctl:
    if st.button("⏻ Sign out", width="stretch", key="signout"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()
region = ss.region
if ss.is_admin and config.password_warning():
    st.warning(config.password_warning(), icon="🔐")


def lob_bar(chips=True):
    """LOB tabs (NORAM / EUROPE / APAC / QBAY) + sub-LOB chips. Returns the chosen sub-LOB or None."""
    if ss.get("region_tab") not in config.REGIONS:
        ss.region_tab = ss.region
    pick = st.segmented_control("Line of business", config.REGIONS, key="region_tab", width="stretch",
                                format_func=lambda r: config.REGION_LABELS.get(r, r), label_visibility="collapsed")
    if pick and pick != ss.region:
        ss.region = pick
        st.rerun()
    subs = LOBS.get(ss.region, [])
    if not chips or not subs:
        return None
    all_label = f"All {ss.region}"
    sub_pick = st.pills("Sub-LOB", [all_label] + subs, key=f"sub_{ss.region}", default=all_label,
                        label_visibility="collapsed")
    return sub_pick if sub_pick and sub_pick != all_label else None


# ---------------------------------------------------------------------------
# ANSWER RENDERING
# ---------------------------------------------------------------------------
def show_sources(sources, key):
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for j, s in enumerate(sources):
            a, b = st.columns([6, 1])
            color = ui.REGION_COLORS.get(s["region"], "#999")
            a.markdown(f"<div class='om-src' style='border-left-color:{color}'><b>{ui.esc(s['title'])}</b> · "
                       f"{ui.esc(s['type'])} · {ui.esc(s['location'])} · <i>{ui.esc(s['region'])}</i></div>",
                       unsafe_allow_html=True)
            with b.popover("View"):
                st.markdown(f"**{s['title']}** — {s['location']}")
                st.text(s["text"][:1500])
                doc = kb.get_document(s["doc_id"]) if s.get("doc_id") else None
                if doc:
                    st.download_button("⬇ Open document", kb.read_file(doc), file_name=doc["filename"],
                                       key=f"dl{key}{j}")


CONF_HELP = ("Confidence % = how sure OpsMate is that the answer comes from your documents. "
             "High is 80-98%, Medium 60-79%. It is based on how closely the best document matches "
             "your question, how many of your key words it covers, and how many documents say the same. "
             "Below Medium, OpsMate does not answer and asks you to consult your supervisor.")


def render_clarify(msg, i):
    """OpsMate asks one short question (which GDS / which process) with buttons, then answers."""
    info = (msg.get("meta") or {}).get("clarify") or {}
    st.markdown(f"<div class='om-clarify'>🤔 {ui.esc(msg['content'])}</div>", unsafe_allow_html=True)
    if msg.get("picked"):
        st.caption(f"You chose: {msg['picked']}")
        return
    opts = info.get("options", [])
    cols = st.columns(len(opts) + 1)
    for k, (c, o) in enumerate(zip(cols, opts)):
        if c.button(o["label"], key=f"cl{region}{i}_{k}", type="primary", width="stretch"):
            q = msg.get("question", "")
            ss.pending_clarify = {"q": q + o.get("append", ""), "sub_lob": o.get("sub_lob") or msg.get("sub_lob")}
            msg["picked"] = o["label"]
            st.rerun()
    if cols[-1].button("Search all", key=f"cl{region}{i}_all", width="stretch",
                       help="Answer without choosing - OpsMate searches everything"):
        ss.pending_clarify = {"q": msg.get("question", ""), "sub_lob": msg.get("sub_lob")}
        msg["picked"] = "Search all"
        st.rerun()


def render_answer(msg, i):
    result = msg.get("result", "answered")
    meta = msg.get("meta") or {}
    pct = msg.get("confidence_pct")
    tag = (f" <span class='om-conf'>{pct}% confident</span>" if pct is not None and result == "answered" else "")
    st.markdown(f"<div class='om-answer-head'>✈ {config.APP_NAME}{tag}</div>", unsafe_allow_html=True)
    if result == "clarify":
        render_clarify(msg, i)
        return
    if result == "not_found":
        st.markdown(f"<div class='om-fallback'>🛈 {ui.esc(msg['content'])}</div>", unsafe_allow_html=True)
        st.caption("Your question has been sent to the process team so the SOPs can be improved.")
    elif result == "conflict":
        st.markdown(f"<div class='om-conflict'>⚠️ <b>Different answers found.</b> {ui.esc(msg['content'])}</div>",
                    unsafe_allow_html=True)
        if meta.get("conflict_text"):
            st.markdown(f"**What each document says:** {meta['conflict_text']}")
        else:
            st.markdown("**What each document says:**")
            for s in msg.get("sources", [])[:2]:
                st.markdown(f"- **{s['title']}** ({s['region']}, {s['location']}): "
                            f"{brain.snippet(msg.get('question', ''), s['text'], s['location'])}")
        st.caption("Reported to the process team as a conflict to fix ✔")
    else:
        gds = brain.gds_context(msg.get("question", ""), msg["content"], msg.get("sources"))
        body = brain.highlight_commands(msg["content"]) if gds else msg["content"]
        st.markdown(("✅ " if not body.lstrip().startswith(("|", "1.", "-")) else "") + body)
        cmds = brain.gds_commands(msg["content"]) if gds else []
        if len(cmds) >= 2:
            st.markdown(ui.terminal(cmds, gds), unsafe_allow_html=True)
        for img in msg.get("images") or []:
            try:
                st.image(img["path"], caption=img["caption"])
            except Exception:                       # picture removed (document deleted / server reset)
                pass
        if meta.get("important"):
            st.markdown(f"<div class='om-important'>❗ <b>Important:</b> {ui.esc(meta['important'])}</div>",
                        unsafe_allow_html=True)
        if meta.get("learned"):
            st.caption("🧠 From an answer OpsMate learned (process team / staff 👍)")
    if result != "not_found":
        show_sources(msg.get("sources"), f"{region}{i}")
    where = meta.get("other_lob") or region
    bits = [f"Answer based on {where} documents" + (f" · {meta['sub_lob']} first" if meta.get("sub_lob") else "")]
    if msg.get("confidence") and result == "answered":
        pct = msg.get("confidence_pct")
        bits.append(f"Confidence: **{pct}%** ({msg['confidence'].title()})" if pct is not None
                    else f"Confidence: **{msg['confidence'].title()}**")
        if meta.get("supporting_docs", 0) > 1:
            bits.append(f"supported by {meta['supporting_docs']} documents")
    bits.append("🤖 AI" if meta.get("mode") == "ai" else "📄 documents")
    if meta.get("seconds") is not None:
        bits.append(f"⏱ {meta['seconds']} s")
    if meta.get("followup"):
        bits.append(f"🔗 follow-up, understood as *{meta['followup']}*")
    elif meta.get("did_you_mean"):
        bits.append(f"🔤 understood as *{meta['did_you_mean']}*")
    st.caption(" · ".join(bits), help=CONF_HELP if result == "answered" else None)
    for note in meta.get("notes", []):
        st.caption(f"ℹ️ {note}")
    if msg.get("qid") and result == "answered":
        if msg.get("rated") is True:
            st.caption("Thanks - OpsMate learns from your feedback ✔")
        elif msg.get("rated") != "pending":
            b1, b2, _ = st.columns([1.6, 2, 4])
            if b1.button("👍 Helpful", key=f"up{region}{i}"):
                kb.give_feedback(msg["qid"], True, trusted=ss.is_admin)
                msg["rated"] = True
                st.rerun()
            if b2.button("👎 Not helpful", key=f"dn{region}{i}"):
                msg["rated"] = "pending"
                st.rerun()
        if msg.get("rated") == "pending":
            with st.form(f"fb{region}{i}"):
                note = st.text_area("What was wrong or missing? (optional)")
                if st.form_submit_button("Send to process team"):
                    kb.give_feedback(msg["qid"], False, note)
                    msg["rated"] = True
                    st.rerun()


def ask_now(text, image=None, sub_lob=None, clarify=True):
    chat = ss.chats[region]
    # short follow-up ("how about Sabre?") -> full question using the previous one in this chat
    prev = next((m.get("asked") or m["content"] for m in reversed(chat) if m["role"] == "user"
                 and not m.get("image")), "")
    sub_names = [x for lst in LOBS.values() for x in lst]
    asked = brain.merge_followup(prev, text, sub_names) if text and not image else text
    chat.append({"role": "user", "content": text or "*(screenshot)*", "image": image, "asked": asked,
                 "lob": " › ".join(x for x in (region, sub_lob) if x)})
    where = f"{sub_lob} documents first" if sub_lob else f"{region} documents"
    with st.spinner("Reading your screenshot and searching SOPs…" if image else f"Searching {where}…"):
        res = kb.ask(region, asked, ss.user, image=image, sub_lob=sub_lob, clarify=clarify)
    if asked != text:
        res["meta"]["followup"] = asked
    chat.append({"role": "assistant", "content": res["answer"], "sources": res["sources"],
                 "qid": res["id"], "result": res["result"], "confidence": res["confidence"],
                 "confidence_pct": res.get("confidence_pct"),
                 "meta": res["meta"], "question": res["question"], "images": res.get("images", []),
                 "sub_lob": sub_lob})
    ss.pasted = None
    ss.paste_n += 1


# ---------------------------------------------------------------------------
# PAGE: Ask OpsMate (home + chat)
# ---------------------------------------------------------------------------
def page_ask():
    chat = ss.chats[region]
    sub_lob = lob_bar()
    if ss.pending_clarify:
        pc, ss.pending_clarify = ss.pending_clarify, None
        ask_now(pc["q"], sub_lob=pc.get("sub_lob"), clarify=False)
        st.rerun()
    if ss.pending_q:
        q, ss.pending_q = ss.pending_q, None
        ask_now(q, sub_lob=sub_lob)
        st.rerun()

    main, side = st.columns([3, 1.15], gap="large")
    with main:
        if not chat:
            ui.hero(ss.user.split()[0])
            # 'Try asking' follows the selected LOB tab and sub-LOB chip (no separate topic buttons -
            # they looked like filters but did not change the search)
            where = f"{region} › {sub_lob}" if sub_lob else region
            tries = kb.suggested_questions(region, sub_lob, 6) or DEFAULT_TRY
            st.markdown(f"<div class='om-section'>💡 Try asking · {ui.esc(where)}</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for n, q in enumerate(tries[:6]):
                if cols[n % 3].button(q, key=f"try{region}{sub_lob}{n}", width="stretch"):
                    ss.pending_q = q
                    st.rerun()
        else:
            h1, h2 = st.columns([4, 1])
            h1.markdown(f"<div class='om-section'>Conversation · {ui.region_pill(region)}</div>",
                        unsafe_allow_html=True)
            if h2.button("＋ New chat", width="stretch"):
                ss.chats[region] = []
                st.rerun()
            for i, msg in enumerate(chat):
                with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "✈️"):
                    if msg.get("image"):
                        st.image(msg["image"], width=320)
                    if msg["role"] == "user":
                        st.markdown(msg["content"])
                        if msg.get("lob"):
                            st.caption(f"LOB: {msg['lob']}")
                    else:
                        render_answer(msg, i)

        # ---- attach a screenshot / ask by voice --------------------------------
        st.write("")
        t1, t2, t3 = st.columns([1.5, 1.6, 2.7])
        ask_shot = False
        with t1:
            if paste_image_button is not None:
                res = paste_image_button("📋 Paste screenshot", background_color=ui.BLUE,
                                         hover_background_color="#174a96", key=f"paste{ss.paste_n}")
                if res.image_data is not None:
                    buf = io.BytesIO()
                    res.image_data.save(buf, format="PNG")
                    ss.pasted = buf.getvalue()
        with t2:
            voice = None
            if llm.provider() == "gemini" and hasattr(st, "audio_input"):
                voice = st.audio_input("🎙 Ask by voice", key=f"voice{ss.paste_n}", label_visibility="collapsed")
            else:
                st.caption("🎙 Voice questions work when a Gemini key is added.")
        with t3:
            if ss.pasted:
                a, b = st.columns([2, 1])
                a.image(ss.pasted, caption="Screenshot attached", width=200)
                ask_shot = b.button("➤ Ask about it", type="primary")
                if b.button("✕ Remove"):
                    ss.pasted = None
                    ss.paste_n += 1
                    st.rerun()
            else:
                st.caption("Error on screen? Copy a screenshot (Win + Shift + S), click 📋 Paste screenshot, "
                           "then ask. Or attach an image with ➕ in the question box.")
        if voice is not None:
            try:
                text = llm.transcribe(voice.getvalue(), voice.type or "audio/wav")
                ask_now(text, sub_lob=sub_lob)
                st.rerun()
            except llm.AIError as e:
                st.warning(f"Could not read the voice note: {e}")

    with side:
        mine = kb.history(region, ss.user, 5)
        body = "".join(
            f"<div style='margin-bottom:10px'><div style='font-size:.9rem'>{ui.esc(r['question'][:70])}</div>"
            f"{ui.result_badge(r.get('result') or ('answered' if r['status'] in ('answered','resolved') else 'not_found'))}"
            f" <span class='om-meta'>{ui.esc(r['asked_at'][5:])}</span></div>" for r in mine) \
            or "<span class='om-meta'>Your questions will appear here.</span>"
        ui.card("Recent Questions", body)
        notes = kb.announcements(region)
        icons = {"Alert": "🔴", "Maintenance": "🛠️", "New SOP": "🟢", "Info": "🔵"}
        body = "".join(f"<div style='margin-bottom:8px'>{icons.get(a['level'], '🔵')} <b>{ui.esc(a['title'])}</b>"
                       f"<div class='om-meta'>{ui.esc(a['body'][:120])}</div></div>" for a in notes[:5]) \
            or "<span class='om-meta'>No updates right now.</span>"
        ui.card("Important Updates", body)
        st.markdown("<div class='om-help'><b>Need more help?</b><br>If you are not sure, please consult "
                    "your supervisor.</div>", unsafe_allow_html=True)
        st.caption(("🤖 AI: " + llm.label()) if llm.ai_on() else "📄 AI off - answers from documents")

    prompt = st.chat_input("Type your question here… (English, Hindi, Kannada, Telugu in English letters)",
                           accept_file=True, file_type=["png", "jpg", "jpeg"])
    if prompt or (ss.pasted and ask_shot):
        text = (prompt.text if prompt else "") or ""
        image = ss.pasted
        if prompt and prompt.files:
            image = prompt.files[0].getvalue()
        if text.strip() or image:
            ask_now(text, image, sub_lob=sub_lob)
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: My Questions
# ---------------------------------------------------------------------------
def page_my_questions():
    lob_bar(chips=False)
    st.header("My Questions")
    rows = kb.history(region, ss.user, 100)
    if not rows:
        st.info("You haven't asked anything in this region yet.")
    for r in rows:
        result = r.get("result") or ("answered" if r["status"] in ("answered", "resolved") else "not_found")
        label = {"answered": "✅", "not_found": "⏳", "conflict": "⚠️"}.get(result, "")
        with st.expander(f"{label} {r['asked_at']} · {r['question']}"):
            st.markdown(ui.result_badge(result), unsafe_allow_html=True)
            st.markdown(r["answer"] or "")
            if r["status"] == "resolved":
                st.success("The process team has added an answer - ask again to see it.")


# ---------------------------------------------------------------------------
# PAGE: Knowledge Library / Quick Guides / Announcements
# ---------------------------------------------------------------------------
def doc_list(docs, key):
    for d in docs:
        with st.container(border=True):
            a, b, c = st.columns([5, 2, 2])
            a.markdown(f"**{d['title']}**  \n{d['doc_type']} · {d['filename']}"
                       + (f" · v{d['version']}" if d["version"] else ""))
            lob_txt = d.get("sub_lob") or "General"
            b.markdown(ui.region_pill(d["region"]) + f"<br><span class='om-meta'>{ui.esc(lob_txt)} · "
                       f"{d['uploaded_at']}</span>", unsafe_allow_html=True)
            c.download_button("⬇ Download", kb.read_file(d), file_name=d["filename"], key=f"{key}{d['id']}")
            with c.popover("👁 View"):
                st.text(kb.document_text(d["id"], 3000))


def page_library():
    lob_bar(chips=False)
    st.header("Knowledge Library")
    docs = kb.list_documents(region, status="Active")
    c1, c2, c3 = st.columns([1, 1, 2])
    lb = c1.selectbox("Sub-LOB", ["All"] + LOBS.get(region, []) + ["General (no sub-LOB)"])
    t = c2.selectbox("Type", ["All"] + config.DOC_TYPES)
    f = c3.text_input("Search by title")
    docs = [d for d in docs if (t == "All" or d["doc_type"] == t) and (not f or f.lower() in d["title"].lower())
            and (lb == "All" or (d.get("sub_lob") or "") == ("" if lb.startswith("General") else lb))]
    if not docs:
        st.info("No documents match.")
    doc_list(docs, "lib")


def page_quick_guides():
    lob_bar(chips=False)
    st.header("Quick Guides")
    st.caption("Short how-to guides. Admins upload them with Document type = Quick Guide.")
    docs = [d for d in kb.list_documents(region, status="Active") if d["doc_type"] == "Quick Guide"]
    if not docs:
        st.info("No quick guides yet for this region.")
    for d in docs:
        with st.expander(f"⚡ {d['title']}"):
            st.text(kb.document_text(d["id"], 4000))
            st.download_button("⬇ Download", kb.read_file(d), file_name=d["filename"], key=f"qg{d['id']}")


def page_announcements():
    st.header("Announcements")
    if ss.is_admin:
        admin_pages.announcements_admin(ss.user)
    items = kb.announcements(region)
    if not items:
        st.info("No announcements right now.")
    for a in items:
        with st.container(border=True):
            st.markdown(f"**{a['title']}** · {a['level']} · {a['region']}  \n{a['body']}")
            st.caption(f"{a['author']} · {a['created_at']}")
            if ss.is_admin and st.button("Hide", key=f"hide{a['id']}"):
                kb.set_announcement(a["id"], False)
                st.rerun()


# ---------------------------------------------------------------------------
# NAVIGATION - a menu bar across the top of every page (works on any screen size)
# ---------------------------------------------------------------------------
def _page(fn, title, icon, url, default=False):
    def run():
        fn()
    run.__name__ = "page_" + url.replace("-", "_")
    return st.Page(run, title=title, icon=icon, url_path=url, default=default)


def _admin(fn):
    return lambda: fn(region, ss.user)


staff = [_page(page_ask, "Ask OpsMate", "💬", "ask", True),
         _page(page_my_questions, "My Questions", "🕘", "my-questions"),
         _page(page_library, "Knowledge Library", "📚", "library"),
         _page(page_quick_guides, "Quick Guides", "⚡", "quick-guides"),
         _page(page_announcements, "Announcements", "📢", "announcements")]
sections = {"OpsMate": staff}
if ss.is_admin:
    n_review, n_gaps = len(kb.review_queue()), len(kb.knowledge_gaps())
    sections["Admin"] = [
        _page(_admin(admin_pages.dashboard), "Dashboard", "📊", "dashboard"),
        _page(_admin(admin_pages.documents), "Document Management", "🗂️", "documents"),
        _page(_admin(admin_pages.review), f"Review Queue ({n_review})", "🧠", "review"),
        _page(_admin(admin_pages.gaps), f"Knowledge Gaps ({n_gaps})", "🕳️", "gaps"),
        _page(_admin(admin_pages.learned), "Learned Answers", "✅", "learned"),
        _page(_admin(admin_pages.settings), "Settings", "⚙️", "settings")]
st.navigation(sections, position="top").run()
