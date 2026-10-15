"""
admin_pages.py - OpsMate admin screens: Dashboard & Analytics, Document Management,
Review Queue, Knowledge Gaps, Learned Answers, Announcements, Settings.
"""
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import config
import knowledge as kb
import llm
import ui

PERIODS = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}


def route(name, region, user):
    pages = {"📊 Dashboard": dashboard, "🗂️ Document Management": documents,
             "🧠 Review Queue": review, "🕳️ Knowledge Gaps": gaps,
             "✅ Learned Answers": learned, "⚙️ Settings": settings}
    fn = pages.get(name)
    if fn:
        fn(region, user)


def _clean_q(q):
    return re.sub(r"\s*\[screenshot\]$", "", str(q)).strip()


# ---------------------------------------------------------------------------
# DASHBOARD & ANALYTICS
# ---------------------------------------------------------------------------
def dashboard(region, user):
    st.header("Dashboard & Analytics")
    c1, c2, c5, c3, c4 = st.columns([1.1, 1.1, 1.2, 1.3, 1.3])
    period = c1.selectbox("Period", list(PERIODS), index=1)
    reg = c2.selectbox("LOB", ["All"] + config.REGIONS)
    lob = c5.selectbox("Sub-LOB", ["All"] + (kb.get_lobs().get(reg, []) if reg != "All" else []), key=f"dsub_{reg}")
    topic = c3.selectbox("Topic", ["All"] + list(config.TOPICS) + ["Other"])
    df = kb.questions_df(reg, PERIODS[period], topic)
    if lob != "All" and not df.empty:
        df = df[df["sub_lob"].fillna("") == lob]
    since = None
    if PERIODS[period]:
        since = (pd.Timestamp.now() - pd.Timedelta(days=PERIODS[period])).strftime("%Y-%m-%d %H:%M")
    gap_list = kb.knowledge_gaps(None if reg == "All" else reg, since)
    c4.write("")
    c4.download_button("⬇ Export to Excel", kb.export_excel(df, gap_list), type="primary",
                       file_name=f"OpsMate_questions_{datetime.now():%Y-%m-%d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")
    if df.empty:
        st.info("No questions in this period yet.")
        return

    total = len(df)
    answered = int((df["result"] == "answered").sum())
    rated = int(df["feedback"].notna().sum())
    helpful = int((df["feedback"] == 1).sum())
    unhelpful = int((df["feedback"] == -1).sum())
    unanswered = int(df["result"].isin(["not_found", "conflict"]).sum())
    k = st.columns(6)
    ui.kpi(k[0], "💬", f"{total:,}", "Total questions")
    ui.kpi(k[1], "✅", f"{100 * answered / total:.0f}%", "Answered")
    ui.kpi(k[2], "👍", f"{100 * helpful / rated:.0f}%" if rated else "–", "Helpful", f"{rated} rated")
    ui.kpi(k[3], "👎", f"{100 * unhelpful / rated:.0f}%" if rated else "–", "Not helpful")
    ui.kpi(k[4], "🕳️", f"{unanswered:,}", "Unanswered / conflict",
           f"{int((df['result'] == 'conflict').sum())} conflicts")
    pcts = pd.to_numeric(df["confidence_pct"], errors="coerce").dropna() if "confidence_pct" in df else pd.Series(dtype=float)
    ui.kpi(k[5], "🎯", f"{pcts.mean():.0f}%" if len(pcts) else "–", "Avg confidence", "of answered questions")
    st.write("")

    colors = alt.Scale(domain=list(ui.REGION_COLORS), range=list(ui.REGION_COLORS.values()))
    a, b, c = st.columns([1, 1.2, 1.5])
    with a:
        st.markdown("<div class='om-section'>Questions by region</div>", unsafe_allow_html=True)
        by_reg = df.groupby("region").size().reset_index(name="questions")
        st.altair_chart(alt.Chart(by_reg).mark_arc(innerRadius=55).encode(
            theta="questions:Q", color=alt.Color("region:N", scale=colors, legend=alt.Legend(orient="bottom")),
            tooltip=["region", "questions"]).properties(height=240), width="stretch")
    with b:
        st.markdown("<div class='om-section'>Top topics</div>", unsafe_allow_html=True)
        by_topic = df.groupby("topic").size().reset_index(name="questions").sort_values("questions", ascending=False)
        st.altair_chart(alt.Chart(by_topic.head(8)).mark_bar(color=ui.BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("questions:Q", title=None), y=alt.Y("topic:N", sort="-x", title=None),
            tooltip=["topic", "questions"]).properties(height=240), width="stretch")
    with c:
        st.markdown("<div class='om-section'>Question trend</div>", unsafe_allow_html=True)
        trend = df.groupby(["date", "region"]).size().reset_index(name="questions")
        st.altair_chart(alt.Chart(trend).mark_line(point=True).encode(
            x=alt.X("date:T", title=None), y=alt.Y("questions:Q", title=None),
            color=alt.Color("region:N", scale=colors, legend=alt.Legend(orient="bottom")),
            tooltip=["date", "region", "questions"]).properties(height=240), width="stretch")

    d, e, f = st.columns([1.4, 1.4, 1])
    with d:
        st.markdown("<div class='om-section'>Most asked questions</div>", unsafe_allow_html=True)
        q = df.assign(q=df["question"].map(_clean_q).str.lower().str.strip(" ?."))
        top = q.groupby("q").size().reset_index(name="times").sort_values("times", ascending=False).head(8)
        st.dataframe(top.rename(columns={"q": "Question", "times": "Times"}), hide_index=True,
                     width="stretch")
    with e:
        st.markdown("<div class='om-section'>Unanswered / needs review</div>", unsafe_allow_html=True)
        g = pd.DataFrame([{"Question": x["question"], "Type": x["type"], "Times": x["count"]} for x in gap_list[:8]])
        if g.empty:
            st.success("No open gaps 🎉")
        else:
            st.dataframe(g, hide_index=True, width="stretch")
    with f:
        st.markdown("<div class='om-section'>Feedback summary</div>", unsafe_allow_html=True)
        fb = pd.DataFrame({"feedback": ["Helpful", "Not helpful"], "votes": [helpful, unhelpful]})
        if rated:
            st.altair_chart(alt.Chart(fb).mark_arc(innerRadius=50).encode(
                theta="votes:Q", color=alt.Color("feedback:N", scale=alt.Scale(
                    domain=["Helpful", "Not helpful"], range=["#16a34a", "#dc2626"]),
                    legend=alt.Legend(orient="bottom")), tooltip=["feedback", "votes"]).properties(height=200),
                width="stretch")
        else:
            st.caption("No ratings yet.")

    with st.expander("All questions in this period"):
        show = df[["asked_at", "region", "sub_lob", "asked_by", "topic", "question", "result", "confidence",
                   "confidence_pct", "helpful"]]
        st.dataframe(show, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# DOCUMENT MANAGEMENT
# ---------------------------------------------------------------------------
def documents(region, user):
    st.header("Document Management")
    lobs = kb.get_lobs()
    with st.expander("⬆ Upload document", expanded=False):
        l1, l2 = st.columns(2)
        up_region = l1.selectbox("LOB", config.REGIONS + [config.GLOBAL_REGION], key="up_region",
                                 index=config.REGIONS.index(region) if region in config.REGIONS else 0,
                                 format_func=lambda r: config.REGION_LABELS.get(r, "🌐 GLOBAL (every LOB)"),
                                 help="GLOBAL = visible in every LOB")
        up_sub = l2.selectbox("Sub-LOB", ["General (whole LOB)"] + lobs.get(up_region, []), key=f"up_sub_{up_region}",
                              disabled=up_region == config.GLOBAL_REGION,
                              help="Answers for this sub-LOB rank this document first.")
        up_sub = "" if up_sub.startswith("General") else up_sub
        with st.form("upload", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            doc_type = c2.selectbox("Document type", config.DOC_TYPES)
            version = c3.text_input("Version", placeholder="e.g. 2.1")
            title = c1.text_input("Title (optional)")
            owner = c2.text_input("Owner (optional)")
            files = st.file_uploader("Files", type=config.ALLOWED_EXTENSIONS, accept_multiple_files=True)
            if st.form_submit_button("Upload & index", type="primary") and files:
                for f in files:
                    try:
                        with st.spinner(f"Reading {f.name}…"):
                            _, n, info = kb.add_document(up_region, doc_type, title if len(files) == 1 else "",
                                                         f.name, f.getvalue(), user, version, owner,
                                                         sub_lob=up_sub)
                        msg = f"✔ {f.name} → {up_region} · {up_sub or 'General'}: {n} knowledge pieces"
                        if info["images_read"]:
                            msg += f", {info['images_read']} image(s) read by AI"
                        if info.get("images_stored"):
                            msg += f", {info['images_stored']} picture(s) kept to show with answers"
                        if info["semantic"] == "on":
                            msg += ", semantic search ✔"
                        st.success(msg)
                        if info["images_skipped"]:
                            st.warning(f"{info['images_skipped']} image(s) in {f.name} were not read - they are "
                                       "shown with answers, but text inside them is searchable only with a "
                                       "Gemini key (add it and upload again).")
                        if info["gaps_rechecked"]:
                            st.info(f"🧠 {info['gaps_rechecked']} open knowledge gap(s) may now be answered "
                                    "by this document - see Knowledge Gaps.")
                    except Exception as e:
                        st.error(f"✖ {f.name}: {e}")
        st.caption("Tip: use the OpsMate SOP / FAQ templates - clear headings, numbered steps, real tables "
                   "and 'Q:' / 'A:' FAQs give the most accurate answers.")

    c1, c0, c2, c3, c4 = st.columns([1, 1.2, 1, 1, 2])
    r = c1.selectbox("LOB", ["All"] + config.REGIONS + [config.GLOBAL_REGION])
    lf = c0.selectbox("Sub-LOB", ["All"] + (lobs.get(r, []) if r in lobs else []) + ["General (no sub-LOB)"],
                      key=f"lf_{r}")
    t = c2.selectbox("Type", ["All"] + config.DOC_TYPES)
    s = c3.selectbox("Status", ["All", "Active", "Archived"])
    q = c4.text_input("Search documents…")
    docs = kb.list_documents()
    docs = [d for d in docs if (r == "All" or d["region"] == r) and (t == "All" or d["doc_type"] == t)
            and (lf == "All" or (d.get("sub_lob") or "") == ("" if lf.startswith("General") else lf))
            and (s == "All" or (d.get("status") or "Active") == s)
            and (not q or q.lower() in (d["title"] + d["filename"]).lower())]
    untagged = sum(1 for d in kb.list_documents() if not d.get("sub_lob") and d["region"] != config.GLOBAL_REGION)
    if untagged:
        st.caption(f"ℹ️ {untagged} document(s) have no sub-LOB (General). They are used for the whole LOB; "
                   "tag them with ✏️ so the right sub-LOB's documents rank first.")
    h = st.columns([3.2, 1, 1.1, 0.8, 1.3, 1, 2.2])
    for col, label in zip(h, ["File name", "LOB · sub-LOB", "Type", "Version", "Updated on", "Status", "Actions"]):
        col.markdown(f"**{label}**")
    for d in docs:
        status = d.get("status") or "Active"
        c = st.columns([3.2, 1, 1.1, 0.8, 1.3, 1, 2.2])
        icon = {"pdf": "📕", "docx": "📘", "xlsx": "📗", "pptx": "📙"}.get(d["filename"].rsplit(".", 1)[-1], "📄")
        c[0].markdown(f"{icon} {d['title']}  \n<span class='om-meta'>{d['filename']} · {d['chunks']} piece{'s' if d['chunks'] != 1 else ''}</span>",
                      unsafe_allow_html=True)
        lob_txt = d.get("sub_lob") or "General"
        c[1].markdown(ui.region_pill(d["region"]) + f"<br><span class='om-meta'>{ui.esc(lob_txt)}</span>",
                      unsafe_allow_html=True)
        c[2].write(d["doc_type"])
        c[3].write(d["version"] or "–")
        c[4].write(d["uploaded_at"][:10])
        c[5].markdown(ui.result_badge("answered").replace("Answered", "Active") if status == "Active"
                      else "<span class='om-badge' style='background:#eef0f4;color:#5b6475'>Archived</span>",
                      unsafe_allow_html=True)
        a1, a2, a3, a4 = c[6].columns(4)
        with a4.popover("✏️", help="Change sub-LOB"):
            subs = ["General (whole LOB)"] + lobs.get(d["region"], [])
            cs = d.get("sub_lob") or ""
            ns = st.selectbox(f"Sub-LOB in {d['region']}", subs, index=subs.index(cs) if cs in subs else 0,
                              key=f"ts{d['id']}", disabled=d["region"] == config.GLOBAL_REGION)
            if st.button("Save", key=f"tsave{d['id']}", type="primary"):
                kb.update_document_tags(d["id"], "" if ns.startswith("General") else ns)
                st.rerun()
        a1.download_button("⬇", kb.read_file(d), file_name=d["filename"], key=f"d{d['id']}", help="Download")
        if a2.button("🗄" if status == "Active" else "♻", key=f"s{d['id']}",
                     help="Archive (stop using in answers)" if status == "Active" else "Make active again"):
            kb.set_document_status(d["id"], "Archived" if status == "Active" else "Active")
            st.rerun()
        if a3.button("🗑", key=f"x{d['id']}", help="Delete"):
            kb.delete_document(d["id"])
            st.rerun()
    if not docs:
        st.info("No documents match.")


# ---------------------------------------------------------------------------
# REVIEW QUEUE
# ---------------------------------------------------------------------------
def review(region, user):
    st.header("Review Queue")
    st.caption("Questions OpsMate could not answer, found conflicting answers for, or staff marked 👎. "
               "Write the correct answer once - it is used for similar questions from then on.")
    scope = st.radio("Show", ["This region", "All regions"], horizontal=True)
    queue = kb.review_queue(region if scope == "This region" else None)
    if not queue:
        st.success("Nothing waiting. 🎉")
    limit = st.session_state.setdefault("rq_limit", 10)
    st.caption(f"Showing {min(limit, len(queue))} of {len(queue)} - newest first. Tip: Knowledge Gaps groups "
               "repeated questions so you can fix many at once.")
    for q in queue[:limit]:
        with st.container(border=True):
            result = q.get("result") or "not_found"
            st.markdown(f"{ui.region_pill(q['region'])} {ui.result_badge(result)} &nbsp; **{ui.esc(q['question'])}**",
                        unsafe_allow_html=True)
            st.caption(f"Asked by {q['asked_by']} · {q['asked_at']} · topic {q.get('topic') or '–'}"
                       + (" · 👎 rated not helpful" if q["feedback"] == -1 else ""))
            if q["comment"]:
                st.markdown(f"> Staff note: {q['comment']}")
            if q.get("image_path"):
                from pathlib import Path
                if Path(q["image_path"]).exists():
                    st.image(q["image_path"], caption="Screenshot the agent pasted", width=360)
            with st.expander("What OpsMate answered"):
                st.markdown(q["answer"] or "")
            with st.form(f"teach{q['id']}"):
                new_q = st.text_input("Question (edit to make it general)", _clean_q(q["question"]))
                ans = st.text_area("Correct answer", height=120,
                                   placeholder="Write the steps / rule exactly as staff should follow them")
                c1, c2, c3 = st.columns(3)
                if c1.form_submit_button("✅ Save & teach", type="primary"):
                    if ans.strip():
                        kb.teach(q["region"], new_q, ans.strip(), user, q["id"])
                        st.rerun()
                    else:
                        st.warning("Please write an answer.")
                if c2.form_submit_button("📝 SOP update needed"):
                    kb.set_gap([q["id"]], status="SOP update needed")
                    st.rerun()
                if c3.form_submit_button("Dismiss"):
                    kb.dismiss(q["id"])
                    st.rerun()
    if len(queue) > limit and st.button("Show 10 more"):
        st.session_state.rq_limit = limit + 10
        st.rerun()


# ---------------------------------------------------------------------------
# KNOWLEDGE GAPS
# ---------------------------------------------------------------------------
def gaps(region, user):
    st.header("Knowledge Gaps")
    st.caption("Questions OpsMate could not answer, where documents disagree, or that staff marked 👎 - "
               "grouped by similar wording, most-asked first. Fix these to improve the process.")
    c1, c2, c3, c4 = st.columns([1, 1.6, 1.2, 1.3])
    reg = c1.selectbox("Region", ["All"] + config.REGIONS)
    statuses = c2.multiselect("Status", kb.GAP_STATUSES,
                              default=["Open", "SOP update needed", "In progress", "Possibly answered"])
    kind = c3.selectbox("Type", ["All", "No answer", "Conflict", "Not helpful"])
    groups = kb.knowledge_gaps(None if reg == "All" else reg, include_closed=True)
    c4.write("")
    c4.download_button("⬇ Export gaps to Excel", kb.export_excel(kb.questions_df(None if reg == "All" else reg),
                                                                [g for g in groups if g["status"] in statuses]),
                       file_name=f"OpsMate_gaps_{datetime.now():%Y-%m-%d}.xlsx", width="stretch")
    shown = [g for g in groups if g["status"] in statuses and (kind == "All" or g["type"] == kind)]
    k = st.columns(4)
    ui.kpi(k[0], "🕳️", len([g for g in groups if g["status"] not in ("Fixed", "Closed")]), "Open gaps")
    ui.kpi(k[1], "🔁", sum(g["count"] for g in shown), "Times asked (shown)")
    ui.kpi(k[2], "⚠️", len([g for g in shown if g["type"] == "Conflict"]), "Conflicts")
    ui.kpi(k[3], "🧠", len([g for g in groups if g["status"] == "Possibly answered"]), "Possibly answered")
    st.write("")
    if not shown:
        st.success("No gaps match. 🎉")
    glimit = st.session_state.setdefault("gap_limit", 10)
    for n, g in enumerate(shown[:glimit]):
        with st.container(border=True):
            badge = {"Conflict": "conflict", "No answer": "not_found", "Not helpful": "not_found"}[g["type"]]
            st.markdown(f"{ui.result_badge(badge).replace(ui.RESULT_BADGE[badge][0], g['type'])} "
                        + " ".join(ui.region_pill(r) for r in g["regions"])
                        + f" &nbsp; **{ui.esc(g['question'])}**", unsafe_allow_html=True)
            st.caption(f"Asked {g['count']}× · topic {g['topic']} · first {g['first']} · last {g['last']} · "
                       f"by {', '.join(g['users'][:4])}")
            if len(g["examples"]) > 1:
                st.markdown("Also asked as: " + " · ".join(f"*{ui.esc(x)}*" for x in g["examples"][1:]))
            if g["note"]:
                st.info(g["note"])
            c1, c2, c3, c4 = st.columns([1.3, 1.2, 2.2, 0.8])
            status = c1.selectbox("Status", kb.GAP_STATUSES, index=kb.GAP_STATUSES.index(g["status"])
                                  if g["status"] in kb.GAP_STATUSES else 0, key=f"gs{n}")
            owner = c2.text_input("Owner", g["owner"], key=f"go{n}")
            note = c3.text_input("Note / action", g["note"], key=f"gn{n}")
            c4.write("")
            if c4.button("Save", key=f"gsv{n}"):
                kb.set_gap(g["ids"], status, owner, note)
                st.rerun()
            with st.expander("✍ Write the answer now"):
                with st.form(f"gt{n}"):
                    qtext = st.text_input("Question", g["question"])
                    ans = st.text_area("Correct answer", height=100)
                    if st.form_submit_button("✅ Save & teach (closes this gap)", type="primary") and ans.strip():
                        for r in g["regions"]:
                            kb.teach(r, qtext, ans.strip(), user)
                        kb.set_gap(g["ids"], "Fixed", owner or user, "Answered by process team")
                        st.rerun()
    if len(shown) > glimit and st.button("Show 10 more gaps"):
        st.session_state.gap_limit = glimit + 10
        st.rerun()


# ---------------------------------------------------------------------------
# LEARNED ANSWERS
# ---------------------------------------------------------------------------
def learned(region, user):
    st.header("Learned Answers")
    st.caption("Verified answers come from the process team. Auto-learned answers are saved by OpsMate "
               f"when {config.AUTO_LEARN_MIN_VOTES} different people (or one admin) mark the same answer 👍; "
               "a 👎 switches an auto-learned answer off automatically.")
    t1, t2, t3 = st.tabs(["✅ Verified", "🧠 Auto-learned (staff 👍)", "➕ Add a Q&A"])
    with t1:
        rows = kb.list_learned(region, "verified") + kb.list_learned(config.GLOBAL_REGION, "verified")
        if not rows:
            st.info("No verified answers yet for this region.")
        for l in rows:
            with st.container(border=True):
                a, b = st.columns([6, 1])
                a.markdown(ui.region_pill(l["region"]) + f" **Q:** {ui.esc(l['question'])}", unsafe_allow_html=True)
                a.markdown(f"**A:** {l['answer']}")
                a.caption(f"by {l['added_by']} · {l['added_at']}")
                if b.button("Delete", key=f"dv{l['id']}"):
                    kb.set_learned(l["id"], delete=True)
                    st.rerun()
    with t2:
        rows = kb.list_learned(region, "auto")
        if not rows:
            st.info("Nothing auto-learned yet. Answers staff mark 👍 appear here.")
        for l in rows:
            with st.container(border=True):
                a, b = st.columns([6, 1.4])
                state = "🟢 in use" if l["active"] else "⚪ switched off"
                a.markdown(f"{state} · 👍 {l.get('votes') or 1} · **Q:** {ui.esc(l['question'])}")
                a.markdown(l["answer"])
                if b.button("Make verified", key=f"mv{l['id']}"):
                    kb.set_learned(l["id"], make_verified=True, admin=user)
                    st.rerun()
                if b.button("Switch off" if l["active"] else "Switch on", key=f"ta{l['id']}"):
                    kb.set_learned(l["id"], active=not l["active"])
                    st.rerun()
                if b.button("Delete", key=f"da{l['id']}"):
                    kb.set_learned(l["id"], delete=True)
                    st.rerun()
    with t3:
        with st.form("add_qa", clear_on_submit=True):
            r = st.selectbox("Region", config.REGIONS + [config.GLOBAL_REGION], index=config.REGIONS.index(region))
            q = st.text_input("Question")
            a = st.text_area("Answer")
            if st.form_submit_button("Save", type="primary") and q and a:
                kb.teach(r, q, a, user)
                st.success("Saved.")


# ---------------------------------------------------------------------------
# ANNOUNCEMENTS (admin form, shown on the Announcements page)
# ---------------------------------------------------------------------------
def announcements_admin(user):
    with st.expander("➕ Post an update (shown under 'Important Updates' on the home screen)"):
        with st.form("ann", clear_on_submit=True):
            c1, c2 = st.columns(2)
            title = c1.text_input("Title", placeholder="New SOP: Refund process updated (NORAM)")
            level = c2.selectbox("Type", ["New SOP", "Alert", "Maintenance", "Info"])
            body = st.text_area("Message")
            reg = st.selectbox("Show to", ["All"] + config.REGIONS)
            if st.form_submit_button("Post", type="primary") and title:
                kb.add_announcement(title, body, level, reg, user)
                st.rerun()


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
def settings(region, user):
    st.header("Settings")
    if config.password_warning():
        st.error(config.password_warning(), icon="🔐")
    src = {"secrets": "✔ from Secrets", "environment": "✔ from environment", "default": "✖ not set"}
    st.caption("Secrets check (values are never shown): "
               + ("Secrets read OK" if config.SECRETS_STATUS["state"] == "ok" else
                  "⚠ Secrets could not be read" if config.SECRETS_STATUS["state"] == "error" else "no Secrets saved")
               + f" · ADMIN_PASSWORD {src[config.SECRET_SOURCE.get('ADMIN_PASSWORD', 'default')]}"
               + f" · GEMINI_API_KEY {src[config.SECRET_SOURCE.get('GEMINI_API_KEY', 'default')]}")
    lob_settings(user)
    backup_settings(user)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='om-section'>AI connection</div>", unsafe_allow_html=True)
        st.write(f"Provider: **{llm.label()}**")
        st.write("Semantic search: **" + ("On (Gemini embeddings)" if llm.embeddings_on() else "Off") + "**")
        if st.button("Test AI connection"):
            ok, msg = llm.test_connection()
            (st.success if ok else st.warning)(msg)
        done, total = kb.semantic_coverage()
        st.write(f"Semantic index: **{done} of {total}** knowledge pieces")
        if llm.embeddings_on() and done < total and st.button("Build semantic index now", type="primary"):
            bar = st.progress(0.0)
            try:
                n = kb.reindex_embeddings(bar.progress)
                st.success(f"Indexed {n} items.")
            except llm.AIError as e:
                st.error(str(e))
        if config.GEMINI_API_KEY:
            st.warning("Gemini free tier: Google may use the content you send to improve its products. "
                       "Use sample or non-confidential SOPs on the free tier; for real company SOPs use a "
                       "paid key or company-approved AI.")
        st.caption("Add keys in Streamlit → App → Settings → Secrets:  GEMINI_API_KEY = \"...\"  "
                   "(or ANTHROPIC_API_KEY). On a laptop:  setx GEMINI_API_KEY \"...\"")
    with c2:
        st.markdown("<div class='om-section'>Standard replies</div>", unsafe_allow_html=True)
        st.write("**When no answer is found:**")
        st.info(config.FALLBACK_TEXT)
        st.write("**When documents disagree:**")
        st.warning(config.CONFLICT_TEXT)
        st.caption("Change these wordings in config.py (FALLBACK_TEXT, CONFLICT_TEXT).")
        st.markdown("<div class='om-section'>Self-learning</div>", unsafe_allow_html=True)
        st.write(f"Auto-learn from 👍: **{'On' if config.AUTO_LEARN else 'Off'}** - after 👍 from "
                 f"{config.AUTO_LEARN_MIN_VOTES} different people, or 1 admin")
        st.write("👍 / 👎 also move passages up / down for similar questions; new uploads re-check open gaps.")


def lob_settings(user):
    st.markdown("<div class='om-section'>Lines of business (tabs) and sub-LOBs</div>", unsafe_allow_html=True)
    st.caption("Each LOB is a tab above the chat; its sub-LOBs are the chips under it. Questions search the "
               "selected LOB first (sub-LOB documents ranked highest), then the other LOBs if needed. "
               "Separate sub-LOBs with commas.")
    lobs = kb.get_lobs()
    base = pd.DataFrame([{"LOB": k, "Sub-LOBs": ", ".join(v)} for k, v in lobs.items()])
    with st.form("lobs"):
        edited = st.data_editor(base, num_rows="fixed", hide_index=True, width="stretch", disabled=["LOB"],
                                column_config={"Sub-LOBs": st.column_config.TextColumn(width="large")})
        if st.form_submit_button("Save sub-LOBs", type="primary"):
            try:
                kb.save_lobs({r["LOB"]: [x.strip() for x in str(r["Sub-LOBs"] or "").split(",")]
                              for _, r in edited.iterrows()})
                st.success("Saved. The chips update immediately.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.caption("Adding a new LOB tab (besides " + ", ".join(config.REGIONS) + ") is a one-line change in "
               "config.py (REGIONS).")


def backup_settings(user):
    st.markdown("<div class='om-section'>Backup & restore</div>", unsafe_allow_html=True)
    st.caption("Free Streamlit hosting can erase uploaded documents and learned answers when the app restarts. "
               "Download a backup after uploading documents, and restore it if the library is ever empty.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Prepare backup", key="mkbackup"):
            st.session_state["backup_bytes"] = kb.make_backup()
        if st.session_state.get("backup_bytes"):
            st.download_button("⬇ Download backup (.zip)", st.session_state["backup_bytes"], type="primary",
                               file_name=f"OpsMate_backup_{datetime.now():%Y-%m-%d_%H%M}.zip", mime="application/zip")
    with c2:
        up = st.file_uploader("Restore from a backup .zip", type=["zip"], key="restore_zip")
        if up is not None and st.button("Restore now (replaces current library)", key="dorestore"):
            try:
                n = kb.restore_backup(up.getvalue())
                st.success(f"Restored {n} documents with their questions, learned answers and settings.")
            except ValueError as e:
                st.error(str(e))
