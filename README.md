# OpsMate - Your AI Operations Companion (NORAM · EUROPE · APAC)

Agents ask questions in plain words (typos and Hindi/Kannada/Telugu in English letters
are fine) or paste a screenshot, and get one short answer with its SOP source and a
confidence level. When OpsMate is not sure it says:
"I do not have answer to this question, please consult with your team supervisor."

## What's new in OpsMate (v3, upgraded from Compass)
- New look based on the OpsMate mock-up: home screen, topic buttons, Recent Questions,
  Important Updates, answer card with confidence, sources with View.
- More accurate: spelling + Hinglish fix, coverage check, re-ranking, optional Gemini
  semantic search. Tested: 26/26 real questions right, 10/10 out-of-scope questions refused.
- Never guesses: low confidence -> the supervisor message (config.FALLBACK_TEXT).
- Conflict check: if two documents give different answers -> shows both + CONFLICT_TEXT.
- Learns by itself: 👍 from 2 people (or 1 admin) saves the answer; 👍/👎 move passages
  up/down for similar questions; 👎 switches off a wrong learned answer; new uploads
  re-check open gaps.
- Knowledge Gaps page: unanswered / conflicting / 👎 questions grouped, with status,
  owner, note, "write the answer now", Excel export.
- Dashboard: KPIs, by region, top topics, trend, most asked, unanswered, feedback,
  Export to Excel (5 sheets).
- Document Management (archive/activate), Announcements, Quick Guides, Settings.
- AI: Gemini (free tier available) or Claude - optional. Voice questions with Gemini.

## Files
| File | What it does |
|---|---|
| `app.py` | Staff screens + menu |
| `admin_pages.py` | Dashboard, Document Management, Review Queue, Knowledge Gaps, Learned Answers, Settings |
| `ui.py` | Colours, header bar, cards |
| `knowledge.py` | Documents, search, 3-layer answering, self-learning, gaps, Excel export |
| `brain.py` | Spelling/Hinglish, topics, confidence, conflict check, short answers |
| `llm.py` | Connection to Gemini or Claude (optional) |
| `prompts.py` | Instructions given to the AI - edit to change answer style |
| `config.py` | All settings: name, standard replies, regions, topics, thresholds, AI keys |
| `.streamlit/config.toml` | Colours (blue theme) |
| `sample_docs/` | Demo documents loaded automatically |
| `demo_files/` | A document that deliberately conflicts with the FAQ - upload it to EUROPE to demo the conflict warning |

## Run on a laptop
Double-click `START_APP.bat` (installs libraries the first time), or:
`python -m pip install -r requirements.txt` then `python -m streamlit run app.py`

## Turn on AI (optional)
Streamlit Cloud → App → Settings → Secrets:
```
ADMIN_PASSWORD = "your-admin-password"
GEMINI_API_KEY = "your-gemini-key"
```
Laptop: `setx GEMINI_API_KEY "your-gemini-key"` once, then restart.
Then Settings → "Test AI connection" and "Build semantic index now".
Gemini free tier: Google may use what you send to improve its products - use sample /
non-confidential SOPs on the free tier.

## What's new in v4
- Menu bar across the top of every page (OpsMate / Admin) and an always-visible **Sign out** button and
  region selector - nothing depends on the side panel any more.
- **LOB tabs** above the chat: NORAM · EUROPE · APAC · QBAY, each with its sub-LOB chips
  (e.g. NORAM: OBT Cancellation, Schedule Change, Ticketing, Refund, Profile Management, Others).
  Questions search the selected LOB (plus GLOBAL documents); the chosen sub-LOB's documents rank first.
  If the LOB has no answer, OpsMate searches the other LOBs and says clearly which LOB the answer came from.
  Admins edit sub-LOBs in Settings and tag documents on upload or with ✏️ in Document Management.
- **Confidence %** on every answer (High 80-98 %, Medium 60-79 %), based on match strength, how much of the
  question is covered, and how many documents agree. Hover the ⓘ next to it for the explanation.
- How-to answers come as numbered steps in the order of the SOP.
- **Backup & restore** in Settings (free hosting can erase uploads on restart).
- Warning while the demo admin password is still in use - set ADMIN_PASSWORD in Secrets.

## What's new in v4.1 - pictures, GDS screens and "which one?" questions
- **Pictures in answers**: pictures inside uploaded Word / PDF / PowerPoint files (e.g. a sample Amadeus screen)
  are kept and shown under the answer that comes from the same section. Works without the AI key.
  Documents uploaded before v4.1 get their pictures automatically the first time the app starts.
  (Finding answers *inside* a picture still needs `GEMINI_API_KEY`.)
- **GDS entries look like a terminal**: commands in answers (RTABC123, FXP, TTP/RT, WETR*1, *T …) are shown
  green-on-dark, with a box listing every entry used in the answer.
- **OpsMate asks before guessing**: if the steps differ by GDS (Amadeus / Sabre / Galileo …) and the agent did not
  say which, or the best answers come from different sub-LOBs and no sub-LOB is selected, OpsMate asks with
  buttons, then answers. "Search all" answers without choosing. Settings: `ASK_CLARIFY`, `CLARIFY_GROUPS` in config.py.
