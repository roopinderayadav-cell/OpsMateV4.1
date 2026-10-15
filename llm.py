"""
llm.py - One connection point for the AI models OpsMate can use.

  Gemini (Google)   - free tier available; also provides EMBEDDINGS (semantic search)
  Claude (Anthropic) - paid
  none              - OpsMate still works offline (word matching + spelling fix)

Every function here fails safely: if the AI is unreachable, out of quota or the
key is wrong, it raises AIError and OpsMate falls back to its offline answer.
"""
import base64
import io
import time

import numpy as np

import config


class AIError(Exception):
    pass


# ===========================================================================
# WHICH PROVIDER?
# ===========================================================================

def provider():
    p = (config.AI_PROVIDER or "auto").lower()
    if p == "off":
        return None
    if p in ("auto", "gemini") and config.GEMINI_API_KEY:
        return "gemini"
    if p in ("auto", "claude") and config.ANTHROPIC_API_KEY:
        return "claude"
    return None


def ai_on():
    return provider() is not None


def embeddings_on():
    """Semantic search needs a Gemini key (Claude has no embedding model)."""
    return (config.AI_PROVIDER or "auto").lower() != "off" and bool(config.GEMINI_API_KEY)


def label():
    p = provider()
    if p == "gemini":
        return f"Gemini ({config.GEMINI_MODEL})"
    if p == "claude":
        return f"Claude ({config.CLAUDE_MODEL})"
    return "Off"


# ===========================================================================
# CLIENTS
# ===========================================================================
_gemini = None
_claude = None


def _gemini_client():
    global _gemini
    if _gemini is None:
        try:
            from google import genai
        except ImportError:
            raise AIError("Gemini library missing - run: pip install -r requirements.txt")
        _gemini = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini


def _types():
    try:
        from google.genai import types
        return types
    except ImportError:
        raise AIError("Gemini library missing - run: pip install -r requirements.txt")


def _claude_client():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=60)
    return _claude


def _png(img_bytes):
    """Resize any image to a sensible size and return PNG bytes."""
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((1568, 1568))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# After a quota / rate-limit error the AI is skipped for a few minutes, so staff get the
# document answer at once instead of waiting for the AI to fail again on every question.
_paused_until = {"generate": 0.0, "embed": 0.0}
QUOTA_TEXT = "The AI's free quota is used up for now - answering from documents."


def paused(kind="generate"):
    return time.time() < _paused_until.get(kind, 0.0)


def _check_pause(kind):
    if paused(kind):
        raise AIError(QUOTA_TEXT)


def _note_failure(kind, err):
    if err.args and err.args[0] == QUOTA_TEXT:
        _paused_until[kind] = time.time() + getattr(config, "AI_PAUSE_SECONDS", 300)


def _retry(fn):
    """Try twice: free tiers sometimes answer 'busy / rate limit' for a moment."""
    last = None
    for attempt in range(2):
        try:
            return fn()
        except Exception as e:                          # noqa: BLE001
            last = e
            msg = str(e).lower()
            if attempt == 0 and any(k in msg for k in ("429", "503", "rate", "overloaded",
                                                       "unavailable", "resource_exhausted")):
                time.sleep(2)
                continue
            break
    raise AIError(_friendly(last))


def _friendly(e):
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "401" in low or "permission" in low or "invalid" in low and "key" in low:
        return "The AI key was rejected - check the key in Secrets."
    if "429" in low or "quota" in low or "resource_exhausted" in low or "rate" in low:
        return QUOTA_TEXT
    return f"AI unavailable ({msg[:120]})"


# ===========================================================================
# GENERATE TEXT (optionally with images / audio)
# ===========================================================================

def generate(prompt, images=None, audio=None, fast=False, json_mode=False, max_tokens=1200):
    p = provider()
    if p is None:
        raise AIError("AI is off")
    _check_pause("generate")
    try:
        return _generate(p, prompt, images, audio, fast, json_mode, max_tokens)
    except AIError as e:
        _note_failure("generate", e)
        raise


def _generate(p, prompt, images, audio, fast, json_mode, max_tokens):
    images = [_png(b) for b in (images or []) if b]

    if p == "gemini":
        types = _types()
        model = config.GEMINI_FAST_MODEL if fast else config.GEMINI_MODEL
        parts = [types.Part.from_bytes(data=b, mime_type="image/png") for b in images]
        if audio:
            parts.append(types.Part.from_bytes(data=audio[0], mime_type=audio[1]))
        parts.append(prompt)

        def call(thinking=True):
            cfg = dict(temperature=0.1, max_output_tokens=max_tokens + 1024)
            if json_mode:
                cfg["response_mime_type"] = "application/json"
            if thinking:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
            r = _gemini_client().models.generate_content(
                model=model, contents=parts, config=types.GenerateContentConfig(**cfg))
            return (r.text or "").strip()

        def run():
            try:
                return call(True)
            except Exception as e:                      # model without thinking options
                if "thinking" in str(e).lower():
                    return call(False)
                raise
        return _retry(run)

    # Claude
    model = config.CLAUDE_FAST_MODEL if fast else config.CLAUDE_MODEL
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(b).decode()}} for b in images]
    content.append({"type": "text", "text": prompt})

    def run_claude():
        msg = _claude_client().messages.create(model=model, max_tokens=max_tokens,
                                               messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    return _retry(run_claude)


# ===========================================================================
# EMBEDDINGS (Gemini) - turn text into meaning-vectors for semantic search
# ===========================================================================

def embed(texts, query=False):
    """Returns an (n, EMBED_DIM) float32 array of unit vectors, or raises AIError."""
    if not embeddings_on():
        raise AIError("Semantic search needs a Gemini key")
    _check_pause("embed")
    try:
        return _embed(texts, query)
    except AIError as e:
        _note_failure("embed", e)
        raise


def _embed(texts, query=False):
    types = _types()
    out = []
    for i in range(0, len(texts), 50):
        batch = [t[:6000] for t in texts[i:i + 50]]

        def run(batch=batch):
            r = _gemini_client().models.embed_content(
                model=config.GEMINI_EMBED_MODEL, contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                    output_dimensionality=config.EMBED_DIM))
            return [e.values for e in r.embeddings]
        out += _retry(run)
    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


# ===========================================================================
# VOICE (Gemini) and CONNECTION TEST
# ===========================================================================

def transcribe(audio_bytes, mime="audio/wav"):
    import prompts
    if provider() != "gemini":
        raise AIError("Voice questions need a Gemini key")
    return generate(prompts.TRANSCRIBE_PROMPT, audio=(audio_bytes, mime), fast=True, max_tokens=300)


def test_connection():
    """Used by the Settings page: returns (ok, message)."""
    if not ai_on():
        return False, "No AI key set - OpsMate is running in free offline mode."
    try:
        reply = generate("Reply with the single word: ready", fast=True, max_tokens=20)
        msg = f"✅ {label()} answered: {reply[:40]}"
        if embeddings_on():
            embed(["test"], query=True)
            msg += " · semantic search ✅"
        return True, msg
    except AIError as e:
        return False, f"❌ {e}"
