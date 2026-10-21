"""
Audit Compass RAG — a free, no-sign-in-required audit readiness Q&A tool.

Architecture:
  - Retrieval: local embedding model (sentence-transformers, free, runs on
    the server, no API key) finds the most relevant corpus chunks for a
    question, scoped to the selected regulator.
  - Generation: a free-tier hosted LLM (via Groq's API, which offers a
    genuinely free, no-credit-card tier for open models) answers using
    ONLY the retrieved chunks as grounding, with instructions to say
    "not covered in my reference material" rather than guess.
  - Interface: Streamlit, deployed on Streamlit Community Cloud (free,
    no payment method required). Visitors need NO account and NO sign-in
    to use the deployed app.

Run locally with: streamlit run app.py
Deploy by pushing this folder's contents (app.py, build_index.py,
requirements.txt, and the corpus/ folder) to a GitHub repo, then creating
an app at https://share.streamlit.io pointed at app.py. No local Python
setup and no pre-built index.json are needed — the first time the app
starts, it builds the search index itself (on Streamlit's own fast
network), then reuses it on every later restart (cached in memory via
st.cache_resource — see note on cold starts in DEPLOY.md). See DEPLOY.md
for exact steps.
"""
import json
import os

import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

from build_index import REGULATOR_FILES, CORPUS_DIR, parse_chunks

INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.json")

# Free, open model served via Groq's API (genuinely free tier, no credit
# card required). No cost to visitors; the app owner's free Groq account
# funds inference.
GENERATION_MODEL = "llama-3.1-8b-instant"

REGULATORS = [
    ("US FDA", "usfda"),
    ("Australia TGA", "tga"),
    ("EU GMP / EMA", "ema"),
    ("UK MHRA", "mhra"),
    ("South Africa SAHPRA", "sahpra"),
    ("Brazil ANVISA", "anvisa"),
    ("WHO Prequalification", "who"),
    ("PIC/S (general)", "picks"),
    ("Health Canada", "hc"),
]
REGULATOR_BY_LABEL = dict((label, rid) for label, rid in REGULATORS)

TOP_K = 4


# ---------------------------------------------------------------------------
# Cached, expensive resources — st.cache_resource keeps these alive across
# Streamlit reruns (Streamlit re-executes the whole script on every click),
# so the embedding model and index are loaded/built only once per server
# process, not once per question.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading reference material (first visit only)…")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building search index (first visit only)…")
def load_chunks():
    embed_model = load_embed_model()

    if not os.path.exists(INDEX_PATH):
        all_chunks = []
        for reg_id, fname in REGULATOR_FILES.items():
            path = os.path.join(CORPUS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            all_chunks.extend(parse_chunks(text, reg_id))

        texts = [f"{c['topic']}. {c['text']}" for c in all_chunks]
        embeddings = embed_model.encode(texts, normalize_embeddings=True)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb.tolist()

        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f)

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    for c in chunks:
        c["_emb"] = np.array(c["embedding"], dtype=np.float32)
    return chunks


@st.cache_resource
def load_groq_client():
    # GROQ_API_KEY is read from Streamlit secrets (Settings → Secrets in
    # the deployed app) or, for local runs, an environment variable.
    # Visitors never see or need this — it belongs to the app owner only.
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    return Groq(api_key=api_key)


def retrieve(question, regulator_id, chunks, embed_model, k=TOP_K):
    q_emb = embed_model.encode([question], normalize_embeddings=True)[0]
    scored = []
    for c in chunks:
        if c["regulator"] != regulator_id:
            continue
        score = float(np.dot(q_emb, c["_emb"]))
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def build_prompt(question, regulator_label, retrieved):
    context_blocks = []
    for score, c in retrieved:
        context_blocks.append(
            f"[Topic: {c['topic']}]\n{c['text']}\n(Source: {c['source']})"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are Audit Compass, an audit-readiness assistant for pharmaceutical quality professionals preparing for a {regulator_label} inspection.

Answer the question below using ONLY the reference material provided. Do not add facts that are not in the reference material. If the reference material does not cover something the question asks, say plainly that it is not covered here and recommend checking the regulator's current official guidance.

Keep the answer concise and practical (under 180 words), in plain language a quality manager would find useful.

REFERENCE MATERIAL ({regulator_label}):
{context}

QUESTION: {question}

ANSWER:"""


def answer_question(question, regulator_label):
    chunks = load_chunks()
    embed_model = load_embed_model()
    groq_client = load_groq_client()
    regulator_id = REGULATOR_BY_LABEL[regulator_label]

    retrieved = retrieve(question, regulator_id, chunks, embed_model)
    if not retrieved:
        return "I don't have reference material loaded for this regulator yet."

    prompt = build_prompt(question, regulator_label, retrieved)

    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GENERATION_MODEL,
            max_tokens=400,
            temperature=0.2,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        reply = (
            "The free model backend is temporarily unavailable or rate-limited. "
            "Please try again shortly. (Technical detail: " + str(e)[:200] + ")"
        )

    sources_used = sorted(set(c["topic"] for _, c in retrieved))
    reply += "\n\n— Grounded in reference topics: " + ", ".join(sources_used)
    return reply


# ---------------------------------------------------------------------------
# Visual design
#
# Typography: IBM Plex Serif for the headline (document/dossier feel
# appropriate to a compliance tool), IBM Plex Sans for body/UI text,
# IBM Plex Mono for small meta labels — loaded via a Google Fonts @import.
#
# Color: a deep ink-navy + muted brass/amber accent, evoking an official
# seal/certification rather than a generic tech-startup blue.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Audit Compass", page_icon="🧭", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif;
}

.block-container {
    max-width: 760px;
    padding-top: 2.2rem;
}

#audit-compass-hero .eyebrow {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 12px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #b45309;
    font-weight: 600;
    margin-bottom: 6px;
}
#audit-compass-hero h1 {
    font-family: 'IBM Plex Serif', serif !important;
    font-weight: 700 !important;
    font-size: 44px !important;
    letter-spacing: -0.01em;
    margin: 0 0 6px 0 !important;
    line-height: 1.1 !important;
    color: #0f172a;
}
#audit-compass-hero .tagline {
    font-size: 16px;
    color: #475569;
    max-width: 62ch;
    line-height: 1.6;
    margin-bottom: 4px;
}
#audit-compass-hero .credit {
    font-size: 13px;
    color: #64748b;
    margin-top: 10px;
    margin-bottom: 22px;
}
#audit-compass-hero .credit a {
    color: #b45309;
    font-weight: 600;
    text-decoration: none;
}

div[data-baseweb="select"] {
    font-size: 16px !important;
}

#audit-compass-footer {
    margin-top: 28px;
    padding-top: 14px;
    border-top: 1px solid #e2e8f0;
    font-size: 13px;
    color: #64748b;
}
#audit-compass-footer a {
    color: #b45309;
    font-weight: 600;
    text-decoration: none;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div id="audit-compass-hero">
        <div class="eyebrow">Quality &amp; Regulatory Affairs</div>
        <h1>Audit Compass</h1>
        <p class="tagline">Audit-readiness Q&amp;A grounded in curated regulatory
        reference material. Free to use, no sign-in required.
        <strong>General guidance only — not legal or regulatory advice; verify
        against current official guidance and your QA/RA team.</strong></p>
        <p class="credit">Built by <strong>Sagar Mali</strong> —
        <a href="mailto:sagarmali0801@gmail.com">sagarmali0801@gmail.com</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

regulator_label = st.selectbox(
    "Regulator",
    options=[label for label, _ in REGULATORS],
    key="regulator_select",
)

# Per-regulator chat history, so switching regulators starts a fresh thread
# (matching the original behavior) without losing earlier conversations.
if "histories" not in st.session_state:
    st.session_state.histories = {label: [] for label, _ in REGULATORS}

history = st.session_state.histories[regulator_label]

for role, content in history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input(f"Ask about {regulator_label} inspection readiness…")

if question and question.strip():
    history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking reference material…"):
            reply = answer_question(question, regulator_label)
        st.markdown(reply)
    history.append(("assistant", reply))

st.markdown(
    """
    <div id="audit-compass-footer">
        Audit Compass is an independent reference tool, built and maintained by
        <strong>Sagar Mali</strong>
        (<a href="mailto:sagarmali0801@gmail.com">sagarmali0801@gmail.com</a>).
        Not affiliated with any regulatory authority named above.
    </div>
    """,
    unsafe_allow_html=True,
)
