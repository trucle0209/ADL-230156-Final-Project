"""MATH CHILL ADVISOR | Streamlit presentation layer.

The existing retrieval, embeddings, model invocation, and conversation prompt
are unchanged from the original Task 4 chat_app.py.
"""

from __future__ import annotations

import random
import re
import time
from html import escape

import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from run_pdf import EMBED_MODEL, OLLAMA_URL, load_and_index, retrieve

st.set_page_config(
    page_title="MATH CHILL ADVISOR",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

MODEL_CHOICES = ["qwen2.5:1.5b", "qwen2.5:0.5b"]
STARTER_QUESTIONS = [
    "How many credits are required for the Applied Mathematics major?",
    "What are the three specialized tracks?",
    "What can students take instead of Capstone I and II?",
]


@st.cache_resource(show_spinner="Loading the existing vector database...")
def get_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    return load_and_index(embeddings)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, max_seconds: int):
    # Synchronous Ollama client avoids cross-rerun event-loop failures.
    # HTTP timeout limits network inactivity; num_predict caps output tokens.
    return OllamaLLM(
        model=model_name,
        base_url=OLLAMA_URL,
        temperature=0,
        num_predict=180,
        keep_alive="1m",
        client_kwargs={"timeout": float(max_seconds)},
    )


def invoke_model(llm, prompt: str) -> str:
    """Use LangChain's synchronous Ollama invocation on every Streamlit rerun."""
    return llm.invoke(prompt).strip()


def is_translation_followup(question: str) -> bool:
    q = question.casefold().strip()
    return bool(re.search(
        r"(\bdịch\b|tiếng anh|\btranslate\b|\benglish\b|"
        r"tiếng việt|say (that|it|this) in)", q
    ))


def translate_previous_answer(question: str, messages: list[dict], llm):
    """Translate last answer; protect handbook entities and quantities."""
    prior_assistants = [
        m["content"] for m in messages
        if m.get("role") == "assistant" and m.get("content")
        and not str(m.get("content")).startswith("Sorry, the request")
    ]
    if not prior_assistants:
        return "I don't have a previous answer to translate.", [], 0.0, []

    previous = prior_assistants[-1].strip()
    target_english = bool(re.search(r"tiếng anh|\benglish\b", question.casefold()))
    target_vietnamese = bool(re.search(r"tiếng việt|\bvietnamese\b", question.casefold()))

    # For a single verified major/minor credit fact, translating with a template
    # is safer and faster than making a fresh LLM request.
    simple_credit = re.search(
        r"(?:Applied Mathematics\s+)?(major|minor).*?\b(\d+)\s*(credits|tín chỉ)\b",
        previous, flags=re.I | re.S,
    )
    if simple_credit and ("Applied Mathematics" in previous or "Toán ứng dụng" in previous):
        area, amount = simple_credit.group(1).lower(), simple_credit.group(2)
        if target_vietnamese:
            area_vi = "ngành phụ" if area == "minor" else "ngành chính"
            return (f"{area_vi.capitalize()} Toán ứng dụng yêu cầu {amount} tín chỉ.",
                    [], 0.0, [])
        if target_english:
            return (f"The Applied Mathematics {area} requires {amount} credits.",
                    [], 0.0, [])

    if target_english and not re.search(r"[À-ỹ]", previous):
        return "The previous answer is already in English.", [], 0.0, []

    started = time.perf_counter()
    target = "Vietnamese" if target_vietnamese else "English" if target_english else "the requested language"
    prompt = (
        f"Translate the following text to {target}. Preserve all proper nouns, "
        "named academic programs, numbers, units (courses vs credits), and "
        "conditions exactly. Applied Mathematics means Toán ứng dụng in Vietnamese. "
        "Minor means ngành phụ. Do not interpret, correct, or add facts. "
        "Output only the translated text.\n\n"
        f"TEXT:\n{previous[:1600]}\n\nTRANSLATION:"
    )
    translated = invoke_model(llm, prompt)

    # Never present a translation that changes cardinal facts or the program.
    previous_numbers = re.findall(r"\b\d+\b", previous)
    translated_numbers = re.findall(r"\b\d+\b", translated)
    program_protected = (
        ("Applied Mathematics" not in previous and "Toán ứng dụng" not in previous)
        or "Applied Mathematics" in translated or "Toán ứng dụng" in translated
    )
    if not program_protected or not set(previous_numbers).issubset(set(translated_numbers)):
        return (
            "I couldn't safely translate the previous answer without changing "
            "its academic facts. Please refer to the original answer:\n\n" + previous,
            [], time.perf_counter() - started, []
        )

    return translated, [], time.perf_counter() - started, []


def out_of_scope_reply(question: str) -> str | None:
    """Refuse explicit requests for another degree's requirements.

    The only indexed source is the Applied Mathematics Major Handbook.
    This intentionally checks explicit, named programs rather than guessing
    unsupported facts from a generic graduation-credit snippet.
    """
    q = " ".join(question.casefold().split())
    named_other_program = re.search(
        r"\b(?:computer\s+science|comp\s*sci|"
        r"khoa\s+học\s+máy\s+tính|khoa\s+hoc\s+may\s+tinh|"
        r"software\s+engineering|kỹ\s+thuật\s+phần\s+mềm|"
        r"business\s+administration|quản\s+trị\s+kinh\s+doanh|"
        r"psychology|tâm\s+lý\s+học|"
        r"economics|kinh\s+tế)\b",
        q,
        re.I,
    )
    if not named_other_program:
        return None

    # When the user explicitly compares programs, respond with the limitation
    # instead of accidentally applying Applied Math requirements to the other.
    asks_about_program = bool(re.search(
        r"tín\s+chỉ|tin\s+chi|tốt\s+nghiệp|tot\s+nghiep|"
        r"\b(credits?|graduat(?:e|ion)|degree|major|minor|"
        r"requirements?|curriculum|courses?|học\s+phần|môn\s+học)\b",
        q,
        re.I,
    ))
    if not asks_about_program:
        return None

    vietnamese = bool(re.search(r"[à-ỹ]|tín chỉ|sinh viên|ngành", q))
    if vietnamese:
        return (
            "Tôi chỉ có tài liệu Applied Mathematics Major Handbook "
            "(ngành Toán ứng dụng), nên không thể xác nhận yêu cầu của "
            "ngành khác từ tài liệu hiện có. Vui lòng tham khảo "
            "handbook chính thức của ngành đó."
        )
    return (
        "I only have the Applied Mathematics Major Handbook. "
        "It does not establish requirements for other majors. "
        "Please consult the official handbook for that program."
    )


def question_type(question: str, messages: list[dict]) -> str | None:
    """Find the topic of a credit-count query; never supply its numeric answer."""
    q = question.casefold()
    minor = bool(re.search(r"\bminor\b|ngành phụ|ngành nhỏ|chuyên ngành phụ", q))
    major = bool(re.search(r"\bmajor\b|ngành toán|toán ứng dụng|applied mathematics", q))
    grad = bool(re.search(r"graduat|tốt nghiệp", q))
    credits = bool(re.search(r"credit|tín chỉ|tính chỉ|tin chi|bao nhiêu", q))
    if minor:
        return "minor" if (credits or len(q.split()) <= 7) else None
    if grad and credits:
        return "graduation"
    if major and credits:
        return "major"
    # Only inherit *the requested property*, not the previous major subject.
    if len(q.split()) <= 8 and re.search(r"how about|what about|còn.*sao", q):
        return None
    return None


def focused_query(question: str, messages: list[dict]) -> str:
    q = question.casefold()
    if re.search(r"capstone", q):
        return "Capstone I and Capstone II OR advanced courses at 300-level alternative major requirements"
    if re.search(r"\bminor\b|ngành phụ|ngành nhỏ", q):
        return "Applied Mathematics minor requirements minimum credits courses"
    if re.search(r"graduat|tốt nghiệp", q):
        return "minimum total credits graduation degree requirements"
    if (re.search(r"credit|tín chỉ|tính chỉ|tin chi", q)
            and re.search(r"\bmajor\b|ngành toán|toán ứng dụng", q)):
        return "Applied Mathematics major requirements credits"
    return question


def extract_fact(question: str, messages: list[dict], hits) -> str | None:
    """Ground simple direct credit questions in a numbered PDF heading.

    Both the number AND its label are read from retrieved PDF excerpts.
    Nothing about credits is hardcoded into the model prompt.
    """
    if re.search(r"capstone", question, re.I) and re.search(
        r"\b(instead|alternative|replace|thay thế|không làm|không học|không chọn)\b",
        question, re.I,
    ):
        # Require evidence of both sides of the explicit handbook "OR".
        for doc, _score in hits:
            passage = doc.page_content
            alternative = re.search(
                r"Capstone\s+I\s+and\s+Capstone\s+II"
                r".{0,210}?\bOR\s+(\d+)\s+advanced\s+courses?"
                r"\s+at\s+(?:the\s+)?(\d{3})[- ]level",
                passage, re.I | re.S,
            )
            if alternative:
                count, level = alternative.groups()
                vietnamese = bool(re.search(
                    r"nếu|không|thay|sinh viên|cần|phải|môn|học|gì",
                    question.casefold(),
                ))
                if vietnamese:
                    return f"Nếu không chọn Capstone I và II, sinh viên cần hoàn thành {count} môn nâng cao ở level {level} thay thế."
                return f"Instead of Capstone I and II, students must complete {count} advanced courses at the {level}-level."
        # Let ordinary evidence-grounded generation handle cases where
        # the exact alternative was not retrieved. Do not invent an answer.

    topic = question_type(question, messages)
    if topic not in {"minor", "major"}:
        return None
    label = "Minor" if topic == "minor" else "Major"
    pattern = re.compile(rf"\b{label} Requirements\s*\(\s*(\d+)\s*credits\s*\)", re.I)
    for doc, _score in hits:
        match = pattern.search(doc.page_content)
        if match:
            target = "minor" if topic == "minor" else "major"
            vietnamese = bool(re.search(r"ngành|toán|tín chỉ|tính chỉ", question.casefold()))
            if vietnamese:
                area_vi = "phụ" if target == "minor" else "chính"
                return f"Ngành {area_vi} Toán ứng dụng yêu cầu {match.group(1)} tín chỉ."
            return f"The Applied Mathematics {target} requires {match.group(1)} credits."
    return None


def answer_question(question: str, messages: list[dict], db, llm, top_k: int):
    started = time.perf_counter()
    scope_response = out_of_scope_reply(question)
    if scope_response is not None:
        return scope_response, [], time.perf_counter() - started, []
    query = focused_query(question, messages)
    hits = retrieve(query, db, top_k=top_k, method="hybrid")
    if not hits:
        return "I couldn't find enough information in the handbook.", [], time.perf_counter()-started, []

    evidence, citations, source_details = [], [], []
    seen = set()
    for idx, (doc, score) in enumerate(hits, 1):
        page = str(doc.metadata.get("page", "?"))
        evidence.append(f"[Evidence {idx} | PDF page {page}]\n{doc.page_content}")
        source_details.append({
            "page": page,
            "source": str(doc.metadata.get("source", "MATH_MajorHandbook.pdf")),
            "score": float(score),
            "text": doc.page_content,
        })
        if page not in seen:
            seen.add(page)
            citations.append(page)

    # Exact handbook headings can be read deterministically, avoiding LLM
    # hallucination and unnecessary GPU inference for simple credit questions.
    exact = extract_fact(question, messages, hits)
    if exact:
        return exact, citations, time.perf_counter()-started, source_details

    # Never include previous assistant answers as evidence. In particular,
    # a hallucinated 120 credits must not contaminate later responses.
    recent_questions = [m["content"] for m in messages if m.get("role") == "user"][-2:]
    previous_text = " | ".join(recent_questions)[:300] or "None"
    prompt = f"""You are a careful Applied Mathematics handbook advisor.
The PDF evidence is the ONLY source of academic facts.
Do not use previous model answers as evidence.
Correctly distinguish major requirements, minor requirements, graduation,
number of courses versus credits, and OR/NOT alternatives.
If the evidence is insufficient, say so, without inventing facts.
Answer in the language of the latest question, in 1-3 concise sentences.
Mention a PDF page if you can verify it in the evidence.

Previous user topic (context only): {previous_text}
LATEST QUESTION: {question}

PDF EVIDENCE:
{chr(10).join(evidence)}

DIRECT ANSWER:"""
    return invoke_model(llm, prompt), citations, time.perf_counter()-started, source_details


FONT_FAMILIES = {
    "Patrick Hand": '"Patrick Hand", cursive',
    "Caveat": '"Caveat", cursive',
    "Kalam": '"Kalam", cursive',
    "Inter": '"Inter", sans-serif',
    "System": 'system-ui, -apple-system, "Segoe UI", sans-serif',
}
FONT_SIZES = {"Small": "15px", "Medium": "17px", "Large": "20px"}


def inject_css(theme: str, font: str, font_size: str) -> None:
    '''One CSS entrypoint called on every rerun; no RAG functions affected.'''
    light = theme == "Light"
    page = "#f6f0e3" if light else "#0c1b19"
    ink = "#1d302b" if light else "#eee8d8"
    muted = "#667772" if light else "#b0bbb4"
    chalk = "#d5ded2" if light else "#e8eddf"
    wood = "#8b5a2b"
    font_stack = FONT_FAMILIES.get(font, FONT_FAMILIES["Patrick Hand"])
    base = FONT_SIZES.get(font_size, "17px")

    st.markdown(
        f'''<style>
@import url('https://fonts.googleapis.com/css2?family=Patrick+Hand&family=Caveat:wght@400;600&family=Kalam:wght@300;400;700&family=Inter:wght@400;500;600;700&display=swap');
:root {{ --paper:{page}; --ink:{ink}; --muted:{muted}; --chalk:{chalk}; --wood:{wood}; --body-font:{font_stack}; --base-font:{base}; }}
#MainMenu, footer, header, header[data-testid="stHeader"],
header [data-testid="stToolbar"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stDeployButton"], [data-testid="collapsedControl"] {{ display:none !important; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
[data-testid="stMain"], .main {{ background:var(--paper) !important; color:var(--ink) !important; }}
html {{ font-size:var(--base-font) !important; }}
[data-testid="stAppViewContainer"] {{ position:relative; isolation:isolate; }}
[data-testid="stMain"] {{ position:relative; z-index:1; background:transparent !important; }}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width:850px !important; padding-top:3.4rem !important; padding-bottom:8rem !important;
  position:relative; z-index:2;
}}
[data-testid="stMarkdownContainer"], [data-testid="stChatMessageContent"],
[data-testid="stChatInput"], p, label, button, textarea {{
  font-family:var(--body-font) !important; font-size:var(--base-font);
}}
p {{ line-height:1.5; }}
.chill-title {{ font-family:"Patrick Hand", cursive; font-size:clamp(2.15rem,5vw,3.5rem);
  line-height:1.07; text-align:center; color:var(--ink); letter-spacing:.08em;
  font-weight:400; margin:0 auto 1.3rem; }}
.chill-disclaimer {{ color:var(--muted); text-align:center; opacity:.6;
  font-size:.75rem !important; margin-top:.3rem; font-family:"Inter",sans-serif !important; }}
.math-drift {{ position:fixed; inset:0; pointer-events:none; z-index:0;
  overflow:hidden; color:{'#6d837c' if light else '#e4eddf'}; }}
.math-drift span {{position:absolute; left:var(--x); top:var(--y); font-family:"Caveat",serif;
  font-size:var(--s); white-space:nowrap; opacity:0; will-change:transform,opacity;
  animation:chalk-float var(--d) ease-in-out var(--delay) infinite alternate; }}
@keyframes chalk-float {{
  0% {{ transform:translate3d(0,13px,0) rotate(-5deg); opacity:0; }}
  25% {{ opacity:var(--opacity); }}
  70% {{opacity:var(--opacity); }}
  100% {{ transform:translate3d(32px,-70px,0) rotate(6deg); opacity:0; }}
}}
@media (prefers-reduced-motion: reduce) {{ .math-drift span {{animation:none; opacity:.09;}} }}
[data-testid="stChatMessage"] {{
  padding:1rem 1.25rem !important; margin:.55rem 0 !important;
  box-shadow:0 9px 24px rgba(0,0,0,.14); border-radius:8px !important;
}}
/* Assistant: framed chalkboard */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
  background:radial-gradient(ellipse at 18% 10%,#334c40 0%,#233a33 45%,#162824 100%) !important;
  border:5px solid var(--wood) !important;
  outline:1px solid rgba(232,202,146,.45);
  color:#f1f0dd !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) span {{
  color:#f1f0dd !important; text-shadow:0 0 1px rgba(255,255,240,.43);
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {{
  font-family:"Patrick Hand",var(--body-font) !important;
}}
/* User: a soft paper note, visually distinct from bot */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
  background:{'#fffaf0' if light else '#d8d2b3'} !important;
  color:#26372d !important; border:1px solid #a39776 !important;
  box-shadow:0 5px 16px rgba(0,0,0,.1);
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span {{ color:#26372d !important; }}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {{
  background:transparent !important; border:1px dashed currentColor !important;
  color:inherit !important; font-family:"Inter",sans-serif !important;
  font-size:.7rem !important; box-shadow:none !important;
}}
/* Retrieved PDF evidence stays legible inside the chalkboard. */
[data-testid="stChatMessage"] [data-testid="stExpander"] {{
  border:1px dashed rgba(243,241,211,.50) !important; border-radius:8px !important;
  color:#f1f0dd !important; background:rgba(0,0,0,.11) !important;
}}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary,
[data-testid="stChatMessage"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{
  color:#f1f0dd !important;
}}
[data-testid="stChatMessage"] [data-testid="stExpander"] [data-testid="stText"] {{
  font-size:.86rem !important; white-space:pre-wrap !important;
  overflow-wrap:anywhere !important; color:#e2e7d9 !important;
}}
/* st.chat_input: preserve native submission/keyboard behavior */
[data-testid="stChatInput"] {{
  background:#1f2d2a !important; border:5px solid var(--wood) !important;
  border-radius:10px !important; box-shadow:0 9px 26px rgba(0,0,0,.2) !important;
}}
[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] textarea::placeholder {{
  color:#f0eedd !important; caret-color:#f0eedd !important;
  font-family:var(--body-font) !important; font-size:var(--base-font) !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ opacity:.7; }}
[data-testid="stChatInput"] button {{color:#e7ebdc !important;}}
/* Quick question: light chalk outline */
[data-testid="stButton"] button {{
  border-radius:8px !important; border:1px dashed #9baf9f !important;
  background:transparent !important; color:var(--ink) !important;
  font-family:var(--body-font) !important;
}}
[data-testid="stButton"] button:hover {{ background:rgba(128,152,136,.12) !important; }}
/* Small fixed settings popover trigger */
[data-testid="stPopover"] {{
  position:fixed !important; top:1rem !important; right:1.3rem !important;
  z-index:999 !important; width:auto !important;
}}
[data-testid="stPopover"] > button,
[data-testid="stPopover"] button[data-testid="stBaseButton-secondary"] {{
  min-width:2.5rem !important; border-radius:50% !important; padding:.45rem !important;
  border:1px solid var(--muted) !important; background:var(--paper) !important;
  color:var(--ink) !important; font-size:1.1rem !important;
}}
[data-testid="stPopoverBody"], [data-testid="stPopoverContent"] {{
  background:var(--paper) !important; color:var(--ink) !important;
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  font-size:.73rem !important; opacity:.64;
}}
@media (max-width:640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding-top:4.3rem !important; }}
  [data-testid="stChatMessage"] {{padding:.7rem !important;}}
  [data-testid="stPopover"] {{right:.75rem !important;}}
}}
</style>''',
        unsafe_allow_html=True,
    )


def inject_math_background() -> None:
    '''Decorative HTML only: no scripts, no network, and no RAG impact.'''
    formulas = [
        "eⁱπ + 1 = 0", "∫ f(x) dx", "Σ 1/n²", "∇ · E", "a² + b² = c²",
        "limₓ→₀ sin x/x = 1", "dy/dx", "π", "√2", "ℱ[f](ω)",
        "det(A)", "∂u/∂t", "∑ₖ₌₁ⁿ k", "x² − 4x + 4", "sin²θ + cos²θ = 1",
        "∫₀¹ x² dx", "AᵀA", "λ₁ + λ₂", "P(A|B)", "∇²φ = 0",
        "f′(x)", "ℝⁿ", "∛8 = 2", "eˣ", "∮ F · dr", "∞", "μ = E[X]",
    ]
    rng = random.Random(311)  # stable between Streamlit reruns
    spans = []
    for formula in formulas:
        style = (
            f"--x:{rng.randrange(2,96)}%;--y:{rng.randrange(4,96)}%;"
            f"--s:{rng.randrange(17,40)}px;--d:{rng.randrange(19,43)}s;"
            f"--delay:-{rng.randrange(0,37)}s;--opacity:{rng.uniform(.08,.18):.3f}"
        )
        spans.append(f'<span style="{style}">{escape(formula)}</span>')
    st.markdown(
        '<div class="math-drift" aria-hidden="true">' + "".join(spans) + '</div>',
        unsafe_allow_html=True,
    )


def export_conversation(messages: list[dict]) -> str:
    """Export only existing session messages, with their supporting retrieval excerpts."""
    lines = ["# MATH CHILL ADVISOR", "", "Chat transcript (local session)", ""]
    for msg in messages:
        role = "You" if msg.get("role") == "user" else "Advisor"
        lines.extend([f"## {role}", "", msg.get("content", ""), ""])
        if role == "Advisor":
            pages = msg.get("pages", [])
            if pages:
                lines.extend(["Source pages: " + ", ".join(map(str, pages)), ""])
            for i, item in enumerate(msg.get("sources", []), 1):
                lines.extend([
                    f"### Evidence {i} — Page {item['page']}", "",
                    item.get("text", ""), "",
                ])
    lines.append("Note: Responses may contain mistakes; verify academic requirements in the handbook.")
    return "\n".join(lines) + "\n"


def show_sources(msg: dict, unique_key: str) -> None:
    """Source transparency only. Never performs another database or model request."""
    items = msg.get("sources", [])
    pages = msg.get("pages", [])
    if not items:
        if pages:
            st.caption("PDF pages: " + ", ".join(map(str, pages)))
        return
    with st.expander(f"View sources ({len(items)})", expanded=False):
        st.caption("Retrieved excerpts · MATH_MajorHandbook.pdf · higher score = stronger match")
        for i, item in enumerate(items, 1):
            st.markdown(f"**{i}. PDF page {item['page']} · Relevance {item['score']:.4f}**")
            # st.text displays the retrieved text literally, without interpreting it as markup.
            st.text(item.get("text", ""))
            if i != len(items):
                st.divider()


# State changes only affect the presentation layer and per-request model controls.
for key, default in {
    "messages": [],
    "ui_theme": "Dark",
    "ui_font": "Patrick Hand",
    "ui_font_size": "Medium",
    "ui_model": MODEL_CHOICES[0],
    "ui_top_k": 5,
    "ui_timeout": 25,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Place settings at the top so widgets update state before CSS is injected.
with st.popover(":material/settings:", help="Settings"):
    st.selectbox("Ollama model", MODEL_CHOICES, key="ui_model")
    st.slider("Top-K retrieved chunks", min_value=1, max_value=10, key="ui_top_k")
    st.select_slider("Ollama network timeout (seconds)", options=[15, 20, 25, 30, 40], key="ui_timeout")
    st.radio("Theme", ["Dark", "Light"], horizontal=True, key="ui_theme")
    st.selectbox("Font", list(FONT_FAMILIES), key="ui_font")
    st.select_slider("Font size", options=list(FONT_SIZES), key="ui_font_size")
    if st.session_state.messages:
        st.download_button(
            "Export conversation (.md)",
            data=export_conversation(st.session_state.messages),
            file_name="math_chill_conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

inject_css(st.session_state.ui_theme, st.session_state.ui_font, st.session_state.ui_font_size)
inject_math_background()

st.markdown('<h1 class="chill-title">MATH CHILL ADVISOR</h1>', unsafe_allow_html=True)

# Rendering: source metadata and response times are kept, without long guidance text.
for index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/smart_toy:"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg, f"history_{index}")
            if "elapsed" in msg:
                st.caption(f"{msg['elapsed']:.2f}s")

# A quick question uses the exact same request path as text typed in chat_input.
pending_question = None
if st.button("Quick ask me", key="quick_ask"):
    pending_question = random.choice(STARTER_QUESTIONS)

typed_question = st.chat_input("Ask me anything about math...")
question = (typed_question or pending_question or "").strip()

if question:
    existing_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        # Friendly greetings are UI behavior, not handbook questions.
        # Avoid unnecessary Ollama/Chroma calls and irrelevant PDF citations.
        normalized = " ".join(question.lower().strip(" \t\r\n.!?~").split())
        greetings = {"hello", "hi", "hey", "hey there", "hello there",
                     "xin chào", "chào", "chào bạn", "helo"}
        if normalized in greetings:
            response = (
                "Chào bạn! Bạn muốn hỏi gì về ngành Toán ứng dụng?"
                if normalized in {"xin chào", "chào", "chào bạn"}
                else "Hello! What would you like to know about Applied Mathematics?"
            )
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant", "content": response, "pages": [], "sources": []
            })
        elif normalized in {"i love you", "love you", "thank you", "thanks", "cảm ơn"}:
            response = ("Thanks! What would you like to know about the mathematics handbook?"
                        if normalized not in {"cảm ơn"} else
                        "Cảm ơn bạn! Bạn muốn hỏi gì về cẩm nang ngành Toán ứng dụng?")
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant", "content": response, "pages": [], "sources": []
            })
        elif (scope_response := out_of_scope_reply(question)) is not None:
            # Do not retrieve irrelevant PDF chunks or invoke Ollama.
            st.markdown(scope_response)
            st.caption("0.00s")
            st.session_state.messages.append({
                "role": "assistant", "content": scope_response,
                "pages": [], "sources": [], "elapsed": 0.0,
            })
        else:
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(st.session_state.ui_model, st.session_state.ui_timeout)
                    if is_translation_followup(question):
                        response, pages, elapsed, sources = translate_previous_answer(
                            question, existing_history, llm
                        )
                    else:
                        db = get_db()
                        response, pages, elapsed, sources = answer_question(
                            question, existing_history, db, llm,
                            st.session_state.ui_top_k
                        )
                except (TimeoutError,):
                    response = (
                        f"Sorry, the request exceeded {st.session_state.ui_timeout} seconds. "
                        "I stopped waiting to protect your computer. "
                        "Try a shorter question or lower Top-K."
                    )
                    pages, sources, elapsed = [], [], float(st.session_state.ui_timeout)
                except Exception as exc:
                    description = str(exc)
                    if "timed out" in description.lower() or "timeout" in description.lower():
                        response = ("Ollama did not respond within the configured network timeout. "
                                    "Please retry with a shorter question.")
                        pages, sources, elapsed = [], [], float(st.session_state.ui_timeout)
                    else:
                        response = ("Something went wrong while contacting Ollama. "
                                    "Check whether Ollama is running.")
                        pages, sources, elapsed = [], [], 0.0
                        st.caption(description[:220])

            # Display-only text streaming; the original llm.invoke call is unchanged.
            def display_chunks(text: str):
                for i in range(0, len(text), 16):
                    yield text[i:i + 16]

            st.write_stream(display_chunks(response))
            show_sources({"sources": sources, "pages": pages}, "latest")
            st.caption(f"{elapsed:.2f}s")
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "pages": pages,
                 "sources": sources, "elapsed": elapsed}
            )

st.markdown('<p class="chill-disclaimer">Don\'t fully trust me.</p>', unsafe_allow_html=True)
