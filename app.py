"""
Universal Note Extractor
-------------------------
Extracts structured notes (Summary, Key Points) from:
  1. YouTube links
  2. Uploaded video/audio files
  3. PDF files
  4. Images (OCR)
  5. Plain text / pasted URL text

Optionally generates a Mermaid flowchart (for step-by-step/process content) and/or
a Mermaid concept map (for content with related ideas) alongside the notes.

Stack: Streamlit UI, Groq API (Whisper for transcription, LLaMA/GPT-OSS for note generation).
Handles long documents via chunked map-reduce summarization to respect API rate limits.
"""

import io
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

# ── Setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Note Extractor",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    """Archive/editing-room theme: warm charcoal base, amber 'REC' accent,
    index-card treatment for output blocks instead of generic rounded cards."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg: #14110D;
            --panel: #1D1912;
            --panel-raised: #241F17;
            --accent: #E8A33D;
            --accent-dim: rgba(232, 163, 61, 0.18);
            --teal: #6E9A8C;
            --text: #EDE6D6;
            --text-muted: #93897A;
            --line: #2C2620;
        }

        .stApp {
            background:
                radial-gradient(ellipse 900px 520px at 12% -8%, rgba(232, 163, 61, 0.10), transparent 60%),
                radial-gradient(ellipse 800px 520px at 105% 8%, rgba(110, 154, 140, 0.07), transparent 55%),
                var(--bg);
        }

        html, body, .stApp, [class*="css"] {
            color: var(--text);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        /* Masthead */
        .unx-masthead {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.4rem;
        }
        .unx-masthead .dot {
            color: var(--accent);
            font-size: 1.1rem;
        }
        .unx-masthead h1 {
            font-size: 2.1rem;
            margin: 0;
            color: var(--text);
        }
        .unx-tagline {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.2rem;
            margin-bottom: 1.4rem;
        }

        /* Index-card treatment for generated output */
        .unx-card {
            background: var(--panel);
            border-left: 3px solid var(--accent);
            border-radius: 2px;
            padding: 1.1rem 1.4rem;
            margin: 0.6rem 0 1.2rem 0;
        }
        .unx-card.teal { border-left-color: var(--teal); }

        /* Sidebar as an archive drawer */
        section[data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] h2 {
            font-size: 1.15rem;
        }

        /* Buttons */
        .stButton > button, .stDownloadButton > button {
            background: var(--accent);
            color: #14110D;
            border: none;
            border-radius: 3px;
            font-weight: 600;
            font-family: 'IBM Plex Sans', sans-serif;
            padding: 0.5rem 1.3rem;
            transition: filter 0.15s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            filter: brightness(1.12);
            color: #14110D;
        }

        /* Inputs, uploader, radio/checkbox groups */
        .stTextInput > div > div > input,
        .stTextArea textarea {
            background: var(--panel-raised);
            color: var(--text);
            border: 1px solid var(--line);
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--panel-raised);
            border: 1px dashed var(--line);
        }
        .stRadio > div, .stCheckbox {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        /* Progress + status */
        .stProgress > div > div > div > div {
            background-color: var(--accent);
        }

        hr { border-color: var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.environ.get("GROQ_API_KEY", "").strip()

if not api_key:
    st.error(
        "⚠️ **GROQ_API_KEY not found.**\n\n"
        f"Checked for a `.env` file at: `{env_path}`\n\n"
        "Create a file named exactly `.env` (not `.env.txt`) in this folder containing:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```\n\n"
        "Get a free key at https://console.groq.com/keys"
    )
    st.stop()

client = Groq(api_key=api_key)
HISTORY_FILE = Path(__file__).resolve().parent / "note_history.json"
TRANSCRIBE_MODEL = "whisper-large-v3"
NOTES_MODEL = "openai/gpt-oss-120b"

# Optional — powers the "Add topic images" feature. Free key at pexels.com/api.
# Feature degrades gracefully (just skipped) if this isn't set.
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "").strip()

# Chunking settings for long documents (keeps each API call small & paced
# to respect the free-tier ~8000 tokens/minute rate limit).
CHUNK_CHAR_SIZE = 6000
CHUNK_TRIGGER_CHARS = 8000  # documents longer than this get chunked
SECONDS_BETWEEN_CHUNK_CALLS = 8

# Cap on how much raw text gets sent to the diagram-generation calls, to
# keep cost/rate-limit impact predictable — diagrams only need the gist.
DIAGRAM_INPUT_CHAR_CAP = 6000

DETAIL_LEVELS = {
    "Brief": {
        "chunk_summary_words": "60-90",
        "final_instructions": (
            "Keep it short: Topic Notes should cover only the 2-4 most important topics, "
            "with 2-4 bullets each. Key Points should be 5-8 of the single most important "
            "bullets only. Summary should be 2-3 sentences."
        ),
    },
    "Standard": {
        "chunk_summary_words": "100-150",
        "final_instructions": (
            "Topic Notes should cover all major topics/sections as their own sub-headings, "
            "with 3-6 bullets each. Key Points should cover all major ideas with 10-15 "
            "bullets. Summary should be 4-6 sentences."
        ),
    },
    "Detailed": {
        "chunk_summary_words": "150-220",
        "final_instructions": (
            "Be thorough. Topic Notes should break the content into all distinct topics as "
            "sub-headings, each with comprehensive bullets/sub-bullets — include specific "
            "facts, numbers, and definitions where present, 20+ bullets total is fine for "
            "long source material. Key Points should be comprehensive. Summary should be a "
            "full paragraph (6-10 sentences) covering scope and purpose."
        ),
    },
}


# ── History helpers ──────────────────────────────────────────────────
def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history_entry(title: str, source_type: str, notes: str, detail_level: str) -> None:
    history = load_history()
    history.insert(
        0,
        {
            "title": title,
            "source_type": source_type,
            "notes": notes,
            "detail_level": detail_level,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )
    history = history[:50]  # keep last 50
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ── Extraction helpers ───────────────────────────────────────────────
def transcribe_audio_file(file_path: str) -> str:
    """Transcribe an audio/video file using Groq's Whisper endpoint."""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 24:
        raise RuntimeError(
            f"This audio file is {file_size_mb:.1f}MB, which exceeds Groq's ~25MB limit. "
            "Try a shorter video, or lower the audio quality further."
        )
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), f.read()),
            model=TRANSCRIBE_MODEL,
            response_format="text",
        )
    return str(result)


def download_youtube_audio(url: str) -> tuple[str, str]:
    """Download a YouTube video's audio track to a temp file using yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    tmp_dir = tempfile.mkdtemp()
    out_template = os.path.join(tmp_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio[filesize<25M]/bestaudio",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "32",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "YouTube Video")
        downloaded_path = os.path.join(tmp_dir, f"{info['id']}.mp3")

    if not os.path.exists(downloaded_path):
        candidates = list(Path(tmp_dir).glob("*"))
        if not candidates:
            raise RuntimeError("yt-dlp did not produce an audio file.")
        downloaded_path = str(candidates[0])

    return downloaded_path, title


def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF; falls back to OCR if the PDF has no text layer (scanned)."""
    from pypdf import PdfReader

    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    text_parts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(text_parts).strip()

    if len(text) > 50:
        return text

    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        uploaded_file.seek(0)
        images = convert_from_bytes(uploaded_file.read())
        ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
        return ocr_text.strip()
    except Exception as e:
        raise RuntimeError(
            "This PDF has no extractable text and OCR fallback failed. "
            "Make sure Poppler and Tesseract are installed on your system. "
            f"Details: {e}"
        )


def extract_image_text(uploaded_file) -> str:
    from PIL import Image
    import pytesseract

    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    if not text.strip():
        raise RuntimeError("No text could be detected in this image via OCR.")
    return text.strip()


def fetch_url_text(url: str) -> str:
    """Fetch a webpage and strip HTML tags down to plain text."""
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", response.text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Chunked map-reduce for long documents ───────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_CHAR_SIZE) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def condense_chunk(chunk: str, detail_level: str) -> str:
    words = DETAIL_LEVELS[detail_level]["chunk_summary_words"]
    completion = client.chat.completions.create(
        model=NOTES_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Condense the following excerpt into {words} words, preserving "
                    "concrete facts, names, numbers, and important ideas. Plain text, "
                    "no headers, no preamble — just the condensed content."
                ),
            },
            {"role": "user", "content": chunk},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


def build_condensed_document(raw_text: str, detail_level: str, progress=None) -> str:
    """Map-reduce a long document into a compact representative version."""
    chunks = chunk_text(raw_text)
    condensed_parts = []
    for i, chunk in enumerate(chunks):
        condensed_parts.append(condense_chunk(chunk, detail_level))
        if progress is not None:
            progress.progress(
                (i + 1) / len(chunks),
                text=f"Reading document — section {i + 1} of {len(chunks)}...",
            )
        if i < len(chunks) - 1:
            time.sleep(SECONDS_BETWEEN_CHUNK_CALLS)
    return "\n\n".join(condensed_parts)


# ── Note generation ──────────────────────────────────────────────────
def generate_notes(raw_text: str, title: str = "", detail_level: str = "Standard") -> str:
    if len(raw_text.strip()) < 20:
        raise RuntimeError("Extracted content is too short to summarize.")

    source_text = raw_text
    used_chunking = False

    if len(raw_text) > CHUNK_TRIGGER_CHARS:
        used_chunking = True
        progress = st.progress(0, text="Reading document...")
        source_text = build_condensed_document(raw_text, detail_level, progress=progress)
        progress.empty()

    detail_instructions = DETAIL_LEVELS[detail_level]["final_instructions"]

    system_prompt = (
        "You are an expert note-taker. Given transcript or document content, "
        "produce clean, well-structured notes in Markdown with exactly these "
        "three sections, IN THIS ORDER:\n\n"
        "## Topic Notes\n"
        "(Break this into sub-headings for each specific, important topic covered — "
        "use '### <topic name>' for each one, with bullet points underneath. This is "
        "the main, most detailed part of the notes.)\n\n"
        "## Key Points\n"
        "(A flat bullet list of the most important takeaways across all topics.)\n\n"
        "## Summary\n"
        "(A short overview, placed LAST.)\n\n"
        f"{detail_instructions}\n\n"
        "Do not add any other sections — in particular, do not include action items, "
        "to-do lists, or follow-up tasks. Do not reorder these sections. Be faithful "
        "to the source content — do not invent facts."
    )

    completion = client.chat.completions.create(
        model=NOTES_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Title: {title}\n\nContent:\n{source_text}"},
        ],
        temperature=0.3,
    )
    notes = completion.choices[0].message.content

    if used_chunking:
        st.caption(
            f"📄 This was a long document — notes were generated from the full "
            f"content via {len(chunk_text(raw_text))} processed sections, not just the beginning."
        )

    return notes


# ── Diagram generation (flowchart / concept map) ─────────────────────
def generate_mermaid_diagram(raw_text: str, diagram_type: str) -> str | None:
    """
    diagram_type: "flowchart" or "concept_map"
    Returns raw Mermaid syntax, or None if the content isn't a good fit for that
    diagram type (the model is instructed to say so rather than force one).
    """
    if diagram_type == "flowchart":
        instruction = (
            "If this content describes a process, sequence of steps, workflow, or "
            "decision flow, output ONLY valid Mermaid syntax starting with "
            "'flowchart TD'. If there is no clear step-by-step process in this "
            "content, output exactly: NONE"
        )
    else:
        instruction = (
            "If this content has key concepts/entities with relationships worth "
            "mapping (e.g. a lecture explaining how ideas connect), output ONLY "
            "valid Mermaid syntax starting with 'graph TD' showing the main "
            "concepts and how they relate. If there is no clear conceptual "
            "structure, output exactly: NONE"
        )

    completion = client.chat.completions.create(
        model=NOTES_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{instruction}\n\nKeep it to 6-12 nodes maximum. No explanation, "
                    "no markdown code fences — just raw Mermaid syntax or the word NONE."
                ),
            },
            {"role": "user", "content": raw_text[:DIAGRAM_INPUT_CHAR_CAP]},
        ],
        temperature=0.2,
    )
    result = completion.choices[0].message.content.strip()
    result = re.sub(r"^```(?:mermaid)?\n?|```$", "", result).strip()
    return None if result.upper() == "NONE" else result


def render_mermaid(mermaid_code: str, height: int = 420) -> None:
    """Render Mermaid syntax inline via the Mermaid CDN — no extra pip dependency."""
    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


# ── Topic images ─────────────────────────────────────────────────────
def extract_topics_from_notes(notes: str, max_topics: int = 6) -> list[str]:
    """Pull the '### <topic>' sub-headings out of the generated Topic Notes section."""
    topics = re.findall(r"^### (.+)$", notes, flags=re.MULTILINE)
    return topics[:max_topics]


def fetch_topic_image(query: str) -> str | None:
    """Look up one relevant photo for a topic via the Pexels API. Returns None on
    any failure (missing key, no results, network error) so callers can degrade
    gracefully instead of breaking note generation."""
    if not PEXELS_API_KEY:
        return None
    try:
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            timeout=10,
        )
        response.raise_for_status()
        photos = response.json().get("photos", [])
        return photos[0]["src"]["medium"] if photos else None
    except Exception:
        return None


# ── UI ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="unx-masthead">
        <span class="dot">●</span>
        <h1>Universal Note Extractor</h1>
    </div>
    <div class="unx-tagline">Video, audio, PDFs, images, or text — developed into structured notes.</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📜 History")
    history = load_history()
    if not history:
        st.caption("No notes generated yet.")
    else:
        for i, entry in enumerate(history):
            with st.expander(f"{entry.get('title', 'Untitled')[:40]} — {entry.get('timestamp', 'unknown')}"):
                st.caption(
                    f"Source: {entry.get('source_type', 'unknown')} • "
                    f"Detail: {entry.get('detail_level', 'Standard')}"
                )
                st.markdown(entry["notes"])

input_type = st.radio(
    "Choose an input type",
    [
        "🔗 YouTube Link",
        "🎙️ Upload Video/Audio",
        "📄 Upload PDF",
        "🖼️ Upload Image",
        "📝 Text or URL",
    ],
    horizontal=True,
)

detail_level = st.radio(
    "Note detail level",
    list(DETAIL_LEVELS.keys()),
    index=1,  # Standard by default
    horizontal=True,
    help="Brief = quick skim. Standard = balanced. Detailed = comprehensive, best for long documents.",
)

col1, col2, col3 = st.columns(3)
with col1:
    include_flowchart = st.checkbox(
        "🔀 Add flowchart", help="Generated only if the content describes a process or sequence of steps."
    )
with col2:
    include_concept_map = st.checkbox(
        "🧠 Add concept map", help="Generated only if the content has related ideas worth mapping."
    )
with col3:
    include_images = st.checkbox(
        "🖼️ Add topic images", help="Pulls one relevant photo per topic from Pexels. Requires PEXELS_API_KEY in .env."
    )

extracted_text = None
source_title = "Untitled"

if input_type == "🔗 YouTube Link":
    yt_url = st.text_input("Paste a YouTube URL")
    if st.button("Extract Notes", type="primary", disabled=not yt_url):
        with st.spinner("Downloading audio..."):
            try:
                audio_path, source_title = download_youtube_audio(yt_url)
            except Exception as e:
                st.error(f"Couldn't download this video: {e}")
                st.stop()
        with st.spinner("Transcribing..."):
            try:
                extracted_text = transcribe_audio_file(audio_path)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

elif input_type == "🎙️ Upload Video/Audio":
    media_file = st.file_uploader(
        "Upload a video or audio file", type=["mp3", "wav", "m4a", "mp4", "mov", "webm"]
    )
    if media_file and st.button("Extract Notes", type="primary"):
        source_title = media_file.name
        with st.spinner("Transcribing..."):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(media_file.name).suffix
                ) as tmp:
                    tmp.write(media_file.read())
                    tmp_path = tmp.name
                extracted_text = transcribe_audio_file(tmp_path)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

elif input_type == "📄 Upload PDF":
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf_file and st.button("Extract Notes", type="primary"):
        source_title = pdf_file.name
        with st.spinner("Extracting text..."):
            try:
                extracted_text = extract_pdf_text(pdf_file)
            except Exception as e:
                st.error(str(e))
                st.stop()

elif input_type == "🖼️ Upload Image":
    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if img_file and st.button("Extract Notes", type="primary"):
        source_title = img_file.name
        with st.spinner("Running OCR..."):
            try:
                extracted_text = extract_image_text(img_file)
            except Exception as e:
                st.error(str(e))
                st.stop()

elif input_type == "📝 Text or URL":
    text_input = st.text_area("Paste text, or a URL to fetch", height=200)
    if text_input and st.button("Extract Notes", type="primary"):
        if text_input.strip().startswith("http"):
            source_title = text_input.strip()
            with st.spinner("Fetching page..."):
                try:
                    extracted_text = fetch_url_text(text_input.strip())
                except Exception as e:
                    st.error(f"Couldn't fetch that URL: {e}")
                    st.stop()
        else:
            source_title = text_input.strip()[:40] or "Pasted Text"
            extracted_text = text_input

# ── Generate + display notes ────────────────────────────────────────
if extracted_text:
    try:
        notes = generate_notes(extracted_text, title=source_title, detail_level=detail_level)
    except Exception as e:
        st.error(f"Note generation failed: {e}")
        st.stop()

    st.success("Notes generated!")
    st.markdown(f'<div class="unx-card">{notes}</div>', unsafe_allow_html=True)

    if include_flowchart or include_concept_map:
        with st.spinner("Checking whether a diagram fits this content..."):
            if include_flowchart:
                flowchart = generate_mermaid_diagram(extracted_text, "flowchart")
                if flowchart:
                    st.subheader("🔀 Flowchart")
                    render_mermaid(flowchart)
                else:
                    st.caption("No clear step-by-step process detected — skipping flowchart.")
            if include_concept_map:
                concept_map = generate_mermaid_diagram(extracted_text, "concept_map")
                if concept_map:
                    st.subheader("🧠 Concept Map")
                    render_mermaid(concept_map)
                else:
                    st.caption("No clear conceptual structure detected — skipping concept map.")

    if include_images:
        topics = extract_topics_from_notes(notes)
        if not PEXELS_API_KEY:
            st.caption(
                "🖼️ Add a `PEXELS_API_KEY` to your `.env` to enable topic images "
                "(free at https://www.pexels.com/api/)."
            )
        elif not topics:
            st.caption("No distinct topics detected to illustrate.")
        else:
            st.subheader("🖼️ Topic Images")
            with st.spinner("Fetching relevant images..."):
                img_cols = st.columns(min(len(topics), 3))
                for i, topic in enumerate(topics):
                    img_url = fetch_topic_image(topic)
                    with img_cols[i % len(img_cols)]:
                        if img_url:
                            st.image(img_url, caption=topic, use_container_width=True)
                        else:
                            st.caption(f"No image found for: {topic}")

    st.download_button(
        "⬇️ Download Notes (.md)",
        data=notes,
        file_name=f"{re.sub(r'[^a-zA-Z0-9]+', '_', source_title)[:50]}_notes.md",
        mime="text/markdown",
    )

    save_history_entry(source_title, input_type, notes, detail_level)