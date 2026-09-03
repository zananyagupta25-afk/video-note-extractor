"""
Universal Note Extractor
-------------------------
Extracts structured notes (Summary, Key Points, Action Items) from:
  1. YouTube links
  2. Uploaded video/audio files
  3. PDF files
  4. Images (OCR)
  5. Plain text / pasted URL text

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
from dotenv import load_dotenv
from groq import Groq

# ── Setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Note Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Chunking settings for long documents (keeps each API call small & paced
# to respect the free-tier ~8000 tokens/minute rate limit).
CHUNK_CHAR_SIZE = 6000
CHUNK_TRIGGER_CHARS = 8000  # documents longer than this get chunked
SECONDS_BETWEEN_CHUNK_CALLS = 8

DETAIL_LEVELS = {
    "Brief": {
        "chunk_summary_words": "60-90",
        "final_instructions": (
            "Keep it short: Summary should be 2-3 sentences. Key Points should be "
            "5-8 of the single most important bullets only. Action Items should be "
            "at most 3 bullets, or 'No specific action items.'"
        ),
    },
    "Standard": {
        "chunk_summary_words": "100-150",
        "final_instructions": (
            "Summary should be 4-6 sentences. Key Points should cover all major "
            "ideas/sections with 10-15 bullets. Action Items should list concrete "
            "follow-ups, or 'No specific action items.'"
        ),
    },
    "Detailed": {
        "chunk_summary_words": "150-220",
        "final_instructions": (
            "Be thorough. Summary should be a full paragraph (6-10 sentences) covering "
            "scope and purpose. Key Points should be comprehensive — organize with "
            "sub-headings by topic/section where appropriate, 20+ bullets is fine for "
            "long source material. Include specific facts, numbers, and definitions "
            "where present. Action Items should be a complete list, or 'No specific "
            "action items.'"
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
        "three sections:\n\n"
        "## Summary\n"
        "## Key Points\n"
        "## Action Items\n\n"
        f"{detail_instructions}\n\n"
        "Do not add any other sections. Be faithful to the source content — do not invent facts."
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


# ── UI ────────────────────────────────────────────────────────────────
st.title("🎬 Universal Note Extractor")
st.caption("Turn videos, audio, PDFs, images, or text into structured notes — powered by Groq.")

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
    st.markdown(notes)

    st.download_button(
        "⬇️ Download Notes (.md)",
        data=notes,
        file_name=f"{re.sub(r'[^a-zA-Z0-9]+', '_', source_title)[:50]}_notes.md",
        mime="text/markdown",
    )

    save_history_entry(source_title, input_type, notes, detail_level)