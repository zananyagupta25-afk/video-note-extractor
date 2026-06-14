import streamlit as st
import json
import os
import re
import subprocess
import glob
import requests
from datetime import datetime
from groq import Groq

st.set_page_config(page_title="Video Note Extractor", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
HISTORY_FILE = "note_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID. Please check the URL.")

def parse_vtt(vtt_content):
    lines = vtt_content.split('\n')
    entries = []
    current_time = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        time_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->', line)
        if time_match:
            if current_time and current_text:
                text = ' '.join(current_text).strip()
                text = re.sub(r'<[^>]+>', '', text)
                if text:
                    entries.append({'timestamp': current_time, 'text': text})
            current_time = time_match.group(1)[:8]
            current_text = []
        elif line and not line.startswith('WEBVTT') and not line.startswith('NOTE') and '-->' not in line and not line.isdigit():
            clean = re.sub(r'<[^>]+>', '', line)
            if clean:
                current_text.append(clean)
    
    if current_time and current_text:
        text = ' '.join(current_text).strip()
        if text:
            entries.append({'timestamp': current_time, 'text': text})
    
    return entries

def get_transcript_yt_dlp(video_id):
    # Clean up old vtt files
    for f in glob.glob("*.vtt"):
        os.remove(f)
    
    result = subprocess.run(
        ['yt-dlp', '--write-auto-sub', '--skip-download', '--sub-lang', 'en', 
         f'https://www.youtube.com/watch?v={video_id}'],
        capture_output=True, text=True
    )
    
    vtt_files = glob.glob("*.vtt")
    if not vtt_files:
        raise ValueError("No transcript found. The video may not have English subtitles.")
    
    with open(vtt_files[0], 'r', encoding='utf-8') as f:
        content = f.read()
    
    for f in vtt_files:
        os.remove(f)
    
    entries = parse_vtt(content)
    if not entries:
        raise ValueError("Could not parse transcript from this video.")
    
    return entries

def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def chunk_entries(entries, chunk_size=60):
    chunks = []
    seen_texts = set()
    unique_entries = []
    
    for e in entries:
        if e['text'] not in seen_texts:
            seen_texts.add(e['text'])
            unique_entries.append(e)
    
    for i in range(0, len(unique_entries), chunk_size):
        chunk = unique_entries[i:i+chunk_size]
        text = " ".join([e['text'] for e in chunk])
        chunks.append({"timestamp": chunk[0]['timestamp'], "text": text})
    
    return chunks

def chunk_plain_text(text, chars_per_chunk=4000):
    """Chunk plain text (no timestamps) into pieces, labeled by position."""
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    for i in range(0, len(text), chars_per_chunk):
        piece = text[i:i+chars_per_chunk]
        label = f"Part {i // chars_per_chunk + 1}"
        chunks.append({"timestamp": label, "text": piece})
    return chunks

def extract_text_from_pdf(file_bytes):
    import io
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        # Fall back to OCR for scanned/image-only PDFs
        full_text = extract_text_from_pdf_via_ocr(file_bytes)

    if not full_text:
        raise ValueError("Could not extract any text from this PDF, even with OCR.")
    return full_text

def extract_text_from_pdf_via_ocr(file_bytes):
    import io
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        raise ValueError(
            "This PDF appears to be scanned (no selectable text), and OCR libraries are not installed. "
            "Run: pip install pdf2image pytesseract (and install Tesseract OCR + Poppler on your system)."
        )

    try:
        images = convert_from_bytes(file_bytes)
    except Exception as e:
        raise ValueError(f"OCR failed to render PDF pages: {str(e)}. Make sure Poppler is installed.")

    text_parts = []
    for img in images:
        try:
            page_text = pytesseract.image_to_string(img)
        except Exception as e:
            raise ValueError(f"OCR failed: {str(e)}. Make sure Tesseract OCR is installed and on PATH.")
        text_parts.append(page_text)

    return "\n".join(text_parts).strip()

def extract_text_from_image(file_bytes):
    import io
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ValueError("Image OCR requires: pip install pillow pytesseract (and install Tesseract OCR on your system).")

    try:
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
    except Exception as e:
        raise ValueError(f"OCR failed: {str(e)}. Make sure Tesseract OCR is installed and on PATH.")

    text = text.strip()
    if not text:
        raise ValueError("No readable text found in this image.")
    return text

def extract_text_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts = []
            self.skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "footer", "header"):
                self.skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "footer", "header"):
                self.skip = False

        def handle_data(self, data):
            if not self.skip:
                cleaned = data.strip()
                if cleaned:
                    self.texts.append(cleaned)

    parser = TextExtractor()
    parser.feed(resp.text)
    full_text = " ".join(parser.texts)
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    if len(full_text) < 100:
        raise ValueError("Could not extract meaningful text from this URL.")

    return full_text

def transcribe_audio_with_groq(file_path):
    with open(file_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )

    entries = []
    if hasattr(transcription, "segments") and transcription.segments:
        for seg in transcription.segments:
            start = seg.get("start", 0) if isinstance(seg, dict) else seg.start
            text = seg.get("text", "") if isinstance(seg, dict) else seg.text
            entries.append({"timestamp": format_time(start), "text": text.strip()})
    else:
        entries = [{"timestamp": "00:00", "text": transcription.text}]

    return entries

def call_groq(prompt, max_tokens):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def extract_notes_with_groq(chunks, detail_level):
    detail_instructions = {
        "Brief": "2-3 bullet points per section only. No sub-points. Very short.",
        "Standard": "4-6 bullet points per section with one sentence explanation each.",
        "Detailed": "8-10 bullet points per section with sub-points, examples, and full explanations."
    }
    max_tokens = {"Brief": 600, "Standard": 1000, "Detailed": 1500}

    # Split chunks into 2 halves
    mid = len(chunks) // 2
    first_half = chunks[:mid] if mid > 0 else chunks
    second_half = chunks[mid:] if mid > 0 else []

    def make_notes_prompt(half_chunks, part_label):
        text = "\n\n".join([f"[{c['timestamp']}] {c['text']}" for c in half_chunks])
        return f"""You are an expert note-taker. Analyze this transcript ({part_label} of the video).

{detail_instructions[detail_level]}

Transcript:
{text}

Format as:
## 🗂️ Notes - {part_label}
[topic sections with bullet points and timestamps]
"""

    notes_part1 = call_groq(make_notes_prompt(first_half, "Part 1"), max_tokens[detail_level])
    notes_part2 = call_groq(make_notes_prompt(second_half, "Part 2"), max_tokens[detail_level]) if second_half else ""

    # Final summary call
    summary_detail = {
        "Brief": "Write 2-3 sentences summary and 3 action items.",
        "Standard": "Write 4-5 sentences summary and 5 action items with one sentence each.",
        "Detailed": "Write 7-8 sentences summary and 8-10 action items with full explanations."
    }
    all_notes = notes_part1 + "\n\n" + notes_part2
    summary_prompt = f"""Based on these video notes, write a summary and action items.

{summary_detail[detail_level]}

Notes:
{all_notes[:3000]}

Format as:
## 📋 Summary
[summary]

## ✅ Action Items
[numbered list]

## ⏱️ Key Timestamps
[8 most important moments from the notes above]
"""
    summary = call_groq(summary_prompt, max_tokens[detail_level])

    return f"{summary}\n\n{all_notes}"

def time_to_seconds(t):
    parts = t.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

def filter_entries_by_time(entries, start_sec, end_sec):
    def ts_to_sec(ts):
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    return [e for e in entries if start_sec <= ts_to_sec(e['timestamp']) <= end_sec]

def process_youtube(url, detail_level, use_full_video, start_time, end_time):
    video_id = extract_video_id(url.strip())
    entries = get_transcript_yt_dlp(video_id)

    if not use_full_video and (start_time or end_time):
        start_sec = time_to_seconds(start_time) if start_time else 0
        end_sec = time_to_seconds(end_time) if end_time else 99999
        entries = filter_entries_by_time(entries, start_sec, end_sec)
        if not entries:
            raise ValueError("No transcript found in the specified time range. Please check your start/end times.")
        time_range = f"{start_time or '00:00'} → {end_time or 'end'}"
    else:
        time_range = "Full video"

    chunks = chunk_entries(entries)
    stats = f"⏱️ Range: {time_range} | Segments: {len(entries)} | Sections: {len(chunks)}"
    notes = extract_notes_with_groq(chunks, detail_level)

    history = load_history()
    history.insert(0, {
        "url": url.strip(),
        "source_type": "YouTube",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": notes,
        "stats": stats
    })
    save_history(history[:20])
    return stats, notes, chunks

def process_pdf(file_bytes, filename, detail_level):
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_plain_text(text)
    stats = f"📄 PDF: {filename} | Characters: {len(text):,} | Sections: {len(chunks)}"
    notes = extract_notes_with_groq(chunks, detail_level)

    history = load_history()
    history.insert(0, {
        "url": filename,
        "source_type": "PDF",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": notes,
        "stats": stats
    })
    save_history(history[:20])
    return stats, notes, chunks

def process_image(file_bytes, filename, detail_level):
    text = extract_text_from_image(file_bytes)
    chunks = chunk_plain_text(text)
    stats = f"🖼️ Image: {filename} | Characters: {len(text):,} | Sections: {len(chunks)}"
    notes = extract_notes_with_groq(chunks, detail_level)

    history = load_history()
    history.insert(0, {
        "url": filename,
        "source_type": "Image",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": notes,
        "stats": stats
    })
    save_history(history[:20])
    return stats, notes, chunks


def process_article_url(url, detail_level):
    text = extract_text_from_url(url.strip())
    chunks = chunk_plain_text(text)
    stats = f"🔗 Article | Characters: {len(text):,} | Sections: {len(chunks)}"
    notes = extract_notes_with_groq(chunks, detail_level)

    history = load_history()
    history.insert(0, {
        "url": url.strip(),
        "source_type": "Article",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": notes,
        "stats": stats
    })
    save_history(history[:20])
    return stats, notes, chunks

def process_audio_file(file_path, filename, detail_level):
    entries = transcribe_audio_with_groq(file_path)
    chunks = chunk_entries(entries, chunk_size=20)
    stats = f"🎙️ Audio: {filename} | Segments: {len(entries)} | Sections: {len(chunks)}"
    notes = extract_notes_with_groq(chunks, detail_level)

    history = load_history()
    history.insert(0, {
        "url": filename,
        "source_type": "Audio/Video",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": notes,
        "stats": stats
    })
    save_history(history[:20])
    return stats, notes, chunks

def find_relevant_chunks(question, chunks, top_k=4):
    """Simple keyword-overlap retrieval over transcript chunks."""
    stopwords = {"the","a","an","is","are","was","were","what","when","where","who",
                  "how","why","does","do","did","of","in","on","at","to","for","and",
                  "or","this","that","it","about","explain","tell","me","video"}
    q_words = set(re.findall(r"\w+", question.lower())) - stopwords

    scored = []
    for c in chunks:
        c_words = set(re.findall(r"\w+", c['text'].lower()))
        score = len(q_words & c_words)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for score, c in scored[:top_k] if score > 0]

    if not top_chunks:
        top_chunks = [c for _, c in scored[:top_k]]

    return top_chunks

def answer_question(question, chunks):
    relevant = find_relevant_chunks(question, chunks)
    context = "\n\n".join([f"[{c['timestamp']}] {c['text']}" for c in relevant])

    prompt = f"""You are answering a question about a video based on its transcript excerpts below.

Transcript excerpts:
{context}

Question: {question}

Answer the question based ONLY on the transcript excerpts above. If the answer isn't in the excerpts, say so.
Mention the relevant timestamp(s) where this is discussed, formatted like [MM:SS].
Keep the answer concise (3-5 sentences).
"""
    return call_groq(prompt, 400)

# ============== UI ==============

# Splash screen on first load
if "app_loaded" not in st.session_state:
    st.session_state.app_loaded = False

if not st.session_state.app_loaded:
    splash = st.empty()
    with splash.container():
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        #MainMenu, footer, header {visibility: hidden;}
        .splash-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 70vh;
            animation: fadeIn 0.6s ease-in;
        }
        .splash-logo {
            font-size: 5rem;
            animation: popIn 0.7s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .splash-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 1rem;
            font-family: 'Inter', sans-serif;
            opacity: 0;
            animation: fadeUp 0.6s ease-out 0.3s forwards;
        }
        .splash-sub {
            font-size: 0.95rem;
            color: #7f77dd;
            margin-top: 0.4rem;
            font-family: 'JetBrains Mono', monospace;
            opacity: 0;
            animation: fadeUp 0.6s ease-out 0.5s forwards;
        }
        @keyframes popIn {
            0% { transform: scale(0.3); opacity: 0; }
            60% { transform: scale(1.15); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
        <div class="splash-wrap">
            <div class="splash-logo">🎬</div>
            <div class="splash-title">Video Note Extractor</div>
            <div class="splash-sub">Powered by LLaMA 3.3 70B</div>
        </div>
        """, unsafe_allow_html=True)

    import time
    time.sleep(1.4)
    st.session_state.app_loaded = True
    splash.empty()
    st.rerun()


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #0e1117;
}

/* Hero header */
.hero-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #232938 100%);
    border: 1px solid #2d3548;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #9aa4b8;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(124, 119, 221, 0.15);
    color: #b3aef0;
    border: 1px solid rgba(124, 119, 221, 0.3);
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Section card */
.section-card {
    background: #161b26;
    border: 1px solid #2d3548;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7f77dd;
    margin-bottom: 0.8rem;
}

/* Stats pill bar */
.stats-pill {
    background: rgba(29, 158, 117, 0.1);
    border: 1px solid rgba(29, 158, 117, 0.3);
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    color: #5dcaa5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* History items */
.history-empty {
    color: #6b7280;
    font-size: 0.9rem;
    text-align: center;
    padding: 1.5rem 0;
}

/* Notes output container */
.notes-output {
    background: #0f1319;
    border: 1px solid #2d3548;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
}
.notes-output h2 {
    color: #85b7eb !important;
    border-bottom: 1px solid #2d3548;
    padding-bottom: 0.5rem;
    margin-top: 1.2rem;
    font-size: 1.15rem;
}
.notes-output h2:first-child {
    margin-top: 0;
}

/* Q&A bubbles */
.qa-question {
    background: rgba(55, 138, 221, 0.1);
    border-left: 3px solid #378add;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin-top: 0.8rem;
    font-weight: 600;
    color: #85b7eb;
}
.qa-answer {
    background: #161b26;
    border-radius: 0 0 8px 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    color: #d1d5db;
    border: 1px solid #2d3548;
    border-top: none;
}

/* Buttons */
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #534ab7 0%, #378add 100%) !important;
    border: none !important;
}
.stButton button[kind="primary"]:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Inputs */
.stTextInput input, .stTextArea textarea {
    border-radius: 8px !important;
    border-color: #2d3548 !important;
    background-color: #0f1319 !important;
}

/* Radio buttons horizontal */
div[role="radiogroup"] {
    gap: 0.5rem;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">🎬 Universal Note Extractor</div>
    <p class="hero-subtitle">Turn YouTube videos, articles, PDFs, or audio/video files into organized notes, key points, and action items</p>
    <span class="hero-badge">⚡ LLaMA 3.3 70B + Whisper via Groq</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2], gap="medium")

with col1:
    with st.container(border=True):
        st.markdown('<div class="section-label">📥 Input</div>', unsafe_allow_html=True)

        source_type = st.radio(
            "Source type",
            ["YouTube", "Article URL", "PDF", "Image (Screenshot)", "Audio/Video File"],
            horizontal=True,
            label_visibility="collapsed"
        )

        url = ""
        use_full_video = True
        start_time = ""
        end_time = ""
        uploaded_pdf = None
        uploaded_audio = None
        uploaded_image = None

        if source_type == "YouTube":
            url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")

            st.markdown("**⏱️ Time Range**")
            use_full_video = st.checkbox("Extract full video", value=True)

            if not use_full_video:
                tc1, tc2 = st.columns(2)
                with tc1:
                    start_time = st.text_input("Start time", placeholder="e.g. 02:30")
                with tc2:
                    end_time = st.text_input("End time", placeholder="e.g. 08:00")
                st.caption("Format: MM:SS or HH:MM:SS")

        elif source_type == "Article URL":
            url = st.text_input("Article/Blog URL", placeholder="https://example.com/article", label_visibility="collapsed")

        elif source_type == "PDF":
            uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
            st.caption("If the PDF is scanned, OCR (Tesseract) will be used automatically.")

        elif source_type == "Image (Screenshot)":
            uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp", "bmp"], label_visibility="collapsed")
            st.caption("Upload a screenshot of notes, slides, or text — OCR will extract the content.")

        elif source_type == "Audio/Video File":
            uploaded_audio = st.file_uploader("Upload audio or video", type=["mp3", "wav", "m4a", "mp4", "webm"], label_visibility="collapsed")
            st.caption("Supports podcasts, lectures, meeting recordings (max ~25MB)")


        st.markdown("**Detail Level**")
        detail_level = st.radio("Detail Level", ["Brief", "Standard", "Detailed"], index=1, horizontal=True, label_visibility="collapsed")

        extract_btn = st.button("⚡ Extract Notes", type="primary", use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown('<div class="section-label">📂 History</div>', unsafe_allow_html=True)

        history = load_history()
        if history:
            choices = [f"{h['timestamp']} — [{h.get('source_type','?')}] {h['url'][:30]}" for h in history]
            selected = st.selectbox("Past extractions", choices, label_visibility="collapsed")
            if st.button("Load Selected", use_container_width=True):
                for i, h in enumerate(history):
                    if choices[i] == selected:
                        st.session_state.loaded_url = h['url']
                        st.session_state.loaded_notes = h['notes']
                        st.session_state.loaded_stats = h.get('stats', '')
                        st.session_state.pop('loaded_chunks', None)
                        st.session_state.qa_history = []
                        st.rerun()
        else:
            st.markdown('<div class="history-empty">No extractions yet.<br>Your history will appear here.</div>', unsafe_allow_html=True)

if extract_btn:
    try:
        if source_type == "YouTube":
            if not url:
                raise ValueError("Please enter a YouTube URL.")
            with st.spinner("Fetching transcript and generating notes..."):
                stats, notes, chunks = process_youtube(url, detail_level, use_full_video, start_time, end_time)

        elif source_type == "Article URL":
            if not url:
                raise ValueError("Please enter an article URL.")
            with st.spinner("Fetching article and generating notes..."):
                stats, notes, chunks = process_article_url(url, detail_level)

        elif source_type == "PDF":
            if not uploaded_pdf:
                raise ValueError("Please upload a PDF file.")
            with st.spinner("Extracting text from PDF and generating notes..."):
                stats, notes, chunks = process_pdf(uploaded_pdf.read(), uploaded_pdf.name, detail_level)

        elif source_type == "Image (Screenshot)":
            if not uploaded_image:
                raise ValueError("Please upload an image.")
            with st.spinner("Reading text from image (OCR) and generating notes..."):
                stats, notes, chunks = process_image(uploaded_image.read(), uploaded_image.name, detail_level)

        elif source_type == "Audio/Video File":
            if not uploaded_audio:
                raise ValueError("Please upload an audio or video file.")
            with st.spinner("Transcribing audio with Whisper and generating notes..."):
                temp_path = f"/tmp/{uploaded_audio.name}" if os.path.exists("/tmp") else uploaded_audio.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_audio.read())
                stats, notes, chunks = process_audio_file(temp_path, uploaded_audio.name, detail_level)
                os.remove(temp_path)

        st.session_state.loaded_stats = stats
        st.session_state.loaded_notes = notes
        st.session_state.loaded_chunks = chunks
        st.session_state.qa_history = []
    except Exception as e:
        st.error(f"⚠️ {str(e)}")


if "loaded_stats" in st.session_state and st.session_state.loaded_stats:
    st.markdown(f'<div class="stats-pill">{st.session_state.loaded_stats}</div>', unsafe_allow_html=True)

if "loaded_notes" in st.session_state and st.session_state.loaded_notes:
    st.markdown('<div class="notes-output">', unsafe_allow_html=True)
    st.markdown(st.session_state.loaded_notes)
    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download Notes (.md)",
        data=st.session_state.loaded_notes,
        file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

    st.divider()
    with st.container(border=True):
        st.markdown('<div class="section-label">💬 Ask About This Video</div>', unsafe_allow_html=True)

        if "loaded_chunks" not in st.session_state:
            st.caption("Q&A is only available right after extraction (not for loaded history).")
        else:
            qa_col1, qa_col2 = st.columns([4, 1])
            with qa_col1:
                question = st.text_input("Your question", placeholder="e.g. What did the speaker say about pricing?", label_visibility="collapsed")
            with qa_col2:
                ask_btn = st.button("Ask", use_container_width=True)

            if "qa_history" not in st.session_state:
                st.session_state.qa_history = []

            if ask_btn and question:
                with st.spinner("Searching transcript..."):
                    try:
                        answer = answer_question(question, st.session_state.loaded_chunks)
                        st.session_state.qa_history.insert(0, (question, answer))
                    except Exception as e:
                        st.error(f"⚠️ {str(e)}")

            for q, a in st.session_state.qa_history:
                st.markdown(f'<div class="qa-question">Q: {q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="qa-answer">{a}</div>', unsafe_allow_html=True)
