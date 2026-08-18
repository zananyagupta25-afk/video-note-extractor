# 🎬 Universal Note Extractor

An AI-powered tool that converts YouTube videos, articles, PDFs, images, and audio/video files into structured notes, key timestamps, and action items — instantly.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)]()
[![Groq](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-violet?style=flat-square)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)]()

---

## ✨ Features

- **Multi-Source Input** — Extract notes from YouTube videos, article URLs, PDFs, images (screenshots), and audio/video files
- **Organized Notes** — Automatically segments content into topic sections with bullet points
- **Key Timestamps** — Identifies the most important moments with precise timestamps (for YouTube/audio sources)
- **Action Items** — Extracts tasks, recommendations, and steps to follow
- **Content Summary** — Generates a concise overview of the entire input
- **Detail Levels** — Choose between Brief, Standard, or Detailed output
- **Automatic OCR Fallback** — Scanned PDFs are automatically run through OCR (Tesseract) if no selectable text is found
- **Time-Range Extraction** — For YouTube videos, extract notes from a specific start/end time instead of the full video
- **Ask About This Video (Q&A)** — Ask follow-up questions about the extracted content; relevant transcript sections are retrieved and answered with cited timestamps
- **Note History** — Saves your last 20 extractions locally for quick access
- **Download Notes** — Export as a clean `.md` file

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| UI | Streamlit |
| LLM (notes & summaries) | LLaMA 3.3 70B via Groq API |
| Speech-to-Text | Whisper-large-v3-turbo via Groq |
| YouTube Transcripts | yt-dlp |
| PDF Text Extraction | pypdf / PyPDF2 |
| OCR (scanned PDFs & images) | Tesseract (pytesseract) + pdf2image |
| Web Scraping (articles) | requests + Python's built-in HTMLParser |
| Q&A Retrieval | Keyword-overlap matching over transcript chunks |
| History Storage | Local JSON file |

---

## 🚀 Getting Started

### 1. Clone the repo

```
git clone https://github.com/zananyagupta25-afk/video-note-extractor
cd video-note-extractor
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

**System dependencies:** OCR and audio/video features also require Tesseract OCR and Poppler installed on your system.

- **Windows:** Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases), then add both to your PATH.
- **macOS:** `brew install tesseract poppler`
- **Linux:** `sudo apt install tesseract-ocr poppler-utils`

### 3. Set your Groq API key

```
# Windows (PowerShell)
$env:GROQ_API_KEY = "your_groq_api_key_here"

# Linux / macOS
export GROQ_API_KEY="your_groq_api_key_here"
```

Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Run the app

```
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

---

## 📋 How It Works

```
Input (YouTube URL / Article URL / PDF / Image / Audio-Video)
    │
    ▼
Source-Specific Extraction
    • YouTube      → yt-dlp fetches auto-captions (.vtt), parsed into timestamped entries
    • Article URL  → requests + HTMLParser extracts readable text
    • PDF          → pypdf extracts text; falls back to OCR if scanned
    • Image        → pytesseract (Tesseract OCR) extracts text
    • Audio/Video   → Groq Whisper transcribes with timestamped segments
    │
    ▼
Chunking
(splits content into manageable sections, deduplicating repeated captions)
    │
    ▼
LLaMA 3.3 70B via Groq (map-reduce)
    • Chunks split into two halves, each summarized independently ("map")
    • Combined and synthesized into a final summary + action items ("reduce")
    │
    ▼
Structured Output
(Summary · Notes · Timestamps · Action Items)
    │
    ▼
Ask About This Video (optional)
    • Keyword-overlap search finds the most relevant chunks for your question
    • LLaMA answers using only those chunks, citing timestamps
```

---

## 🖼️ Demo

Select a source type → provide input (URL/file) → choose detail level → click **Extract Notes**

Supported YouTube URL formats:

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

Supported file uploads:

- **PDF:** `.pdf`
- **Image:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- **Audio/Video:** `.mp3`, `.wav`, `.m4a`, `.mp4`, `.webm` (max ~25MB)

---

## 📁 Project Structure

```
video-note-extractor/
├── app.py              # Main application (UI + logic)
├── requirements.txt    # Python dependencies
├── note_history.json   # Auto-generated note history (gitignored)
└── README.md
```

---

## ⚠️ Limitations

- YouTube extraction only works with videos that have English auto-generated or manual captions
- Very long videos (3h+) may hit Groq rate limits — try "Brief" detail level for those
- Videos with disabled transcripts are not supported
- Q&A retrieval uses keyword overlap, not semantic/embedding-based search — it can miss relevant content phrased differently than the question
- Note history is a local JSON file (not a database), so it isn't safe for concurrent multi-user use
- On some cloud hosting platforms, YouTube may rate-limit or block requests from cloud IP ranges, which can cause `yt-dlp` extraction to fail even though it works locally

---

## 🔮 Possible Extensions

- [ ] Embedding-based (semantic) RAG instead of keyword-overlap retrieval
- [ ] Multi-language transcript support
- [ ] Batch processing for playlists
- [ ] Export to PDF / Notion / Obsidian
- [ ] Speaker diarization for meeting/lecture recordings
- [ ] Move history storage from JSON to a proper database (SQLite/Postgres)

---

## 👩‍💻 Author

**Ananya Gupta** — B.Tech AI & ML, Galgotias College of Engineering and Technology
[GitHub](https://github.com/zananyagupta25-afk)

---

## 📄 License

MIT License — free to use and modify.
