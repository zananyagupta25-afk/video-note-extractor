# 🎬 Video Note Extractor

An AI-powered tool that converts YouTube videos into structured notes, key timestamps, and action items — instantly.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Groq](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-violet?style=flat-square)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

- **Organized Notes** — Automatically segments video content into topic sections with bullet points
- **Key Timestamps** — Identifies the 5–8 most important moments in the video with precise timestamps
- **Action Items** — Extracts tasks, recommendations, and steps to follow
- **Video Summary** — Generates a concise 3–4 sentence overview of the entire video
- **Detail Levels** — Choose between Brief, Standard, or Detailed output
- **Note History** — Saves your last 20 extractions locally for quick access
- **Download Notes** — Export as a clean `.md` file

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B (via Groq API) |
| Transcript Extraction | `youtube-transcript-api` |
| UI | Gradio |
| Language | Python 3.9+ |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/zananyagupta25-afk/video-note-extractor
cd video-note-extractor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY = "your_groq_api_key_here"

# Linux / macOS
export GROQ_API_KEY="your_groq_api_key_here"
```

Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## 📋 How It Works

```
YouTube URL
    │
    ▼
youtube-transcript-api
(fetches timestamped transcript)
    │
    ▼
Transcript Chunking
(groups into ~50-line segments with timestamps)
    │
    ▼
LLaMA 3.3 70B via Groq
(summarization + note extraction + action items)
    │
    ▼
Structured Output
(Summary · Notes · Timestamps · Action Items)
```

---

## 🖼️ Demo

Paste any YouTube URL → Select detail level → Click **Extract Notes**

Supported formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

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

- Only works with videos that have English transcripts (auto-generated or manual)
- Very long videos (3h+) may hit Groq rate limits — try "Brief" detail level for those
- Videos with disabled transcripts are not supported

---

## 🔮 Possible Extensions

- [ ] Multi-language transcript support
- [ ] RAG-based Q&A over extracted notes
- [ ] Batch processing for playlists
- [ ] Export to PDF / Notion / Obsidian
- [ ] Speaker diarization for meeting videos

---

## 👩‍💻 Author

**Ananya Gupta** — B.Tech AI & ML, Galgotias College of Engineering and Technology  
[GitHub](https://github.com/zananyagupta25-afk)

---

## 📄 License

MIT License — free to use and modify.
