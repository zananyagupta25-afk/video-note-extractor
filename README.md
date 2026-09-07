# 🎬 Universal Note Extractor

Turn video, audio, PDFs, images, or plain text/URLs into clean, structured notes — powered by Groq (Whisper for transcription, LLaMA/GPT-OSS for note generation).

## Features

- **Five input types** — YouTube link, uploaded video/audio, PDF, image (OCR), or pasted text/URL
- **Three note detail levels** — Brief, Standard, Detailed, each controlling how much depth the notes go into
- **Topic-first structure** — notes are organized as:
  1. **Topic Notes** — the source content broken into its own topics, each as a sub-heading with detailed bullets
  2. **Key Points** — a flat list of the most important takeaways across all topics
  3. **Summary** — a short overview, last
- **Optional flowchart** — auto-generated (via Mermaid) only when the content actually describes a step-by-step process; skipped otherwise
- **Optional concept map** — auto-generated (via Mermaid) only when the content has related ideas worth mapping; skipped otherwise
- **Optional topic images** — pulls one relevant photo per topic from the Pexels API
- **Long-document handling** — automatically chunks and condenses very long transcripts/documents via map-reduce summarization to stay within API rate limits
- **History sidebar** — the last 50 generated notes are saved locally and browsable from the sidebar
- **Download as Markdown** — export any generated notes as a `.md` file

## Tech Stack

- **UI**: Streamlit
- **Transcription**: Groq Whisper (`whisper-large-v3`)
- **Note generation**: Groq LLaMA/GPT-OSS (`openai/gpt-oss-120b`)
- **Diagrams**: Mermaid.js (rendered via CDN, no extra dependency)
- **Images**: Pexels API
- **PDF/OCR**: pypdf, pdf2image, pytesseract
- **YouTube audio**: yt-dlp

## Setup

1. Clone the repo and create a virtual environment:
   ```
   git clone https://github.com/zananyagupta25-afk/video-note-extractor.git
   cd video-note-extractor
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_key_here
   PEXELS_API_KEY=your_pexels_key_here
   ```
   - Get a free Groq key at https://console.groq.com/keys
   - Get a free Pexels key at https://www.pexels.com/api/ (only needed if you want topic images — the app runs fine without it, that feature just stays off)

4. Run the app:
   ```
   streamlit run app.py
   ```

## Usage

1. Pick an input type and choose your note detail level (Brief / Standard / Detailed).
2. Optionally tick **Add flowchart**, **Add concept map**, and/or **Add topic images**.
3. Provide your source (link, file, or text) and click **Extract Notes**.
4. Review the generated notes, diagrams, and images, then download the notes as Markdown if needed.

## Notes on rate limits

Long documents are automatically chunked and condensed before the final note-generation call to stay within Groq's free-tier rate limits. Diagram and image generation each add one extra API call per option enabled, so enabling all three roughly triples the calls used per note.