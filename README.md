🎓 Universal Note Extractor
Turn videos, audio, PDFs, images, or plain text into clean, structured notes in one click, powered by Whisper and LLaMA 3.3 70B.

Python Streamlit Groq Whisper License

Report a Bug

Home

📌 Overview
Revising from long lectures, tutorials, and documents is slow. Universal Note Extractor accepts content from five input types, converts it to text, and uses LLaMA 3.3 70B (via Groq) to generate structured, easy-to-revise notes. Long content is processed section by section, so notes cover the full material and not just the beginning.

✨ Key Features
🎥 Five input types: YouTube link, video/audio upload, PDF, image, and text or URL
🗣️ Accurate transcription of speech using OpenAI Whisper
🎚️ Adjustable detail level: Brief, Standard, or Detailed notes
🔀 Optional flowchart to visualise processes
🧠 Optional concept map to show how ideas connect
🖼️ Optional topic images to enrich the notes
📚 Long-document handling: content is split into sections and processed fully
📜 History sidebar to revisit previously generated notes
⚡ Fast inference thanks to the Groq API
🖼️ Screenshots
Choose an input type, detail level, and extras

Input screen

Notes generated from a YouTube lecture

Extraction

Structured output with headings and key points

Notes output

🏗️ How It Works
Architecture

Input: the user picks a source (YouTube, video/audio, PDF, image, or text/URL).
Extraction: Whisper transcribes audio and video; PDFs, images, and text are parsed into plain text.
Chunking: long content is split into sections so nothing is skipped.
Generation: LLaMA 3.3 70B writes structured notes at the chosen detail level, with optional flowchart, concept map, and images.
Output: notes are displayed in the Streamlit app and saved to history.
🧰 Tech Stack
Layer	Tools
Language	Python
LLM	LLaMA 3.3 70B via Groq API
Speech-to-Text	OpenAI Whisper
Frontend	Streamlit
Version Control	Git, GitHub
📁 Project Structure
video-note-extractor/
├── app.py
├── requirements.txt
├── .env.example
├── assets/
│   ├── home.png
│   ├── extraction.png
│   ├── notes.png
│   └── architecture.png
└── README.md
Add your other source files here to match the real repo.

🚀 Getting Started
Prerequisites
Python 3.10+
FFmpeg installed and on your PATH
A free Groq API key
Installation
git clone https://github.com/zananyagupta25-afk/video-note-extractor.git
cd video-note-extractor
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
Configuration
Create a .env file in the project root:

GROQ_API_KEY=your_api_key_here
Run
streamlit run app.py
Open http://localhost:8501 in your browser.

📖 Usage
Select an input type.
Choose the note detail level: Brief, Standard, or Detailed.
Optionally tick Add flowchart, Add concept map, or Add topic images.
Paste a link or upload a file, then click Extract Notes.
Read your notes, or reopen older ones from the History sidebar.
🧗 Challenges and Learnings
Unifying five input formats into one clean text pipeline.
Handling long content that exceeds the model's context limit by processing it in sections.
Deployment: moved the interface to Streamlit for simple hosting and a smoother experience.
🔮 Future Improvements
[ ] Question-answering over the generated notes
[ ] Export notes to PDF / Markdown
[ ] Multi-language transcription and translation
[ ] Timestamped notes linking back to the video
👩‍💻 Author
Ananya Gupta, B.Tech AI & ML LinkedIn · GitHub

📄 License
This project is open source under the MIT License.
