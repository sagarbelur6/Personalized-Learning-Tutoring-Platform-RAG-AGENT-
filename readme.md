# Personalized Learning & Tutoring Platform (RAG + FAISS + OpenAI)

This project provides a console-based AI tutor that uses Retrieval-Augmented Generation (RAG) to answer student questions using a knowledge base of textbooks, PDFs, code examples and images.

## Features

- Ingest multi-modal learning materials (PDFs, text, code files, images).
- Use FAISS for vector search over content chunks.
- OpenAI embeddings (multilingual) and OpenAI GPT for generation.
- Simple student profile + adaptive difficulty (beginner/intermediate/advanced).
- Web fallback (DuckDuckGo) for missing information.
- Conversation memory for follow-up questions.
- Simple console UI for interactive tutoring.

## Setup

1. Clone this repo and move into folder:
```bash
git clone <repo-url>
cd <repo-folder>
Create materials/ folder and put your learning content there:

markdown
Copy code
materials/
  - chapter1.pdf
  - algebra_notes.txt
  - example_code.py
  - diagram1.png
Install system dependencies:

Install Tesseract OCR (optional, only if you want image OCR):

macOS (Homebrew): brew install tesseract

Ubuntu/Debian: sudo apt-get install tesseract-ocr

Windows: install Tesseract from https://github.com/tesseract-ocr/tesseract

Install Python dependencies:

bash
Copy code
pip install -r requirements.txt
Add your OpenAI API key in .env:

ini
Copy code
OPENAI_API_KEY=sk-...
Run the app:

bash
Copy code
python app.py
Follow prompts:

Enter a student id (e.g., alice). The system maintains a small JSON file students.json with profile and history.

Ask questions. The system will fetch relevant content, optionally search the web, and answer in a style adapted to the student's skill level.

Give feedback (y/n) to help adapt the difficulty.
