# app.py
"""
Personalized Learning & Tutoring Platform (single-file app)
- RAG pipeline (FAISS)
- Multi-document, multi-modal ingestion (PDF, .txt, code, images)
- Image OCR to include diagrams/figure text in retrieval
- OpenAI embeddings + OpenAI LLM (GPT)
- DuckDuckGo web fallback (external tool execution)
- Simple per-student profile & adaptive difficulty
- Conversation memory (LangChain)
Run:
    export OPENAI_API_KEY="sk-..."
    python app.py
"""

import os
import glob
import json
import io
import sys
from typing import List, Dict, Any

# --- Libraries for documents, OCR, embeddings, LLM, vector DB, search, memory ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from PIL import Image
import pytesseract

# load .env if present
from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: Please set OPENAI_API_KEY in environment or .env file.")
    sys.exit(1)

MATERIALS_DIR = "materials"       # folder where you place PDFs, txt, images, code examples
STUDENT_DB = "students.json"      # simple JSON file to store student profiles (skill level, history)
FAISS_INDEX_PATH = "faiss_index"  # optional persistence (we store in-memory; you may extend)

# Embedding model (multi-lingual support)
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI multilingual embedding

# LLM model
LLM_MODEL = "gpt-4o-mini"  # change to available GPT model like "gpt-4o-mini" or another

# Chunking params
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Retrieval params
TOP_K = 5      # how many chunks to retrieve normally
WEB_TOP = 5    # how many web results to include when falling back

# === Utility: Manage student profiles ===
def load_students_db(path: str = STUDENT_DB) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_students_db(db: Dict[str, Any], path: str = STUDENT_DB):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def get_or_create_student(student_id: str, db: Dict[str, Any]) -> Dict[str, Any]:
    if student_id not in db:
        db[student_id] = {
            "id": student_id,
            "skill_level": "beginner",    # beginner | intermediate | advanced
            "history": []                 # list of {"query":..., "answer":..., "correct":bool}
        }
    return db[student_id]

def update_student_skill(student: Dict[str, Any], correct: bool):
    # very simple heuristic: promote/demote by example
    if correct:
        if student["skill_level"] == "beginner":
            student["skill_level"] = "intermediate"
        elif student["skill_level"] == "intermediate":
            student["skill_level"] = "advanced"
    else:
        if student["skill_level"] == "advanced":
            student["skill_level"] = "intermediate"
        elif student["skill_level"] == "intermediate":
            student["skill_level"] = "beginner"

# === Document ingestion: load PDFs, TXT, code files, images ===
print(">> Scanning materials folder for documents...")
docs: List[Document] = []

# 1) PDFs -> use PyPDFLoader
pdf_paths = glob.glob(os.path.join(MATERIALS_DIR, "*.pdf"))
for p in sorted(pdf_paths):
    try:
        loader = PyPDFLoader(p)
        loaded = loader.load()
        # add filename metadata
        for d in loaded:
            if d.metadata is None:
                d.metadata = {}
            d.metadata["source"] = os.path.basename(p)
        docs.extend(loaded)
        print(f"  - Loaded PDF: {p} ({len(loaded)} pages)")
    except Exception as e:
        print(f"  ! Failed to load PDF {p}: {e}")

# 2) TXT -> TextLoader
txt_paths = glob.glob(os.path.join(MATERIALS_DIR, "*.txt"))
for p in sorted(txt_paths):
    try:
        loader = TextLoader(p, encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            if d.metadata is None:
                d.metadata = {}
            d.metadata["source"] = os.path.basename(p)
        docs.extend(loaded)
        print(f"  - Loaded TXT: {p}")
    except Exception as e:
        print(f"  ! Failed to load TXT {p}: {e}")

# 3) Code examples (.py, .java, .cpp, .md) treat as text
code_paths = []
for ext in ("*.py", "*.java", "*.cpp", "*.md"):
    code_paths += glob.glob(os.path.join(MATERIALS_DIR, ext))
for p in sorted(set(code_paths)):
    try:
        loader = TextLoader(p, encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            if d.metadata is None:
                d.metadata = {}
            d.metadata["source"] = os.path.basename(p)
            d.metadata["type"] = "code"
        docs.extend(loaded)
        print(f"  - Loaded code/text: {p}")
    except Exception as e:
        print(f"  ! Failed to load code/text {p}: {e}")

# 4) Images (.png, .jpg, .jpeg) -> OCR text included + metadata to reference image
image_paths = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    image_paths += glob.glob(os.path.join(MATERIALS_DIR, ext))
for p in sorted(set(image_paths)):
    try:
        img = Image.open(p).convert("RGB")
        # run OCR to capture text/diagram labels (this makes images searchable)
        ocr_text = pytesseract.image_to_string(img).strip()
        page_content = f"[IMAGE: {os.path.basename(p)}]\n"
        if ocr_text:
            page_content += "\nOCR_TEXT:\n" + ocr_text
        # create a Document for the image with metadata
        d = Document(page_content=page_content, metadata={"source": os.path.basename(p), "type": "image", "path": p})
        docs.append(d)
        print(f"  - Loaded image (OCR): {p} (OCR chars: {len(ocr_text)})")
    except Exception as e:
        print(f"  ! Failed to load image {p}: {e}")

if not docs:
    print("No documents loaded. Please add files into the 'materials' folder and re-run.")
    sys.exit(1)

# === Chunking documents ===
print(">> Chunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)
print(f"  - Created {len(chunks)} chunks from materials")

# === Build embeddings + FAISS vectorstore ===
print(">> Building embeddings and FAISS vectorstore... (this may take some time)")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

# FAISS.from_documents uses embeddings internally
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
print("  - FAISS vectorstore ready")

# === Initialize LLM, memory, and web search tool ===
llm = OpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
web = DuckDuckGoSearchAPIWrapper()

# === Helper: build context + structured prompt tuned for student level ===
def build_prompt(student_profile: Dict[str, Any], query: str, retrieved_docs: List[Document], include_web: bool=False, web_results: str = "") -> str:
    # Compose context (include sources and small snippets)
    context_items = []
    for i, d in enumerate(retrieved_docs, 1):
        src = d.metadata.get("source", "unknown")
        typ = d.metadata.get("type", "text")
        snippet = d.page_content.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        context_items.append(f"[{i}] source: {src} type: {typ}\n{snippet}")
    context = "\n\n".join(context_items)
    if include_web and web_results:
        context += "\n\n[WEB_RESULTS]\n" + web_results

    # Adapt explanation/difficulty style
    style_map = {
        "beginner": "Explain step-by-step in simple language, give analogies, and include 1-2 simple exercises.",
        "intermediate": "Provide concise explanations, some derivations or examples, and 2-3 practice tasks with moderate difficulty.",
        "advanced": "Provide precise, technical explanation, point to edge-cases, and propose 2-3 challenging exercises or proofs."
    }
    instruction = style_map.get(student_profile.get("skill_level", "beginner"), style_map["beginner"])

    prompt = f"""
You are an AI tutor specialized in teaching technical subjects and code. Follow these rules precisely:

Student profile:
- id: {student_profile.get('id')}
- skill_level: {student_profile.get('skill_level')}

Task:
- Use the INTERNAL CONTEXT below (from uploaded materials). If the internal context is insufficient, also use the WEB_RESULTS appended.
- Answer the student's question. {instruction}
- When you provide exercises, also show model answers for self-check.
- Cite sources by filename whenever you use a fact from the internal materials.
- If the question asks for code feedback, produce suggested corrected code with short explanation.

INTERNAL CONTEXT:
{context}

Question:
{query}

Answer:
"""
    return prompt

# === Main interactive loop (console) ===
students_db = load_students_db()
print("\n>> Personalized Tutor Ready. Type 'exit' to quit.")
student_id = input("Enter student id (e.g., alice): ").strip() or "default_student"
student = get_or_create_student(student_id, students_db)
save_students_db(students_db)  # ensure exists

while True:
    user_q = input(f"\n[{student_id}/{student['skill_level']}] Ask a question (or type 'exit'): ").strip()
    if user_q.lower() in ("exit", "quit", "q"):
        print("Goodbye!")
        break
    if not user_q:
        continue

    # 1) Convert question to semantic query: (we simply use the question directly, but could transform)
    semantic_query = user_q  # placeholder for more complex question-to-query translation

    # 2) Retrieve from FAISS
    retrieved = retriever.get_relevant_documents(semantic_query)
    # Quick heuristic: if retrieved content is very small, do web fallback
    combined_text_len = sum(len(d.page_content.strip()) for d in retrieved)
    use_web = combined_text_len < 200  # threshold; tune as needed

    web_text = ""
    if use_web:
        print("  - Internal content limited → performing web search fallback")
        try:
            web_results_list = web.results(semantic_query, max_results=WEB_TOP)
            # DuckDuckGoSearchAPIWrapper exposes .results or .run; handle both
            if isinstance(web_results_list, list):
                web_text = "\n\n".join([f"{r['title']}\n{r.get('snippet', '')}\n{r.get('link','')}" if isinstance(r, dict)
                                       else str(r) for r in web_results_list])
            else:
                web_text = str(web_results_list)
        except Exception as e:
            print(f"  ! Web search failed: {e}")
            web_text = ""

    # 3) Build prompt tailored to student skill
    prompt = build_prompt(student, user_q, retrieved, include_web=bool(web_text), web_results=web_text)

    # 4) Call LLM
    try:
        print("  - Asking LLM...")
        resp = llm.invoke(prompt)
        answer = resp.content
    except Exception as e:
        answer = f"LLM request failed: {e}"

    # 5) Display answer with simple source listing
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Sources (top results) ===")
    for d in retrieved[:min(5, len(retrieved))]:
        print(f" - {d.metadata.get('source','unknown')} (type: {d.metadata.get('type','text')})")

    # 6) Simple feedback loop: ask if student says answer was correct
    feedback = input("\nWas this helpful/correct? (y/n/skip): ").strip().lower()
    if feedback == "y":
        correct = True
    elif feedback == "n":
        correct = False
    else:
        correct = None

    # 7) Update student history & possibly skill level
    entry = {"query": user_q, "answer": answer, "correct": correct}
    student["history"].append(entry)
    if correct is True:
        update_student_skill(student, True)
    elif correct is False:
        update_student_skill(student, False)
    save_students_db(students_db)
    print(f"  - Student skill now: {student['skill_level']}")
