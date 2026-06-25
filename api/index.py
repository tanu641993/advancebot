import os
os.environ["GOOGLE_API_VERSION"] = "v1"
os.environ["GOOGLE_API_ENDPOINT"] = "https://generativelanguage.googleapis.com/v1/"

import hashlib
import logging
import re
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import deque

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.genai import types

try:
    import pandas as pd
except ImportError:
    pd = None

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "index.html"
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY")
    raise RuntimeError("Missing GOOGLE_API_KEY")

app = FastAPI(title="Gemini RAG Data API", version="1.0")

# ─── Helper classes and functions ─────────────────────────────────────────────

class LRUCache:
    def __init__(self, max_size: int = 200):
        self.cache: Dict[str, str] = {}
        self.access_order = deque()
        self.max_size = max_size

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            lru_key = self.access_order.popleft()
            del self.cache[lru_key]
        self.cache[key] = value
        self.access_order.append(key)

    def clear(self) -> None:
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        return len(self.cache)

GLOBAL_STATE = {
    "file_hash": None,
    "filename": None,
    "retriever": None,
    "prompt_cache": LRUCache(max_size=200),
    "all_chunks": [],
    "chunk_metadata": [],
    "embeddings_cache": {},
    "query_history": deque(maxlen=50),
    "raw_data": None,
}

class GeminiEmbeddings:
    def __init__(self, model="gemini-embedding-2"):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = model

    def __call__(self, text):
        return self.embed_query(text)

    def embed_documents(self, texts):
        result = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="retrieval_document"),
            )
            result.append(response.embeddings[0].values)
        return result

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="retrieval_query"),
        )
        return response.embeddings[0].values

class ChatGemini:
    def __init__(self, model="gemini-2.5-flash", temperature=0.0, max_output_tokens=2048):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def invoke(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        )
        return response.text

llm = ChatGemini(model="gemini-2.5-flash", temperature=0.0, max_output_tokens=2048)

# ─── Helper functions ──────────────────────────────────────────────────────

def compute_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def prompt_hash(p: str) -> str:
    return hashlib.md5(p.encode()).hexdigest()

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    return text.strip()

def deduplicate_chunks(chunks):
    if not chunks:
        return chunks
    seen = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique

def compute_cosine_similarity(vec1, vec2):
    dot = sum(a*b for a,b in zip(vec1, vec2))
    n1 = (sum(a**2 for a in vec1)**0.5) + 1e-9
    n2 = (sum(b**2 for b in vec2)**0.5) + 1e-9
    return dot / (n1 * n2)

def load_excel_file(file_path: str):
    if pd is None:
        raise HTTPException(400, "Pandas not installed")
    xls = pd.ExcelFile(file_path)
    documents = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        for idx, row in df.iterrows():
            row_text = f"Sheet: {sheet} | Row {idx+2}\n"
            for col, val in row.items():
                row_text += f"{col}: {val}\n"
            documents.append({
                "page_content": clean_text(row_text),
                "metadata": {"sheet": sheet, "row": idx+2, "source": "Excel"}
            })
    return documents

def llm_invoke(prompt: str) -> str:
    p_hash = prompt_hash(prompt)
    cache = GLOBAL_STATE["prompt_cache"]
    cached = cache.get(p_hash)
    if cached:
        return cached
    raw = llm.invoke(prompt)
    result = strip_think_tags(raw)
    cache.put(p_hash, result)
    return result

# ─── Document Processing ──────────────────────────────────────────────────

def process_document(file_path: str, ext: str):
    logger.info(f"Processing {file_path} with extension {ext}")
    try:
        docs = []

        # ─── CSV ────────────────────────────────────────────
        if ext == ".csv":
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                raise HTTPException(400, "CSV is empty.")
            GLOBAL_STATE["raw_data"] = rows
            block_size = 20
            for i in range(0, len(rows), block_size):
                block = rows[i:i+block_size]
                text = f"Rows {i+1} to {min(i+block_size, len(rows))}:\n"
                for row in block:
                    text += ", ".join(f"{col}: {val}" for col, val in row.items()) + "\n"
                class Doc:
                    def __init__(self, content, meta):
                        self.page_content = content
                        self.metadata = meta
                docs.append(Doc(text, {"source": "CSV", "row_start": i+1}))

        # ─── PDF ────────────────────────────────────────────
        elif ext == ".pdf":
            try:
                docs = PyPDFLoader(file_path=file_path).load()
            except Exception as e:
                if "decrypted" in str(e).lower():
                    logger.warning("PDF encrypted – trying empty password.")
                    docs = PyPDFLoader(file_path=file_path, password="").load()
                else:
                    raise

        # ─── Excel ──────────────────────────────────────────
        elif ext in (".xlsx", ".xls"):
            if pd is None:
                raise HTTPException(400, "Pandas not installed")
            xls = pd.ExcelFile(file_path)
            all_rows = []
            block_size = 20
            for sheet in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                if df.empty:
                    continue
                rows = df.to_dict(orient='records')
                all_rows.extend(rows)
                for i in range(0, len(df), block_size):
                    block = df.iloc[i:i+block_size]
                    text = f"Sheet: {sheet} | Rows {i+2} to {min(i+block_size, len(df))+1}:\n"
                    for _, row in block.iterrows():
                        text += ", ".join(f"{col}: {row[col]}" for col in df.columns) + "\n"
                    class Doc:
                        def __init__(self, content, meta):
                            self.page_content = content
                            self.metadata = meta
                    docs.append(Doc(text, {"sheet": sheet, "row_start": i+2, "source": "Excel"}))
            GLOBAL_STATE["raw_data"] = all_rows

        # ─── Plain Text ──────────────────────────────────────
        elif ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if not text.strip():
                raise HTTPException(400, "Text file is empty.")
            class Doc:
                def __init__(self, content, meta):
                    self.page_content = content
                    self.metadata = meta
            docs.append(Doc(text, {"source": "TXT"}))

        # ─── Word (.docx) ──────────────────────────────────
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                import docx2txt
                text = docx2txt.process(file_path)
            if not text.strip():
                raise HTTPException(400, "Word document is empty.")
            class Doc:
                def __init__(self, content, meta):
                    self.page_content = content
                    self.metadata = meta
            docs.append(Doc(text, {"source": "DOCX"}))

        # ─── Word (.doc) – legacy format ────────────────────
        elif ext == ".doc":
            try:
                import docx2txt
                text = docx2txt.process(file_path)
            except Exception:
                raise HTTPException(400, "Cannot read .doc file. Please convert to .docx or .txt.")
            if not text.strip():
                raise HTTPException(400, "Word document is empty.")
            class Doc:
                def __init__(self, content, meta):
                    self.page_content = content
                    self.metadata = meta
            docs.append(Doc(text, {"source": "DOC"}))

        # ─── IMAGE processing ──────────────────────────────────────────
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            import base64
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=f"image/{ext[1:]}"),
                    "Extract all text from this image. If there is no text, describe the image content in detail."
                ]
            )
            text = response.text
            if not text.strip():
                raise HTTPException(400, "No text or content extracted from image")
            class Doc:
                def __init__(self, content, meta):
                    self.page_content = content
                    self.metadata = meta
            docs.append(Doc(text, {"source": "Image", "filename": Path(file_path).name}))

        else:
            raise HTTPException(400, "Unsupported format")

        if not docs:
            raise HTTPException(400, "No content extracted")

        # ─── Clean ──────────────────────────────────────────
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        # ─── Split ──────────────────────────────────────────
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")

        chunks = deduplicate_chunks(chunks)
        logger.info(f"After dedup: {len(chunks)}")

        chunks = [c for c in chunks if c.page_content.strip() != ""]
        logger.info(f"After filtering empty: {len(chunks)}")

        if not chunks:
            raise HTTPException(400, "No valid chunks extracted.")

        GLOBAL_STATE["all_chunks"] = [c.page_content for c in chunks]
        GLOBAL_STATE["chunk_metadata"] = [c.metadata for c in chunks]

        embeddings = GeminiEmbeddings(model="gemini-embedding-2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        GLOBAL_STATE["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 8})

        GLOBAL_STATE["file_hash"] = compute_hash(open(file_path, "rb").read())
        GLOBAL_STATE["filename"] = Path(file_path).name
        logger.info(f"Indexed {GLOBAL_STATE['filename']} successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("process_document failed")
        raise HTTPException(500, f"Processing error: {str(e)}")

# ─── API Endpoints ──────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".pdf", ".xlsx", ".xls", ".txt", ".docx", ".doc",
                   ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        raise HTTPException(400, "Unsupported file type")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        process_document(tmp_path, ext)
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    return {"message": "File uploaded and indexed successfully.", "filename": file.filename}

@app.get("/dashboard-data")
async def get_dashboard_data():
    try:
        raw = GLOBAL_STATE.get("raw_data")
        if raw is None or len(raw) == 0:
            raise HTTPException(400, "No data uploaded yet.")
        if pd is None:
            raise HTTPException(500, "Pandas is not installed.")
        try:
            df = pd.DataFrame(raw)
        except Exception:
            cleaned = []
            for row in raw:
                cleaned.append({str(k): str(v) if v is not None else "" for k, v in row.items()})
            df = pd.DataFrame(cleaned)
        if df.empty:
            raise HTTPException(400, "DataFrame is empty – no data to display.")
        total_rows = len(df)
        total_cols = len(df.columns)
        column_names = list(df.columns)
        dtypes = {}
        for col in df.columns:
            try:
                dtypes[col] = str(df[col].dtype)
            except:
                dtypes[col] = "unknown"
        numeric_summary = {}
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_summary[col] = {
                        "mean": float(df[col].mean()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "std": float(df[col].std()),
                        "count": int(df[col].count()),
                        "missing": int(df[col].isna().sum())
                    }
            except:
                pass
        categorical_summary = {}
        for col in df.columns:
            try:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    freq = df[col].value_counts().head(5).to_dict()
                    categorical_summary[col] = {
                        "top_values": freq,
                        "unique_count": int(df[col].nunique()),
                        "missing": int(df[col].isna().sum())
                    }
            except:
                pass
        missing_per_col = {}
        for col in df.columns:
            try:
                missing_per_col[col] = int(df[col].isna().sum())
            except:
                missing_per_col[col] = 0
        sample = []
        for _, row in df.head(10).iterrows():
            sample.append(row.to_dict())
        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "column_names": column_names,
            "dtypes": dtypes,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "missing_per_col": missing_per_col,
            "sample": sample
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Dashboard data error")
        raise HTTPException(500, f"Error processing data: {str(e)}")

@app.post("/query")
async def query_rag(request: dict):
    import time
    start_time = time.time()

    question = request.get("question")
    if not question:
        raise HTTPException(400, "Missing question")
    if GLOBAL_STATE["retriever"] is None:
        raise HTTPException(400, "No document indexed yet")

    docs = GLOBAL_STATE["retriever"].invoke(question)
    if not docs:
        return {"answer": "No relevant information found.", "source_chunks": []}

    context = "\n\n".join([d.page_content for d in docs])
    source_chunks = [d.page_content[:150] + "..." for d in docs]

    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the provided context.

    IMPORTANT: Before giving the final answer, you MUST think step‑by‑step and list your reasoning as bullet points (each on a new line, starting with "- ").

    After your reasoning, provide the final answer.

    Output format MUST be exactly as follows (do not change the labels):

    REASONING:
    - (first reasoning point)
    - (second reasoning point)
    ...

    ANSWER:
    (your final answer)

    Context:
    {context}

    Question: {question}
    """
    raw = llm_invoke(prompt)

    # Log raw response for debugging
    logger.info(f"Raw response from Gemini: {raw[:500]}...")  # first 500 chars

    # ─── Parse reasoning and answer ──────────────────────────────
    reasoning = ""
    answer = raw

    # Try standard labels
    if "REASONING:" in raw and "ANSWER:" in raw:
        parts = raw.split("ANSWER:")
        reasoning = parts[0].replace("REASONING:", "").strip()
        answer = parts[1].strip()
    elif "Reasoning:" in raw and "Answer:" in raw:
        parts = raw.split("Answer:")
        reasoning = parts[0].replace("Reasoning:", "").strip()
        answer = parts[1].strip()
    else:
        # Try regex fallback (case‑insensitive)
        import re
        match = re.search(r'(?:reasoning|REASONING):?\s*(.*?)\s*(?:answer|ANSWER):?\s*(.*)', raw, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
        else:
            # Last resort: if no split, assume the entire response is the answer
            answer = raw

    # If reasoning is empty, we can still show a placeholder
    if not reasoning:
        reasoning = "No explicit reasoning provided, but here is the answer."

    elapsed = time.time() - start_time
    thought_seconds = round(elapsed)

    return {
        "answer": answer,
        "reasoning": reasoning,
        "thought_seconds": thought_seconds,
        "source_chunks": source_chunks
    }

@app.get("/summary")
async def get_summary():
    if not GLOBAL_STATE["all_chunks"]:
        raise HTTPException(400, "No document has been indexed yet.")

    full_text = "\n\n".join(GLOBAL_STATE["all_chunks"])

    # ─── Short documents ──────────────────────────────────────────
    if len(full_text) < 10000:
        prompt = f"""
        Read the entire document and produce a concise summary as a numbered list of the most important points.
        Output ONLY the numbered list – no introduction, no extra text, no explanation.
        Each point must be on its own line, starting with a number (1., 2., 3., ...).
        Keep each point clear and brief.

        Document:
        {full_text}

        Numbered list:
        """
        summary = llm_invoke(prompt)
        # Ensure each number starts on a new line
        summary = re.sub(r'(\d+\.\s*)', r'\n\1', summary).strip()
        return {"summary": summary}

    # ─── Long documents – Map‑Reduce ──────────────────────────────
    segment_size = 2500
    overlap = 300
    segments = []
    start = 0
    while start < len(full_text):
        end = min(start + segment_size, len(full_text))
        segments.append(full_text[start:end])
        start = end - overlap

    segment_summaries = []
    for i, seg in enumerate(segments):
        prompt = f"""
        Read this document section (part {i+1}/{len(segments)}) and produce a numbered list of the key points.
        Output ONLY the numbered list – no introduction, no extra text.
        Each point on a new line, starting with a number (1., 2., ...).

        Section:
        {seg}

        Numbered list:
        """
        seg_summary = llm_invoke(prompt)
        segment_summaries.append(seg_summary)

    combined = "\n\n".join(segment_summaries)
    final_prompt = f"""
    Combine the following section summaries into one final numbered list of the most important points from the entire document.
    Output ONLY the numbered list – no introduction, no extra text.
    Remove duplicates, keep logical order.
    Each point must be on its own line, starting with a number (1., 2., 3., ...).

    Section summaries:
    {combined}

    Final numbered list:
    """
    final_summary = llm_invoke(final_prompt)
    final_summary = re.sub(r'(\d+\.\s*)', r'\n\1', final_summary).strip()
    return {"summary": final_summary}

@app.post("/reset")
async def reset_state():
    GLOBAL_STATE["retriever"] = None
    GLOBAL_STATE["file_hash"] = None
    GLOBAL_STATE["filename"] = None
    GLOBAL_STATE["all_chunks"] = []
    GLOBAL_STATE["prompt_cache"].clear()
    return {"message": "State reset"}

@app.get("/dashboard")
async def dashboard_page():
    try:
        return FileResponse("templates/dashboard.html")
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading dashboard</h1><p>{str(e)}</p>", status_code=500)

# ─── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def home():
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: index.html not found</h1>")
    current_file = GLOBAL_STATE.get("filename") or "No document indexed yet"
    html_content = html_content.replace("{{ current_file }}", current_file)
    return HTMLResponse(content=html_content)

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "file_indexed": GLOBAL_STATE["filename"] is not None,
        "current_file": GLOBAL_STATE["filename"],
        "chunks_count": len(GLOBAL_STATE["all_chunks"]),
    }

@app.get("/debug-raw")
async def debug_raw():
    raw = GLOBAL_STATE.get("raw_data")
    if raw is None:
        return {"raw_data": None}
    return {
        "raw_data_type": str(type(raw)),
        "raw_data_length": len(raw),
        "first_row": raw[0] if raw and len(raw) > 0 else None,
        "last_row": raw[-1] if raw and len(raw) > 0 else None,
        "sample_rows": raw[:3] if raw else []
    }