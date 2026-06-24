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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from pathlib import Path

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

def process_document(file_path: str, ext: str):
    """Load, clean, chunk, embed, and index a document."""
    logger.info(f"Processing {file_path} with extension {ext}")
    try:
        docs = []

        # ─── CSV ────────────────────────────────────────────
        if ext == ".csv":
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                #GLOBAL_STATE["raw_data"] = rows
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
            logger.info(f"Created {len(docs)} blocks from CSV.")

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
            logger.info(f"Created {len(docs)} blocks from Excel.")

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

        # ─── Deduplicate ────────────────────────────────────
        chunks = deduplicate_chunks(chunks)
        logger.info(f"After dedup: {len(chunks)}")

        # ─── Filter only empty ──────────────────────────────
        chunks = [c for c in chunks if c.page_content.strip() != ""]
        logger.info(f"After filtering empty: {len(chunks)}")

        if not chunks:
            raise HTTPException(400, "No valid chunks extracted.")

        # ─── Store and index ──────────────────────────────
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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".pdf", ".xlsx", ".xls"):
        raise HTTPException(400, "Only CSV, PDF, or Excel allowed")
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
    if GLOBAL_STATE["raw_data"] is None or len(GLOBAL_STATE["raw_data"]) == 0:
        raise HTTPException(400, "No data uploaded yet.")
    
    raw = GLOBAL_STATE["raw_data"]
    # Convert to pandas DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(raw)
    
    # Basic info
    total_rows = len(df)
    total_cols = len(df.columns)
    column_names = list(df.columns)
    
    # Data types
    dtypes = df.dtypes.astype(str).to_dict()
    
    # Summary for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_summary = {}
    for col in numeric_cols:
        numeric_summary[col] = {
            "mean": df[col].mean(),
            "min": df[col].min(),
            "max": df[col].max(),
            "std": df[col].std(),
            "count": df[col].count(),
            "missing": df[col].isna().sum()
        }
    
    # Summary for categorical columns (top 5 frequencies)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_summary = {}
    for col in categorical_cols:
        freq = df[col].value_counts().head(5).to_dict()
        categorical_summary[col] = {
            "top_values": freq,
            "unique_count": df[col].nunique(),
            "missing": df[col].isna().sum()
        }
    
    # Overall missing values per column
    missing_per_col = df.isna().sum().to_dict()
    
    # Sample data (first 10 rows) for preview
    sample = df.head(10).to_dict(orient='records')
    
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

@app.post("/query")
async def query_rag(request: dict):
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

    prompt = f"Answer based only on the context:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = llm_invoke(prompt)
    return {"answer": answer, "source_chunks": source_chunks}


@app.get("/summary")
async def get_summary():
    if not GLOBAL_STATE["all_chunks"]:
        raise HTTPException(400, "No document indexed")
    full_text = "\n\n".join(GLOBAL_STATE["all_chunks"])
    prompt = f"Provide a comprehensive summary of the document:\n\n{full_text}\n\nSummary:"
    summary = llm_invoke(prompt)
    return {"summary": summary}

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
    return FileResponse("templates/dashboard.html")

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

# ─── Keep only the minimal endpoints for now ────────────────────────────────

@app.get("/")
async def home():
    # Read the HTML template
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: index.html not found</h1>")
    
    # Get the current filename from global state
    current_file = GLOBAL_STATE.get("filename") or "No document indexed yet"
    
    # Replace placeholder in HTML with the actual filename
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