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

# ─── Keep only the minimal endpoints for now ────────────────────────────────

@app.get("/")
async def root():
    return HTMLResponse("<h1>✅ Helpers loaded</h1><p>All classes and helpers are working.</p>")



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