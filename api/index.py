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

# ─── Keep only the minimal endpoints for now ───
@app.get("/")
async def root():
    return HTMLResponse("<h1>✅ Imports loaded</h1><p>Your imports and API key check passed.</p>")

@app.get("/ping")
async def ping():
    return {"status": "ok"}