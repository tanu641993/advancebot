# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: SETUP & IMPORTS - Initialize Google API and import required libraries
# ═══════════════════════════════════════════════════════════════════════════════

import os
# Configure Google AI API version and endpoint for the Gemini API
os.environ["GOOGLE_API_VERSION"] = "v1"
os.environ["GOOGLE_API_ENDPOINT"] = "https://generativelanguage.googleapis.com/v1/"

import hashlib          # For computing file hashes (detect duplicate uploads)
import logging          # For logging application events and errors
import re               # For regex operations (strip XML tags)
import tempfile         # For creating temporary files during upload
import json             # For JSON serialization
from pathlib import Path  # For file path operations
from typing import List, Optional, Dict, Any  # Type hints
from collections import deque  # For efficient memory management

from fastapi import FastAPI, HTTPException, UploadFile, File  # Web framework
from fastapi.responses import HTMLResponse, JSONResponse  # Serve responses
from pydantic import BaseModel  # Data validation and serialization

from langchain_community.document_loaders import CSVLoader, PyPDFLoader  # Load docs
from langchain_community.vectorstores import FAISS  # Vector database
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Chunk documents

# Google's Generative AI SDK for Gemini
from google import genai
from google.genai import types

# For Excel support
try:
    import pandas as pd
except ImportError:
    pd = None

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LOGGING CONFIGURATION - Setup application logging
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("uvicorn.error")  # Get logger for FastAPI/Uvicorn
logger.setLevel(logging.INFO)  # Set minimum log level to INFO

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: CONFIGURATION CONSTANTS - Settings for document processing and AI
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 800              # Larger chunks (800 chars) for better semantic coherence
CHUNK_OVERLAP = 100           # More overlap (100 chars) to preserve context between chunks
MAX_TOKENS = 2048             # Balanced tokens for accurate, focused responses
TOP_K = 8                     # Retrieve top 8 chunks for better coverage
SIMILARITY_THRESHOLD = 0.5    # Filter chunks with relevance score < 0.5
RE_RANK_TOP_K = 5             # Re-rank and keep only top 5 most relevant chunks
PROMPT_CACHE_MAX_SIZE = 200   # Larger cache (200) for better hit rate
DEDUPLICATION_THRESHOLD = 0.95  # Remove chunks > 95% similar to already indexed chunks

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: ENVIRONMENT VALIDATION - Load and validate Google API key
# ═══════════════════════════════════════════════════════════════════════════════

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Load API key from environment
if not GOOGLE_API_KEY:  # Fail early if not found
    logger.error("Missing GOOGLE_API_KEY")
    raise RuntimeError("Missing GOOGLE_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: FASTAPI SETUP - Create the web application instance
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Gemini RAG Data API", version="1.0")  # Web API for RAG system

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: GLOBAL STATE - Store application data in memory during runtime
# ═══════════════════════════════════════════════════════════════════════════════

# Memory-efficient cache with LRU eviction
class LRUCache:
    """LRU Cache for prompt responses with memory optimization."""
    def __init__(self, max_size: int = 200):
        self.cache: Dict[str, str] = {}
        self.access_order = deque()  # Track access order for LRU
        self.max_size = max_size

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
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
    "file_hash": None,           # MD5 hash of uploaded file (detect duplicates)
    "filename": None,            # Name of currently indexed document
    "retriever": None,           # FAISS retriever for finding relevant chunks
    "prompt_cache": LRUCache(max_size=PROMPT_CACHE_MAX_SIZE),  # LRU cache for responses
    "all_chunks": [],            # Store all text chunks from document
    "chunk_metadata": [],        # Store metadata (source, position) for chunks
    "embeddings_cache": {},      # Cache embeddings to avoid recomputation
    "query_history": deque(maxlen=50),  # Keep last 50 queries for context
}

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: EMBEDDINGS CLASS - Convert text to vector embeddings using Gemini API
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Transform text into numerical vectors for similarity search

class GeminiEmbeddings:
    def __init__(self, model="gemini-embedding-2"):
        # Initialize Gemini client for embedding operations
        self.client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1'})
        self.model = model  # Embedding model to use

    def __call__(self, text):
        """Make the instance callable – used by FAISS for query embeddings."""
        return self.embed_query(text)  # Delegate to embed_query for consistency

    def embed_documents(self, texts):
        """Embed a list of documents (document chunks) for storage in vector DB."""
        result = []
        for text in texts:
            # Call Gemini API to convert document text to embedding vector
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="retrieval_document"),
            )
            # Extract the embedding values and store in result list
            result.append(response.embeddings[0].values)
        return result

    def embed_query(self, text):
        """Embed a single user query to search for similar documents."""
        # Call Gemini API to convert query to embedding vector
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="retrieval_query"),
        )
        # Return the embedding vector for similarity search
        return response.embeddings[0].values

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: CHAT CLASS - Wrapper for Gemini LLM to generate AI responses
# ═══════════════════════════════════════════════════════════════════════════════
# Purpose: Send prompts to Gemini and get generated text responses

class ChatGemini:
    def __init__(self, model="gemini-2.5-flash", temperature=0.0, max_output_tokens=1024):
        # Initialize Gemini client for chat/generation operations
        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options={'api_version': 'v1'}   # Force API v1
        )
        self.model = model                       # Model to use (fast, multimodal)
        self.temperature = temperature           # Creativity (0=deterministic, 1=random)
        self.max_output_tokens = max_output_tokens  # Max response length

    def invoke(self, prompt: str) -> str:
        """Send a prompt to Gemini and get the generated response."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        )
        return response.text  # Return the generated text

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: LLM INITIALIZATION - Create global AI model instance
# ═══════════════════════════════════════════════════════════════════════════════

llm = ChatGemini(
    model="gemini-2.5-flash",    # Fast multimodal model
    temperature=0.0,             # Deterministic responses (best for RAG)
    max_output_tokens=MAX_TOKENS, # Match config above
)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10: HTML TEMPLATE LOADING - Load the frontend UI
# ═══════════════════════════════════════════════════════════════════════════════

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "index.html"
try:
    # Try to load the HTML file for the web interface
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        HTML_TEMPLATE = f.read()
except FileNotFoundError:
    # Fallback if template file doesn't exist
    HTML_TEMPLATE = "<html><body><h1>Gemini RAG</h1><p>Template missing.</p></body></html>"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11: DATA MODELS - Define request/response structures
# ═══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """Single message in conversation history."""
    user: str          # What the user asked
    assistant: str     # What the AI responded

class QueryRequest(BaseModel):
    """Input for a RAG query endpoint."""
    question: str                              # User's question
    chat_history: Optional[List[ChatMessage]] = []  # Previous messages (optional)

class QueryResponse(BaseModel):
    """Output from RAG query endpoint."""
    answer: str              # AI-generated answer
    source_chunks: List[str] # Relevant document excerpts used

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12: HELPER FUNCTIONS - Utility functions for the application
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hash(b: bytes) -> str:
    """Generate MD5 hash of file bytes (detect duplicate uploads)."""
    return hashlib.md5(b).hexdigest()

def prompt_hash(p: str) -> str:
    """Generate MD5 hash of prompt string (for cache lookup)."""
    return hashlib.md5(p.encode()).hexdigest()

def strip_think_tags(text: str) -> str:
    """Remove Gemini's internal <think> tags from response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def deduplicate_chunks(chunks: List) -> List:
    """Remove near-duplicate chunks using embedding similarity."""
    if not chunks:
        return chunks
    
    seen_hashes = set()
    unique_chunks = []
    
    for chunk in chunks:
        # Hash chunk content for quick comparison
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks

def clean_text(text: str) -> str:
    """Clean and normalize text before processing."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    return text.strip()

def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = (sum(a ** 2 for a in vec1) ** 0.5) + 1e-9
    norm2 = (sum(b ** 2 for b in vec2) ** 0.5) + 1e-9
    return dot_product / (norm1 * norm2)

def load_excel_file(file_path: str) -> List[Dict[str, Any]]:
    """Load Excel file with pandas and convert to documents."""
    if pd is None:
        raise HTTPException(status_code=400, detail="Pandas not installed for Excel support")
    
    try:
        # Read Excel file
        xls = pd.ExcelFile(file_path)
        documents = []
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Convert DataFrame to text documents
            for idx, row in df.iterrows():
                # Create a formatted document from each row
                row_text = f"Sheet: {sheet_name} | Row {idx + 2}\n"
                for col, value in row.items():
                    row_text += f"{col}: {value}\n"
                
                documents.append({
                    "page_content": clean_text(row_text),
                    "metadata": {"sheet": sheet_name, "row": idx + 2, "source": "Excel"}
                })
        
        return documents
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading Excel: {str(e)}")

def llm_invoke(prompt: str) -> str:
    """Invoke LLM with LRU caching to avoid duplicate API calls and manage memory."""
    p_hash = prompt_hash(prompt)  # Hash the prompt for caching
    cache = GLOBAL_STATE["prompt_cache"]
    
    # Check if we've already processed this exact prompt
    cached_result = cache.get(p_hash)
    if cached_result:
        logger.info(f"Cache hit for prompt hash {p_hash[:8]}")
        return cached_result  # Return cached result
    
    # Call the LLM to generate response
    logger.info(f"Cache miss - calling LLM for prompt hash {p_hash[:8]}")
    raw = llm.invoke(prompt)
    result = strip_think_tags(raw)  # Clean up internal tags
    
    # Store result in LRU cache (automatically evicts least used if full)
    cache.put(p_hash, result)
    logger.info(f"Cached result. Cache size: {cache.size()}/{PROMPT_CACHE_MAX_SIZE}")
    
    # Track query in history
    GLOBAL_STATE["query_history"].append({
        "prompt_hash": p_hash,
        "timestamp": str(Path(".")),
        "cached": False
    })
    
    return result

def process_document(file_path: str, ext: str):
    """Main pipeline: Load file -> Clean -> Split -> Dedupe -> Embed -> Index."""
    logger.info(f"Processing {file_path} with extension {ext}")
    try:
        docs = []  # Will hold Document objects

        # ─── CSV handling ──────────────────────────────────────────────
        if ext == ".csv":
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                raise HTTPException(status_code=400, detail="CSV is empty.")

            block_size = 20  # adjust as needed
            for i in range(0, len(rows), block_size):
                block = rows[i:i+block_size]
                text = f"Rows {i+1} to {min(i+block_size, len(rows))}:\n"
                for row in block:
                    text += ", ".join(f"{col}: {val}" for col, val in row.items()) + "\n"
                # Create a Document object with metadata
                class Doc:
                    def __init__(self, content, meta):
                        self.page_content = content
                        self.metadata = meta
                docs.append(Doc(text, {"source": "CSV", "row_start": i+1}))
            logger.info(f"Created {len(docs)} blocks from CSV.")

        # ─── PDF handling ──────────────────────────────────────────────
        elif ext == ".pdf":
            try:
                docs = PyPDFLoader(file_path=file_path).load()
            except Exception as e:
                if "decrypted" in str(e).lower():
                    logger.warning("PDF is encrypted – trying with empty password.")
                    try:
                        docs = PyPDFLoader(file_path=file_path, password="").load()
                    except Exception as inner_e:
                        raise HTTPException(status_code=400, detail=f"PDF encrypted: {str(inner_e)}")
                else:
                    raise

        # ─── Excel handling ──────────────────────────────────────────────
        elif ext in (".xlsx", ".xls"):
            if pd is None:
                raise HTTPException(status_code=400, detail="Pandas not installed for Excel support")
            xls = pd.ExcelFile(file_path)
            block_size = 20
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                if df.empty:
                    continue
                # Group rows into blocks
                for i in range(0, len(df), block_size):
                    block = df.iloc[i:i+block_size]
                    text = f"Sheet: {sheet_name} | Rows {i+2} to {min(i+block_size, len(df))+1}:\n"
                    for idx, row in block.iterrows():
                        row_text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                        text += row_text + "\n"
                    class Doc:
                        def __init__(self, content, meta):
                            self.page_content = content
                            self.metadata = meta
                    docs.append(Doc(text, {"sheet": sheet_name, "row_start": i+2, "source": "Excel"}))
            logger.info(f"Created {len(docs)} blocks from Excel.")

        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from file")

        # ─── Clean all documents ──────────────────────────────────────
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        # ─── Split into chunks ──────────────────────────────────────
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")

        # ─── Deduplicate ──────────────────────────────────────────────
        chunks = deduplicate_chunks(chunks)
        logger.info(f"After deduplication: {len(chunks)} unique chunks")

        # ─── Filter only truly empty chunks ──────────────────────────
        chunks = [c for c in chunks if len(c.page_content.strip()) > 0]
        logger.info(f"After filtering empty chunks: {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No valid chunks extracted. Document may be empty or text could not be extracted."
            )

        # ─── Store and embed ──────────────────────────────────────────
        GLOBAL_STATE["all_chunks"] = [chunk.page_content for chunk in chunks]
        GLOBAL_STATE["chunk_metadata"] = [getattr(chunk, 'metadata', {}) for chunk in chunks]

        embeddings = GeminiEmbeddings(model="gemini-embedding-2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        GLOBAL_STATE["retriever"] = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        GLOBAL_STATE["file_hash"] = compute_hash(open(file_path, "rb").read())
        GLOBAL_STATE["filename"] = Path(file_path).name
        logger.info(f"Indexed {GLOBAL_STATE['filename']} successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("process_document failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/summary")
async def get_summary():
    """Generate a comprehensive summary of the entire indexed document."""
    # Check if document has been indexed
    if not GLOBAL_STATE["all_chunks"]:
        raise HTTPException(status_code=400, detail="No document has been indexed yet.")

    # Combine all chunks into single text
    full_text = "\n\n".join(GLOBAL_STATE["all_chunks"])
    
    # STRATEGY 1: For short documents, summarize directly (efficient)
    if len(full_text) < 10000:  # Reduced threshold for better quality
        prompt = f"""
        Provide a comprehensive and well-structured summary of the document.
        
        INCLUDE:
        - Main topics and key points
        - Important findings and conclusions
        - Critical details and numbers
        - Any recommendations or implications
        
        FORMAT: Use clear sections with headers
        TONE: Professional and concise
        
        Document:
        {full_text}
        
        Summary:
        """
        summary = llm_invoke(prompt)
        return {"summary": summary}

    # STRATEGY 2: For long documents, use map-reduce pattern
    # Split text into overlapping segments
    segment_size = 2500      # Smaller segments for better quality
    overlap = 300            # More overlap to preserve context
    segments = []
    start = 0
    while start < len(full_text):
        end = min(start + segment_size, len(full_text))
        segments.append(full_text[start:end])  # Create segment
        start = end - overlap  # Move start forward with overlap

    # MAP PHASE: Summarize each segment independently
    segment_summaries = []
    for i, seg in enumerate(segments):
        prompt = f"""
        Summarize this document section (part {i+1}/{len(segments)}).
        
        FOCUS ON:
        - Main ideas and key information
        - Important facts and conclusions
        - Relationships between concepts
        
        KEEP IT: Clear, complete, and concise
        
        Section:
        {seg}
        
        Summary:
        """
        seg_summary = llm_invoke(prompt)  # Summarize segment
        segment_summaries.append(seg_summary)

    # REDUCE PHASE: Combine all segment summaries into final summary
    combined = "\n\n".join(segment_summaries)
    final_prompt = f"""
    Create a cohesive final summary by combining these section summaries.
    
    ENSURE:
    - Logical flow and structure
    - No repetition or redundancy
    - All key information preserved
    - Clear sections with headers
    
    Section Summaries:
    {combined}
    
    Final Summary:
    """
    final_summary = llm_invoke(final_prompt)  # Generate final summary
    return {"summary": final_summary}

@app.post("/reset")
async def reset_state():
    """Clear all indexed data and caches."""
    # Reset all global state variables
    GLOBAL_STATE["retriever"] = None
    GLOBAL_STATE["file_hash"] = None
    GLOBAL_STATE["filename"] = None
    GLOBAL_STATE["all_chunks"] = []  # Clear cached chunks
    GLOBAL_STATE["chunk_metadata"] = []  # Clear metadata
    GLOBAL_STATE["prompt_cache"].clear()  # Clear LLM response cache (LRU)
    GLOBAL_STATE["embeddings_cache"].clear()  # Clear embedding cache
    GLOBAL_STATE["query_history"].clear()  # Clear query history
    logger.info("Application state reset - all caches cleared")
    return {"message": "State reset successfully - all caches cleared."}

@app.get("/dashboard")
async def get_dashboard():
    """Get dashboard data for rendering in frontend."""
    if GLOBAL_STATE["filename"] is None:
        raise HTTPException(status_code=400, detail="No file indexed. Upload a file first.")
    
    filename = GLOBAL_STATE["filename"]
    
    try:
        # Return dashboard data as JSON - frontend will render it
        return {
            "filename": filename,
            "chunks_count": len(GLOBAL_STATE["all_chunks"]),
            "avg_chunk_size": round(sum(len(c) for c in GLOBAL_STATE["all_chunks"]) / max(1, len(GLOBAL_STATE["all_chunks"]))),
            "total_characters": sum(len(c) for c in GLOBAL_STATE["all_chunks"]),
            "prompt_cache_size": GLOBAL_STATE["prompt_cache"].size(),
            "prompt_cache_max": PROMPT_CACHE_MAX_SIZE,
            "embeddings_cache_size": len(GLOBAL_STATE["embeddings_cache"]),
            "query_history_count": len(GLOBAL_STATE["query_history"]),
            "retrieval_model": "FAISS + Gemini Embeddings",
            "llm_model": "Gemini 2.5 Flash",
            "re_rank_top_k": RE_RANK_TOP_K,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
    except Exception as e:
        logger.exception("Dashboard data generation failed")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.get("/health")
async def health():
    """Check API health and indexing status with detailed metrics."""
    return {
        "status": "ok",
        "file_indexed": GLOBAL_STATE["filename"] is not None,  # Is a document loaded?
        "current_file": GLOBAL_STATE["filename"],
        "chunks_count": len(GLOBAL_STATE["all_chunks"]),
        "prompt_cache_size": GLOBAL_STATE["prompt_cache"].size(),
        "prompt_cache_max": PROMPT_CACHE_MAX_SIZE,
        "embeddings_cache_size": len(GLOBAL_STATE["embeddings_cache"]),
        "query_history_count": len(GLOBAL_STATE["query_history"]),
        "total_characters": sum(len(c) for c in GLOBAL_STATE["all_chunks"]),
        "vectoring": "Gemini Embedding 2",
        "reranking": True,
        "memory_status": "optimized"
    }

@app.get("/ping")
async def ping():
    return {"status": "ok"}
