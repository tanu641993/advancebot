# Gemini RAG Data API

A production-ready **Retrieval-Augmented Generation (RAG)** system that combines document processing, semantic search, and AI-powered question answering with enterprise-grade caching and optimization.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Performance Optimization](#performance-optimization)

---

## 🎯 Overview

The Gemini RAG Data API enables users to:
1. **Upload documents** (CSV, PDF, Excel)
2. **Ask questions** about document content using natural language
3. **Get accurate answers** grounded in the documents with source citations
4. **Generate summaries** of entire documents
5. **View system metrics** through an interactive dashboard

The system uses **semantic search** to find relevant information and **AI generation** to craft accurate, context-aware responses.

---

## 🛠️ Technology Stack

### Backend (Python/FastAPI)

| Technology | Purpose | Why Used |
|-----------|---------|----------|
| **FastAPI** | Web framework for REST API | Fast, modern async Python framework with auto-documentation |
| **Google Gemini API** | LLM (Large Language Model) for text generation | State-of-the-art generative AI with excellent instruction following |
| **Gemini Embeddings 2** | Convert text to vectors | Semantic understanding; enables similarity search |
| **FAISS** | Vector database for fast similarity search | Facebook's highly optimized vector similarity search library |
| **LangChain** | Document loading and text splitting | Industry-standard library for RAG pipeline orchestration |
| **Pandas** | Excel/CSV file parsing | Robust data processing and dataframe manipulation |
| **Uvicorn** | ASGI server for FastAPI | High-performance async server |

### Frontend (HTML/JavaScript)

| Component | Purpose |
|-----------|---------|
| **HTML5** | Semantic markup for chatbot UI |
| **CSS3** | Responsive styling with gradient backgrounds |
| **Vanilla JavaScript** | Client-side chat logic, file upload, dashboard rendering |
| **Fetch API** | Asynchronous communication with backend |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   USER INTERFACE (Browser)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ • File Upload Area (CSV, PDF, Excel)                 │  │
│  │ • Chat Interface with Message History                │  │
│  │ • Real-time Typing Indicator                         │  │
│  │ • Dashboard with System Metrics                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTP/JSON
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Document Processing Pipeline                         │  │
│  │  1. Load (CSV/PDF/Excel)                             │  │
│  │  2. Clean (normalize text)                           │  │
│  │  3. Split (chunk with overlap)                       │  │
│  │  4. Deduplicate (remove similar chunks)              │  │
│  │  5. Embed (convert to vectors)                       │  │
│  │  6. Index (store in FAISS)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ In-Memory Cache System                               │  │
│  │  • LRU Cache (responses) - 200 max entries            │  │
│  │  • Embedding Cache (vectors) - no size limit         │  │
│  │  • Query History - 50 latest queries                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Query Processing (RAG Pipeline)                      │  │
│  │  1. Retrieve (8 chunks from FAISS)                   │  │
│  │  2. Re-rank (score by similarity)                    │  │
│  │  3. Filter (keep top 5 above 0.5 threshold)          │  │
│  │  4. Contextualize (build prompt with chunks)         │  │
│  │  5. Generate (call Gemini LLM)                       │  │
│  │  6. Cache (store for reuse)                          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ API Calls
         ┌────────────────────────────────┐
         │   Google Gemini API            │
         │  • gemini-2.5-flash (LLM)      │
         │  • gemini-embedding-2 (vectors)│
         └────────────────────────────────┘
```

---

## ✨ Key Features

### 1. **Multi-Format Document Support**
- **CSV Files**: Row-by-row parsing, column headers preserved
- **PDF Files**: Text extraction with page numbers
- **Excel Files**: Per-sheet processing, row/column metadata preserved

### 2. **Intelligent Text Processing**
- **Chunking**: 800-character semantic chunks with 100-char overlap
- **Deduplication**: Removes near-identical chunks automatically
- **Cleaning**: Normalizes whitespace and removes control characters
- **Filtering**: Removes chunks under 50 characters (noise)

### 3. **Advanced RAG Pipeline**
- **Two-Stage Retrieval**: Retrieve 8 chunks, re-rank to top 5 most relevant
- **Semantic Similarity**: Cosine similarity scoring (0-1 scale)
- **Relevance Filtering**: Configurable threshold (default 0.5)
- **Context Assembly**: Combines top chunks into coherent context

### 4. **Performance Optimization**
- **LRU Cache**: Stores 200 most recent LLM responses (automatic eviction)
- **Embedding Cache**: Avoids redundant embedding API calls
- **Query History**: Tracks 50 latest queries for analysis
- **Smart Summarization**: Map-reduce pattern for large documents (>10K chars)

### 5. **Production-Ready**
- **XSS Protection**: HTML escaping in dashboard data
- **Error Handling**: Comprehensive try-catch with detailed error messages
- **Logging**: Full operation logging with timestamps
- **Health Checks**: `/health` endpoint with system metrics
- **State Management**: Graceful reset functionality

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Google API Key (get from [Google AI Studio](https://aistudio.google.com/))
- pip package manager

### Step 1: Install Dependencies

```bash
pip install fastapi uvicorn google-genai langchain-community faiss-cpu pandas
```

**Package Details:**
- `fastapi`: Web API framework
- `uvicorn`: ASGI server
- `google-genai`: Google Gemini API client
- `langchain-community`: Document loaders and splitters
- `faiss-cpu`: Vector similarity search (CPU version)
- `pandas`: Excel and CSV parsing

### Step 2: Set Environment Variable

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your-api-key-here"

# Windows Command Prompt
set GOOGLE_API_KEY=your-api-key-here

# Linux/Mac
export GOOGLE_API_KEY="your-api-key-here"
```

### Step 3: Run the Server

```bash
python index.py
```

Or with Uvicorn directly:

```bash
uvicorn index:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Open in Browser

Navigate to: `http://localhost:8000`

---

## ⚙️ Configuration

Edit these constants in `index.py` to customize behavior:

```python
CHUNK_SIZE = 800              # Characters per chunk (larger = more context)
CHUNK_OVERLAP = 100           # Overlap between chunks (more = better context flow)
MAX_TOKENS = 2048             # Max response length (lower = faster)
TOP_K = 8                     # Initial chunks to retrieve (more = broader search)
SIMILARITY_THRESHOLD = 0.5    # Min relevance score (higher = stricter filtering)
RE_RANK_TOP_K = 5             # Final chunks to use (lower = focused answers)
PROMPT_CACHE_MAX_SIZE = 200   # LRU cache size (higher = more memory used)
```

### Tuning Guide

| Goal | Adjustment |
|------|-------------|
| **Faster responses** | Reduce MAX_TOKENS, RE_RANK_TOP_K |
| **Better accuracy** | Increase TOP_K, CHUNK_SIZE, RE_RANK_TOP_K |
| **Lower API costs** | Increase PROMPT_CACHE_MAX_SIZE, reduce MAX_TOKENS |
| **More memory** | Reduce PROMPT_CACHE_MAX_SIZE, enable embeddings pagination |
| **Better context** | Increase CHUNK_SIZE, CHUNK_OVERLAP |

---

## 🚀 Usage

### Basic Workflow

#### 1. **Upload a Document**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.csv"
```

#### 2. **Ask a Question**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings?",
    "chat_history": []
  }'
```

Response:
```json
{
  "answer": "Based on the document, the main findings are...",
  "source_chunks": [
    "[Chunk 1] Relevant excerpt from document...",
    "[Chunk 2] Another relevant section..."
  ]
}
```

#### 3. **Get Document Summary**
```bash
curl "http://localhost:8000/summary"
```

#### 4. **View Dashboard**
Open browser to `/dashboard` to see system metrics

#### 5. **Reset System**
```bash
curl -X POST "http://localhost:8000/reset"
```

---

## 📡 API Endpoints

### Serving the UI

**GET** `/`
- Returns HTML chatbot interface
- Status: 200 OK
- No authentication required

### File Management

**POST** `/upload`
- Upload document (CSV, PDF, or Excel)
- Supported formats: `.csv`, `.pdf`, `.xlsx`, `.xls`
- Returns: `{"message": "...", "filename": "..."}`
- Response: 200 on success, 400 on invalid format

### RAG Query

**POST** `/query`
- Ask question about indexed document
- Request body:
  ```json
  {
    "question": "Your question here",
    "chat_history": [] // optional previous messages
  }
  ```
- Response:
  ```json
  {
    "answer": "Generated answer",
    "source_chunks": ["excerpt1", "excerpt2", ...]
  }
  ```

### Summarization

**GET** `/summary`
- Generate comprehensive summary of entire document
- Uses map-reduce for documents over 10K characters
- Response: `{"summary": "..."}`

### Metrics & Status

**GET** `/health`
- Check system status with detailed metrics
- Returns 9+ fields including cache sizes, file status
- Response: `{"status": "ok", "file_indexed": true, ...}`

**GET** `/dashboard`
- Get system metrics as JSON (for frontend rendering)
- Returns 13 fields with file info, cache usage, config
- Response: `{"filename": "...", "chunks_count": 42, ...}`

### System Control

**POST** `/reset`
- Clear all indexed data and caches
- Resets: retriever, cache, embeddings, history
- Response: `{"message": "State reset successfully"}`

**GET** `/list-generative-models`
- List all available Gemini models
- Response: `{"all_models": ["models/gemini-2.5-flash", ...]}`

---

## 🚄 Performance Optimization

### Caching Strategy

The system implements **three-tier caching**:

#### 1. **LRU Prompt Cache** (200 entries)
- **What**: Stores exact LLM responses
- **When**: Same question asked multiple times
- **Benefit**: ~100ms lookup vs 3-5s API call
- **Eviction**: Least-recently-used item removed when full

#### 2. **Embedding Cache** (unlimited)
- **What**: Stores vector embeddings
- **When**: Same text needs embedding multiple times
- **Benefit**: Avoids redundant embedding API calls
- **Usage**: Tracks 50 query embeddings + chunk embeddings

#### 3. **Query History** (50 entries)
- **What**: Stores recent query hashes
- **When**: Analyzing query patterns
- **Benefit**: Can identify frequently asked questions
- **Use Case**: Optimize chunk selection for common queries

### Memory Management

- **Automatic Eviction**: LRU cache removes least-used items when reaching 200
- **Bounded Memory**: No unbounded growth of cache dictionaries
- **Efficient Deduplication**: Hash-based chunk deduplication
- **Streaming**: Large files processed in chunks (not loaded entirely)

### Retrieval-Augmented Generation Pipeline

```
User Question
    ↓
[RETRIEVE] Query FAISS → 8 chunks
    ↓
[RE-RANK] Score by cosine similarity
    ↓
[FILTER] Keep top 5 above 0.5 threshold
    ↓
[CONTEXTUALIZE] Build prompt with chunks
    ↓
[GENERATE] Call Gemini LLM
    ↓
[CACHE] Store response in LRU cache
    ↓
Answer + Sources
```

### Cost Optimization

1. **Reduce Token Usage**
   - Smaller MAX_TOKENS (2048 vs 4096)
   - Fewer re-ranked chunks (5 vs 10)
   - More aggressive similarity threshold (0.5)

2. **Increase Cache Hit Rate**
   - Larger PROMPT_CACHE_MAX_SIZE (200 entries)
   - Process similar questions together
   - Cache embeddings for reuse

3. **Batch Processing**
   - Process multiple files before querying
   - Handle queries efficiently with caching
   - Use summaries instead of multiple queries

---

## 📊 What Gets Cached vs Not Cached

| Item | Cached | Cache Type | TTL |
|------|--------|-----------|-----|
| LLM responses | ✅ Yes | LRU (200 max) | Session |
| Embeddings | ✅ Yes | Dict (unlimited) | Session |
| Query history | ✅ Yes | Deque (50 max) | Session |
| FAISS index | ✅ Yes | In-memory | Per upload |
| Document chunks | ✅ Yes | Memory array | Per upload |
| Uploaded files | ❌ No | Temp file | Deleted after index |

---

## 🔍 Troubleshooting

### Issue: "Missing GOOGLE_API_KEY"
**Solution**: Set environment variable before running
```bash
$env:GOOGLE_API_KEY = "your-key-here"
python index.py
```

### Issue: "No document indexed yet"
**Solution**: Upload a file first using the UI or `/upload` endpoint

### Issue: Slow responses
**Solution**: 
- Check cache hit rate in logs
- Reduce MAX_TOKENS
- Increase SIMILARITY_THRESHOLD to filter irrelevant chunks

### Issue: Out of memory
**Solution**:
- Reduce PROMPT_CACHE_MAX_SIZE
- Clear cache with `/reset` endpoint
- Process smaller documents

### Issue: Inaccurate answers
**Solution**:
- Increase TOP_K (retrieve more chunks)
- Decrease SIMILARITY_THRESHOLD (keep more chunks)
- Increase CHUNK_SIZE for better context
- Check document quality

---

## 📈 Monitoring

### Log Important Metrics

The system logs:
- ✅ Cache hits/misses
- ✅ Chunk counts and sizes
- ✅ Processing times
- ✅ Error messages with stack traces

Example log output:
```
Cache hit for prompt hash abc123d8
Split into 42 chunks
After deduplication: 40 unique chunks
After filtering short chunks: 38 chunks
Cached result. Cache size: 85/200
```

### Check Health

```bash
curl http://localhost:8000/health | jq
```

Returns:
```json
{
  "status": "ok",
  "file_indexed": true,
  "current_file": "document.csv",
  "chunks_count": 38,
  "prompt_cache_size": 85,
  "embeddings_cache_size": 142,
  "query_history_count": 23,
  "vectoring": "Gemini Embedding 2",
  "reranking": true,
  "memory_status": "optimized"
}
```

---

## 🎓 Learning Resources

### How RAG Works
1. **Retrieve**: Find relevant documents using vector similarity
2. **Augment**: Combine retrieval results with user query
3. **Generate**: Feed augmented prompt to LLM for response

### Key Concepts

- **Embeddings**: Vector representation of text (captures semantic meaning)
- **Vector Database**: Fast lookup of similar embeddings (FAISS)
- **Chunking**: Breaking documents into overlapping pieces
- **Re-ranking**: Second-pass filtering for better accuracy
- **Caching**: Store results to avoid expensive recomputation

---

## 📝 License

This project uses Google's Gemini API. Ensure compliance with Google's terms of service.

---

## 💡 Tips for Best Results

1. **Upload Quality Documents**: Clean, well-formatted files yield better results
2. **Ask Specific Questions**: "What is X?" works better than "Tell me about the document"
3. **Check Sources**: Always review provided source chunks
4. **Use Summaries**: For getting document overview without specific questions
5. **Monitor Cache**: Watch `prompt_cache_size` for hit rates
6. **Adjust Thresholds**: Experiment with SIMILARITY_THRESHOLD based on document type

---

**Version**: 1.0  
**Last Updated**: 2026-06-23  
**Status**: Production Ready ✅
