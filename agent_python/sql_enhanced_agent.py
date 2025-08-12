#!/usr/bin/env python3
"""
Enhanced RAG Chatbot with SQL Database
Uses MySQL for document storage, chunking, and chat history
"""

import os
import logging
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import uvicorn
from graph_router import build_graph  # at top
# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()
router = None
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '123456')
DB_NAME = os.getenv('DB_NAME', 'internship_chat1')

# Configuration flags
DISABLE_WEB_SEARCH = os.getenv('DISABLE_WEB_SEARCH', 'false').lower() == 'true'

# Global variables
llm = None
embeddings = None
search_tool = None
agent = None
db_connection = None

# Pydantic models
class ChatMessage(BaseModel):
    userId: int
    message: str
    system_prompt: Optional[str] = "You are a helpful AI assistant that can search through documents and the web to provide accurate information. Be CONCISE and DIRECT in your responses. Keep answers brief and to the point without unnecessary elaboration."

class DocumentUpload(BaseModel):
    title: str
    content: str
    userId: int

class ChatResponse(BaseModel):
    reply: str
    sources: List[Dict[str, Any]] = []
    web_results: List[Dict[str, Any]] = []

# Database connection
def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            autocommit=True
        )
        return connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        raise

def initialize_embeddings():
    """Initialize embeddings with fallback options"""
    embedding_models = [
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]

    for model_name in embedding_models:
        try:
            logger.info(f"🔄 Trying embeddings model: {model_name}")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder="./embeddings_cache"
            )
            logger.info(f"✅ Embeddings initialized with {model_name}")
            return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {model_name}: {e}")
            continue

    # If all models fail, try with a simple model
    try:
        logger.info("🔄 Trying simple embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            cache_folder="./embeddings_cache",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("✅ Simple embeddings initialized")
        return embeddings
    except Exception as e:
        logger.error(f"❌ All embedding models failed: {e}")
        raise Exception("Could not initialize any embedding model")

def initialize_components():
    """Initialize all components"""
    global llm, embeddings, search_tool, agent, router
    
    try:
        # Initialize LLM
        logger.info("🔄 Initializing LLM...")
        if not GEMINI_API_KEY:
            raise Exception("GEMINI_API_KEY not set")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        logger.info("✅ LLM initialized")
        
        # Initialize embeddings
        logger.info("🔄 Initializing embeddings...")
        embeddings = initialize_embeddings()
        
        # Initialize search tool
        if TAVILY_API_KEY and not DISABLE_WEB_SEARCH:
            logger.info("🔄 Initializing Tavily search tool...")
            search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
            logger.info("✅ Tavily search tool initialized")
        else:
            logger.warning("⚠️ TAVILY_API_KEY not set or DISABLE_WEB_SEARCH is true - web search disabled")
            search_tool = None
        
        # Initialize agent
        logger.info("🔄 Initializing agent...")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        tools = []
        
        # Document search tool
        doc_search_tool = Tool(
            name="document_search",
            description="Search through uploaded documents and knowledge base",
            func=lambda q: search_documents(q)
        )
        tools.append(doc_search_tool)
        
        # Web search tool
        if search_tool:
            tools.append(search_tool)
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
        
        logger.info("✅ Agent initialized successfully")

        # Build router graph with injected dependencies
        try:
            router = build_graph(llm_instance=llm, retrieve_top_chunks_fn=retrieve_top_chunks, web_search_fn=web_search)
            logger.info("✅ Router graph initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing router graph: {e}")
            router = None
        
    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")
        raise

def search_documents(query: str, top_k: int = 3):
    """Search documents using embeddings (formatted string)."""
    try:
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search in database
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Search in document chunks
        cursor.execute("""
            SELECT dc.id, dc.content, dc.metadata, d.title as document_title, dc.embedding
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
        """)
        
        chunks = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not chunks:
            return "No documents found in the knowledge base."
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            if chunk['embedding']:
                chunk_embedding = json.loads(chunk['embedding'])
                # Simple cosine similarity
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                similarities.append({
                    'content': chunk['content'],
                    'title': chunk['document_title'],
                    'similarity': similarity
                })
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:top_k]
        
        if not top_results:
            return "No relevant documents found."
        
        # Format results
        result = "Relevant documents found:\n\n"
        for i, doc in enumerate(top_results, 1):
            result += f"{i}. {doc['title']}\n"
            result += f"   Content: {doc['content'][:200]}...\n"
            result += f"   Relevance: {doc['similarity']:.3f}\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"Error searching documents: {str(e)}"

def retrieve_top_chunks(query: str, top_k: int = 3):
    """Return a list of top matching chunks with scores for a query."""
    try:
        query_embedding = embeddings.embed_query(query)
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT dc.id, dc.content, dc.metadata, d.title as document_title, dc.embedding
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
        """)
        chunks = cursor.fetchall()
        cursor.close()
        connection.close()
        if not chunks:
            return []
        matches = []
        for chunk in chunks:
            emb_json = chunk.get('embedding')
            if not emb_json:
                continue
            try:
                chunk_embedding = json.loads(emb_json)
                score = cosine_similarity(query_embedding, chunk_embedding)
                matches.append({
                    'title': chunk['document_title'],
                    'content': chunk['content'],
                    'similarity': float(score)
                })
            except Exception:
                continue
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]
    except Exception as e:
        logger.error(f"retrieve_top_chunks error: {e}")
        return []

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def save_chat_message(user_id: int, role: str, content: str, metadata: dict = None):
    """Save chat message to database"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO chat_messages (user_id, role, content, metadata)
            VALUES (%s, %s, %s, %s)
        """, (user_id, role, content, metadata_json))
        
        cursor.close()
        connection.close()
        logger.info(f"Saved {role} message for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")

def get_chat_history(user_id: int, limit: int = 50):
    """Get chat history for user"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT role, content, metadata, created_at
            FROM chat_messages
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))
        
        messages = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return messages
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

def web_search(query: str, max_results: int = 3):
    """Use Tavily to search the web. Returns list of {title, content, url}."""
    results: List[Dict[str, Any]] = []
    
    # Check if web search is disabled
    if DISABLE_WEB_SEARCH:
        logger.info("Web search is disabled")
        return results
    
    try:
        if not search_tool:
            return results
        
        # Try to invoke the search tool
        try:
            raw = search_tool.invoke(query)
        except Exception as e:
            # Check if it's a connection error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['connection', 'timeout', 'network', 'unreachable', 'maxretry']):
                logger.warning(f"Tavily connection error: {e}")
                return results  # Return empty results instead of error message
            # For other errors, try the run method
            try:
                raw = search_tool.run(query)
            except Exception as e2:
                logger.warning(f"Tavily search failed: {e2}")
                return results
        
        # Process results only if we got valid data
        if isinstance(raw, list) and raw:
            for r in raw[:max_results]:
                # Limit content length to prevent overly verbose responses
                content = r.get("content") or r.get("snippet") or ""
                if len(content) > 300:  # Limit to 300 characters
                    content = content[:300] + "..."
                
                results.append({
                    "title": r.get("title") or r.get("source") or "",
                    "content": content,
                    "url": r.get("url") or r.get("link") or ""
                })
        elif isinstance(raw, str) and raw and not raw.startswith("ConnectionError"):
            # Only process string responses that aren't error messages
            if len(raw) > 300:
                raw = raw[:300] + "..."
            results.append({"title": "web", "content": raw, "url": ""})
        
        # Log if no valid results were found
        if not results:
            logger.info(f"No valid web search results for query: {query}")
            
    except Exception as e:
        logger.warning(f"web_search error: {e}")
    
    return results

def trim_verbose_response(response: str, max_words: int = 50) -> str:
    """Trim overly verbose responses to keep them concise"""
    words = response.split()
    if len(words) <= max_words:
        return response
    
    # Take first max_words and try to end at a sentence boundary
    trimmed = " ".join(words[:max_words])
    
    # Look for the last sentence ending in the trimmed text
    last_period = trimmed.rfind('.')
    last_exclamation = trimmed.rfind('!')
    last_question = trimmed.rfind('?')
    
    # Find the latest sentence ending
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    if last_sentence_end > max_words * 0.6:  # If we have a reasonable sentence ending
        trimmed = trimmed[:last_sentence_end + 1]
    else:
        # Otherwise just add ellipsis
        trimmed += "..."
    
    return trimmed

def test_tavily_connectivity():
    """Test if Tavily API is accessible"""
    try:
        if not search_tool:
            return False, "Tavily not configured"
        
        # Try a simple search
        test_results = web_search("test", max_results=1)
        if test_results:
            return True, "Tavily working"
        else:
            return False, "Tavily returned no results"
    except Exception as e:
        return False, f"Tavily error: {str(e)}"

# FastAPI app
app = FastAPI(title="Enhanced RAG Chatbot with SQL Database")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("🚀 Starting Enhanced RAG Chatbot with SQL...")
    logger.info(f"GEMINI_API_KEY: {'✅ SET' if GEMINI_API_KEY else '❌ NOT SET'}")
    logger.info(f"TAVILY_API_KEY: {'✅ SET' if TAVILY_API_KEY else '❌ NOT SET'}")
    logger.info(f"DISABLE_WEB_SEARCH: {'✅ ENABLED' if not DISABLE_WEB_SEARCH else '❌ DISABLED'}")
    
    try:
        initialize_components()
        logger.info("✅ All components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    """Main chat endpoint"""
    try:
        if not router:
            raise HTTPException(status_code=500, detail="Router not initialized. Check server logs.")

        logger.info(f"Received chat request from user {request.userId}")
        logger.info(f"Message: {request.message}")

        # Save user message
        save_chat_message(request.userId, 'user', request.message)

        # Route the query
        try:
            route_result = router.invoke({"input": request.message})
            reply_text = route_result.get("answer") or "I couldn't generate a response."
        except Exception as e:
            logger.warning(f"Router failed, using fallback: {e}")
            # Fallback: try to use knowledge base directly
            try:
                chunks = retrieve_top_chunks(request.message, top_k=3)
                if chunks and any(c.get('similarity', 0) > 0.3 for c in chunks):
                    # We have relevant knowledge base content
                    ctx = "".join([f"\n[KB {i+1}: {c.get('title','')}]\n{c.get('content','')}\n" for i, c in enumerate(chunks)])
                    prompt = (
                        "You are a helpful AI assistant. Use ONLY the knowledge base context provided below to answer the question.\n"
                        "IMPORTANT: Be CONCISE and DIRECT. Keep your answer brief and to the point.\n"
                        "If the information is not found in the knowledge base, simply say 'I don't have that information in my knowledge base.'\n\n"
                        f"Context:\n{ctx}\n\nQuestion: {request.message}\n\n"
                        "Answer (be concise):"
                    )
                    resp = llm.invoke(prompt)
                    reply_text = getattr(resp, "content", None) or str(resp)
                else:
                    # No relevant knowledge base content
                    reply_text = "I'm having trouble accessing external information right now. Please try again later or ask me about something in my knowledge base."
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                reply_text = "I'm experiencing technical difficulties. Please try again later."
        
        # Trim verbose responses to keep them concise
        reply_text = trim_verbose_response(str(reply_text), max_words=50)
        
        raw_sources = route_result.get("sources", []) or []

        # Save assistant response
        save_chat_message(request.userId, 'assistant', str(reply_text))

        # Format sources and web_results
        formatted_sources = []
        web_results = []
        for src in raw_sources:
            title = src.get("title") or src.get("document_title") or src.get("source") or ""
            content = src.get("content") or src.get("page_content") or ""
            score = src.get("similarity")
            url = src.get("url")
            formatted_sources.append({
                "title": title,
                "content": (content[:200] + "...") if content else "",
                "score": score,
            })
            if url:
                web_results.append({
                    "title": title,
                    "content": content,
                    "url": url,
                })

        return ChatResponse(
            reply=str(reply_text),
            sources=formatted_sources,
            web_results=web_results[:5],
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_document")
async def upload_document(doc: DocumentUpload):
    """Upload a document to the knowledge base"""
    try:
        logger.info(f"Uploading document: {doc.title}")
        
        # Save document
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            INSERT INTO documents (title, content, user_id)
            VALUES (%s, %s, %s)
        """, (doc.title, doc.content, doc.userId))
        
        document_id = cursor.lastrowid
        
        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(doc.content)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks with embeddings
        for i, chunk in enumerate(chunks):
            try:
                # Get embedding for chunk
                chunk_embedding = embeddings.embed_query(chunk)
                
                # Save chunk
                cursor.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    document_id,
                    i,
                    chunk,
                    json.dumps(chunk_embedding),
                    json.dumps({"chunk_index": i, "document_title": doc.title})
                ))
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                # Save chunk without embedding
                cursor.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, metadata)
                    VALUES (%s, %s, %s, %s)
                """, (
                    document_id,
                    i,
                    chunk,
                    json.dumps({"chunk_index": i, "document_title": doc.title})
                ))
        
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully uploaded document with {len(chunks)} chunks")
        
        return {
            "message": f"Document uploaded successfully with {len(chunks)} chunks",
            "document_id": document_id,
            "chunks": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all documents"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT d.id, d.title, d.content, d.created_at,
                   COUNT(dc.id) as chunk_count
            FROM documents d
            LEFT JOIN document_chunks dc ON d.id = dc.document_id
            GROUP BY d.id, d.title, d.content, d.created_at
            ORDER BY d.created_at DESC
        """)
        
        documents = cursor.fetchall()
        cursor.close()
        connection.close()
        
        logger.info(f"Found {len(documents)} documents")
        
        # Format response
        formatted_docs = []
        for doc in documents:
            try:
                formatted_docs.append({
                    "id": doc.get('id'),
                    "title": doc.get('title', 'Unknown'),
                    "chunks": doc.get('chunk_count', 0) or 0,
                    "created_at": doc.get('created_at').isoformat() if doc.get('created_at') else None
                })
            except Exception as e:
                logger.error(f"Error formatting document {doc}: {e}")
                # Add a safe fallback
                formatted_docs.append({
                    "id": doc.get('id', 0),
                    "title": doc.get('title', 'Unknown'),
                    "chunks": 0,
                    "created_at": None
                })
        
        logger.info(f"Successfully formatted {len(formatted_docs)} documents")
        return {"documents": formatted_docs}
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{title}")
async def delete_document(title: str):
    """Delete a document"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("DELETE FROM documents WHERE title = %s", (title,))
        
        deleted_count = cursor.rowcount
        cursor.close()
        connection.close()
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document '{title}' deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{user_id}")
async def get_chat_history_endpoint(user_id: int, limit: int = 50):
    """Get chat history for a user"""
    try:
        messages = get_chat_history(user_id, limit)
        return {"messages": messages}
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/toggle_web_search")
async def toggle_web_search():
    """Toggle web search on/off for troubleshooting"""
    global DISABLE_WEB_SEARCH
    DISABLE_WEB_SEARCH = not DISABLE_WEB_SEARCH
    
    # Reinitialize components to reflect the change
    try:
        initialize_components()
        status = "disabled" if DISABLE_WEB_SEARCH else "enabled"
        logger.info(f"Web search {status}")
        return {"message": f"Web search {status}", "web_search_enabled": not DISABLE_WEB_SEARCH}
    except Exception as e:
        logger.error(f"Error reinitializing components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        connection.close()
        
        # Test Tavily connectivity
        tavily_status, tavily_message = test_tavily_connectivity()

        return {
            "status": "healthy",
            "database": "connected",
            "llm": "initialized" if llm else "not initialized",
            "embeddings": "initialized" if embeddings else "not initialized",
            "agent": "initialized" if agent else "not initialized",
            "tavily": {
                "status": tavily_status,
                "message": tavily_message
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
