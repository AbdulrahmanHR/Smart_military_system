"""
Military Training Chatbot MVP - FastAPI Backend
Supporting both Arabic and English languages with RAG system
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Fix SQLite compatibility for ChromaDB (required for some deployment environments)
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
import pyarabic.araby as araby
import chromadb
from sentence_transformers import SentenceTransformer

# Configure logging for production deployment
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log', encoding='utf-8') if os.getenv("LOG_TO_FILE", "false").lower() == "true" else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("Starting Military Training Chatbot...")
    
    # Initialize Gemini
    gemini_initialized = initialize_gemini()
    if not gemini_initialized:
        logger.warning("Gemini not initialized - check GOOGLE_API_KEY")
    
    # Initialize RAG system
    await initialize_rag_system()
    logger.info("Chatbot startup completed")
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down Military Training Chatbot...")

# FastAPI app initialization
app = FastAPI(
    title="Military Training Chatbot",
    description="MVP Chatbot for Military Training with Arabic and English support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access - simplified for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - simplest for deployment
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    language: str = "en"  # "en" or "ar"
    selected_files: List[str] = []  # List of specific files to search in
    session_id: str = "default"  # Session identifier for conversation memory

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    language: str
    timestamp: str
    available_files: List[str] = []  # List of available files for selection

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables for RAG system
vectorstore: Optional[Chroma] = None
embeddings: Optional[HuggingFaceEmbeddings] = None
text_splitter: Optional[RecursiveCharacterTextSplitter] = None

# Conversation memory for each session
session_memories: Dict[str, ConversationBufferWindowMemory] = {}
max_memory_length = 10  # Keep last 10 exchanges

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

def initialize_gemini():
    """Initialize Google Gemini AI"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found in environment variables")
        return False
    
    try:
        genai.configure(api_key=api_key)
        logger.info("Google Gemini AI initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return False

def clean_arabic_text(text: str) -> str:
    """Clean and normalize Arabic text"""
    if not text:
        return text
    
    # Remove extra whitespaces
    text = " ".join(text.split())
    
    # Normalize Arabic text
    text = araby.strip_diacritics(text)
    text = araby.normalize_hamza(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_teh(text)
    
    return text

def detect_language(text: str) -> str:
    """Simple language detection"""
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return "en"
    
    arabic_ratio = arabic_chars / total_chars
    return "ar" if arabic_ratio > 0.3 else "en"

def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Get or create conversation memory for a session"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(
            k=max_memory_length,
            return_messages=True
        )
    return session_memories[session_id]

def get_available_files() -> List[str]:
    """Get list of available document files"""
    documents_dir = Path("documents")
    if not documents_dir.exists():
        return []
    
    available_files = []
    for file_path in documents_dir.glob("*"):
        if file_path.suffix.lower() in ['.pdf', '.txt']:
            available_files.append(file_path.name)
    
    return sorted(available_files)

async def initialize_rag_system():
    """Initialize the RAG system with multilingual embeddings"""
    global vectorstore, embeddings, text_splitter
    
    try:
        # Initialize multilingual embeddings
        logger.info("Loading multilingual embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # Initialize Chroma vectorstore with string path (ChromaDB requirement)
        persist_directory = Path("database/chroma_db").resolve()
        persist_directory.mkdir(parents=True, exist_ok=True)
        
        vectorstore = Chroma(
            persist_directory=str(persist_directory),  # Convert Path to string for ChromaDB
            embedding_function=embeddings,
            collection_name="military_training"
        )
        
        logger.info("RAG system initialized successfully")
        
        # Load initial documents if any exist
        await load_initial_documents()
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

async def load_initial_documents():
    """Load documents from the documents folder"""
    documents_dir = Path("documents")
    if not documents_dir.exists():
        documents_dir.mkdir(exist_ok=True)
        logger.info("Created documents directory")
        return
    
    document_files = list(documents_dir.glob("*.pdf")) + list(documents_dir.glob("*.txt"))
    
    if not document_files:
        logger.info("No initial documents found in documents folder")
        return
    
    for file_path in document_files:
        try:
            await process_document(file_path)
            logger.info(f"Loaded document: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load document {file_path.name}: {e}")

async def process_document(file_path: Path) -> bool:
    """Process and add document to vectorstore"""
    try:
        # Load document based on file type
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return False
        
        documents = loader.load()
        
        # Process each document
        processed_docs = []
        for doc in documents:
            # Clean text based on language
            content = doc.page_content
            detected_lang = detect_language(content)
            
            if detected_lang == "ar":
                content = clean_arabic_text(content)
            
            # Split into chunks
            chunks = text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                processed_doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path.name,
                        "language": detected_lang,
                        "chunk_id": i,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                processed_docs.append(processed_doc)
        
        # Add to vectorstore
        if processed_docs:
            vectorstore.add_documents(processed_docs)
            logger.info(f"Added {len(processed_docs)} chunks from {file_path.name}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error processing document {file_path.name}: {e}")
        return False

def get_relevant_context(question: str, language: str, selected_files: List[str] = None, k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant context from vectorstore with optional file filtering"""
    if not vectorstore:
        return []
    
    try:
        # Clean question if Arabic
        if language == "ar":
            question = clean_arabic_text(question)
        
        # Search for relevant documents
        if selected_files and len(selected_files) > 0:
            logger.info(f"Filtering by selected files: {selected_files}")
            # Get all results first, then filter manually for more reliable filtering
            all_results = vectorstore.similarity_search_with_score(question, k=k*3)  # Get more results to filter from
            
            # Filter results to only include selected files
            results = []
            for doc, score in all_results:
                if doc.metadata.get("source", "") in selected_files:
                    results.append((doc, score))
                    if len(results) >= k:  # Stop when we have enough results
                        break
            
            if not results:
                logger.warning(f"No results found for selected files {selected_files}, searching all files")
                results = vectorstore.similarity_search_with_score(question, k=k)
        else:
            logger.info("Searching all documents")
            results = vectorstore.similarity_search_with_score(question, k=k)
        
        context_docs = []
        for doc, score in results:
            context_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "language": doc.metadata.get("language", "en"),
                "score": score
            })
        
        return context_docs
    
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []

async def generate_response(question: str, context_docs: List[Dict[str, Any]], language: str, session_id: str) -> str:
    """Generate response using Gemini with conversation memory"""
    try:
        # Get conversation memory for this session
        memory = get_session_memory(session_id)
        
        # Prepare context from documents
        context_text = "\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['content']}"
            for doc in context_docs
        ])
        
        # Get conversation history from memory
        conversation_history = ""
        if memory.chat_memory.messages:
            conversation_history = "\n\nPrevious conversation:\n"
            for msg in memory.chat_memory.messages[-6:]:  # Last 3 exchanges (6 messages)
                if isinstance(msg, HumanMessage):
                    conversation_history += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    conversation_history += f"Assistant: {msg.content}\n"
        
        # Create system prompt based on language
        if language == "ar":
            system_prompt = """أنت مساعد تدريب عسكري مفيد. مهمتك هي الإجابة على الأسئلة حول الإجراءات العسكرية باستخدام الوثائق المقدمة.

إرشادات:
- استخدم فقط المعلومات من الوثائق المسترجعة
- راعي سياق المحادثة السابقة إذا كان متوفراً
- أجب باللغة العربية بوضوح ودقة
- قدم إجابات مباشرة ومفيدة
- اذكر مصادر المعلومات عند الإمكان
- إذا لم تجد المعلومة في الوثائق، قل ذلك بوضوح

السياق المسترجع:
{context}

{conversation_history}

السؤال الحالي: {question}

الإجابة:"""
        else:
            system_prompt = """You are a helpful military training assistant. Your task is to answer questions about military procedures using the provided documents.

Guidelines:
- Use only information from the retrieved documents
- Consider the previous conversation context if available
- Respond in English clearly and accurately
- Provide direct and helpful answers
- Mention source documents when applicable
- If information is not found in documents, state this clearly

Retrieved Context:
{context}

{conversation_history}

Current Question: {question}

Answer:"""
        
        # Format prompt
        prompt = system_prompt.format(
            context=context_text, 
            question=question,
            conversation_history=conversation_history
        )
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        
        answer = response.text if response.text else ("عذراً، لا يمكنني تقديم إجابة مناسبة." if language == "ar" else "Sorry, I cannot provide a suitable answer.")
        
        # Save conversation to memory
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        
        return answer
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_msg = "حدث خطأ في توليد الإجابة." if language == "ar" else "An error occurred while generating the response."
        return error_msg

# API Endpoints


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found. Please create frontend/index.html</h1>")

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    return {"message": "No favicon"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Military Training Chatbot is running"
    )

@app.get("/files")
async def get_files_endpoint():
    """Get list of available files for selection"""
    try:
        available_files = get_available_files()
        return {"files": available_files}
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with file selection and conversation memory"""
    try:
        logger.info(f"Chat request: question='{request.question}', selected_files={request.selected_files}, session_id={request.session_id}")
        
        # Detect language if not specified
        if not request.language:
            request.language = detect_language(request.question)
        
        # Get relevant context with optional file filtering
        context_docs = get_relevant_context(
            request.question, 
            request.language, 
            request.selected_files if request.selected_files else None
        )
        
        # Generate response with conversation memory
        answer = await generate_response(
            request.question, 
            context_docs, 
            request.language,
            request.session_id
        )
        
        # Extract sources
        sources = list(set([doc["source"] for doc in context_docs]))
        
        # Get available files for the response
        available_files = get_available_files()
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            language=request.language,
            timestamp=datetime.now().isoformat(),
            available_files=available_files
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process new document"""
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Save file
        file_path = Path("documents") / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        success = await process_document(file_path)
        
        if success:
            return {"message": f"Document {file.filename} uploaded and processed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process document")
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload document")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process the message (similar to chat endpoint)
            # For now, just echo back
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
