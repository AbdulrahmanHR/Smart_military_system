"""
Configuration settings for the Military Training Chatbot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))  # Number of documents to retrieve
    
    # LLM Parameters
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
    
    # Chroma Configuration
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "military_training_docs")
    PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", str(VECTOR_DB_DIR))
    
    # Streamlit Configuration
    PAGE_TITLE = os.getenv("PAGE_TITLE", "Military Training Assistant")
    PAGE_ICON = os.getenv("PAGE_ICON", "⚔️")
    LAYOUT = os.getenv("LAYOUT", "wide")
    
    # Training Categories
    TRAINING_CATEGORIES = [
        "All Categories",
        "Tactical Procedures",
        "Equipment Training", 
        "Emergency Protocols",
        "Leadership & Coordination",
        "Physical Training",
        "Safety Procedures",
        "Communication Protocols",
        "Mission Planning"
    ]
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        cls.ensure_directories()
        return True

# Initialize configuration
config = Config()
