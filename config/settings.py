"""
Configuration settings for the Military Training Chatbot
"""
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

def get_secret(key: str, default=None):
    """Get secret from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets.get(key, os.getenv(key, default))
    except:
        # Fallback to environment variables (for local development)
        return os.getenv(key, default)

class Config:
    """Application configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    
    # API Configuration
    GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
    
    # Model Configuration (Arabic-supporting defaults)
    EMBEDDING_MODEL = get_secret("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    LLM_MODEL = get_secret("LLM_MODEL", "gemini-1.5-flash")
    
    # RAG Configuration
    CHUNK_SIZE = int(get_secret("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(get_secret("CHUNK_OVERLAP", "100"))
    RETRIEVAL_K = int(get_secret("RETRIEVAL_K", "5"))  # Number of documents to retrieve
    
    # LLM Parameters
    TEMPERATURE = float(get_secret("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(get_secret("MAX_TOKENS", "1024"))
    
    # Chroma Configuration
    COLLECTION_NAME = get_secret("COLLECTION_NAME", "military_training_docs")
    PERSIST_DIRECTORY = get_secret("PERSIST_DIRECTORY", str(VECTOR_DB_DIR))
    
    # Streamlit Configuration
    PAGE_TITLE = get_secret("PAGE_TITLE", "Military Training Assistant")
    PAGE_ICON = get_secret("PAGE_ICON", "⚔️")
    LAYOUT = get_secret("LAYOUT", "wide")
    
    # Training Categories (English and Arabic)
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
    
    # Arabic Training Categories
    TRAINING_CATEGORIES_AR = [
        "جميع الفئات",
        "الإجراءات التكتيكية",
        "تدريب المعدات",
        "بروتوكولات الطوارئ", 
        "القيادة والتنسيق",
        "التدريب البدني",
        "إجراءات السلامة",
        "بروتوكولات الاتصال",
        "تخطيط المهام"
    ]
    
    # Language settings
    SUPPORTED_LANGUAGES = ["en", "ar"]
    DEFAULT_LANGUAGE = "en"
    
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
