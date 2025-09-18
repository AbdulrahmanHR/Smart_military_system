#!/usr/bin/env python3
"""
Arabic Support Setup Script for Smart Military System
This script helps set up Arabic language support for the military training system.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required Arabic language dependencies"""
    try:
        logger.info("Installing Arabic language support dependencies...")
        
        # Install basic requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Install additional Arabic support packages
        arabic_packages = [
            "arabic-reshaper>=3.0.0",
            "python-bidi>=0.4.2"
        ]
        
        for package in arabic_packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        logger.info("✅ Arabic dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during installation: {e}")
        return False

def check_arabic_documents():
    """Check if Arabic documents exist and are properly encoded"""
    documents_dir = Path("data/documents")
    arabic_files = [
        "tactical_procedures_ar.txt",
        "equipment_training_ar.txt"
    ]
    
    logger.info("Checking Arabic documents...")
    
    for filename in arabic_files:
        file_path = documents_dir / filename
        if not file_path.exists():
            logger.warning(f"Arabic document missing: {filename}")
            continue
            
        try:
            # Test if file can be read with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if file contains Arabic characters
            arabic_chars = sum(1 for char in content if '\u0600' <= char <= '\u06FF')
            if arabic_chars > 0:
                logger.info(f"✅ {filename}: Found {arabic_chars} Arabic characters")
            else:
                logger.warning(f"⚠️ {filename}: No Arabic characters detected")
                
        except UnicodeDecodeError:
            logger.error(f"❌ {filename}: Encoding error - file may not be UTF-8")
        except Exception as e:
            logger.error(f"❌ {filename}: Error reading file - {e}")
    
    return True

def setup_arabic_embedding_model():
    """Download and cache the Arabic-supporting embedding model"""
    try:
        logger.info("Setting up Arabic-supporting embedding model...")
        
        from sentence_transformers import SentenceTransformer
        
        # Try to load the multilingual model
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        logger.info(f"Downloading and caching model: {model_name}")
        
        model = SentenceTransformer(model_name)
        
        # Test with Arabic text
        test_text = "هذا نص تجريبي باللغة العربية"
        embedding = model.encode(test_text)
        
        logger.info(f"✅ Arabic embedding model ready! Embedding dimension: {len(embedding)}")
        return True
        
    except ImportError:
        logger.error("❌ sentence-transformers not installed. Please install requirements first.")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to setup embedding model: {e}")
        return False

def setup_vector_database():
    """Initialize vector database with Arabic documents"""
    try:
        logger.info("Setting up vector database with Arabic support...")
        
        from setup_database import main as setup_db
        
        # Run the database setup
        success = setup_db()
        
        if success:
            logger.info("✅ Vector database initialized with Arabic documents!")
        else:
            logger.error("❌ Failed to initialize vector database")
            
        return success
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Make sure all dependencies are installed")
        return False
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def test_arabic_functionality():
    """Test basic Arabic functionality"""
    try:
        logger.info("Testing Arabic language detection...")
        
        # Test language detection
        from src.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test Arabic text
        arabic_text = "هذا نص باللغة العربية للاختبار"
        detected_lang = processor.detect_language(arabic_text)
        
        if detected_lang == "ar":
            logger.info("✅ Arabic language detection working correctly")
        else:
            logger.warning(f"⚠️ Language detection returned: {detected_lang} (expected: ar)")
        
        # Test English text
        english_text = "This is English text for testing"
        detected_lang = processor.detect_language(english_text)
        
        if detected_lang == "en":
            logger.info("✅ English language detection working correctly")
        else:
            logger.warning(f"⚠️ Language detection returned: {detected_lang} (expected: en)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Arabic functionality test failed: {e}")
        return False

def check_configuration():
    """Check if configuration is properly set up"""
    try:
        logger.info("Checking configuration...")
        
        from config.settings import config
        
        # Check if API key is set
        if not config.GOOGLE_API_KEY:
            logger.error("❌ GOOGLE_API_KEY not set in environment variables")
            logger.info("Please create a .env file with your Google API key:")
            logger.info("GOOGLE_API_KEY=your_api_key_here")
            return False
        
        # Check directories
        config.ensure_directories()
        logger.info("✅ Configuration and directories OK")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration check failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Setting up Arabic support for Smart Military System...")
    logger.info("=" * 60)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Checking configuration", check_configuration),
        ("Checking Arabic documents", check_arabic_documents),
        ("Setting up embedding model", setup_arabic_embedding_model),
        ("Testing Arabic functionality", test_arabic_functionality),
        ("Initializing vector database", setup_vector_database),
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        logger.info(f"\n📋 {step_name}...")
        try:
            success = step_function()
            if not success:
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            failed_steps.append(step_name)
    
    logger.info("\n" + "=" * 60)
    
    if failed_steps:
        logger.error(f"❌ Setup completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            logger.error(f"   - {step}")
        logger.info("\nPlease review the errors above and try again.")
        return False
    else:
        logger.info("🎉 Arabic support setup completed successfully!")
        logger.info("\nYou can now:")
        logger.info("1. Run the application: streamlit run app.py")
        logger.info("2. Select Arabic language from the sidebar")
        logger.info("3. Ask questions in Arabic")
        logger.info("4. Upload Arabic documents for training")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
