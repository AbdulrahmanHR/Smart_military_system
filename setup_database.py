"""
Database Setup Script for Military Training Chatbot
This script initializes the vector database with sample training documents
"""
import logging
from pathlib import Path

from config.settings import config
from src.document_processor import DocumentProcessor
from src.vector_database import VectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database with sample documents"""
    try:
        logger.info("Starting database setup...")
        
        # Ensure directories exist
        config.ensure_directories()
        
        # Initialize components
        doc_processor = DocumentProcessor()
        vector_db = VectorDatabase()
        
        # Process sample documents
        documents_dir = config.DOCUMENTS_DIR
        if not documents_dir.exists():
            logger.error(f"Documents directory not found: {documents_dir}")
            return False
        
        logger.info(f"Processing documents from: {documents_dir}")
        all_documents = doc_processor.process_directory(documents_dir)
        
        if not all_documents:
            logger.warning("No documents found to process")
            return False
        
        # Add documents to vector database
        logger.info(f"Adding {len(all_documents)} document chunks to vector database...")
        success = vector_db.add_documents(all_documents)
        
        if success:
            logger.info("✅ Database setup completed successfully!")
            
            # Display statistics
            stats = doc_processor.get_document_stats(all_documents)
            logger.info(f"📊 Database Statistics:")
            logger.info(f"   Total chunks: {stats['total_chunks']}")
            logger.info(f"   Categories: {list(stats['categories'].keys())}")
            logger.info(f"   Source files: {len(stats['sources'])}")
            
            for category, count in stats['categories'].items():
                logger.info(f"   - {category}: {count} chunks")
            
            return True
        else:
            logger.error("❌ Failed to add documents to database")
            return False
            
    except Exception as e:
        logger.error(f"Error during database setup: {e}")
        return False

if __name__ == "__main__":
    # Validate configuration
    try:
        config.validate_config()
        main()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        print("\n🔧 Setup Instructions:")
        print("1. Create a .env file in the project root")
        print("2. Add your Google API key: GOOGLE_API_KEY=your_key_here")
        print("3. Get API key from: https://makersuite.google.com/app/apikey")
        print("4. Run this script again after configuration")
