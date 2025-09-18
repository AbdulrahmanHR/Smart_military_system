"""
Document processing module for military training documents
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document ingestion, processing, and chunking for military training materials"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from Word documents"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def determine_document_category(self, filename: str, content: str) -> str:
        """Determine document category based on filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        if any(keyword in filename_lower or keyword in content_lower 
               for keyword in ['tactical', 'combat', 'strategy', 'maneuver']):
            return "Tactical Procedures"
        elif any(keyword in filename_lower or keyword in content_lower 
                 for keyword in ['equipment', 'weapon', 'gear', 'maintenance']):
            return "Equipment Training"
        elif any(keyword in filename_lower or keyword in content_lower 
                 for keyword in ['emergency', 'crisis', 'evacuation', 'medical']):
            return "Emergency Protocols"
        elif any(keyword in filename_lower or keyword in content_lower 
                 for keyword in ['leadership', 'command', 'coordination', 'team']):
            return "Leadership & Coordination"
        elif any(keyword in filename_lower or keyword in content_lower 
                 for keyword in ['physical', 'fitness', 'training', 'exercise']):
            return "Physical Training"
        elif any(keyword in filename_lower or keyword in content_lower 
                 for keyword in ['safety', 'security', 'protection', 'risk']):
            return "Safety Procedures"
        elif any(keyword in filename_lower or keyword in content_lower 
                 for keyword in ['communication', 'radio', 'signal', 'intel']):
            return "Communication Protocols"
        elif any(keyword in filename_lower or keyword in content_lower 
                 for keyword in ['mission', 'planning', 'operation', 'briefing']):
            return "Mission Planning"
        else:
            return "General Training"
    
    def process_document(self, file_path: Path) -> List[LangchainDocument]:
        """Process a single document and return chunked documents"""
        logger.info(f"Processing document: {file_path}")
        
        # Extract text based on file type
        file_extension = file_path.suffix.lower()
        if file_extension == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return []
        
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return []
        
        # Determine document category
        category = self.determine_document_category(file_path.name, text)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create LangChain documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            documents.append(LangchainDocument(page_content=chunk, metadata=metadata))
        
        logger.info(f"Created {len(documents)} chunks from {file_path}")
        return documents
    
    def process_directory(self, directory_path: Path) -> List[LangchainDocument]:
        """Process all supported documents in a directory"""
        supported_extensions = {'.pdf', '.docx', '.txt'}
        all_documents = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents = self.process_document(file_path)
                all_documents.extend(documents)
        
        logger.info(f"Processed {len(all_documents)} total chunks from {directory_path}")
        return all_documents
    
    def get_document_stats(self, documents: List[LangchainDocument]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not documents:
            return {"total_chunks": 0, "categories": {}, "sources": []}
        
        categories = {}
        sources = set()
        
        for doc in documents:
            category = doc.metadata.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
            sources.add(doc.metadata.get('filename', 'Unknown'))
        
        return {
            "total_chunks": len(documents),
            "categories": categories,
            "sources": list(sources)
        }
