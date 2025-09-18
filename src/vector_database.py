"""
Vector database module using Chroma for military training document storage and retrieval
"""
import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangchainDocument

from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages the Chroma vector database for military training documents"""
    
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.client = None
        self.vectorstore = None
        self._initialize_database()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings with Arabic and multilingual support"""
        try:
            # Priority list of Arabic-supporting embedding models
            arabic_models = [
                config.EMBEDDING_MODEL,
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # Supports Arabic
                "sentence-transformers/distiluse-base-multilingual-cased",      # Supports Arabic
                "sentence-transformers/LaBSE",                                  # Language-agnostic model
                "sentence-transformers/all-MiniLM-L6-v2"                       # Fallback
            ]
            
            for model_name in arabic_models:
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={
                            'device': 'cpu',
                            'trust_remote_code': True
                        },
                        encode_kwargs={
                            'normalize_embeddings': True
                        }
                    )
                    logger.info(f"Successfully initialized Arabic-supporting embeddings: {model_name}")
                    return embeddings
                except Exception as model_error:
                    logger.warning(f"Failed to initialize {model_name}: {model_error}")
                    continue
            
            raise Exception("All Arabic-supporting embedding models failed to initialize")
            
        except Exception as e:
            logger.error(f"Error initializing Arabic embeddings: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize Chroma database with persistent storage"""
        try:
            # Ensure the directory exists
            config.ensure_directories()
            
            # Create Chroma client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(config.PERSIST_DIRECTORY),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize vectorstore with error handling
            try:
                self.vectorstore = Chroma(
                    client=self.client,
                    collection_name=config.COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=str(config.PERSIST_DIRECTORY)
                )
            except Exception as vectorstore_error:
                logger.warning(f"Error creating vectorstore, attempting to reset: {vectorstore_error}")
                # Try to delete and recreate the collection
                try:
                    self.client.delete_collection(config.COLLECTION_NAME)
                except:
                    pass  # Collection might not exist
                
                self.vectorstore = Chroma(
                    client=self.client,
                    collection_name=config.COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=str(config.PERSIST_DIRECTORY)
                )
            
            logger.info(f"Initialized vector database at {config.PERSIST_DIRECTORY}")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise
    
    def add_documents(self, documents: List[LangchainDocument]) -> bool:
        """Add documents to the vector database"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            # Add documents to vectorstore
            self.vectorstore.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        category_filter: Optional[str] = None
    ) -> List[LangchainDocument]:
        """Perform similarity search with optional category filtering"""
        try:
            k = k or config.RETRIEVAL_K
            
            # Prepare filter if category is specified
            filter_dict = None
            if category_filter and category_filter != "All Categories":
                filter_dict = {"category": category_filter}
            
            # Perform similarity search
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query, 
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = None,
        category_filter: Optional[str] = None
    ) -> List[tuple]:
        """Perform similarity search and return documents with similarity scores"""
        try:
            k = k or config.RETRIEVAL_K
            
            # Prepare filter if category is specified
            filter_dict = None
            if category_filter and category_filter != "All Categories":
                filter_dict = {"category": category_filter}
            
            # Perform similarity search with scores
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query, 
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            logger.info(f"Retrieved {len(results)} documents with scores for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            collection = self.client.get_collection(config.COLLECTION_NAME)
            count = collection.count()
            
            # Get sample of metadata to understand categories
            if count > 0:
                sample_results = collection.get(limit=min(100, count))
                categories = set()
                sources = set()
                
                for metadata in sample_results.get('metadatas', []):
                    if metadata:
                        categories.add(metadata.get('category', 'Unknown'))
                        sources.add(metadata.get('filename', 'Unknown'))
                
                return {
                    "total_documents": count,
                    "categories": list(categories),
                    "sources": list(sources)
                }
            else:
                return {"total_documents": 0, "categories": [], "sources": []}
                
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"total_documents": 0, "categories": [], "sources": []}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection (use with caution)"""
        try:
            self.client.delete_collection(config.COLLECTION_NAME)
            logger.info(f"Deleted collection: {config.COLLECTION_NAME}")
            
            # Reinitialize the vectorstore
            self._initialize_database()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def update_documents(self, documents: List[LangchainDocument]) -> bool:
        """Update existing documents in the database"""
        try:
            # For Chroma, we typically delete and re-add
            # This is a simplified approach - for production, implement proper updates
            self.delete_collection()
            return self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            return False
    
    def search_by_metadata(
        self, 
        metadata_filter: Dict[str, Any], 
        limit: int = 10
    ) -> List[LangchainDocument]:
        """Search documents by metadata filters"""
        try:
            collection = self.client.get_collection(config.COLLECTION_NAME)
            results = collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            # Convert results back to LangchainDocument format
            documents = []
            for i, (doc_id, document, metadata) in enumerate(zip(
                results.get('ids', []),
                results.get('documents', []),
                results.get('metadatas', [])
            )):
                documents.append(LangchainDocument(
                    page_content=document,
                    metadata=metadata or {}
                ))
            
            logger.info(f"Found {len(documents)} documents matching metadata filter")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
