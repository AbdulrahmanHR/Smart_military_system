"""
RAG (Retrieval-Augmented Generation) system for military training chatbot
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document as LangchainDocument
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.settings import config
from src.vector_database import VectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilitaryTrainingRAG:
    """RAG system specialized for military training scenarios"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
        self.rag_chain = self._create_rag_chain()
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize Google Gemini LLM"""
        try:
            llm = ChatGoogleGenerativeAI(
                model=config.LLM_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            logger.info("Initialized Google Gemini LLM")
            return llm
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the PTCF (Persona-Task-Context-Format) prompt template"""
        
        system_prompt = """
You are a seasoned Military Training Instructor with extensive experience across multiple domains of military operations. You possess the expertise of a tactical training specialist and maintain the professional bearing of a military education officer, while remaining approachable, patient, and committed to soldier development.

## YOUR MISSION
Your primary mission is to provide comprehensive military training guidance and answer questions about military procedures, tactics, and protocols. You must:
- Analyze training scenarios and provide step-by-step tactical guidance
- Break down complex military concepts for soldiers at various experience levels
- Provide practical, actionable advice based on established military doctrine
- Assess trainee understanding and suggest additional training resources
- Focus exclusively on training and education, not operational planning or classified information

## OPERATIONAL CONTEXT
You are operating in a professional military education environment, serving soldiers at various levels of experience and specialization. Your responses draw from a comprehensive knowledge base of military training manuals, standard operating procedures, and tactical guidelines. You must:
- Prioritize information from retrieved military training documents over general knowledge
- Consider real-world application and safety in all recommendations
- Focus on standard, non-classified training materials and procedures
- Maintain situational awareness appropriate for training environments

## RESPONSE FORMAT AND PROTOCOLS
Structure your responses with military precision and educational clarity:
- Provide clear, organized answers with logical flow
- Use professional military terminology balanced with educational explanations
- Format procedures as bullet points, sequences as numbered lists
- Highlight critical information in bold text
- ALWAYS cite source documents when available in this format: [Source: Document Name]
- When relevant training materials aren't found in the knowledge base, clearly state this limitation
- Recommend consulting human instructors or official channels for complex operational matters

## AVAILABLE CONTEXT
{context}

## SOLDIER'S QUESTION
{question}

## YOUR RESPONSE
Provide a comprehensive, authoritative response based on the retrieved training materials. Include specific references to source documents and maintain professional military communication standards while ensuring educational value.
"""
        
        return ChatPromptTemplate.from_template(system_prompt)
    
    def _format_retrieved_documents(self, docs: List[LangchainDocument]) -> str:
        """Format retrieved documents for context injection"""
        if not docs:
            return "No relevant training materials found in the knowledge base."
        
        formatted_context = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('filename', 'Unknown Document')
            category = doc.metadata.get('category', 'General')
            content = doc.page_content.strip()
            
            formatted_context.append(f"""
Document {i} - {source} (Category: {category}):
{content}
""")
        
        return "\n".join(formatted_context)
    
    def _create_rag_chain(self):
        """Create the RAG chain for processing queries"""
        
        def retrieve_documents(query_dict):
            """Retrieve relevant documents based on query"""
            query = query_dict.get("question", "")
            category_filter = query_dict.get("category_filter")
            
            docs = self.vector_db.similarity_search(
                query, 
                k=config.RETRIEVAL_K,
                category_filter=category_filter
            )
            
            return {
                "context": self._format_retrieved_documents(docs),
                "question": query,
                "retrieved_docs": docs
            }
        
        # Create the RAG chain
        chain = (
            RunnablePassthrough() |
            retrieve_documents |
            self.prompt_template |
            self.llm |
            StrOutputParser()
        )
        
        return chain
    
    def query(
        self, 
        question: str, 
        category_filter: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """Process a training query and return response with sources"""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Prepare input for the chain
            chain_input = {
                "question": question,
                "category_filter": category_filter
            }
            
            # Get retrieved documents first for source information
            retrieved_docs = self.vector_db.similarity_search_with_score(
                question,
                k=config.RETRIEVAL_K,
                category_filter=category_filter
            )
            
            # Generate response using the RAG chain
            response = self.rag_chain.invoke(chain_input)
            
            # Prepare source information
            sources = []
            if return_sources and retrieved_docs:
                for doc, score in retrieved_docs:
                    sources.append({
                        "filename": doc.metadata.get('filename', 'Unknown'),
                        "category": doc.metadata.get('category', 'General'),
                        "similarity_score": float(score),
                        "content_preview": doc.page_content[:200] + "..."
                    })
            
            return {
                "response": response,
                "sources": sources,
                "query": question,
                "category_filter": category_filter,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your question. Please try again or rephrase your query.",
                "sources": [],
                "query": question,
                "category_filter": category_filter,
                "num_sources": 0,
                "error": str(e)
            }
    
    def get_training_suggestions(self, topic: str) -> List[str]:
        """Get training suggestions based on a topic"""
        try:
            suggestions_query = f"What are the key training areas and procedures for {topic}?"
            
            docs = self.vector_db.similarity_search(suggestions_query, k=10)
            
            suggestions = []
            for doc in docs:
                category = doc.metadata.get('category', 'General')
                if category not in [s.split(':')[0] for s in suggestions]:
                    preview = doc.page_content[:100].replace('\n', ' ')
                    suggestions.append(f"{category}: {preview}...")
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error getting training suggestions: {e}")
            return []
    
    def add_training_documents(self, documents: List[LangchainDocument]) -> bool:
        """Add new training documents to the knowledge base"""
        try:
            return self.vector_db.add_documents(documents)
        except Exception as e:
            logger.error(f"Error adding training documents: {e}")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the current knowledge base"""
        try:
            return self.vector_db.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {"total_documents": 0, "categories": [], "sources": []}
    
    def validate_system(self) -> Dict[str, bool]:
        """Validate that all system components are working"""
        validation_results = {
            "llm_connection": False,
            "vector_database": False,
            "retrieval_system": False
        }
        
        try:
            # Test LLM connection
            test_response = self.llm.invoke("Test")
            validation_results["llm_connection"] = bool(test_response)
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
        
        try:
            # Test vector database
            info = self.vector_db.get_collection_info()
            validation_results["vector_database"] = isinstance(info, dict)
        except Exception as e:
            logger.error(f"Vector database validation failed: {e}")
        
        try:
            # Test retrieval system
            test_docs = self.vector_db.similarity_search("test query", k=1)
            validation_results["retrieval_system"] = isinstance(test_docs, list)
        except Exception as e:
            logger.error(f"Retrieval system validation failed: {e}")
        
        return validation_results
