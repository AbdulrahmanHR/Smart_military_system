"""
Military Training Chatbot - Streamlit Interface
"""
import streamlit as st
import logging
from datetime import datetime
from typing import List, Dict, Any
import json
import os

from config.settings import config
from src.rag_system import MilitaryTrainingRAG
from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Simplified CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #228B22);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .stats-metric {
        background-color: #f0f9f0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #d4e8d4;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Simplified chat styling using Streamlit's built-in components */
    .stChatMessage {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "rag_system" not in st.session_state:
        with st.spinner("Initializing Military Training Assistant..."):
            try:
                st.session_state.rag_system = MilitaryTrainingRAG()
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.stop()
    
    if "knowledge_base_stats" not in st.session_state:
        st.session_state.knowledge_base_stats = {}
    
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = "All Categories"

def display_header():
    """Display the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>⚔️ Military Training Assistant</h1>
        <p>AI-Powered Training Support for Military Personnel</p>
    </div>
    """, unsafe_allow_html=True)

def display_chat_interface():
    """Display the main chat interface using Streamlit's native chat components"""
    st.subheader("💬 Training Discussion")
    
    # Display chat history using Streamlit's chat message component
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            st.caption(f"⏰ {message['timestamp']}")
            
            # Display sources if available (for assistant messages)
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"📚 Sources ({len(message['sources'])} documents)", expanded=False):
                    for j, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        **Document {j}:** {source['filename']}  
                        **Category:** {source['category']}  
                        **Relevance Score:** {source['similarity_score']:.3f}  
                        **Preview:** {source['content_preview']}
                        """)
                        if j < len(message["sources"]):
                            st.divider()

def handle_user_input():
    """Handle user input and generate responses"""
    # User input
    user_input = st.chat_input("Ask your military training question here...")
    
    if user_input:
        # Add user message to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Generate response
        with st.spinner("🤔 Analyzing training materials..."):
            try:
                response_data = st.session_state.rag_system.query(
                    question=user_input,
                    category_filter=st.session_state.selected_category if st.session_state.selected_category != "All Categories" else None,
                    return_sources=True
                )
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "sources": response_data["sources"],
                    "timestamp": timestamp
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error processing your question. Please try again.",
                    "timestamp": timestamp
                })
        
        # Rerun to display new messages
        st.rerun()

def display_sidebar():
    """Display sidebar with controls and information"""
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("🎯 Training Controls")
        
        # Category selection
        st.session_state.selected_category = st.selectbox(
            "Training Category",
            config.TRAINING_CATEGORIES,
            index=config.TRAINING_CATEGORIES.index(st.session_state.selected_category)
        )
        
        # Retrieval settings
        st.subheader("⚙️ Search Settings")
        num_sources = st.slider("Number of sources to retrieve", 1, 10, config.RETRIEVAL_K)
        
        # Update config if changed
        if num_sources != config.RETRIEVAL_K:
            config.RETRIEVAL_K = num_sources
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Knowledge base stats
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📊 Knowledge Base")
        
        if st.button("🔄 Refresh Stats"):
            st.session_state.knowledge_base_stats = st.session_state.rag_system.get_knowledge_base_stats()
        
        if st.session_state.knowledge_base_stats:
            stats = st.session_state.knowledge_base_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stats-metric">
                    <h3>{stats.get('total_documents', 0)}</h3>
                    <p>Documents</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-metric">
                    <h3>{len(stats.get('categories', []))}</h3>
                    <p>Categories</p>
                </div>
                """, unsafe_allow_html=True)
            
            if stats.get('categories'):
                st.write("**Available Categories:**")
                for category in stats['categories']:
                    st.write(f"• {category}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Document upload section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload Training Documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx'],
            help="Upload military training documents to expand the knowledge base"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents"):
                process_uploaded_documents(uploaded_files)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat management
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("💬 Chat Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("💾 Export Chat"):
                export_chat_history()
        
        st.markdown('</div>', unsafe_allow_html=True)

def process_uploaded_documents(uploaded_files):
    """Process uploaded documents and add to knowledge base"""
    try:
        doc_processor = DocumentProcessor()
        all_documents = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            temp_path = config.DOCUMENTS_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Process document
            documents = doc_processor.process_document(temp_path)
            all_documents.extend(documents)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Add documents to knowledge base
        if all_documents:
            success = st.session_state.rag_system.add_training_documents(all_documents)
            if success:
                st.success(f"✅ Successfully processed {len(all_documents)} document chunks from {len(uploaded_files)} files!")
                # Refresh stats
                st.session_state.knowledge_base_stats = st.session_state.rag_system.get_knowledge_base_stats()
            else:
                st.error("❌ Failed to add documents to knowledge base")
        else:
            st.warning("⚠️ No content extracted from uploaded files")
            
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Error processing documents: {e}")

def export_chat_history():
    """Export chat history as JSON"""
    try:
        chat_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_messages": len(st.session_state.chat_history),
            "chat_history": st.session_state.chat_history
        }
        
        # Convert to JSON
        json_data = json.dumps(chat_data, indent=2, ensure_ascii=False)
        
        # Create download button
        st.download_button(
            label="📥 Download Chat History",
            data=json_data,
            file_name=f"military_training_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting chat history: {e}")

def display_welcome_message():
    """Display welcome message and instructions"""
    if not st.session_state.chat_history:
        st.markdown("""
        ### 🎖️ Welcome to the Military Training Assistant
        
        I'm your AI-powered military training instructor, ready to help you with:
        
        - **Tactical Procedures**: Battle tactics, formations, and strategic planning
        - **Equipment Training**: Weapon systems, gear operation, and maintenance
        - **Emergency Protocols**: Crisis response, evacuation procedures, and medical emergencies
        - **Leadership & Coordination**: Team management and command strategies
        - **Physical Training**: Fitness routines and conditioning programs
        - **Safety Procedures**: Risk assessment and protection protocols
        - **Communication**: Radio procedures and intelligence protocols
        - **Mission Planning**: Operational planning and briefing procedures
        
        **How to use this system:**
        1. Select a training category from the sidebar (optional)
        2. Ask your question in the chat input below
        3. Review the response and source documents provided
        4. Ask follow-up questions for clarification
        
        **Example questions:**
        - "What are the standard procedures for establishing a defensive perimeter?"
        - "How should I maintain my M4 rifle in desert conditions?"
        - "What are the emergency evacuation protocols for a medical emergency?"
        
        Ready to begin your training session? Ask me anything!
        """)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Validate configuration
    try:
        config.validate_config()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.info("Please check your .env file and ensure GOOGLE_API_KEY is set")
        st.stop()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area - simplified layout
    display_welcome_message()
    display_chat_interface()
    
    # Chat input must be outside any containers (columns, expanders, etc.)
    handle_user_input()
    
    # System status section (moved to bottom to avoid layout conflicts)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔧 System Status")
        
        if st.button("🔍 Run Diagnostics"):
            with st.spinner("Running system diagnostics..."):
                validation_results = st.session_state.rag_system.validate_system()
                
                for component, status in validation_results.items():
                    status_icon = "✅" if status else "❌"
                    component_name = component.replace("_", " ").title()
                    st.write(f"{status_icon} {component_name}")
    
    with col2:
        st.subheader("💡 Quick Topics")
        quick_topics = [
            "Tactical formations",
            "Equipment maintenance", 
            "Emergency procedures",
            "Leadership principles",
            "Safety protocols"
        ]
        
        for topic in quick_topics:
            if st.button(f"📖 {topic}", key=f"quick_{topic}"):
                # Add topic as user message
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"Tell me about {topic}",
                    "timestamp": timestamp
                })
                st.rerun()

if __name__ == "__main__":
    main()
