"""
Military Training Chatbot - Streamlit Interface
"""
import streamlit as st
import logging
from datetime import datetime

from config.settings import config
from src.rag_system import MilitaryTrainingRAG

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

# Minimal CSS styling
st.markdown("""
<style>
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
    
    
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = "All Categories"

def display_header():
    """Display the main application header"""
    st.title("⚔️ Military Training Assistant")
    st.caption("AI-Powered Training Support")

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
                with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                    for source in message["sources"]:
                        st.caption(f"• {source['filename']}")

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
    """Display simplified sidebar with essential controls only"""
    with st.sidebar:
        st.subheader("🎯 Controls")
        
        # Category selection
        st.session_state.selected_category = st.selectbox(
            "Training Category",
            config.TRAINING_CATEGORIES,
            index=config.TRAINING_CATEGORIES.index(st.session_state.selected_category)
        )
        
        # Chat management
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def display_welcome_message():
    """Display welcome message and instructions"""
    if not st.session_state.chat_history:
        st.markdown("""
        ### 🎖️ Welcome to the Military Training Assistant
        
        I'm here to help with military training questions covering tactics, equipment, protocols, and more.
        
        **How to use:** Select a category (optional) and ask your question below.
        
        **Example:** "What are the procedures for establishing a defensive perimeter?"
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

if __name__ == "__main__":
    main()
