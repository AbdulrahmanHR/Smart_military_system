"""
Military Training Chatbot - Streamlit Interface
"""
import streamlit as st
import logging
from datetime import datetime

from config.settings import config
from src.rag_system import MilitaryTrainingRAG
from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_database_if_needed():
    """Initialize database with sample documents if it's empty - cached for performance"""
    try:
        doc_processor = DocumentProcessor()
        
        # Check if documents directory exists
        if not config.DOCUMENTS_DIR.exists():
            logger.warning(f"Documents directory not found: {config.DOCUMENTS_DIR}")
            return False
        
        # Process all documents
        documents = doc_processor.process_directory(config.DOCUMENTS_DIR)
        
        if documents:
            # Add to the existing RAG system
            if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system:
                success = st.session_state.rag_system.add_training_documents(documents)
                if success:
                    logger.info(f"Successfully initialized database with {len(documents)} documents")
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# CSS styling with Arabic support
st.markdown("""
<style>
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    /* Arabic text support */
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', 'Tahoma', 'Microsoft Sans Serif', sans-serif;
    }
    
    /* Language selector styling */
    .language-selector {
        margin-bottom: 1rem;
    }
    
    /* RTL support for Arabic interface */
    .rtl {
        direction: rtl;
        text-align: right;
    }
    
    /* Better Arabic font rendering */
    .stSelectbox > div > div {
        font-family: 'Arial', 'Tahoma', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables with Arabic support"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "en"
    
    if "rag_system" not in st.session_state:
        init_msg = ("تهيئة مساعد التدريب العسكري..." if st.session_state.ui_language == "ar" 
                   else "Initializing Military Training Assistant...")
        success_msg = ("✅ تم تهيئة النظام بنجاح!" if st.session_state.ui_language == "ar" 
                      else "✅ System initialized successfully!")
        error_msg = ("❌ فشل في تهيئة النظام" if st.session_state.ui_language == "ar" 
                    else "❌ Failed to initialize system")
        
        with st.spinner(init_msg):
            try:
                # Initialize the system with error handling for cloud deployment
                st.session_state.rag_system = MilitaryTrainingRAG()
                
                # Check if database is empty and needs initialization
                stats = st.session_state.rag_system.get_knowledge_base_stats()
                if stats.get("total_documents", 0) == 0:
                    init_db_msg = ("تهيئة قاعدة البيانات..." if st.session_state.ui_language == "ar" 
                                 else "Initializing database...")
                    with st.spinner(init_db_msg):
                        initialize_database_if_needed()
                
                st.success(success_msg)
            except Exception as e:
                st.error(f"{error_msg}: {e}")
                if st.session_state.ui_language == "ar":
                    st.info("يرجى التحقق من ملف .env والتأكد من تعيين GOOGLE_API_KEY")
                    st.info("أو إضافة المفاتيح في إعدادات Streamlit Cloud")
                else:
                    st.info("Please check your .env file and ensure GOOGLE_API_KEY is set")
                    st.info("Or add secrets in Streamlit Cloud settings")
                st.stop()
    
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = "All Categories"

def display_header():
    """Display the main application header with language support"""
    if st.session_state.ui_language == "ar":
        st.markdown('<div class="rtl">', unsafe_allow_html=True)
        st.title("⚔️ مساعد التدريب العسكري")
        st.caption("دعم التدريب بالذكاء الاصطناعي")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.title("⚔️ Military Training Assistant")
        st.caption("AI-Powered Training Support")

def display_chat_interface():
    """Display the main chat interface with Arabic support"""
    if st.session_state.ui_language == "ar":
        st.markdown('<div class="rtl">', unsafe_allow_html=True)
        st.subheader("💬 مناقشة التدريب")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.subheader("💬 Training Discussion")
    
    # Display chat history using Streamlit's chat message component
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            # Detect if message content is Arabic and apply RTL styling
            content = message["content"]
            if any('\u0600' <= char <= '\u06FF' for char in content):
                st.markdown(f'<div class="arabic-text">{content}</div>', unsafe_allow_html=True)
            else:
                st.write(content)
            st.caption(f"⏰ {message['timestamp']}")
            
            # Display sources if available (for assistant messages)
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                sources_label = f"📚 المصادر ({len(message['sources'])})" if st.session_state.ui_language == "ar" else f"📚 Sources ({len(message['sources'])})"
                with st.expander(sources_label, expanded=False):
                    for source in message["sources"]:
                        language_flag = "🇸🇦" if source.get('language') == 'ar' else "🇬🇧"
                        st.caption(f"• {language_flag} {source['filename']}")

def handle_user_input():
    """Handle user input and generate responses with Arabic support"""
    # User input with language-appropriate placeholder
    placeholder = ("اسأل سؤالك حول التدريب العسكري هنا..." if st.session_state.ui_language == "ar" 
                  else "Ask your military training question here...")
    user_input = st.chat_input(placeholder)
    
    if user_input:
        # Add user message to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Generate response with language-appropriate messages
        spinner_msg = ("🤔 تحليل مواد التدريب..." if st.session_state.ui_language == "ar" 
                      else "🤔 Analyzing training materials...")
        with st.spinner(spinner_msg):
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
                error_msg = f"خطأ في توليد الإجابة: {e}" if st.session_state.ui_language == "ar" else f"Error generating response: {e}"
                st.error(error_msg)
                
                fallback_msg = ("أعتذر، واجهت خطأ في معالجة سؤالك. يرجى المحاولة مرة أخرى." if st.session_state.ui_language == "ar" 
                               else "I apologize, but I encountered an error processing your question. Please try again.")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": fallback_msg,
                    "timestamp": timestamp
                })
        
        # Rerun to display new messages
        st.rerun()

def display_sidebar():
    """Display sidebar with language selection and essential controls"""
    with st.sidebar:
        # Language selection
        language_options = {"English": "en", "العربية": "ar"}
        selected_lang = st.selectbox(
            "Language / اللغة",
            options=list(language_options.keys()),
            index=0 if st.session_state.ui_language == "en" else 1,
            key="language_selector"
        )
        
        # Update UI language
        new_language = language_options[selected_lang]
        if new_language != st.session_state.ui_language:
            st.session_state.ui_language = new_language
            st.rerun()
        
        # Localized headers
        if st.session_state.ui_language == "ar":
            st.markdown('<div class="rtl">', unsafe_allow_html=True)
            st.subheader("🎯 عناصر التحكم")
            
            # Category selection in Arabic
            category_options = dict(zip(config.TRAINING_CATEGORIES, config.TRAINING_CATEGORIES_AR))
            selected_ar_category = st.selectbox(
                "فئة التدريب",
                options=list(category_options.values()),
                index=list(category_options.keys()).index(st.session_state.selected_category)
            )
            # Map back to English for backend processing
            st.session_state.selected_category = list(category_options.keys())[list(category_options.values()).index(selected_ar_category)]
            
            # Chat management in Arabic
            if st.button("🗑️ مسح المحادثة"):
                st.session_state.chat_history = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.subheader("🎯 Controls")
            
            # Category selection in English
            st.session_state.selected_category = st.selectbox(
                "Training Category",
                config.TRAINING_CATEGORIES,
                index=config.TRAINING_CATEGORIES.index(st.session_state.selected_category)
            )
            
            # Chat management in English
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


def display_welcome_message():
    """Display welcome message and instructions with language support"""
    if not st.session_state.chat_history:
        if st.session_state.ui_language == "ar":
            st.markdown("""
            <div class="rtl arabic-text">
            
            ### 🎖️ مرحباً بك في مساعد التدريب العسكري
            
            أنا هنا لمساعدتك في أسئلة التدريب العسكري التي تغطي التكتيكات والمعدات والبروتوكولات وأكثر.
            
            **كيفية الاستخدام:** اختر فئة (اختياري) واسأل سؤالك أدناه.
            
            **مثال:** "ما هي إجراءات إنشاء محيط دفاعي؟"
            
            </div>
            """, unsafe_allow_html=True)
        else:
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
        if st.session_state.ui_language == "ar":
            st.error(f"خطأ في التكوين: {e}")
            st.info("يرجى التحقق من ملف .env والتأكد من تعيين GOOGLE_API_KEY")
        else:
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
