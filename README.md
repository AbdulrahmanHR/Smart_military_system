# Military Training Chatbot

An AI-powered military training assistant using Retrieval-Augmented Generation (RAG) to provide accurate, contextual responses based on military training documents.

## 🎯 Features

- **Intelligent Q&A**: Ask questions about military procedures, tactics, and protocols
- **Document-Based Responses**: Answers backed by actual training materials with source citations
- **Category Filtering**: Focus on specific training areas (Tactical, Equipment, Emergency, etc.)
- **Military-Styled Interface**: Professional web interface designed for military users
- **Document Upload**: Add new training materials to expand the knowledge base
- **Chat History**: Persistent conversation history with export capabilities
- **Source Attribution**: See exactly which documents support each answer

## 🛠️ Technology Stack

- **Python 3.8+**: Core programming language
- **LangChain**: RAG pipeline orchestration
- **Chroma**: Vector database for document storage
- **Sentence Transformers**: Document embeddings
- **Streamlit**: Web interface
- **Google Gemini 2.5 Flash**: Large language model
- **PyPDF2 & python-docx**: Document processing

## 📋 Prerequisites

1. **Python 3.8 or higher**
2. **Google AI Studio API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key for Gemini access

## 🚀 Quick Setup (3 Steps)

### 1. Install Dependencies

```bash
# Navigate to project directory
cd Smart_military_system

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env file and add your Google API key
# Replace 'your_google_api_key_here' with your actual API key
```

### 3. Initialize and Run

```bash
# Initialize the database with sample documents
python setup_database.py

# Launch the application
streamlit run app.py
```

**🎉 Done!** Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
Smart_military_system/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── setup_database.py          # Database initialization script
├── HOW_TO_RUN.md              # Simple run instructions
├── env.example                # Environment configuration template
├── config/
│   └── settings.py            # Configuration management
├── src/
│   ├── document_processor.py  # Document ingestion and processing
│   ├── vector_database.py     # Chroma vector database management
│   └── rag_system.py         # RAG pipeline implementation
└── data/
    ├── documents/             # Training documents
    └── vector_db/            # Persistent vector database storage
```

## 🎓 Training Categories

- **Tactical Procedures**: Battle tactics, formations, and strategic planning
- **Equipment Training**: Weapon systems, gear operation, and maintenance
- **Emergency Protocols**: Crisis response and medical emergencies
- **Leadership & Coordination**: Team management and command strategies
- **Physical Training**: Fitness routines and conditioning programs
- **Safety Procedures**: Risk assessment and protection protocols
- **Communication Protocols**: Radio procedures and intelligence protocols
- **Mission Planning**: Operational planning and briefing procedures

## 🔍 Example Questions

1. **Tactical**: "What are the standard procedures for establishing a defensive perimeter?"
2. **Equipment**: "How do I maintain my M4 rifle in desert conditions?"
3. **Emergency**: "What is the proper procedure for treating a sucking chest wound?"
4. **Leadership**: "What are the key principles of military leadership?"

## 📚 Adding New Documents

### Via Web Interface
1. Use the "Document Management" section in the sidebar
2. Upload PDF, TXT, or DOCX files
3. Click "Process Documents" to add to knowledge base

### Via File System
1. Add documents to `data/documents/` directory
2. Run: `python setup_database.py`

## 🔧 Configuration

Edit `env.example` and copy to `.env` to customize:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional customizations
TEMPERATURE=0.3          # Response creativity (0.0-1.0)
MAX_TOKENS=1024          # Maximum response length
CHUNK_SIZE=500           # Document chunk size
RETRIEVAL_K=5            # Number of sources to retrieve
```

## 🐛 Troubleshooting

### "Configuration Error: GOOGLE_API_KEY is required"
- Make sure you created `.env` file with your API key

### "No documents found to process"
- Ensure documents exist in `data/documents/`
- Verify file formats are supported (PDF, TXT, DOCX)

### Import errors
- Make sure you're in the correct directory
- Try: `pip install -r requirements.txt`

### App won't start
- Check: `python -c "import streamlit; print('OK')"`
- See `HOW_TO_RUN.md` for detailed troubleshooting

## 🔒 Security Considerations

- **Non-Classified Only**: Use only publicly available training materials
- **Educational Purpose**: Designed for training and education, not operational planning
- **Content Filtering**: System focuses on educational content only

## 📄 Files Overview

- **`app.py`** - Main Streamlit web application
- **`setup_database.py`** - Initialize vector database with sample documents
- **`HOW_TO_RUN.md`** - Simple step-by-step running instructions
- **`env.example`** - Environment configuration template
- **`requirements.txt`** - All required Python packages

## 🆘 Need Help?

1. Check `HOW_TO_RUN.md` for step-by-step instructions
2. Verify your Google API key is correct
3. Ensure all packages are installed: `pip list | grep streamlit`

---

**Ready to enhance military training with AI?** 🎖️

Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and follow the 3-step setup above!