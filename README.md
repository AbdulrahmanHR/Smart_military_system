# Military Training Chatbot MVP

A simple bilingual (Arabic/English) military training chatbot that answers questions about military procedures using your training documents.

## Quick Start

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Set your Google API key
Create a `.env` file (copy from `env_example.txt`) and add your API key:
```
GOOGLE_API_KEY=your_api_key_here
```
Get your API key from: https://aistudio.google.com/

### 3. Run the application
```bash
python app.py
```
Open your browser to: http://localhost:8000

## Features

- **Bilingual**: Works in Arabic and English
- **Smart search**: Finds relevant info from your documents  
- **Easy upload**: Add new training documents through the web interface
- **Clean interface**: Simple chat interface with RTL support for Arabic

## Example Questions

**English:**
- "What are the basic safety rules for weapons handling?"
- "How do I perform equipment maintenance?"

**Arabic:**
- "ما هي قواعد الأمان الأساسية للتعامل مع الأسلحة؟"
- "كيف أقوم بصيانة المعدات؟"

## Files Structure
```
├── app.py              # Main application
├── requirements.txt    # Python packages
├── env_example.txt     # Environment variables template
├── HOW_TO_RUN.md      # Simple setup guide
├── frontend/
│   └── index.html     # Web interface
├── documents/         # Training documents (PDF/TXT)
└── database/          # Vector database (auto-created)
```

## Adding Documents
- Put PDF or TXT files in the `documents/` folder
- Restart the app or use the web upload feature
- Both Arabic and English documents are supported

## Tech Stack
- **Backend**: FastAPI + LangChain + Chroma
- **LLM**: Google Gemini 2.5 Flash  
- **Frontend**: Simple HTML/CSS/JS
