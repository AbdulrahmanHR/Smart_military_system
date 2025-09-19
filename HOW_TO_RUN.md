# How to Run the Military Training Chatbot

## Quick Setup (3 steps)

### 1. Install Python packages
```bash
pip install -r requirements.txt

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   - Copy `env_example.txt` to `.env`
   - Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_gemini_api_key_here
     ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Application**
   - Open your browser and go to: http://localhost:8000
   - The application supports both Arabic and English languages

## Available Documents
- `basic_military_procedures_ar.txt` - Basic military procedures in Arabic
- `basic_military_procedures_en.txt` - Basic military procedures in English  
- `equipment_operation_en.txt` - Equipment operation guide in English
- `security_protocols_ar.txt` - Security protocols in Arabic (NEW!)

## Testing File Selection

### Method 1: Using the Web Interface
1. Open http://localhost:8000
2. In the file selection dropdown, choose specific files
3. Ask questions - responses will only use selected documents
4. View which files were used in the "Sources" section

### Method 2: Using the Test Script
```bash
python test_files.py
```

## New Features Usage

### File Selection
- **All Files**: Leave "Search in all files" selected (default)
- **Specific Files**: Select individual documents from the dropdown
- **Multiple Files**: Hold Ctrl/Cmd to select multiple files
- **Clear Selection**: Click the red X button to reset

### Conversation Memory
- Ask follow-up questions like "Can you explain that in more detail?"
- Reference previous answers: "What about the Arabic version of that?"
- Each browser tab maintains its own conversation session

### Arabic Support
- Switch language using the toggle in the top-right
- New Arabic security protocols document available
- RTL (right-to-left) text support

## Troubleshooting

### File Selection Not Working?
1. Check browser console for JavaScript errors
2. Verify files are loaded: visit http://localhost:8000/files
3. Look for backend logs showing selected files

### Memory Not Working?
- Each browser tab has its own session
- Clear browser cache if issues persist
- Check backend logs for session management

## Testing Commands
```bash
# Test file endpoint
curl http://localhost:8000/files

# Test chat with specific file
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What are security protocols?","language":"en","selected_files":["security_protocols_ar.txt"],"session_id":"test"}'
- Then access the app at http://localhost:8000
- The frontend must be accessed through the server, not as a local file

**Problem**: "Failed to fetch" errors
**Solution**: 
- Ensure the FastAPI server is running (`python app.py`)
- Check that you can access http://localhost:8000 in your browser
- Restart the server if needed