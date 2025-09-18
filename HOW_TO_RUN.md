# 🚀 How to Run the Military Training Chatbot

## Quick Start (3 Steps)

### 1. 📋 Make sure you have the environment ready

```bash
# You should be in the project directory
cd Smart_military_system

# Activate your Python environment (if using conda)
conda activate pymain
```

### 2. 🔑 Set up your Google API Key

**First, get your API key:**
- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Click "Create API Key"
- Copy the key

**Then, edit your .env file:**
```bash
# Edit the .env file (use any text editor)
notepad .env
```

**Add your API key:**
```
GOOGLE_API_KEY=your_actual_api_key_here
```
Save and close the file.

### 3. 🎯 Initialize and Run

```bash
# Initialize the database with sample documents
python setup_database.py

# Launch the chatbot
streamlit run app.py
```

**🎉 That's it!** Your browser should open to `http://localhost:8501`

---

## 📝 If You Get Errors

### "Configuration Error: GOOGLE_API_KEY is required"
- Make sure you edited the `.env` file
- Check that your API key is correct (no extra spaces)

### Import Errors or "Modality" Attribute Error
```bash
# This should already be fixed, but if you get errors, run:
pip install google-generativeai==0.7.2 langchain-google-genai==1.0.10 --force-reinstall
```

### "No documents found to process"
```bash
# Run the database setup again
python setup_database.py
```

### Port Already in Use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

---

## 🎖️ Using the Chatbot

Once running, you can:

1. **Ask military training questions:**
   - "How do I establish a defensive perimeter?"
   - "What's the procedure for M4 rifle maintenance?"
   - "How do I request a medical evacuation?"

2. **Filter by category:**
   - Use the sidebar to select specific training areas
   - Choose from Tactical, Equipment, Emergency, etc.

3. **Upload documents:**
   - Add new training materials via the sidebar
   - Supports PDF, Word, and text files

4. **View sources:**
   - Each answer shows which documents it came from
   - Check the expandable "Sources" section

---

## 🔧 Troubleshooting

### App won't start
1. Check Python environment is activated
2. Ensure all packages are installed: `pip list | grep streamlit`
3. Try: `python -c "import streamlit; print('Streamlit OK')"`

### No responses from chatbot
1. Verify your API key is correct
2. Check internet connection
3. Try the "Run Diagnostics" button in the app

### Slow responses
1. Reduce "Number of sources to retrieve" in sidebar
2. Check your internet connection
3. Try with shorter questions

---

## 🏃‍♂️ Quick Test

Run this to verify everything works:

```bash
python -c "
from config.settings import config
config.validate_config()
print('✅ Configuration OK')
"
```

If you see "✅ Configuration OK", you're ready to go!

---

## 📱 Access from Other Devices

The app also runs on your local network:
- Check the terminal output for "Network URL"
- Use that URL from other devices on the same WiFi

---

**Need help?** Check the `README.md` for detailed documentation or `SIMPLE_INSTALL.md` for installation troubleshooting.
