# Mental Health Support Chatbot

An AI-powered web chatbot that provides **text-based emotional support** with optional voice interaction — built to make mental health conversations more accessible.

> This project is for educational and early-support purposes only. It is not a replacement for professional mental healthcare.

## What it does

* Accepts text input (and optional voice via speech recognition)
* Responds with empathetic, NLP-driven messages
* Detects crisis keywords and surfaces safety messaging
* Analyses emotion from user input to tailor responses
* Runs as a Flask web application — no installation required for end users

## Tech Stack
Python, Flask, speech\_recognition, pyttsx3, HTML/CSS

## Pipeline Structure
User Input (text/voice) → Emotion Analysis → Crisis Detection → Response Generation → Voice Output (optional)

## Project Structure
MENTAL_HEALTH_SUPPORT/
├── app.py                  # Main Flask app
├── chatbot.py              # Core response logic
├── emotion_analyzer.py     # Emotion classification
├── crisis_detector.py      # Crisis keyword detection
├── voice_handler.py        # Speech recognition & TTS
├── templates/              # HTML frontend
├── datasets/               # Training/reference data
└── requirements.txt

## How to Run

```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:5000
```

## Author
**Mohammad Ayesha Summaiyya** — msumaiya03579@gmail.com
