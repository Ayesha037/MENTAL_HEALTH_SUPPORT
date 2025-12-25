# Mental Health Support Chatbot

A comprehensive mental health support chatbot that provides both text and voice-based interactions, with emotion analysis and crisis detection capabilities.

## Features

- Text-based chat interface
- Voice interaction support
- Emotion analysis using NLP
- Crisis detection and emergency contact information
- Responsive web interface
- Real-time voice-to-text and text-to-voice conversion

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Microphone (for voice interaction)
- Speakers (for voice responses)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-support
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. You can interact with the chatbot using either:
   - Text input: Type your message and press Enter or click Send
   - Voice input: Click "Start Voice Recording" to begin speaking, then click "Stop Recording" when done

## Features in Detail

### Emotion Analysis
The chatbot analyzes your messages to detect emotions such as:
- Sadness
- Anxiety
- Anger
- Joy
- Fear
- Disgust

### Crisis Detection
The system monitors for:
- Suicidal ideation
- Self-harm intentions
- Severe depression
- Extreme anxiety
- Immediate danger indicators

### Voice Interaction
- Speech-to-text conversion for user input
- Text-to-speech for bot responses
- Adjustable voice settings (speed and volume)

## Safety Notice

This chatbot is designed to provide support and guidance but is not a replacement for professional mental health care. If you are experiencing a mental health crisis:

1. Call emergency services (911 in the US)
2. Contact the National Suicide Prevention Lifeline (988 in the US)
3. Reach out to a mental health professional
4. Contact a trusted friend or family member

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 