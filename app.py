from flask import Flask, render_template, request, jsonify
import os
import tempfile
import json
import re
import random
import speech_recognition as sr

app = Flask(__name__)

recognizer = None
engine = None
training_data = None

def normalize_text_shortcuts(text):
    shortcuts = {
        " u ": " you ", "ur ": "your ", " r ": " are ", " rn ": " right now ",
        "im": "I am", "ive": "I have", "ill": "I will", "dont": "do not",
        "cant": "cannot", "wont": "will not", "idk": "I don't know",
        "pls": "please", "bc": "because", "cuz": "because",
        "af": "very", "lol": "laughing out loud", "brb": "be right back",
        "tbh": "to be honest", "lmk": "let me know", "rofl": "rolling on floor laughing",
        "ngl": "not gonna lie", "gtg": "got to go", "ttyl": "talk to you later",
        "fomo": "fear of missing out", "imo": "in my opinion", "smh": "shaking my head",
        "finna": "going to", "slay": "succeed", "vibes": "feelings",
        "lowkey": "kind of", "highkey": "really", "sus": "suspicious",
        "deadass": "seriously", "no cap": "no lie", "cap": "lie",
        "goat": "greatest of all time", "bet": "alright", "yass": "yes",
        "mood": "relatable", "extra": "too much", "basic": "mainstream",
        "boujee": "fancy", "dope": "cool", "lit": "awesome"
    }
    text = f" {text} "
    for shortcut, full_form in shortcuts.items():
        text = re.sub(rf'\\b{re.escape(shortcut.strip())}\\b', full_form.strip(), text, flags=re.IGNORECASE)
    return text.strip()

def classify_emotion(text):
    emotion_keywords = {
        "happy": ["happy", "excited", "yay", "glad", "good", "great", "joy", "awesome", "love", "delighted", "pleased", "content"],
        "sad": ["sad", "upset", "depressed", "cry", "down", "miserable", "hurt", "unhappy", "grief", "blue", "heartbroken"],
        "angry": ["angry", "mad", "furious", "irritated", "pissed", "annoyed", "rage", "frustrat", "bitter", "resentful"],
        "anxious": ["anxious", "nervous", "scared", "worried", "stress", "panic", "fear", "tense", "restless", "uneasy", "dread"],
        "confused": ["confused", "lost", "unsure", "uncertain", "puzzled", "perplexed", "bewildered", "doubt"],
        "grateful": ["thank", "grateful", "appreciate", "thanks", "thankful", "blessed", "appreciative"],
        "overwhelmed": ["overwhelmed", "too much", "can't handle", "exhausted", "burnout", "tired", "drained", "fatigued"],
        "lonely": ["lonely", "alone", "isolated", "abandoned", "disconnected", "rejected", "left out"]
    }
    
    text_lower = text.lower()
    matched_emotions = {}
    
    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if word in text_lower:
                if emotion in matched_emotions:
                    matched_emotions[emotion] += 1
                else:
                    matched_emotions[emotion] = 1
                    
    if matched_emotions:
        # Return the emotion with the most matches
        return max(matched_emotions.items(), key=lambda x: x[1])[0]
    return "neutral"

def get_response(message):
    def journal_prompt():
        prompts = [
            "Grab your journal and reflect on this: 'What does feeling safe mean to me today?'",
            "Try writing: 'Right now, my mind feels...' and just let it all out.",
            "Scribble about a moment today you want to remember — even something tiny."
        ]
        return random.choice(prompts)

    def breathing_exercise():
        exercises = [
            "Let's try the 4-7-8 breathing technique: Inhale through your nose for 4 counts... hold for 7... and exhale slowly through your mouth for 8. Repeat 3-4 times. This helps activate your parasympathetic nervous system. 🧘‍♀️",
            
            "Try box breathing: Inhale for 4 counts... hold for 4... exhale for 4... hold for 4. Visualize tracing a square with your breath. This technique is used by many, including Navy SEALs, to stay calm under pressure. 🔄",
            
            "Diaphragmatic breathing can be powerful: Place one hand on your chest and one on your belly. Breathe deeply so your belly pushes your hand out while your chest remains relatively still. Exhale slowly through pursed lips. This activates your body's relaxation response. 💨",
            
            "Let's practice alternate nostril breathing: Use your right thumb to close your right nostril, inhale through your left. Then close your left nostril with your finger, release your thumb, and exhale through your right. Continue alternating for 5-10 cycles. This balances your energy. 👃",
            
            "The 4-4-4-4 breathing technique can help ground you: Breathe in for 4 counts, hold for 4, exhale for 4, and rest for 4 before beginning again. Repeat this cycle 4 times while focusing only on your breath. This creates a rhythm that calms your mind. ✨",
            
            "Try the 5-5-5 technique: Inhale deeply for 5 seconds, hold for 5 seconds, then exhale completely for 5 seconds. As you breathe, imagine tension leaving your body with each exhale. This simple pattern can quickly reduce anxiety. 🌬️"
        ]
        return random.choice(exercises)

    def affirmation():
        affirmations = [
            "Affirm this: 'I am doing my best, and that is enough.'",
            "Repeat to yourself: 'I deserve kindness — from others and from myself.'",
            "Say it with me: 'Even when I struggle, I am worthy of love.'"
        ]
        return random.choice(affirmations)
    
    # Try-except block to handle any issues
    try:
        # Check for crisis keywords
        crisis_keywords = ["suicide", "kill myself", "end my life", "don't want to live", "want to die", "harm myself", "hurt myself"]
        is_crisis = any(keyword in message.lower() for keyword in crisis_keywords)
        
        normalized_input = normalize_text_shortcuts(message.lower().strip())
        emotion = classify_emotion(normalized_input)
        
        # Improved question detection
        is_question = ("?" in normalized_input) or any(normalized_input.startswith(word) for word in 
                                                     ["what", "how", "why", "can", "could", "when", "where", 
                                                      "who", "which", "is", "are", "do", "does", "did", 
                                                      "should", "would", "will"])
        
        # Check for greeting
        greetings = ["hi", "hello", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening"]
        if any(greeting in normalized_input.split() for greeting in greetings):
            return random.choice([
                "Hello! How are you feeling today?",
                "Hi there! I'm here to support you. What's on your mind?",
                "Hey! How can I help you today?",
                "Welcome back! How are you doing right now?",
                "Hello! I'm here for you. How's your day going?"
            ]), "neutral", False
        
        # Crisis response takes precedence
        if is_crisis:
            return "I'm concerned about what you're sharing. If you're in immediate danger, please call emergency services or a crisis helpline right away. Would you like me to provide some crisis resources?", "concerned", True
        
        # Direct answers to specific mental health questions
        if "depression" in normalized_input and is_question:
            return "Depression is a mental health condition that causes persistent feelings of sadness and loss of interest. It can affect how you feel, think, and handle daily activities. It's more than just feeling sad temporarily - it's a serious medical condition that usually requires treatment. Have you been experiencing symptoms of depression?", "informative", False
        
        if "anxiety" in normalized_input and is_question:
            return "Anxiety is your body's natural response to stress. It's a feeling of fear or apprehension about what's to come. Everyone experiences anxiety at times, but anxiety disorders involve excessive worry and fear that interfere with daily activities. Is anxiety something you've been struggling with?", "informative", False
        
        if "meditation" in normalized_input and is_question:
            return "Meditation is a practice where you focus your mind on a particular object, thought, or activity to train attention and awareness. It can help reduce stress, control anxiety, improve sleep, and promote emotional health. Would you like me to guide you through a simple meditation exercise?", "calm", False
        
        if "therapy" in normalized_input and is_question:
            return "Therapy or counseling is a process where you work with a mental health professional to address emotional issues, develop coping skills, and improve well-being. There are many types, including cognitive-behavioral therapy (CBT), dialectical behavior therapy (DBT), and psychodynamic therapy. Have you considered talking to a therapist?", "informative", False
        
        if "mindfulness" in normalized_input and is_question:
            return "Mindfulness is the practice of being fully present and engaged in the moment, aware of your thoughts and feelings without judgment. It can help reduce stress, improve focus, and promote well-being. A simple way to practice is by focusing on your breath or paying attention to your senses. Would you like to try a quick mindfulness exercise?", "calm", False
        
        if "sleep" in normalized_input and is_question:
            return "Good sleep is crucial for mental health. Adults typically need 7-9 hours of quality sleep. Try keeping a consistent schedule, creating a relaxing bedtime routine, limiting screen time before bed, and making your bedroom comfortable. If you're having persistent sleep problems, it might be worth discussing with a healthcare provider. What's your sleep routine like?", "helpful", False
        
        if "self care" in normalized_input and is_question:
            return "Self-care includes activities that help maintain your physical, emotional, and mental health. It could be as simple as taking a walk, reading a book, or spending time with loved ones. It also includes basics like proper nutrition, exercise, and adequate sleep. What self-care activities do you enjoy or would like to try?", "supportive", False
        
        # More detailed emotion-specific responses
        if emotion == "happy":
            if is_question:
                return f"I'm glad you're feeling good! To answer your question about {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. Is there anything else about this that you'd like to explore?", "happy", False
            return "That's wonderful to hear! I'm genuinely happy for you. What specifically brought you joy today? I'd love to hear more about what's working well for you right now.", "happy", False
        
        elif emotion == "sad":
            if is_question:
                return f"I understand you might be feeling down right now. Regarding your question about {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. I hope that helps, and I'm here to talk more about how you're feeling too.", "sad", False
            return "I hear that you're feeling down, and I want you to know that's completely valid. Sometimes life gets heavy. Would you like to talk about what's contributing to these feelings? I'm here to listen without judgment.", "sad", False
        
        elif emotion == "angry":
            if is_question:
                return f"I can sense your frustration. To address your question about {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. Would you like to discuss what's causing these feelings of frustration too?", "angry", False
            return "I can hear that you're feeling frustrated or angry, which is a completely valid response. Sometimes anger protects us or signals that boundaries have been crossed. What triggered these feelings? Talking it through might help.", "angry", False
        
        elif emotion == "anxious":
            if is_question:
                return f"I understand anxiety can make it hard to focus. For your question about {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. Taking slow, deep breaths might help with the anxiety you're feeling. Would you like to try a quick breathing exercise?", "anxious", False
            return "It sounds like you're feeling anxious, which can be really uncomfortable. Sometimes just naming what we're worried about can reduce its power over us. What specific concerns are on your mind right now? We can tackle them one by one.", "anxious", False
        
        elif emotion == "confused":
            if is_question:
                return f"It's okay to feel uncertain. Regarding your question about {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. Does that clarify things, or would you like me to explain further?", "confused", False
            return "It sounds like things feel a bit unclear right now, which happens to all of us. Let's try to break this down together. What's the most confusing aspect of what you're dealing with? Sometimes talking it through step by step can help find clarity.", "confused", False
        
        elif emotion == "grateful":
            if is_question:
                return f"It's wonderful to hear that positive tone! To answer your question about {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. I appreciate you sharing these positive reflections with me.", "grateful", False
            return "I really appreciate you sharing that gratitude. Noticing the positive things, even small ones, is so powerful for our wellbeing. What other aspects of your life bring you feelings of appreciation? Building on these positive elements can be really helpful.", "grateful", False
        
        elif emotion == "overwhelmed":
            if is_question:
                return f"I can tell things feel like a lot right now. For your question about {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. When feeling overwhelmed, breaking things down into smaller steps can really help. What feels most urgent to address?", "overwhelmed", False
            return "I hear that you're feeling overwhelmed, which is completely understandable. When everything feels too much, let's just focus on one thing at a time. What's the most pressing concern right now? We can start there and work through things step by step.", "overwhelmed", False
        
        elif emotion == "lonely":
            if is_question:
                return f"I'm sorry you're feeling disconnected. About your question on {get_topic(normalized_input)}: {get_mental_health_info(normalized_input)}. Feelings of loneliness are common but can be really difficult. Would you like to talk about ways to feel more connected?", "lonely", False
            return "I hear that you're feeling lonely, which can be really painful. Connection is a fundamental human need. What kind of connection are you missing most right now? Sometimes even small interactions or reaching out to one person can help reduce these feelings.", "lonely", False

        # If it's a question but no specific emotion
        if is_question:
            return get_mental_health_info(normalized_input), "informative", False

        # Default neutral response
        return "I appreciate you sharing that with me. I'm here to support you through whatever you're experiencing. What's on your mind right now? Feel free to ask me any questions or just talk about what you're going through.", "neutral", False
    
    except Exception as e:
        # Fallback response in case of any error
        print(f"Error in get_response: {str(e)}")
        return "I'm here to listen and support you. Could you tell me more about what's on your mind?", "neutral", False

def get_topic(text):
    """Extract the main topic from the user's question"""
    if "depression" in text:
        return "depression"
    elif "anxiety" in text:
        return "anxiety"
    elif "therapy" in text:
        return "therapy"
    elif "meditation" in text:
        return "meditation"
    elif "sleep" in text:
        return "sleep"
    elif "stress" in text:
        return "stress"
    elif "self care" in text or "selfcare" in text:
        return "self-care"
    else:
        return "your question"

def get_mental_health_info(text):
    """Provide relevant mental health information based on the input text"""
    # Mental health information
    if "depression" in text:
        return "Depression is a common but serious mood disorder that affects how you feel, think, and handle daily activities. It's more than just feeling sad temporarily - it often requires professional support. Symptoms can include persistent sadness, loss of interest in activities, fatigue, and changes in sleep or appetite."
    
    if "anxiety" in text:
        return "Anxiety is a normal response to stress, but can become overwhelming. When it interferes with daily life, it might be an anxiety disorder. Symptoms can include excessive worry, restlessness, and physical symptoms like increased heart rate. Many effective treatments exist, including therapy and sometimes medication."
    
    if "therapy" in text:
        return "Therapy provides a safe space to work through challenges with a trained professional. Different approaches work for different people, but research shows therapy is effective for many mental health conditions. It's a sign of strength to seek help when needed."
    
    if "meditation" in text:
        return "Meditation is a powerful practice that can reduce stress and anxiety while improving focus and emotional regulation. Even 5-10 minutes daily can make a difference. Basic meditation involves focusing on your breath and gently returning your attention when your mind wanders."
    
    if "sleep" in text:
        return "Quality sleep is essential for mental health. Aim for 7-9 hours, maintain a consistent schedule, create a relaxing bedtime routine, limit screen time before bed, and ensure your sleeping environment is comfortable. If sleep problems persist, consider consulting a healthcare provider."
    
    if "stress" in text:
        return "Stress is your body's response to demands or threats. Some stress can be motivating, but chronic stress can affect your health. Managing stress through exercise, relaxation techniques, social connection, and sometimes professional support is important for wellbeing."
    
    if "self care" in text or "selfcare" in text:
        return "Self-care means taking deliberate actions to care for your physical, mental, and emotional health. It includes basics like proper nutrition, exercise, and sleep, as well as activities that bring you joy and relaxation. Regular self-care is essential, not a luxury."
    
    if "playlist" in text or "music" in text or "calm" in text or "peace" in text or "relax" in text:
        return get_calm_playlist()
    
    # If no specific mental health topic is identified
    return "I'm here to provide support and information about mental health, emotions, and wellbeing. I can discuss topics like anxiety, depression, stress management, and self-care strategies. Feel free to ask me about any specific concerns you have."

def get_calm_playlist():
    """Return a suggestion for calming Spotify playlists"""
    playlists = [
        "Peaceful Piano: https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO - Perfect for gentle background music while relaxing or working.",
        "Calm Vibes: https://open.spotify.com/playlist/37i9dQZF1DXaImRpG7HXqp - A mix of gentle acoustic and ambient tracks to help you find peace.",
        "Deep Sleep: https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp - Designed to help you drift off to sleep with soft, ambient sounds.",
        "Ambient Relaxation: https://open.spotify.com/playlist/37i9dQZF1DX3Ogo9pFvBkY - Instrumental ambient music that creates a peaceful atmosphere.",
        "Stress Relief: https://open.spotify.com/playlist/37i9dQZF1DWXe9gFZP0gtP - Curated to help reduce anxiety with calming melodies.",
        "Nature Sounds: https://open.spotify.com/playlist/37i9dQZF1DX4PP3DA4J0N8 - Peaceful natural sounds to help you connect with tranquility.",
        "Meditation Moments: https://open.spotify.com/playlist/37i9dQZF1DX0jgyAiPl8Af - Perfect companion for mindfulness practices and meditation."
    ]
    return f"Music can be incredibly helpful for finding calm. Here's a playlist that might help you relax: {random.choice(playlists)} Would you like to try listening to this while practicing some deep breathing?"

def get_detailed_breathing_exercise():
    """Return a detailed breathing exercise with step-by-step instructions"""
    exercises = [
        {
            "name": "Progressive Relaxation Breathing",
            "steps": [
                "1. Find a comfortable seated position or lie down.",
                "2. Take a deep breath in through your nose for 4 counts.",
                "3. As you inhale, tense your feet and toes.",
                "4. Hold your breath and the tension for 2 counts.",
                "5. Exhale slowly through your mouth for 6 counts while releasing the tension.",
                "6. Take another deep breath, and this time, tense your calves.",
                "7. Continue this pattern, moving up through your body: thighs, abdomen, chest, arms, shoulders, and face.",
                "8. Finish with three deep breaths, relaxing your entire body."
            ],
            "benefits": "This technique combines deep breathing with progressive muscle relaxation to release physical tension and mental stress."
        },
        {
            "name": "4-7-8 Breathing Technique",
            "steps": [
                "1. Sit in a comfortable position with your back straight.",
                "2. Place the tip of your tongue against the ridge behind your upper front teeth.",
                "3. Exhale completely through your mouth, making a whoosh sound.",
                "4. Close your mouth and inhale quietly through your nose for 4 counts.",
                "5. Hold your breath for 7 counts.",
                "6. Exhale completely through your mouth for 8 counts, making the whoosh sound.",
                "7. This completes one breath cycle. Repeat for 3-4 complete breaths."
            ],
            "benefits": "Developed by Dr. Andrew Weil, this technique acts as a natural tranquilizer for the nervous system, helping with anxiety, sleep issues, and stress response."
        },
        {
            "name": "Diaphragmatic (Belly) Breathing",
            "steps": [
                "1. Lie on your back with knees bent (or sit comfortably).",
                "2. Place one hand on your upper chest and the other on your abdomen.",
                "3. Breathe in slowly through your nose, feeling your stomach push against your hand.",
                "4. Your chest should remain relatively still.",
                "5. Purse your lips as if sipping through a straw.",
                "6. Exhale slowly through pursed lips while gently pressing on your abdomen.",
                "7. Repeat for 5-10 minutes, focusing on the movement of your breath."
            ],
            "benefits": "This activates the parasympathetic nervous system, reduces blood pressure, improves core muscle stability, and increases oxygen supply to your body."
        }
    ]
    
    exercise = random.choice(exercises)
    response = f"**{exercise['name']}**\n\n"
    response += "\n".join(exercise['steps']) + "\n\n"
    response += f"**Benefits:** {exercise['benefits']}\n\n"
    response += "Would you like to try this exercise now? Let me know how it goes for you."
    
    return response

def load_speech_components():
    """Load the necessary components for speech recognition and synthesis"""
    global recognizer, engine
    try:
        from voice_handler import VoiceHandler
        voice_handler = VoiceHandler()
        print("Loaded speech recognition and synthesis components")
        return voice_handler
    except Exception as e:
        print(f"Error loading speech components: {str(e)}")
        return None

# Load the voice handler when the app starts
voice_handler = load_speech_components()

@app.route('/')
def home():
    return render_template('simple_index.html', welcome_message="Hi, I'm here for you. This space is yours — what's on your heart today?")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    
    # Check if user is specifically asking about breathing exercises
    if any(word in message.lower() for word in ["breathing", "breathe", "breath", "breathing exercise", "calm breathing", "relaxation breathing"]):
        return jsonify({
            'response': get_detailed_breathing_exercise(),
            'emotion': 'calm',
            'is_crisis': False
        })
    
    # Check if user is specifically asking for playlists or music
    if any(word in message.lower() for word in ["playlist", "music", "spotify", "song", "calm", "relax", "peace"]):
        return jsonify({
            'response': get_calm_playlist(),
            'emotion': 'calm',
            'is_crisis': False
        })
    
    try:
        response, emotion, is_crisis = get_response(message)
        return jsonify({'response': response, 'emotion': emotion, 'is_crisis': is_crisis})
    except Exception as e:
        return jsonify({
            'response': "Something feels off on my end. Let's try that again?",
            'emotion': 'neutral',
            'is_crisis': False
        })

@app.route('/voice', methods=['POST'])
def voice():
    """Handle voice input from the user"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    try:
        audio_file = request.files['audio']
        
        # Validate that the file has content
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to beginning
        
        if file_size == 0:
            return jsonify({'error': 'Empty audio file provided'}), 400
        
        # Process the audio
        if voice_handler:
            try:
                # Read the audio data
                audio_data = audio_file.read()
                
                # Validate audio data
                if len(audio_data) < 100:  # Minimum size for valid audio
                    return jsonify({'error': 'Audio file too small or corrupted'}), 400
                
                # Convert the audio data to a format that speech_recognition can use
                try:
                    audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
                except Exception as e:
                    print(f"Error converting audio data: {str(e)}")
                    return jsonify({'error': 'Invalid audio format. Please try recording again.'}), 400
                
                # Recognize speech using Google Speech Recognition
                try:
                    text = voice_handler.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    return jsonify({'error': "Sorry, I couldn't understand the audio. Please speak clearly and try again."}), 400
                except sr.RequestError as e:
                    print(f"Google Speech Recognition error: {str(e)}")
                    return jsonify({'error': "Could not request results from speech recognition service. Please check your internet connection and try again."}), 500
                
                # Process the text and get response, emotion, and crisis flag
                response_text, emotion, is_crisis = get_response(text)
                
                # Convert response to speech (text-to-speech)
                try:
                    audio_response = voice_handler.text_to_speech(response_text)
                except Exception as e:
                    print(f"Text-to-speech error: {str(e)}")
                    audio_response = None
                
                return jsonify({
                    'text': text,
                    'response': response_text,
                    'emotion': emotion,
                    'is_crisis': is_crisis,
                    'audio_response': audio_response
                })
            except Exception as e:
                print(f"Voice processing error: {str(e)}")
                return jsonify({'error': f'Error processing voice: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Speech recognition not available at the moment.'}), 500
    except Exception as e:
        print(f"Voice processing error: {str(e)}")
        return jsonify({'error': f'Error processing voice: {str(e)}'}), 500

@app.route('/speak', methods=['POST'])
def speak():
    """Convert text to speech"""
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400
    
    text = request.json['text']
    
    try:
        if voice_handler:
            audio_base64 = voice_handler.text_to_speech(text)
            return jsonify({'audio': audio_base64})
        else:
            return jsonify({'error': 'Text-to-speech not available'}), 500
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return jsonify({'error': f'Error converting text to speech: {str(e)}'}), 500

if __name__ == '__main__':
    # Load training data
    try:
        with open('datasets/processed_training_data.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training data: {str(e)}")
        # Create a minimal training data set if the file doesn't exist
        if not os.path.exists('datasets'):
            os.makedirs('datasets')
        
        if not os.path.exists('datasets/processed_training_data.json'):
            minimal_data = [
                {"input": "I feel sad", "emotion": "sad", "response": "I'm sorry to hear you're feeling down. Would you like to talk about what's bothering you?"},
                {"input": "I'm anxious", "emotion": "anxious", "response": "Anxiety can be really challenging. Let's take a moment to breathe deeply together."},
                {"input": "I'm happy", "emotion": "happy", "response": "That's wonderful! I'd love to hear what's bringing you joy."}
            ]
            with open('datasets/processed_training_data.json', 'w', encoding='utf-8') as f:
                json.dump(minimal_data, f, indent=2)
            training_data = minimal_data
    
    # Load speech components after app initialization
    voice_handler = load_speech_components()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
    