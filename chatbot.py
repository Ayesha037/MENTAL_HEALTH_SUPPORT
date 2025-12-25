import random
from textblob import TextBlob

class MentalHealthChatbot:
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello! I'm here to support you. How are you feeling today?",
                "Hi there! I'm your mental health support companion. What's on your mind?",
                "Welcome! I'm here to listen and help. What would you like to talk about?"
            ],
            'sadness': [
                "I hear that you're feeling sad. Would you like to talk more about what's bothering you?",
                "It's okay to feel sad. I'm here to listen if you want to share more.",
                "I understand that you're going through a difficult time. Would you like to explore what's making you feel this way?"
            ],
            'anxiety': [
                "Anxiety can be really challenging. Would you like to try some breathing exercises together?",
                "I understand anxiety can be overwhelming. Would you like to talk about what's making you anxious?",
                "Remember that anxiety is a natural response. Would you like to discuss some coping strategies?"
            ],
            'anger': [
                "I can sense that you're feeling angry. Would you like to talk about what happened?",
                "It's okay to feel angry. Would you like to explore what triggered these feelings?",
                "I'm here to listen without judgment. Would you like to share what's making you angry?"
            ],
            'crisis': [
                "I'm concerned about your safety. Would you like me to provide you with emergency contact numbers?",
                "Your safety is important. Would you like to talk to a crisis counselor?",
                "I want to make sure you're safe. Would you like me to connect you with professional help?"
            ],
            'default': [
                "I'm here to listen. Could you tell me more about that?",
                "Thank you for sharing. How does that make you feel?",
                "I understand. Would you like to explore that further?"
            ]
        }

    def get_response(self, user_input, emotion):
        # Convert user input to lowercase for easier matching
        user_input = user_input.lower()
        
        # Check for greetings
        if any(word in user_input for word in ['hi', 'hello', 'hey']):
            return random.choice(self.responses['greeting'])
        
        # Check for crisis keywords
        if any(word in user_input for word in ['suicide', 'kill myself', 'end it all', 'want to die']):
            return random.choice(self.responses['crisis'])
        
        # Use emotion to select appropriate response category
        if emotion in self.responses:
            return random.choice(self.responses[emotion])
        
        # Analyze sentiment for more nuanced responses
        sentiment = TextBlob(user_input).sentiment.polarity
        if sentiment < -0.5:
            return random.choice(self.responses['sadness'])
        elif sentiment > 0.5:
            return random.choice(self.responses['default'])
        
        return random.choice(self.responses['default']) 