import re
from textblob import TextBlob

class CrisisDetector:
    def __init__(self):
        # Crisis-related keywords and phrases
        self.crisis_keywords = {
            'suicide': [
                'suicide', 'kill myself', 'end my life', 'want to die',
                'don\'t want to live', 'life is not worth living',
                'better off dead', 'no reason to live'
            ],
            'self_harm': [
                'cut myself', 'self-harm', 'hurt myself', 'burn myself',
                'self injury', 'self-injury'
            ],
            'depression': [
                'deep depression', 'severe depression', 'can\'t get out of bed',
                'no hope', 'hopeless', 'worthless', 'useless'
            ],
            'anxiety': [
                'panic attack', 'severe anxiety', 'can\'t breathe',
                'heart racing', 'extreme anxiety', 'overwhelming anxiety'
            ]
        }
        
        # Emergency contact information
        self.emergency_contacts = {
            'National Suicide Prevention Lifeline': '988',
            'Crisis Text Line': '741741',
            'Emergency Services': '911'
        }

    def detect_crisis(self, text):
        # Convert text to lowercase for case-insensitive matching
        text = text.lower()
        
        # Check for crisis keywords
        for category, keywords in self.crisis_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return True
        
        # Analyze sentiment for extreme negative emotions
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment < -0.8:  # Very negative sentiment
            return True
        
        # Check for patterns indicating immediate danger
        danger_patterns = [
            r'going to (kill|hurt) myself',
            r'right now',
            r'immediately',
            r'this instant'
        ]
        
        for pattern in danger_patterns:
            if re.search(pattern, text):
                return True
        
        return False

    def get_emergency_contacts(self):
        return self.emergency_contacts 