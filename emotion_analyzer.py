from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class EmotionAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Emotion keywords
        self.emotion_keywords = {
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'lonely', 'hopeless'],
            'anxiety': ['anxious', 'worried', 'nervous', 'fear', 'panic', 'stress'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed'],
            'joy': ['happy', 'joyful', 'excited', 'delighted', 'cheerful', 'glad'],
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'horrified'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled']
        }

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return tokens

    def analyze_emotion(self, text):
        # Preprocess the text
        tokens = self.preprocess_text(text)
        
        # Get sentiment polarity
        sentiment = TextBlob(text).sentiment.polarity
        
        # Count emotion keywords
        emotion_counts = {emotion: 0 for emotion in self.emotion_keywords}
        
        for token in tokens:
            for emotion, keywords in self.emotion_keywords.items():
                if token in keywords:
                    emotion_counts[emotion] += 1
        
        # Find the emotion with the highest count
        max_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        # If no emotion keywords found, use sentiment analysis
        if max_emotion[1] == 0:
            if sentiment < -0.3:
                return 'sadness'
            elif sentiment > 0.3:
                return 'joy'
            else:
                return 'neutral'
        
        return max_emotion[0] 