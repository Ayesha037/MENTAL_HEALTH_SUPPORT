from flask import Flask, request, jsonify, render_template, Response
import random
import re
import speech_recognition as sr
import pyttsx3
import json
import base64
import os
import io
import tempfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import pickle
import datetime
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Training dataset for mental health conversations
training_data = [
    # Anxiety and Stress
    {
        'input': "I'm feeling really anxious about my upcoming exam",
        'emotion': 'anxiety',
        'response': "Exam anxiety is very common. Let's try a quick grounding exercise: name 5 things you can see in your room right now. Would you like to talk about what specific aspects of the exam are worrying you?"
    },
    {
        'input': "I can't sleep at night, my mind keeps racing",
        'emotion': 'stress',
        'response': "Sleep difficulties can be really frustrating. Have you tried any relaxation techniques before bed? We could explore some calming exercises that might help."
    },
    {
        'input': "I'm worried about my future",
        'emotion': 'anxiety',
        'response': "The future can feel uncertain and scary. Let's focus on what's in your control right now. What small steps could you take today?"
    },
    {
        'input': "I feel so stressed about work",
        'emotion': 'stress',
        'response': "Work stress can be really challenging. Have you noticed any physical symptoms like tension or changes in sleep? Let's talk about what's specifically stressful."
    },
    
    # Depression and Sadness
    {
        'input': "I feel so sad and empty inside",
        'emotion': 'sadness',
        'response': "I hear that you're feeling sad and empty. Depression can make everything feel heavy. Would you like to talk about what might be contributing to these feelings?"
    },
    {
        'input': "Nothing brings me joy anymore",
        'emotion': 'sadness',
        'response': "When depression takes away our joy, it can be really difficult. Let's explore what used to bring you happiness and what small things might help now."
    },
    {
        'input': "I feel worthless and hopeless",
        'emotion': 'sadness',
        'response': "Those feelings of worthlessness and hopelessness are really painful. Remember that depression can distort our thoughts. Would you like to talk about what makes you feel this way?"
    },
    
    # Loneliness and Social Issues
    {
        'input': "I feel so lonely, even when I'm around people",
        'emotion': 'loneliness',
        'response': "That's a really difficult feeling to experience. Sometimes we can feel lonely even in a crowd. Would you like to talk about what kind of connections you're missing?"
    },
    {
        'input': "I don't have any close friends",
        'emotion': 'loneliness',
        'response': "Building close friendships can be challenging. Let's talk about what kind of connections you're looking for and what small steps might help you build them."
    },
    {
        'input': "Everyone seems to have someone except me",
        'emotion': 'loneliness',
        'response': "Social media and society can make it seem like everyone has perfect relationships. Would you like to talk about what kind of connection you're looking for?"
    },
    
    # Self-esteem and Self-doubt
    {
        'input': "I feel like I'm not good enough at anything",
        'emotion': 'self_doubt',
        'response': "Self-doubt can be really challenging. What would you say to a friend who shared these same feelings? Sometimes we're much kinder to others than to ourselves."
    },
    {
        'input': "I keep comparing myself to others",
        'emotion': 'self_doubt',
        'response': "Comparison can be really damaging to our self-esteem. Let's focus on your unique strengths and qualities. What's something you're proud of about yourself?"
    },
    {
        'input': "I feel like a failure",
        'emotion': 'self_doubt',
        'response': "Those feelings of failure can be really painful. Remember that everyone makes mistakes and faces challenges. Would you like to talk about what's making you feel this way?"
    },
    
    # Grief and Loss
    {
        'input': "I'm grieving the loss of my pet",
        'emotion': 'grief',
        'response': "I'm so sorry for your loss. Pets become such important parts of our lives. Would you like to share some memories about your pet?"
    },
    {
        'input': "I lost someone close to me",
        'emotion': 'grief',
        'response': "I'm so sorry for your loss. Grief is a complex journey that takes time. Would you like to talk about your loved one or how you're coping?"
    },
    {
        'input': "The anniversary of my loss is coming up",
        'emotion': 'grief',
        'response': "Anniversaries can bring up strong emotions. It's okay to feel whatever you're feeling. Would you like to talk about how you'd like to honor this day?"
    },
    
    # Anger and Frustration
    {
        'input': "I'm so angry at my friend for betraying my trust",
        'emotion': 'anger',
        'response': "Betrayal of trust can be really painful. It's natural to feel angry. Would you like to talk about what happened and how it's affecting you?"
    },
    {
        'input': "Everything makes me angry lately",
        'emotion': 'anger',
        'response': "When we're feeling angry frequently, it can be a sign of underlying stress or hurt. Would you like to explore what might be contributing to these feelings?"
    },
    {
        'input': "I can't control my anger",
        'emotion': 'anger',
        'response': "Anger can feel overwhelming. Let's talk about some healthy ways to express and manage your anger. Would you like to learn some calming techniques?"
    },
    
    # Fear and Anxiety
    {
        'input': "I'm scared of failing",
        'emotion': 'fear',
        'response': "Fear of failure is a common experience. Remember that failure is often a stepping stone to growth. Would you like to talk about what success means to you?"
    },
    {
        'input': "I have panic attacks",
        'emotion': 'fear',
        'response': "Panic attacks can be really frightening. Let's talk about what triggers them and some techniques that might help you manage them. Would you like to learn some grounding exercises?"
    },
    {
        'input': "I'm afraid of the future",
        'emotion': 'fear',
        'response': "The future can feel scary and uncertain. Let's focus on what's in your control right now. What small steps could you take today?"
    },
    
    # Overwhelm and Burnout
    {
        'input': "Everything feels overwhelming right now",
        'emotion': 'overwhelm',
        'response': "When everything feels overwhelming, it helps to break things down into smaller steps. What's one small thing you could do right now to take care of yourself?"
    },
    {
        'input': "I'm completely burned out",
        'emotion': 'overwhelm',
        'response': "Burnout can be really exhausting. Let's talk about what's contributing to your burnout and what might help you recover. Would you like to explore some self-care options?"
    },
    {
        'input': "I can't handle all these responsibilities",
        'emotion': 'overwhelm',
        'response': "Having too many responsibilities can feel crushing. Let's identify what's most important and what might be able to wait. Would you like to make a priority list together?"
    }
]

# ML model variables
vectorizer = None
classifier = None
confidence_threshold = 0.3
response_templates = {}
conversation_history = []
conversation_count = 0
retraining_frequency = 10  # Retrain after every 10 conversations

def train_model():
    """Train the ML model for emotion classification"""
    global vectorizer, classifier, response_templates, conversation_history
    
    # Create initial training data if none exists
    if not conversation_history:
        # Initialize with training data
        for item in training_data:
            conversation_history.append({
                'input': item['input'],
                'response': item['response'],
                'emotion': item['emotion']
            })
            
            # Initialize response templates
            if item['emotion'] not in response_templates:
                response_templates[item['emotion']] = []
            
            if item['response'] not in response_templates[item['emotion']]:
                response_templates[item['emotion']].append(item['response'])
    
    # Extract data for training
    texts = [item['input'] for item in conversation_history]
    emotions = [item['emotion'] for item in conversation_history]
    
    # Initialize and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # Train a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, emotions)
    
    # Save the model
    with open('ml_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, classifier, response_templates, conversation_history), f)
    
    print("Model trained successfully with", len(conversation_history), "examples")

def load_or_create_model():
    """Load existing model or create a new one"""
    global vectorizer, classifier, response_templates, conversation_history
    
    try:
        # Try to load the existing model
        with open('ml_model.pkl', 'rb') as f:
            vectorizer, classifier, response_templates, conversation_history = pickle.load(f)
        print("Model loaded successfully")
    except (FileNotFoundError, EOFError):
        print("No model found, creating a new one...")
        train_model()

# Initialize the model when the app starts
load_or_create_model()

# Initialize learning parameters
learning_rate = 0.1
min_confidence_threshold = 0.6

# Extended responses for different emotions and therapeutic needs
responses = {
    'greeting': [
        "Hi there! 👋 I'm your friendly mental health companion. I'm here to listen, support, and chat with you. How are you feeling today?",
        "Hello! 🌟 I'm so glad you're here. It's a safe space to share your thoughts and feelings. What's on your mind?",
        "Welcome! 💫 I'm here to be your chat buddy. No judgment, just support and understanding. How can I help you today?",
        "Hey! 🌈 I'm here to make you feel comfortable and supported. What would you like to talk about?",
        "Hi friend! 🤗 I'm here to listen and support you. It's okay to share whatever you're feeling - the good, the bad, or the in-between."
    ],
    'sadness': [
        "I hear that you're feeling sad. Would you like to talk more about what's bothering you?",
        "It's okay to feel sad. Remember that emotions are temporary and will pass with time.",
        "I understand that you're going through a difficult time. Could we explore some small positive steps you might take today?",
        "Depression can make everything feel overwhelming. Let's break things down into smaller, manageable parts.",
        "Sometimes sadness is a natural response to life's challenges. Would it help to talk about what triggered these feelings?"
    ],
    'anxiety': [
        "Anxiety can be really challenging. Let's try a quick breathing exercise: breathe in for 4 counts, hold for 4, and exhale for 6.",
        "I understand anxiety can feel overwhelming. Try grounding yourself by naming 5 things you can see right now.",
        "Remember that anxiety is a natural response, but you can learn to manage it. What specific worries are on your mind?",
        "It sounds like you're feeling anxious. Sometimes writing down our worries can help put them in perspective. Have you tried journaling?",
        "When anxiety strikes, it can help to focus on what's in your control and what isn't. Would you like to talk about that?"
    ],
    'anger': [
        "I can sense that you're feeling angry. Anger is often a signal that something important to us has been threatened.",
        "It's okay to feel angry. Would you like to explore what triggered these feelings?",
        "I'm here to listen without judgment. Sometimes expressing anger in a safe way can be healthy - would you like to tell me more?",
        "When we're angry, our body tends to tense up. Try taking a few deep breaths and notice any tension you're holding.",
        "Anger is often a secondary emotion. Sometimes it helps to identify what might be beneath it - perhaps hurt or fear?"
    ],
    'fear': [
        "Fear is our mind's way of trying to protect us. What feels threatening right now?",
        "It takes courage to face our fears. Would you like to explore small steps to approach this fear gradually?",
        "Sometimes our fears can seem bigger than they really are. Let's talk about what's specifically concerning you.",
        "When we name our fears, they often become less overwhelming. Can you describe what you're afraid of?",
        "I hear that you're feeling scared. Remember that you've overcome difficult situations before."
    ],
    'loneliness': [
        "Feeling lonely can be really painful. Would you like to talk about ways to connect with others?",
        "Even when we're surrounded by people, we can feel lonely. Are there specific relationships you're missing?",
        "Loneliness is a common human experience. What kinds of connections would feel meaningful to you right now?",
        "I'm here with you. While I'm not a replacement for human connection, I'm listening to everything you say.",
        "Sometimes loneliness can be an opportunity to reconnect with ourselves. Have you tried spending quality time with yourself lately?"
    ],
    'grief': [
        "Grief is a natural response to loss. It's okay to take the time you need to process your feelings.",
        "I'm so sorry for your loss. Would you like to share some memories about what/who you're missing?",
        "Everyone experiences grief differently. There's no right or wrong way to feel.",
        "Grief can come in waves. On difficult days, what small things might bring you comfort?",
        "It's okay to hold both joy and sadness together. Finding moments of peace doesn't mean you're forgetting what matters."
    ],
    'stress': [
        "Stress can affect both our mind and body. Have you noticed physical symptoms like tension or changes in sleep?",
        "When we're stressed, it helps to prioritize what truly needs our attention. Could we make a list together?",
        "Taking short breaks throughout the day can help manage stress. What small moments of calm could you build into your routine?",
        "Sometimes stress comes from trying to control things beyond our power. Can we explore what's within your control right now?",
        "Stress is often about perceived demands exceeding our resources. What support might help lighten your load?"
    ],
    'joy': [
        "It's wonderful to hear you're feeling happy! What specifically brought you joy today?",
        "Positive emotions are worth savoring. Could you tell me more about what's going well for you?",
        "Joy is a beautiful emotion to experience. How might you carry this feeling with you throughout your day?",
        "I'm glad you're feeling good! Sometimes writing down positive moments helps us remember them during harder times.",
        "That sounds really positive! Is there someone in your life you could share this good feeling with?"
    ],
    'self_doubt': [
        "I hear you questioning yourself. Remember that self-doubt is common, but it doesn't define your capabilities.",
        "We all have an inner critic sometimes. What would you say to a friend who shared these same doubts?",
        "It takes courage to recognize self-doubt. Could we explore evidence that might contradict these negative thoughts?",
        "Sometimes our minds present thoughts as facts. Let's practice noticing thoughts without automatically believing them.",
        "Self-compassion can be a powerful antidote to self-doubt. How might you speak to yourself more kindly today?"
    ],
    'overwhelm': [
        "When everything feels overwhelming, it helps to focus on just the next small step. What's one tiny thing you could do?",
        "I hear that you're feeling overwhelmed. Let's break things down - what's the most pressing concern right now?",
        "Sometimes overwhelm comes from trying to hold too much in our minds. Would writing things down help create some mental space?",
        "It's okay to set boundaries when you're feeling overwhelmed. Are there commitments you might need to pause?",
        "Taking care of basic needs becomes even more important when we're overwhelmed. How are you doing with sleep, food, and movement?"
    ],
    'crisis': [
        "I'm concerned about your safety right now. Would it help to talk about what's happening and explore some immediate steps?",
        "Your life has value, even if it doesn't feel that way right now. Have you thought about reaching out to a crisis hotline?",
        "I want to make sure you're safe. Do you have someone you trust who could be with you right now?",
        "These intense feelings won't last forever, even though they feel overwhelming right now. Let's focus on getting through just the next hour safely.",
        "Thank you for trusting me with these difficult thoughts. Getting professional support is important - would you consider calling emergency services?"
    ],
    'default': [
        "I'm here to listen. Could you tell me more about that?",
        "Thank you for sharing. How does that make you feel?",
        "I understand. Would you like to explore that further?",
        "That sounds challenging. What thoughts come up for you when you experience this?",
        "I appreciate you opening up. How long have you been feeling this way?"
    ],
    'coping_strategies': [
        "When we're struggling, having a toolkit of coping strategies can help. Would you like to explore some options that might work for you?",
        "Deep breathing can help calm your nervous system. Try breathing in for 4 counts, hold for 1, and exhale for 5.",
        "Physical movement, even just a short walk, can sometimes shift our emotional state. Would that be possible for you today?",
        "Grounding exercises can help when emotions feel intense. Try naming 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
        "Mindfulness means bringing attention to the present moment without judgment. What do you notice in your body right now?"
    ],
    'gratitude': [
        "Even in difficult times, noticing small things we're grateful for can be helpful. Is there anything small that brought you comfort today?",
        "Gratitude practices have been shown to improve mental wellbeing. Would you like to try naming three things you appreciate?",
        "Sometimes shifting our focus to what's going well, even tiny things, can create a little breathing room from challenges.",
        "What's something small you've enjoyed recently? Maybe a cup of tea, a moment of sunshine, or a kind word?",
        "Our brains naturally focus on problems, but we can train them to also notice positive things. What small good moments have you experienced lately?"
    ]
}

# Crisis keywords - expanded
crisis_keywords = [
    'suicide', 'kill myself', 'end my life', 'want to die', 'don\'t want to live', 
    'life is not worth living', 'better off dead', 'no reason to live',
    'cut myself', 'self-harm', 'hurt myself', 'plan to die',
    'how to commit suicide', 'ending it all', 'no way out',
    'can\'t go on', 'too much pain', 'overdose', 'nobody would miss me',
    'hopeless', 'worthless', 'burden', 'unbearable'
]

# Emotion keywords - expanded
emotion_keywords = {
    'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'lonely', 'hopeless', 'blue', 'down', 'gloomy', 'heartbroken', 'grieving', 'melancholy', 'despair', 'disappointed', 'discouraged', 'defeated'],
    'anxiety': ['anxious', 'worried', 'nervous', 'fear', 'panic', 'stress', 'tense', 'restless', 'uneasy', 'dread', 'frightened', 'apprehensive', 'paranoid', 'overwhelmed', 'flustered', 'on edge'],
    'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated', 'frustrated', 'resentful', 'bitter', 'outraged', 'hostile', 'enraged', 'seething', 'offended', 'disgusted'],
    'joy': ['happy', 'joyful', 'excited', 'delighted', 'cheerful', 'glad', 'elated', 'thrilled', 'ecstatic', 'content', 'pleased', 'satisfied', 'hopeful', 'optimistic', 'grateful', 'peaceful'],
    'fear': ['scared', 'terrified', 'afraid', 'petrified', 'alarmed', 'spooked', 'startled', 'horrified', 'fearful', 'threatened', 'intimidated', 'insecure'],
    'loneliness': ['lonely', 'alone', 'isolated', 'abandoned', 'rejected', 'disconnected', 'left out', 'unwanted', 'unloved', 'forgotten', 'solitary'],
    'grief': ['grief', 'grieving', 'mourning', 'loss', 'bereavement', 'devastated', 'heartache', 'missing', 'yearning', 'shattered'],
    'stress': ['stressed', 'overwhelmed', 'pressured', 'burdened', 'strained', 'frazzled', 'exhausted', 'burned out', 'overloaded', 'swamped'],
    'self_doubt': ['worthless', 'inadequate', 'failure', 'incompetent', 'insecure', 'doubt', 'useless', 'inferior', 'unworthy', 'undeserving', 'not good enough'],
    'overwhelm': ['overwhelmed', 'too much', 'can\'t cope', 'drowning', 'crushed', 'can\'t handle', 'at my limit', 'exhausted', 'drained', 'breaking point']
}

# Therapeutic intervention patterns
therapy_patterns = {
    'needs_validation': [
        r'nobody understands',
        r'no one gets it',
        r'i feel so alone in this',
        r'no one listens',
        r'i\'m not being heard'
    ],
    'needs_perspective': [
        r'always',
        r'never',
        r'everyone',
        r'no one',
        r'nothing works',
        r'everything is'
    ],
    'needs_coping': [
        r'don\'t know what to do',
        r'can\'t handle',
        r'how do i deal',
        r'how to cope',
        r'need help with'
    ],
    'needs_gratitude': [
        r'nothing good',
        r'everything is bad',
        r'life sucks',
        r'nothing positive',
        r'everything\'s terrible'
    ]
}

# Add real-time response patterns
real_time_patterns = {
    'time_based': {
        'morning': ['good morning', 'morning', 'wake up', 'start of day'],
        'afternoon': ['good afternoon', 'afternoon', 'lunch', 'midday'],
        'evening': ['good evening', 'evening', 'dinner', 'night'],
        'night': ['good night', 'night', 'sleep', 'bedtime']
    },
    'activity_based': {
        'work': ['work', 'job', 'office', 'career', 'business'],
        'study': ['study', 'school', 'college', 'university', 'exam'],
        'social': ['friend', 'party', 'social', 'meet', 'hang out'],
        'exercise': ['exercise', 'workout', 'gym', 'sport', 'run']
    },
    'health_based': {
        'sleep': ['sleep', 'tired', 'insomnia', 'rest', 'bed'],
        'diet': ['food', 'eat', 'diet', 'hungry', 'meal'],
        'physical': ['pain', 'ache', 'sick', 'ill', 'health'],
        'mental': ['mind', 'thought', 'brain', 'mental', 'psychology']
    }
}

# Crisis helpline numbers
crisis_helplines = {
    'general': {
        'National Suicide Prevention Lifeline': '988',
        'Crisis Text Line': '741741',
        'Emergency Services': '911'
    },
    'mental_health': {
        'SAMHSA National Helpline': '1-800-662-4357',
        'NAMI Helpline': '1-800-950-6264',
        'Mental Health America': '1-800-273-8255'
    },
    'youth': {
        'Youth Crisis Line': '1-800-448-4663',
        'Teen Line': '1-800-852-8336',
        'Youth Mental Health Line': '1-800-273-8255'
    }
}

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level

def analyze_emotion(text):
    """Enhanced emotion analysis with confidence scores"""
    text = text.lower()
    
    # Create a score dictionary for each emotion
    emotion_scores = {emotion: 0 for emotion in emotion_keywords.keys()}
    
    # Calculate scores for each emotion
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text:
                # Exact match gets higher score
                if re.search(r'\b' + keyword + r'\b', text):
                    emotion_scores[emotion] += 2
                else:
                    emotion_scores[emotion] += 1
    
    # Determine the primary emotion
    if max(emotion_scores.values()) > 0:
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
    else:
        primary_emotion = 'neutral'
    
    return primary_emotion

def detect_crisis(text):
    """Enhanced crisis detection with severity rating"""
    text = text.lower()
    
    # Count crisis keywords
    crisis_count = 0
    matched_keywords = []
    
    for keyword in crisis_keywords:
        if keyword in text:
            crisis_count += 1
            matched_keywords.append(keyword)
    
    # Determine severity
    is_crisis = crisis_count > 0
    
    return is_crisis

def detect_therapeutic_need(text):
    """Detect specific therapeutic needs based on patterns"""
    text = text.lower()
    
    for need, patterns in therapy_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return need
    
    return None

def get_time_based_greeting():
    """Get appropriate friendly greeting based on time of day"""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning! 🌅 I'm here to chat and support you. How are you feeling today? Remember, it's okay to share whatever's on your mind."
    elif 12 <= hour < 17:
        return "Good afternoon! ☀️ I'm glad you're here. How's your day going? I'm here to listen and support you."
    elif 17 <= hour < 22:
        return "Good evening! 🌙 Welcome! I'm here to be your friendly chat companion. What would you like to talk about?"
    else:
        return "It's late, but I'm still here for you! 🌙 Sometimes the quiet hours are the best time to talk. How are you feeling?"

def analyze_context(text):
    """Analyze context of the conversation"""
    context = {
        'time_of_day': None,
        'activity': None,
        'health_focus': None
    }
    
    text = text.lower()
    
    # Check time-based patterns
    for time, keywords in real_time_patterns['time_based'].items():
        if any(keyword in text for keyword in keywords):
            context['time_of_day'] = time
            break
    
    # Check activity-based patterns
    for activity, keywords in real_time_patterns['activity_based'].items():
        if any(keyword in text for keyword in keywords):
            context['activity'] = activity
            break
    
    # Check health-based patterns
    for health, keywords in real_time_patterns['health_based'].items():
        if any(keyword in text for keyword in keywords):
            context['health_focus'] = health
            break
    
    return context

def get_contextual_response(text, emotion, context):
    """Generate contextual response based on conversation analysis"""
    response_parts = []
    
    # Add time-based greeting if appropriate
    if context['time_of_day']:
        response_parts.append(get_time_based_greeting())
    
    # Add activity-specific response
    if context['activity']:
        if context['activity'] == 'work':
            response_parts.append("Work can be challenging. Would you like to talk about what's happening at work?")
        elif context['activity'] == 'study':
            response_parts.append("Academic pressure can be stressful. How are you managing your studies?")
        elif context['activity'] == 'social':
            response_parts.append("Social situations can bring up different emotions. How are you feeling about your social life?")
        elif context['activity'] == 'exercise':
            response_parts.append("Physical activity can help with mental wellbeing. Are you finding exercise helpful?")
    
    # Add health-specific response
    if context['health_focus']:
        if context['health_focus'] == 'sleep':
            response_parts.append("Sleep is crucial for mental health. How has your sleep been lately?")
        elif context['health_focus'] == 'diet':
            response_parts.append("Nutrition can affect our mood. How are you feeling about your eating habits?")
        elif context['health_focus'] == 'physical':
            response_parts.append("Physical health and mental health are connected. Would you like to talk about how you're feeling physically?")
        elif context['health_focus'] == 'mental':
            response_parts.append("Mental health is just as important as physical health. How are you taking care of your mental wellbeing?")
    
    # Add emotion-based response
    if emotion in responses:
        response_parts.append(random.choice(responses[emotion]))
    
    return " ".join(response_parts)

def update_response_templates(user_input, response, emotion):
    """Update response templates based on successful interactions"""
    # Initialize response templates for the emotion if it doesn't exist
    if emotion not in response_templates:
        response_templates[emotion] = []
    
    # Add new template
    response_templates[emotion].append({
        'input': user_input,
        'response': response,
        'timestamp': datetime.datetime.now()
    })
    
    # Keep only the most recent 100 templates per emotion
    if len(response_templates[emotion]) > 100:
        response_templates[emotion] = response_templates[emotion][-100:]

def get_ml_response(text, emotion):
    """Get response using ML model and templates"""
    # Initialize response templates for the emotion if it doesn't exist
    if emotion not in response_templates:
        response_templates[emotion] = []
    
    # If no templates for this emotion, use contextual response
    if len(response_templates[emotion]) == 0:
        return get_contextual_response(text, emotion, analyze_context(text))
    
    # Find similar past interactions
    similar_responses = []
    for template in response_templates[emotion]:
        if isinstance(template, dict) and 'input' in template:
            similarity = cosine_similarity(
                vectorizer.transform([text]),
                vectorizer.transform([template['input']])
            )[0][0]
            
            if similarity > 0.3:  # Similarity threshold
                similar_responses.append((template['response'], similarity))
    
    if similar_responses:
        # Sort by similarity and get the best match
        similar_responses.sort(key=lambda x: x[1], reverse=True)
        return similar_responses[0][0]
    
    return get_contextual_response(text, emotion, analyze_context(text))

def update_conversation_history(user_input, response, emotion):
    """Update conversation history and ML model"""
    conversation_history.append({
        'user_input': user_input,
        'response': response,
        'emotion': emotion,
        'timestamp': datetime.datetime.now()
    })
    
    # Update response templates
    update_response_templates(user_input, response, emotion)
    
    # Retrain model periodically
    if len(conversation_history) % 10 == 0:
        train_model()
    
    # Save updated history
    with open('conversation_history.pkl', 'wb') as f:
        pickle.dump(conversation_history, f)

def get_crisis_helpline_info():
    """Get formatted crisis helpline information"""
    helpline_info = "Here are some helpline numbers that might help:\n\n"
    
    for category, helplines in crisis_helplines.items():
        helpline_info += f"{category.replace('_', ' ').title()} Helplines:\n"
        for name, number in helplines.items():
            helpline_info += f"- {name}: {number}\n"
        helpline_info += "\n"
    
    return helpline_info

def get_response(text, emotion):
    """Enhanced response generation with ML"""
    text = text.lower()
    
    # Check for greetings with more friendly variations
    greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'good night']
    if any(word in text for word in greeting_words):
        return get_time_based_greeting()
    
    # Check for crisis - highest priority
    if detect_crisis(text):
        crisis_response = random.choice(responses['crisis'])
        return f"{crisis_response}\n\n{get_crisis_helpline_info()}"
    
    # If emotion is neutral, use contextual response
    if emotion == 'neutral':
        return get_contextual_response(text, emotion, analyze_context(text))
    
    # Get ML-enhanced response
    response = get_ml_response(text, emotion)
    
    # Update conversation history
    update_conversation_history(text, response, emotion)
    
    return response

def text_to_speech(text):
    """Convert text to speech and return as base64 audio data"""
    try:
        # Create a temporary file to save the audio
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # Save speech to the temporary file
        engine.save_to_file(text, temp_filename)
        engine.runAndWait()
        
        # Read the file and convert to base64
        with open(temp_filename, 'rb') as f:
            audio_data = f.read()
        
        # Remove the temporary file
        os.unlink(temp_filename)
        
        # Convert to base64
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return None

def get_welcome_message():
    """Get a warm, friendly welcome message"""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "🌅 Good morning! I'm your friendly mental health companion. I'm here to listen, support, and chat with you. How are you feeling today? Remember, this is a safe space to share whatever's on your mind. Take your time - I'm here for you."
    elif 12 <= hour < 17:
        return "☀️ Good afternoon! I'm so glad you're here. I'm your friendly chat buddy, ready to listen and support you. How's your day going? Feel free to share anything - the good, the bad, or the in-between. I'm here to listen without judgment."
    elif 17 <= hour < 22:
        return "🌙 Good evening! Welcome to our safe space! I'm here to be your friendly companion and support you. What would you like to talk about? Remember, it's okay to share whatever you're feeling - I'm here to listen and help."
    else:
        return "🌙 It's late, but I'm still here for you! Sometimes the quiet hours are the best time to talk. I'm your friendly mental health companion, ready to listen and support you. How are you feeling? Take your time - I'm here all night."

@app.route('/')
def home():
    # Get initial welcome message
    welcome_message = get_welcome_message()
    return render_template('index.html', welcome_message=welcome_message)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    # If this is the first message, send a warm welcome
    if user_input.lower() in ['', 'hi', 'hello', 'hey']:
        return jsonify({
            'response': get_welcome_message(),
            'emotion': 'neutral',
            'is_crisis': False
        })
    
    # Analyze emotion
    emotion = analyze_emotion(user_input)
    
    # Check for crisis
    is_crisis = detect_crisis(user_input)
    
    # Get response
    response = get_response(user_input, emotion)
    
    return jsonify({
        'response': response,
        'emotion': emotion,
        'is_crisis': is_crisis
    })

@app.route('/voice', methods=['POST'])
def voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    
    try:
        # Save the uploaded file temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_audio.name)
        temp_audio.close()
        
        # Convert speech to text
        with sr.AudioFile(temp_audio.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Remove temporary file
        os.unlink(temp_audio.name)
        
        # Process the text
        emotion = analyze_emotion(text)
        is_crisis = detect_crisis(text)
        response = get_response(text, emotion)
        
        # Convert response to speech
        audio_response = text_to_speech(response)
        
        return jsonify({
            'text': text,
            'response': response,
            'emotion': emotion,
            'is_crisis': is_crisis,
            'audio_response': audio_response
        })
    
    except sr.UnknownValueError:
        return jsonify({'error': "Could not understand the audio"}), 400
    except sr.RequestError:
        return jsonify({'error': "Could not request results from speech recognition service"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_emotion(text):
    """Predict emotion using ML model"""
    if len(conversation_history) < 10:
        return analyze_emotion(text)
    
    # Transform input text
    X = vectorizer.transform([text])
    
    # Get prediction and probability
    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]
    confidence = max(probabilities)
    
    if confidence < min_confidence_threshold:
        return analyze_emotion(text)
    
    return prediction

if __name__ == '__main__':
    app.run(debug=True) 