import os
import pandas as pd
import requests
import zipfile
from io import BytesIO
import json

def download_kaggle_dataset():
    """Download the Mental Health in Tech Survey dataset from Kaggle"""
    # Kaggle API credentials should be set in environment variables
    # KAGGLE_USERNAME and KAGGLE_KEY
    
    # Create datasets directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # Download the dataset
    dataset_url = "https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey/download"
    response = requests.get(dataset_url)
    
    if response.status_code == 200:
        # Extract the zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall('datasets')
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Failed to download dataset. Please check your Kaggle API credentials.")

def process_dataset():
    """Process the dataset and convert it to a format suitable for our chatbot"""
    # Read the dataset
    df = pd.read_csv('datasets/survey.csv')
    
    # Create training data format
    training_data = []
    
    # Process each row and create conversation pairs
    for _, row in df.iterrows():
        # Create input based on mental health concerns
        if pd.notna(row['comments']):
            input_text = row['comments']
            
            # Determine emotion based on various factors
            emotion = determine_emotion(row)
            
            # Generate appropriate response
            response = generate_response(emotion, row)
            
            training_data.append({
                'input': input_text,
                'emotion': emotion,
                'response': response
            })
    
    # Save processed data
    with open('datasets/processed_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(training_data)} training examples")

def determine_emotion(row):
    """Determine the emotion based on survey responses"""
    # Map survey responses to emotions
    if pd.notna(row['treatment']):
        if row['treatment'] == 'Yes':
            return 'anxiety'
    
    if pd.notna(row['work_interfere']):
        if row['work_interfere'] in ['Often', 'Sometimes']:
            return 'stress'
    
    if pd.notna(row['mental_health_consequence']):
        if row['mental_health_consequence'] == 'Yes':
            return 'fear'
    
    if pd.notna(row['coworkers']):
        if row['coworkers'] == 'No':
            return 'loneliness'
    
    return 'neutral'

def generate_response(emotion, row):
    """Generate appropriate response based on emotion and context"""
    responses = {
        'anxiety': "I understand you're feeling anxious about seeking treatment. It's a brave step to consider getting help. Would you like to talk about what's making you feel this way?",
        'stress': "Work-related stress can be really challenging. Let's talk about how this is affecting you and what might help.",
        'fear': "It's natural to feel concerned about consequences. Would you like to explore some ways to manage these feelings?",
        'loneliness': "Feeling disconnected at work can be difficult. Would you like to talk about ways to build more supportive relationships?",
        'neutral': "Thank you for sharing. How are you feeling about your current situation?"
    }
    
    return responses.get(emotion, responses['neutral'])

if __name__ == "__main__":
    print("Starting dataset download and processing...")
    download_kaggle_dataset()
    process_dataset()
    print("Dataset processing completed!") 