import pandas as pd
import json
import os

def process_kaggle_dataset(csv_path):
    """Process the Kaggle Mental Health in Tech Survey dataset"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Create training data
        training_data = []
        
        # Process each row
        for _, row in df.iterrows():
            # Create input based on comments and other relevant fields
            if pd.notna(row['comments']):
                input_text = str(row['comments']).strip()
                
                # Skip empty comments
                if not input_text:
                    continue
                
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
        output_path = 'datasets/processed_training_data.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {len(training_data)} training examples")
        print(f"Data saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

def determine_emotion(row):
    """Determine the emotion based on survey responses"""
    # Map survey responses to emotions
    emotions = []
    
    # Check treatment status
    if pd.notna(row.get('treatment')):
        if row['treatment'] == 'Yes':
            emotions.append('anxiety')
    
    # Check work interference
    if pd.notna(row.get('work_interfere')):
        if row['work_interfere'] in ['Often', 'Sometimes']:
            emotions.append('stress')
    
    # Check mental health consequences
    if pd.notna(row.get('mental_health_consequence')):
        if row['mental_health_consequence'] == 'Yes':
            emotions.append('fear')
    
    # Check coworker support
    if pd.notna(row.get('coworkers')):
        if row['coworkers'] == 'No':
            emotions.append('loneliness')
    
    # Check interview concerns
    if pd.notna(row.get('mental_health_interview')):
        if row['mental_health_interview'] == 'No':
            emotions.append('fear')
    
    # Check help-seeking behavior
    if pd.notna(row.get('seek_help')):
        if row['seek_help'] == 'No':
            emotions.append('anxiety')
    
    # Return the most common emotion or neutral if none found
    if emotions:
        from collections import Counter
        return Counter(emotions).most_common(1)[0][0]
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
    # Get the path to the dataset
    dataset_path = input("Please enter the path to your downloaded Kaggle dataset (survey.csv): ").strip()
    
    # Validate the path
    if not os.path.exists(dataset_path):
        print(f"Error: File not found at {dataset_path}")
    else:
        process_kaggle_dataset(dataset_path) 