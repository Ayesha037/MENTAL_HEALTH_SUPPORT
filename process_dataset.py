import pandas as pd
import json
import os

def process_kaggle_dataset(csv_path):
     
    try:
        
        df = pd.read_csv(csv_path)
        
        l
        training_data = []
        
        
        for _, row in df.iterrows():
           
            if pd.notna(row['comments']):
                input_text = str(row['comments']).strip()
                
                
                if not input_text:
                    continue
                
           
                emotion = determine_emotion(row)
                
                
                response = generate_response(emotion, row)
                
                training_data.append({
                    'input': input_text,
                    'emotion': emotion,
                    'response': response
                })
        
       
        output_path = 'datasets/processed_training_data.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {len(training_data)} training examples")
        print(f"Data saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

def determine_emotion(row):
    
    emotions = []
    
    
    if pd.notna(row.get('treatment')):
        if row['treatment'] == 'Yes':
            emotions.append('anxiety')
    
    
    if pd.notna(row.get('work_interfere')):
        if row['work_interfere'] in ['Often', 'Sometimes']:
            emotions.append('stress')
    
  
    if pd.notna(row.get('mental_health_consequence')):
        if row['mental_health_consequence'] == 'Yes':
            emotions.append('fear')
    
    
    if pd.notna(row.get('coworkers')):
        if row['coworkers'] == 'No':
            emotions.append('loneliness')
    
    
    if pd.notna(row.get('mental_health_interview')):
        if row['mental_health_interview'] == 'No':
            emotions.append('fear')
    
    
    if pd.notna(row.get('seek_help')):
        if row['seek_help'] == 'No':
            emotions.append('anxiety')
    
 
    if emotions:
        from collections import Counter
        return Counter(emotions).most_common(1)[0][0]
    return 'neutral'

def generate_response(emotion, row):
    responses = {
        'anxiety': "I understand you're feeling anxious about seeking treatment. It's a brave step to consider getting help. Would you like to talk about what's making you feel this way?",
        'stress': "Work-related stress can be really challenging. Let's talk about how this is affecting you and what might help.",
        'fear': "It's natural to feel concerned about consequences. Would you like to explore some ways to manage these feelings?",
        'loneliness': "Feeling disconnected at work can be difficult. Would you like to talk about ways to build more supportive relationships?",
        'neutral': "Thank you for sharing. How are you feeling about your current situation?"
    }
    
    return responses.get(emotion, responses['neutral'])

if __name__ == "__main__":
    
    dataset_path = input("Please enter the path to your downloaded Kaggle dataset (survey.csv): ").strip()
    
    
    if not os.path.exists(dataset_path):
        print(f"Error: File not found at {dataset_path}")
    else:
        process_kaggle_dataset(dataset_path) 