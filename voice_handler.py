import speech_recognition as sr
import pyttsx3
import wave
import io
import base64
import os
import tempfile

class VoiceHandler:
    def __init__(self):
        try:
            self.recognizer = sr.Recognizer()
            self.engine = pyttsx3.init()
            
            # Configure the text-to-speech engine
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level
            
            # Test the recognizer
            self.recognizer.energy_threshold = 4000  # Adjust for ambient noise
            self.recognizer.dynamic_energy_threshold = True
            
            print("Voice handler initialized successfully")
        except Exception as e:
            print(f"Error initializing voice handler: {str(e)}")
            raise

    def speech_to_text(self, audio_data):
        try:
            # Create a temporary file to save the audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio_data.save(temp_file.name)
            temp_file.close()
            
            # Process the audio file
            with sr.AudioFile(temp_file.name) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Record the audio
                audio = self.recognizer.record(source)
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Sorry, there was an error with the speech recognition service."
        except Exception as e:
            print(f"Speech recognition error: {str(e)}")
            return f"An error occurred: {str(e)}"

    def text_to_speech(self, text):
        try:
            # Create a temporary file to save the audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_filename = temp_file.name
            temp_file.close()
            
            # Save speech to the temporary file
            self.engine.save_to_file(text, temp_filename)
            self.engine.runAndWait()
            
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

    def adjust_voice_settings(self, rate=None, volume=None):
        """
        Adjust the voice settings for text-to-speech
        """
        try:
            if rate is not None:
                self.engine.setProperty('rate', rate)
            if volume is not None:
                self.engine.setProperty('volume', volume)
        except Exception as e:
            print(f"Error adjusting voice settings: {str(e)}") 