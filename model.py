import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import random
from spellchecker import SpellChecker

class Chatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.intents = []
        self.max_len = 50
        self.min_confidence = 0.6  # Lowered threshold
        self.spell = SpellChecker()
        self.context = {}
        self.load_model()

    def load_model(self):
        """Load all required model artifacts"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, 'backend', 'chatbot_model', 'data')
            
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
            
            with open(os.path.join(model_dir, 'tokenizer.pickle'), 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            
            with open(os.path.join(model_dir, 'intents.json'), 'r') as f:
                self.intents = json.load(f)['intents']
            
            self.tag_to_idx = {intent['tag']: idx for idx, intent in enumerate(self.intents)}
            self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def preprocess_input(self, text):
        """Convert raw text to model input format with spell checking"""
        corrected = self.correct_spelling(text.lower())
        sequence = self.tokenizer.texts_to_sequences([corrected])
        return pad_sequences(sequence, maxlen=self.max_len)

    def correct_spelling(self, text):
        """Correct common spelling mistakes"""
        words = text.split()
        corrected = [self.spell.correction(word) if self.spell.unknown([word]) else word 
                    for word in words]
        return ' '.join(filter(None, corrected))

    def predict_intent(self, text):
        """Predict the most likely intent"""
        padded_seq = self.preprocess_input(text)
        prediction = self.model.predict(padded_seq, verbose=0)[0]
        predicted_idx = np.argmax(prediction)
        confidence = prediction[predicted_idx]
        return self.idx_to_tag[predicted_idx], confidence

    def get_response(self, tag):
        """Get random response for a given intent tag"""
        for intent in self.intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        return self.get_fallback_response(tag)

    def get_fallback_response(self, user_input):
        """Context-aware fallback responses"""
        user_input = user_input.lower()
        
        # Handle weather-related queries
        weather_terms = ['weather', 'whether', 'wheather', 'temperature', 'forecast']
        if any(term in user_input for term in weather_terms):
            return "Are you asking about the weather? Please tell me your location."
            
        # Handle greetings
        greeting_terms = ['hello', 'hi', 'hey', 'greetings']
        if any(term in user_input for term in greeting_terms):
            return random.choice(["Hello! How can I help?", "Hi there! What can I do for you?"])
            
        return random.choice([
            "I'm not sure I understand. Could you rephrase that?",
            "I'm still learning about that. Could you ask differently?",
            "I didn't quite get that. Can you ask in another way?"
        ])

    def generate_response(self, message, context=None):
        """Generate appropriate response for user input with context"""
        try:
            if not message.strip():
                return "Please type something so I can help you!"
            
            self.context = context or {}
            tag, confidence = self.predict_intent(message)
            
            if confidence < self.min_confidence:
                return self.get_fallback_response(message)
            
            response = self.get_response(tag)
            self.context['last_intent'] = tag
            return response
            
        except Exception as e:
            print(f"Response generation error: {str(e)}")
            return "I'm having trouble understanding right now. Please try again later."