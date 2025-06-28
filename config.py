import os

class Config:
    def __init__(self):
        # Get the absolute path to this config file's directory
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Model paths
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'backend', 'chatbot_model', 'data')
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'backend', 'chatbot_model', 'data')
        
        # File paths
        self.INTENTS_FILE = os.path.join(self.DATA_DIR, 'intents.json')
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, 'model.h5')
        self.TOKENIZER_PATH = os.path.join(self.MODEL_DIR, 'tokenizer.pickle')
        self.LABEL_MAP_PATH = os.path.join(self.MODEL_DIR, 'label_map.pickle')
        
        # Model parameters
        self.MAX_SEQUENCE_LENGTH = 50
        self.EMBEDDING_DIM = 64
        self.EPOCHS = 100
        self.VALIDATION_SPLIT = 0.2
        self.MIN_CONFIDENCE = 0.7
        
        # Create directories if they don't exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

# Singleton configuration instance
config = Config()