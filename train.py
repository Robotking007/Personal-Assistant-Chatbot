import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import Counter
import logging
from spellchecker import SpellChecker
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotTrainer:
    def __init__(self):
        self.config = self._load_config()
        self.tokenizer = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
        self.spell = SpellChecker()
        self._ensure_dirs_exist()
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def _load_config(self):
        return {
            'intents_path': os.path.join(os.path.dirname(__file__), 'data', 'intents.json'),
            'model_dir': os.path.join(os.path.dirname(__file__), 'data'),
            'max_seq_len': 20,
            'embedding_dim': 128,
            'lstm_units': 64,
            'dense_units': 64,
            'dropout_rate': 0.4,
            'l2_reg': 0.001,
            'epochs': 100,
            'batch_size': 16,
            'validation_split': 0.2,
            'min_confidence': 0.6,
            'min_samples_per_class': 2
        }

    def _ensure_dirs_exist(self):
        os.makedirs(self.config['model_dir'], exist_ok=True)

    def _validate_intents(self, intents_data):
        """Validate and filter intents to ensure minimum samples per class"""
        valid_intents = []
        class_counts = Counter()
        
        for intent in intents_data['intents']:
            if not all(key in intent for key in ['tag', 'patterns', 'responses']):
                logger.warning(f"Skipping malformed intent: {intent.get('tag', 'unknown')}")
                continue
                
            # Spell check and clean patterns
            clean_patterns = []
            for pattern in intent['patterns']:
                cleaned = self._clean_text(pattern)
                if cleaned and cleaned not in clean_patterns:
                    clean_patterns.append(cleaned)
            
            if len(clean_patterns) < self.config['min_samples_per_class']:
                logger.warning(f"Skipping intent '{intent['tag']}' - needs at least {self.config['min_samples_per_class']} unique patterns")
                continue
                
            # Ensure responses exist
            if not intent['responses']:
                intent['responses'] = [f"I can help with {intent['tag']}"]
                logger.warning(f"Added default response for intent '{intent['tag']}'")
            
            valid_intents.append({
                'tag': intent['tag'],
                'patterns': clean_patterns,
                'responses': intent['responses']
            })
            class_counts[intent['tag']] = len(clean_patterns)
        
        if len(valid_intents) < 2:
            raise ValueError(f"Need at least 2 valid intents, found {len(valid_intents)}")
            
        logger.info(f"Class distribution:\n{json.dumps(class_counts, indent=2)}")
        return {'intents': valid_intents}

    def _clean_text(self, text):
        """Clean and normalize text with spell checking"""
        text = text.lower().strip()
        words = text.split()
        corrected = []
        
        for word in words:
            if word.isalpha():  # Only check spelling for alphabetic words
                correction = self.spell.correction(word)
                corrected.append(correction if correction else word)
            else:
                corrected.append(word)
                
        return ' '.join(corrected)

    def _prepare_training_data(self, intents):
        """Convert intents to training data with balanced classes"""
        patterns = []
        labels = []
        tag_to_idx = {intent['tag']: idx for idx, intent in enumerate(intents['intents'])}
        
        # Create balanced dataset
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern)
                labels.append(tag_to_idx[intent['tag']])
        
        return patterns, np.array(labels), tag_to_idx

    def _build_model(self, vocab_size, num_classes):
        """Construct the neural network architecture"""
        model = Sequential([
            Embedding(
                input_dim=vocab_size,
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_seq_len'],
                mask_zero=True
            ),
            Bidirectional(LSTM(
                self.config['lstm_units'],
                return_sequences=True,
                kernel_regularizer=l2(self.config['l2_reg'])
            )),
            Dropout(self.config['dropout_rate']),
            Bidirectional(LSTM(
                self.config['lstm_units'] // 2,
                kernel_regularizer=l2(self.config['l2_reg'])
            )),
            Dense(
                self.config['dense_units'],
                activation='relu',
                kernel_regularizer=l2(self.config['l2_reg'])
            ),
            Dropout(self.config['dropout_rate'] / 2),
            Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=l2(self.config['l2_reg'])
            )
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self):
        """Complete training workflow with validation and model saving"""
        try:
            logger.info("\n=== Starting Training Session ===")
            logger.info(f"Configuration:\n{json.dumps(self.config, indent=2)}")
            
            # Load and validate data
            with open(self.config['intents_path'], 'r', encoding='utf-8') as f:
                intents = self._validate_intents(json.load(f))
            
            logger.info(f"Training with {len(intents['intents'])} intents")
            
            # Prepare training data
            patterns, labels, tag_to_idx = self._prepare_training_data(intents)
            self.tokenizer.fit_on_texts(patterns)
            sequences = self.tokenizer.texts_to_sequences(patterns)
            padded = pad_sequences(sequences, maxlen=self.config['max_seq_len'])
            
            # Split data with stratification
            X_train, X_val, y_train, y_val = train_test_split(
                padded, labels,
                test_size=self.config['validation_split'],
                stratify=labels,
                random_state=42
            )
            
            # Build and train model
            model = self._build_model(
                vocab_size=len(self.tokenizer.word_index) + 1,
                num_classes=len(tag_to_idx)
            )
            
            callbacks = [
                ModelCheckpoint(
                    os.path.join(self.config['model_dir'], 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            logger.info("\nModel Architecture:")
            model.summary(print_fn=lambda x: logger.info(x))
            
            logger.info("\nStarting training...")
            history = model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save all artifacts
            self._save_artifacts(model, tag_to_idx, intents)
            
            # Training report
            best_epoch = np.argmax(history.history['val_accuracy'])
            final_accuracy = history.history['accuracy'][best_epoch]
            val_accuracy = history.history['val_accuracy'][best_epoch]
            
            logger.info("\n=== Training Complete ===")
            logger.info(f"Best Validation Accuracy: {val_accuracy:.2%} (epoch {best_epoch + 1})")
            logger.info(f"Final Training Accuracy: {final_accuracy:.2%}")
            
            return {
                'accuracy': val_accuracy,
                'training_accuracy': final_accuracy,
                'epochs': best_epoch + 1,
                'num_intents': len(tag_to_idx),
                'vocab_size': len(self.tokenizer.word_index) + 1
            }
            
        except Exception as e:
            logger.error(f"\n!!! Training Failed !!!\nError: {str(e)}")
            raise

    def _save_artifacts(self, model, tag_to_idx, intents):
        """Save all model artifacts to disk"""
        artifacts = {
            'model': os.path.join(self.config['model_dir'], 'model.h5'),
            'tokenizer': os.path.join(self.config['model_dir'], 'tokenizer.pickle'),
            'tag_map': os.path.join(self.config['model_dir'], 'tag_to_idx.pickle'),
            'intents': os.path.join(self.config['model_dir'], 'intents_processed.json'),
            'metadata': os.path.join(self.config['model_dir'], 'metadata.json')
        }
        
        try:
            # Save model
            model.save(artifacts['model'])
            
            # Save tokenizer
            with open(artifacts['tokenizer'], 'wb') as f:
                pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save tag mapping
            with open(artifacts['tag_map'], 'wb') as f:
                pickle.dump(tag_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save processed intents
            with open(artifacts['intents'], 'w', encoding='utf-8') as f:
                json.dump(intents, f, indent=2, ensure_ascii=False)
            
            # Save training metadata
            metadata = {
                'training_date': datetime.datetime.now().isoformat(),
                'config': self.config,
                'statistics': {
                    'num_intents': len(tag_to_idx),
                    'vocab_size': len(self.tokenizer.word_index) + 1,
                    'total_samples': len(self.tokenizer.texts_to_sequences(intents['intents'][0]['patterns']))
                }
            }
            with open(artifacts['metadata'], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Artifacts saved to:\n{json.dumps(artifacts, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to save artifacts: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        trainer = ChatbotTrainer()
        results = trainer.train()
        
        print("\n=== Training Results ===")
        print(f"Validation Accuracy: {results['accuracy']:.2%}")
        print(f"Training Accuracy: {results['training_accuracy']:.2%}")
        print(f"Epochs Trained: {results['epochs']}")
        print(f"Intents Learned: {results['num_intents']}")
        print(f"Vocabulary Size: {results['vocab_size']}")
        
    except Exception as e:
        print(f"\n!!! Training Failed !!!\nError: {str(e)}")
        exit(1)