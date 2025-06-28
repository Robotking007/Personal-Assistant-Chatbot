import os
import json
from flask import Flask, render_template, request, jsonify, session
from chatbot_model.train import ChatbotTrainer
from chatbot_model.model import Chatbot

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'frontend', 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'frontend', 'static')

app = Flask(__name__,
           template_folder=TEMPLATE_DIR,
           static_folder=STATIC_DIR)
app.secret_key = 'your-secret-key-here'  # Required for session

# Initialize chatbot
chatbot = Chatbot()

@app.route('/')
def home():
    """Render main chat interface"""
    try:
        template_path = os.path.join(TEMPLATE_DIR, 'index.html')
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found at: {template_path}")
        
        # Clear session on new chat
        session.clear()
        return render_template('index.html')
    except Exception as e:
        return f"Error loading page: {str(e)}", 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with context"""
    try:
        message = request.json.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get context from session
        context = session.get('chat_context', {})
        
        # Get response from chatbot
        response = chatbot.generate_response(message, context)
        
        # Update context in session
        session['chat_context'] = {
            'last_message': message,
            'last_response': response,
            'last_intent': chatbot.context.get('last_intent')
        }
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Handle model training requests"""
    try:
        data = request.json
        required_fields = ['tag', 'patterns', 'responses']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        intents_path = os.path.join(BASE_DIR, 'backend', 'chatbot_model', 'data', 'intents.json')
        with open(intents_path, 'r') as f:
            intents = json.load(f)
        
        # Add new intent
        intents['intents'].append({
            'tag': data['tag'],
            'patterns': data['patterns'],
            'responses': data['responses'],
            'context': ['']
        })
        
        with open(intents_path, 'w') as f:
            json.dump(intents, f, indent=2)
        
        trainer = ChatbotTrainer()
        history = trainer.train()
        
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully',
            'accuracy': history.history['accuracy'][-1]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return app.send_static_file(filename)

if __name__ == '__main__':
    # Print debug information
    print(f"Project root: {BASE_DIR}")
    print(f"Template directory: {TEMPLATE_DIR}")
    print(f"Static directory: {STATIC_DIR}")
    
    # Verify important paths
    required_paths = [
        TEMPLATE_DIR,
        STATIC_DIR,
        os.path.join(TEMPLATE_DIR, 'index.html'),
        os.path.join(BASE_DIR, 'backend', 'chatbot_model', 'data', 'intents.json')
    ]
    
    for path in required_paths:
        exists = "EXISTS" if os.path.exists(path) else "MISSING"
        print(f"{exists}: {path}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)