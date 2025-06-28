import json
from chatbot_model.train import ChatbotTrainer

def add_intent(tag, patterns, responses):
    intents_path = 'backend/chatbot_model/data/intents.json'
    
    with open(intents_path, 'r') as f:
        data = json.load(f)
    
    data['intents'].append({
        "tag": tag,
        "patterns": patterns,
        "responses": responses
    })
    
    with open(intents_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added new intent: {tag}")
    print("Starting training...")
    
    trainer = ChatbotTrainer()
    history = trainer.train()
    
    print(f"Training complete. Final accuracy: {history.history['accuracy'][-1]:.2%}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True, help='Intent tag')
    parser.add_argument('--patterns', required=True, nargs='+', help='User phrases')
    parser.add_argument('--responses', required=True, nargs='+', help='Bot responses')
    
    args = parser.parse_args()
    
    add_intent(
        tag=args.tag,
        patterns=args.patterns,
        responses=args.responses
    )