document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const trainBtn = document.getElementById('train-btn');
    const trainModal = document.getElementById('training-section');
    const closeBtn = document.querySelector('.close-btn');
    const submitTraining = document.getElementById('submit-training');
    const userQuery = document.getElementById('user-query');
    const botResponse = document.getElementById('bot-response');
    const trainingStatus = document.getElementById('training-status');

    // State variables
    let isWaitingForResponse = false;
    let conversationHistory = [];

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendMessage();
    });

    trainBtn.addEventListener('click', function() {
        trainModal.style.display = 'flex';
        userQuery.focus();
    });

    closeBtn.addEventListener('click', closeTrainingModal);
    window.addEventListener('click', function(event) {
        if (event.target === trainModal) {
            closeTrainingModal();
        }
    });

    submitTraining.addEventListener('click', trainModel);

    // Functions
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '' || isWaitingForResponse) return;

        addMessage(message, 'user-message');
        userInput.value = '';
        isWaitingForResponse = true;
        showTypingIndicator();

        // Add to conversation history
        conversationHistory.push({
            type: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });

        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message,
                history: conversationHistory.slice(-5) // Send last 5 messages as context
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            addMessage(data.response, 'bot-message');
            
            // Add bot response to history
            conversationHistory.push({
                type: 'bot',
                content: data.response,
                timestamp: new Date().toISOString()
            });
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage("Sorry, I encountered an error. Please try again.", 'bot-message');
        })
        .finally(() => {
            isWaitingForResponse = false;
            removeTypingIndicator();
        });
    }

    function addMessage(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', className);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'bot-message');
        typingDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('typing-indicator');
        contentDiv.innerHTML = '<span></span><span></span><span></span>';
        
        typingDiv.appendChild(contentDiv);
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async function trainModel() {
        const query = userQuery.value.trim();
        const response = botResponse.value.trim();
        
        if (!query || !response) {
            showTrainingStatus('Please fill in both fields', 'error');
            return;
        }

        showTrainingStatus('Starting training process...', 'info');
        submitTraining.disabled = true;
        
        try {
            const result = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    tag: `user_${Date.now()}`,
                    patterns: [query],
                    responses: [response]
                })
            });

            const data = await result.json();
            
            if (data.error) {
                handleTrainingError(data.error);
            } else {
                showTrainingStatus(
                    `Training successful! Final accuracy: ${(data.accuracy * 100).toFixed(1)}%`, 
                    'success'
                );
                // Add sample of the trained response
                setTimeout(() => {
                    addMessage(`I just learned how to respond to: "${query}"`, 'bot-message');
                    closeTrainingModal();
                }, 1500);
            }
        } catch (error) {
            handleTrainingError(error.message);
        } finally {
            submitTraining.disabled = false;
        }
    }

    function handleTrainingError(error) {
        let userMessage = error;
        
        // Friendly error messages
        if (error.includes('least populated class')) {
            userMessage = 'Please provide at least 2 different examples for each type of question';
        } else if (error.includes('insufficient training data')) {
            userMessage = 'More training examples needed (minimum 2 per intent)';
        } else if (error.includes('Failed to fetch')) {
            userMessage = 'Connection problem. Please check your network';
        }

        showTrainingStatus(`Error: ${userMessage}`, 'error');
    }

    function showTrainingStatus(message, type) {
        trainingStatus.textContent = message;
        trainingStatus.className = `status-${type}`;
    }

    function closeTrainingModal() {
        trainModal.style.display = 'none';
        userQuery.value = '';
        botResponse.value = '';
        trainingStatus.textContent = '';
        trainingStatus.className = '';
    }

    // Initial greeting
    setTimeout(() => {
        addMessage("Hello! I'm your personal assistant. How can I help you today?", 'bot-message');
        conversationHistory.push({
            type: 'bot',
            content: "Hello! I'm your personal assistant. How can I help you today?",
            timestamp: new Date().toISOString()
        });
    }, 800);
});