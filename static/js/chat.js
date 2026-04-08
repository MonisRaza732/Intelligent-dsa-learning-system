/**
 * AI Study Assistant — Frontend Logic
 * Handles chat widget toggling, message rendering, and API communication.
 */

document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('ai-chat-toggle');
    const closeBtn = document.getElementById('ai-chat-close');
    const chatWindow = document.getElementById('ai-chat-window');
    const messagesContainer = document.getElementById('ai-chat-messages');
    const chatInput = document.getElementById('ai-chat-input');
    const sendBtn = document.getElementById('ai-chat-send');

    let isChatOpen = false;
    let isWaitingForResponse = false;

    // --- Markdown formatting helper ---
    function formatMarkdown(text) {
        // Simple markdown parser for bold and code blocks
        let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        formatted = formatted.replace(/`(.*?)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 4px;border-radius:4px;">$1</code>');
        return formatted.replace(/\n/g, '<br>');
    }

    // --- UI Interactions ---
    function toggleChat() {
        isChatOpen = !isChatOpen;
        if (isChatOpen) {
            chatWindow.classList.add('active');
            chatInput.focus();
            
            // Enable input if we have user data loaded
            if (window.currentData) {
                chatInput.disabled = false;
                sendBtn.disabled = false;
            }
        } else {
            chatWindow.classList.remove('active');
        }
    }

    toggleBtn.addEventListener('click', toggleChat);
    closeBtn.addEventListener('click', toggleChat);

    // Watch for global data loaded (from app.js) to enable chat
    const checkDataInterval = setInterval(() => {
        if (window.currentData && chatInput.disabled) {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.placeholder = "Ask about your stats, weak topics, or interview tips...";
            appendMessage("bot", `I see you're analyzing **${window.currentData.user_data.username}**. How can I help you improve?`);
            clearInterval(checkDataInterval);
        }
    }, 1000);

    // --- Messaging Logic ---
    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${sender}`;
        msgDiv.innerHTML = formatMarkdown(text);
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-msg bot typing-indicator-container';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function hideTyping() {
        const typingDiv = document.getElementById('typing-indicator');
        if (typingDiv) typingDiv.remove();
    }

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message || isWaitingForResponse || !window.currentData) return;

        // Add user message to UI
        appendMessage('user', message);
        chatInput.value = '';
        
        isWaitingForResponse = true;
        chatInput.disabled = true;
        sendBtn.disabled = true;
        showTyping();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    context_data: window.currentData
                })
            });

            const data = await response.json();
            hideTyping();

            if (data.success) {
                appendMessage('bot', data.reply);
            } else {
                appendMessage('bot', `⚠️ Error: ${data.error}`);
            }
        } catch (error) {
            hideTyping();
            appendMessage('bot', "⚠️ Connection error. Please try again.");
            console.error('Chat error:', error);
        } finally {
            isWaitingForResponse = false;
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    // --- Event Listeners ---
    sendBtn.addEventListener('click', sendMessage);
    
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
});

// Expose currentData from app.js globally so chat.js can read it
// (Need to patch app.js to attach currentData to window)
