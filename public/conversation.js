// conversation.js
// Refactored ConversationMode class with state machine and improved UI management

class ConversationMode {
    constructor() {
        console.log('[ConversationMode] Initializing...');
        this.state = 'idle'; // idle, listening, processing, speaking
        this.ws = null;
        this.recognition = null;
        this.userWaveformCtx = null;
        this.aiWaveformCtx = null;
        this.animationFrameId = null;
        this.ttsBuffer = '';
        this.ttsTimer = null;
        this.ttsFlushMs = 300;
        this.ttsVoice = null;
        this.showAIText = false;
        this.conversationId = null;
        this.setupSection = document.getElementById('conversation-setup');
        this.mainSection = document.getElementById('conversation-main');
        this.setupStatusEl = document.getElementById('conversation-setup-status');
        this.startButton = document.getElementById('start-conversation-btn');
        this.classSelect = document.getElementById('conversation-class-select');
        this.subjectSelect = document.getElementById('conversation-subject-select');
        this.isLaunching = false;

        this.icons = {
            mic: `<svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="white" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>`,
            stop: `<svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="white" viewBox="0 0 24 24" stroke="white" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>`,
            spinner: `<svg class="animate-spin h-10 w-10 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`,
            speaker: `<svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="white" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.858 15.858a5 5 0 01-2.828-7.072m9.9 9.9a9 9 0 01-12.728 0" /></svg>`,
        };

        this.initializeSpeechRecognition();
        this.bindEventHandlers();
        this.initializeSetupForm();
    }

    setState(newState) {
        if (this.state === newState) return;
        console.log(`[ConversationMode] State change: ${this.state} -> ${newState}`);
        this.state = newState;
        this.updateUI();
    }

    updateUI() {
        const micButton = document.getElementById('mic-button');
        const statusIndicator = document.getElementById('status-indicator');
        const userWaveform = document.getElementById('user-waveform');
        const aiWaveform = document.getElementById('ai-waveform');

        if (!micButton || !statusIndicator || !userWaveform || !aiWaveform) return;

        // Reset classes
        micButton.className = 'mic-button';
        userWaveform.style.opacity = 0;
        aiWaveform.style.opacity = 0;
        this.stopWaveformAnimation();

        switch (this.state) {
            case 'idle':
                micButton.classList.add('idle');
                micButton.innerHTML = this.icons.mic;
                micButton.disabled = false;
                statusIndicator.textContent = 'Click the mic to speak';
                break;
            case 'listening':
                micButton.classList.add('listening');
                micButton.innerHTML = this.icons.stop;
                micButton.disabled = false;
                statusIndicator.textContent = 'Listening...';
                userWaveform.style.opacity = 1;
                this.animateWaveform(this.userWaveformCtx, '#4a90e2');
                break;
            case 'processing':
                micButton.classList.add('processing');
                micButton.innerHTML = this.icons.spinner;
                micButton.disabled = true;
                statusIndicator.textContent = 'AI is thinking...';
                break;
            case 'speaking':
                micButton.classList.add('speaking');
                micButton.innerHTML = this.icons.mic; // Show mic to allow barge-in
                micButton.disabled = false; // Allow barge-in
                statusIndicator.textContent = 'AI is speaking...';
                aiWaveform.style.opacity = 1;
                this.animateWaveform(this.aiWaveformCtx, '#50c878');
                break;
        }
    }

    toggleMic() {
        if (this.state === 'listening') {
            this.stopListening();
        } else if (this.state === 'idle') {
            this.startListening();
        } else if (this.state === 'speaking') {
            console.log('[ConversationMode] Interrupting AI.');
            if (window.speechSynthesis) window.speechSynthesis.cancel();
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'interrupt' }));
            }

            // Explicitly stop any ongoing recognition before starting a new one.
            // This helps prevent race conditions.
            try {
                this.recognition.stop();
            } catch (e) {
                // Ignore errors if recognition is not running
            }

            // A small delay to ensure everything has stopped before starting again.
            setTimeout(() => {
                this.startListening();
            }, 250); // Increased delay
        }
    }

    initializeSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.onresult = (event) => {
                let final_transcript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        final_transcript += event.results[i][0].transcript;
                    }
                }
                if (final_transcript) {
                    this.handleTranscript(final_transcript);
                }
            };
            this.recognition.onend = () => {
                if (this.state === 'listening') {
                    this.stopListening();
                }
            };
            this.recognition.onerror = (event) => {
                console.error('Speech recognition error', event.error);
                this.showError('Speech recognition error: ' + event.error);
                this.setState('idle');
            };
        } else {
            console.warn("Speech Recognition not supported");
        }
    }

    startListening() {
        if (!this.recognition || this.state === 'listening') return;
        try {
            this.recognition.start();
            this.setState('listening');
        } catch (e) {
            console.warn('[Recording] startListening failed:', e);
            this.showError('Microphone unavailable');
            this.setState('idle');
        }
    }

    stopListening() {
        if (!this.recognition || this.state !== 'listening') return;
        this.recognition.stop();
        this.setState('processing');
    }

    bindEventHandlers() {
        const micButton = document.getElementById('mic-button');
        if (micButton) {
            micButton.addEventListener('click', () => this.toggleMic());
        }

        const exitButton = document.getElementById('exit-conversation');
        if (exitButton) {
            exitButton.addEventListener('click', () => this.exitConversationMode());
        }
        
        const showTextToggle = document.getElementById('show-ai-text-toggle');
        const conversationBody = document.getElementById('conversation-body');
        if (conversationBody) {
            conversationBody.style.display = this.showAIText ? 'flex' : 'none';
        }
        if (showTextToggle) {
            showTextToggle.checked = !!this.showAIText;
            showTextToggle.addEventListener('change', (e) => {
                this.showAIText = e.target.checked;
                if (conversationBody) {
                    conversationBody.style.display = this.showAIText ? 'flex' : 'none';
                }
            });
        }
    }
    
    async startConversationMode(bookUuid) {
        console.log('[ConversationMode] Starting for book:', bookUuid);
        const modal = document.getElementById('conversation-modal');
        if (!modal) return;

        this.userWaveformCtx = document.getElementById('user-waveform').getContext('2d');
        this.aiWaveformCtx = document.getElementById('ai-waveform').getContext('2d');
        
        const conversationBody = document.getElementById('conversation-body');
        if (conversationBody) {
            conversationBody.innerHTML = '';
            conversationBody.style.display = this.showAIText ? 'flex' : 'none';
        }

        if (this.setupSection) {
            this.setupSection.style.display = 'none';
        }
        if (this.mainSection) {
            this.mainSection.style.display = 'flex';
        }
        modal.style.display = 'flex';
        this.setState('idle');

        this.conversationId = 'conv_' + Date.now();
        const wsUrl = `ws://${window.location.host}/ws/conversation/${this.conversationId}?book_uuid=${bookUuid}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('[WebSocket] Connected');
            this.startListening();
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('[WebSocket] Message parse error:', error);
            }
        };

        this.ws.onclose = () => {
            console.log('[WebSocket] Closed');
            this.exitConversationMode();
        };
    }

    handleTranscript(transcript) {
        if (!transcript.trim()) {
            this.setState('idle');
            return;
        };
        console.log('[Transcript] Sending:', transcript);
        this.setState('processing');
        if (this.showAIText) this.addMessage('user', transcript);
        this.ws.send(JSON.stringify({ type: 'query', query: transcript }));
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'chunk':
                if (this.state !== 'speaking') {
                    this.setState('speaking');
                }
                const cleanChunk = this.normalizeChunkText(data.content);
                this.speakChunk(cleanChunk);
                if (this.showAIText) this.addOrUpdateAIMessage(data.content);
                break;
            case 'done':
                this.flushTTSBuffer();
                break;
            case 'error':
                this.showError(data.message);
                this.setState('idle');
                break;
        }
    }

    speakChunk(text) {
        if (!text || !text.trim()) return;
        this.ttsBuffer += (this.ttsBuffer ? ' ' : '') + text;
        if (this.ttsTimer) clearTimeout(this.ttsTimer);
        this.ttsTimer = setTimeout(() => this.flushTTSBuffer(), this.ttsFlushMs);
    }

    flushTTSBuffer() {
        if (!this.ttsBuffer || !this.ttsBuffer.trim() || !window.speechSynthesis) {
            if (this.state === 'speaking') this.setState('idle');
            return;
        }
        const textToSpeak = this.ttsBuffer.trim();
        this.ttsBuffer = '';
        if (this.ttsTimer) {
            clearTimeout(this.ttsTimer);
            this.ttsTimer = null;
        }

        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        if (this.ttsVoice) utterance.voice = this.ttsVoice;
        utterance.onend = () => {
            if (this.state === 'speaking') {
                this.setState('idle');
            }
        };
        speechSynthesis.speak(utterance);
    }

    animateWaveform(ctx, color) {
        if (!ctx) return;
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        let shouldContinue = true;

        const drawFrame = () => {
            if (!shouldContinue) return;
            
            const time = Date.now() / 200;
            ctx.clearRect(0, 0, width, height);
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            
            for (let x = 0; x < width; x++) {
                const y = height/2 + 
                        Math.sin(x/50 + time) * 15 * Math.sin(time/3) + 
                        Math.sin(x/30 + time*2) * 10 * Math.cos(time/2) +
                        Math.sin(x/20 + time*1.5) * 5 * Math.sin(time/5);
                
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            
            ctx.stroke();
            this.animationFrameId = requestAnimationFrame(drawFrame);
        };
        
        this.stopWaveformAnimation = () => {
            shouldContinue = false;
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }
            if (ctx && ctx.canvas) {
                ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            }
        }

        drawFrame();
    }
    
    stopWaveformAnimation() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
         try {
            if (this.userWaveformCtx && this.userWaveformCtx.canvas) {
                this.userWaveformCtx.clearRect(0, 0, this.userWaveformCtx.canvas.width, this.userWaveformCtx.canvas.height);
            }
            if (this.aiWaveformCtx && this.aiWaveformCtx.canvas) {
                this.aiWaveformCtx.clearRect(0, 0, this.aiWaveformCtx.canvas.width, this.aiWaveformCtx.canvas.height);
            }
        } catch (e) {
            console.warn('[Waveform] Failed to clear canvases:', e);
        }
    }
    
    addMessage(type, content) {
        const conversationBody = document.getElementById('conversation-body');
        if (!conversationBody) return;
        if (type === 'ai' && !this.showAIText) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + type + '-message';
        
        if (type === 'ai') {
            messageDiv.id = 'current-ai-message';
        }
        messageDiv.textContent = content;

        conversationBody.appendChild(messageDiv);
        conversationBody.scrollTop = conversationBody.scrollHeight;
    }
    
    addOrUpdateAIMessage(content) {
        const conversationBody = document.getElementById('conversation-body');
        if (!conversationBody) return;

        let currentMessage = document.getElementById('current-ai-message');
        if (currentMessage) {
            currentMessage.textContent += content;
        } else {
            this.addMessage('ai', content);
        }
        
        conversationBody.scrollTop = conversationBody.scrollHeight;
    }
    
    showError(message) {
        const statusIndicator = document.getElementById('status-indicator');
        if(statusIndicator) {
            statusIndicator.textContent = 'Error: ' + message;
            statusIndicator.style.color = '#ff6b6b';
            
            setTimeout(() => {
                this.setState('idle');
            }, 3000);
        }
    }
    
    exitConversationMode() {
        console.log('[ConversationMode] Exiting conversation mode');
        if (this.recognition && this.state === 'listening') {
            this.recognition.stop();
        }
        
        this.stopWaveformAnimation();
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        const modal = document.getElementById('conversation-modal');
        if (modal) {
            modal.style.display = 'none';
        }
        
        if (this.mainSection) {
            this.mainSection.style.display = 'none';
        }
        this.prepareSetupView();

        this.setState('idle');
    }

    normalizeChunkText(chunk) {
        try {
            return chunk
                .replace(/\*\*/g, '')
                .replace(/\*/g, '')
                .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
                .replace(/#{1,6}\s/g, '')
                .replace(/\|/g, ', ')
                .replace(/---/g, '')
                .replace(/\s+/g, ' ')
                .trim();
        } catch (e) {
            console.warn('[Response] Text normalization failed:', e);
            return chunk;
        }
    }

    initializeSetupForm() {
        if (!this.setupSection) return;
        this.populateClassOptions();
        this.prepareSetupView();

        if (this.classSelect) {
            this.classSelect.addEventListener('change', () => this.handleClassChange());
        }

        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.handleConversationLaunch());
        }
    }

    populateClassOptions() {
        if (!this.classSelect) return;
        const classes = ['6', '7', '8', '9', '10'];
        const previousValue = this.classSelect.value;
        this.classSelect.innerHTML = '<option value="" disabled selected>Select class...</option>';
        classes.forEach(cls => {
            const option = document.createElement('option');
            option.value = cls;
            option.textContent = cls;
            this.classSelect.appendChild(option);
        });
        if (classes.includes(previousValue)) {
            this.classSelect.value = previousValue;
        }
    }

    prepareSetupView() {
        if (!this.setupSection) return;
        if (this.setupSection) {
            this.setupSection.style.display = 'grid';
        }
        if (this.startButton) {
            this.startButton.disabled = false;
            this.startButton.textContent = 'Start Conversation';
        }
        this.clearSetupStatus();
        if (this.classSelect) {
            this.classSelect.selectedIndex = 0;
        }
        if (this.subjectSelect) {
            this.subjectSelect.disabled = true;
            this.subjectSelect.innerHTML = '<option value="" disabled selected>Select subject...</option>';
        }
        if (this.mainSection) {
            this.mainSection.style.display = 'none';
        }
        const conversationBody = document.getElementById('conversation-body');
        if (conversationBody) {
            conversationBody.style.display = 'none';
            conversationBody.innerHTML = '';
        }
    }

    clearSetupStatus() {
        this.setSetupStatus('');
    }

    setSetupStatus(message, type = 'info') {
        if (!this.setupStatusEl) return;
        this.setupStatusEl.textContent = message || '';
        this.setupStatusEl.classList.remove('error', 'success');
        if (!message) return;
        if (type === 'error') {
            this.setupStatusEl.classList.add('error');
        } else if (type === 'success') {
            this.setupStatusEl.classList.add('success');
        }
    }

    async handleClassChange() {
        if (!this.classSelect || !this.subjectSelect) return;
        const className = this.classSelect.value;
        this.clearSetupStatus();
        if (!className) {
            this.subjectSelect.disabled = true;
            this.subjectSelect.innerHTML = '<option value="" disabled selected>Select subject...</option>';
            return;
        }

        this.subjectSelect.disabled = true;
        this.subjectSelect.innerHTML = '<option value="" disabled selected>Loading subjects...</option>';

        try {
            const subjects = await this.fetchSubjects(className);
            if (!subjects.length) {
                this.subjectSelect.innerHTML = '<option value="" disabled selected>No books found</option>';
                this.setSetupStatus('No books found for the selected class.', 'error');
                return;
            }

            this.subjectSelect.innerHTML = '<option value="" disabled selected>Select subject...</option>';
            subjects.forEach(subject => {
                const option = document.createElement('option');
                option.value = subject;
                option.textContent = subject;
                this.subjectSelect.appendChild(option);
            });
            this.subjectSelect.disabled = false;
        } catch (error) {
            console.error('[ConversationMode] Failed to load subjects:', error);
            this.subjectSelect.innerHTML = '<option value="" disabled selected>Unable to load subjects</option>';
            this.setSetupStatus(error.message || 'Failed to load subjects. Please try again.', 'error');
        }
    }

    async fetchSubjects(className) {
        const response = await fetch(`/api/books?class_name=${encodeURIComponent(className)}`);
        if (!response.ok) {
            throw new Error('Failed to load subjects. Please try again.');
        }
        const books = await response.json();
        const subjects = [...new Set((books || []).map(book => book.subject).filter(Boolean))];
        return subjects.sort((a, b) => a.localeCompare(b));
    }

    async handleConversationLaunch() {
        if (this.isLaunching || !this.startButton) return;

        const className = this.classSelect ? this.classSelect.value : '';
        const subject = this.subjectSelect ? this.subjectSelect.value : '';

        if (!className) {
            this.setSetupStatus('Please select a class to continue.', 'error');
            return;
        }
        if (!subject) {
            this.setSetupStatus('Please select a subject to continue.', 'error');
            return;
        }

        this.clearSetupStatus();
        this.isLaunching = true;
        this.startButton.disabled = true;
        this.startButton.textContent = 'Connecting...';

        try {
            const book = await this.fetchBook(className, subject);
            window.selectedBook = book;
            this.startConversationMode(book.id);
        } catch (error) {
            console.error('[ConversationMode] Conversation launch failed:', error);
            this.setSetupStatus(error.message || 'Unable to start conversation. Please try again.', 'error');
        } finally {
            if (this.setupSection && this.setupSection.style.display !== 'none') {
                this.startButton.disabled = false;
                this.startButton.textContent = 'Start Conversation';
            }
            this.isLaunching = false;
        }
    }

    async fetchBook(className, subject) {
        const params = new URLSearchParams({
            class_name: className,
            subject: subject
        });
        const response = await fetch(`/api/books?${params.toString()}`);
        if (!response.ok) {
            throw new Error('Unable to load the selected book. Please try again.');
        }
        const books = await response.json();
        if (!books || !books.length) {
            throw new Error('No processed book available for the chosen class and subject.');
        }
        return books[0];
    }

    openSetupModal() {
        const modal = document.getElementById('conversation-modal');
        if (!modal || !this.setupSection) return;
        this.populateClassOptions();
        this.prepareSetupView();
        modal.style.display = 'flex';
        this.setState('idle');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('conversation-modal');
    
    if (modal) {
        window.conversationMode = new ConversationMode();
        const button = document.getElementById('conversational-mode-btn');
        if (button) {
            button.addEventListener('click', () => {
                window.conversationMode.openSetupModal();
            });
        }
    }
});