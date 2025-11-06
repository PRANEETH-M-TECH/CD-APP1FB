// Conversation Mode Functionality
class ConversationMode {
    constructor() {
        console.log('[ConversationMode] Initializing...');
        this.ws = null;
        this.isRecording = false;
        this.recognition = null;
        this.animationFrameId = null;
        this.currentSpeaker = null; // 'user' or 'ai'
        this.shouldStop = false;
        this.lastTranscript = '';

        // Processing flags
        this.isProcessing = false; // true while waiting for LLM to start streaming
        this.receivedFirstChunk = false;

    // Buffer to assemble the full response (when user requests whole-answer speaking)
    this.responseTextBuffer = '';
    // Progress tracking
    this.progressFill = null; // will be set to DOM element

        // TTS buffering and voice
        this.ttsBuffer = '';
        this.ttsTimer = null;
        this.ttsFlushMs = 300; // buffer flush window in ms
        this.ttsVoice = null;
        this.voicesReady = false;

        // Debug
        this.showDebug = false;
        this.debugPanel = null;

        // Progress UI
        this.progressInterval = null;
        this.progressFill = null;

        // Control whether AI text is shown on-screen. Default: false -> audio-only with waveform
        this.showAIText = false;
        // Track if recognition was active before TTS so we can restart it
        this.wasRecordingBeforeTTS = false;
        
        // Initialize Web Speech API
        this.initializeSpeechRecognition();
        
        // Initialize as null, will be set when conversation starts
        this.userWaveformCtx = null;
        this.aiWaveformCtx = null;
        
        // Bind event handlers
        this.bindEventHandlers();
    }
    
    initializeSpeechRecognition() {
        console.log('[Speech] Initializing speech recognition...');
        this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    this.recognition.interimResults = true;
    // Use non-continuous recognition: start/stop controlled by mic button
    this.recognition.continuous = false;
        this.recognition.lang = 'en-US';
        
        // Handle interim results to show user feedback
        this.recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            
            const statusIndicator = document.getElementById('status-indicator');
            
            // Show interim results
            if (!event.results[0].isFinal) {
                console.log('[Speech] Interim transcript:', transcript);
                statusIndicator.textContent = '🎤 Hearing: ' + transcript;
                statusIndicator.style.color = '#2196F3';
                return;
            }
            
            // Handle final result
            console.log('[Speech] Final transcript:', transcript);
            statusIndicator.textContent = '🎤 Processing: ' + transcript;
            statusIndicator.style.color = '#4CAF50';
            this.handleTranscript(transcript);
        };
        
        this.recognition.onstart = () => {
            console.log('[Speech] Recognition started');
        };

        // When recognition ends we simply mark isRecording false; we don't auto-restart
        this.recognition.onend = () => {
            console.log('[Speech] Recognition ended');
            this.isRecording = false;
            const micButton = document.getElementById('mic-button');
            if (micButton) micButton.classList.remove('recording');
        };
        
        this.recognition.onerror = (event) => {
            console.error('[Speech] Recognition error:', event.error);
            const statusIndicator = document.getElementById('status-indicator');
            statusIndicator.textContent = '❌ Error: ' + event.error;
            statusIndicator.style.color = '#f44336';
        };
    }

    // Start recognition and set UI to recording state
    startListening() {
        if (this.isRecording) return;
        const micButton = document.getElementById('mic-button');
        const statusIndicator = document.getElementById('status-indicator');

        try {
            this.recognition.start();
            this.isRecording = true;
            if (micButton) micButton.classList.add('recording');
            if (micButton) micButton.style.transform = 'scale(0.96)';
            if (micButton) micButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="white">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            `;
            if (statusIndicator) {
                statusIndicator.textContent = '🎤 Listening...';
                statusIndicator.style.color = '#10B981';
            }
            this.currentSpeaker = 'user';
            if (this.userWaveformCtx) this.animateWaveform(this.userWaveformCtx, '#4a90e2');
        } catch (e) {
            console.warn('[Recording] startListening failed:', e);
            if (statusIndicator) {
                statusIndicator.textContent = '❌ Microphone unavailable';
                statusIndicator.style.color = '#f44336';
            }
        }
    }

    // Stop recognition and set UI to non-recording state
    stopListening() {
        if (!this.isRecording) return;
        const micButton = document.getElementById('mic-button');
        const statusIndicator = document.getElementById('status-indicator');

        try { this.recognition.stop(); } catch (e) { /* ignore */ }
        this.isRecording = false;
        if (micButton) micButton.classList.remove('recording');
        if (micButton) micButton.style.transform = '';
        if (micButton) micButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="white">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 1v11m0 0a3 3 0 01-3 3H7a3 3 0 01-3-3V6a3 3 0 013-3h2a3 3 0 013 3z" />
            </svg>
        `;
        if (statusIndicator) {
            statusIndicator.textContent = 'Processing your question...';
            statusIndicator.style.color = '#6B7280';
        }
        this.stopWaveformAnimation();
    }
    
    bindEventHandlers() {
        // Mic button
        const micButton = document.getElementById('mic-button');
        if (micButton) {
            // Set initial mic button state
            micButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
            `;
            
            micButton.addEventListener('click', () => {
                this.toggleRecording();
            });
        }
        
        // Exit button
        const exitButton = document.getElementById('exit-conversation');
        if (exitButton) {
            exitButton.addEventListener('click', () => {
                this.exitConversationMode();
            });
        }
        
        // Show AI text toggle
        const showTextToggle = document.getElementById('show-ai-text-toggle');
        if (showTextToggle) {
            // initialize toggle UI from current state
            showTextToggle.checked = !!this.showAIText;
            showTextToggle.addEventListener('change', (e) => {
                this.showAIText = e.target.checked;
                // If user enabled text mid-conversation, show current AI message if any
                if (this.showAIText) {
                    const currentMessage = document.getElementById('current-ai-message');
                    if (!currentMessage) {
                        // If there's last transcript or last AI chunks, we don't have stored text here,
                        // so just show a small notice
                        const conversationBody = document.getElementById('conversation-body');
                        if (conversationBody) {
                            const notice = document.createElement('div');
                            notice.className = 'message ai-message fade-in';
                            notice.textContent = 'AI text enabled';
                            conversationBody.appendChild(notice);
                        }
                    }
                } else {
                    // If disabling text, remove AI messages
                    const aiMessages = document.querySelectorAll('#conversation-body .ai-message');
                    aiMessages.forEach(m => m.remove());
                }
            });
        }
        // Debug toggle wiring
        const showDebugToggle = document.getElementById('show-debug-toggle');
        if (showDebugToggle) {
            showDebugToggle.checked = !!this.showDebug;
            showDebugToggle.addEventListener('change', (e) => {
                this.showDebug = e.target.checked;
                const panel = document.getElementById('debug-panel');
                if (panel) panel.style.display = this.showDebug ? 'block' : 'none';
            });
        }
        // Voice selector
        const voiceSelect = document.getElementById('voice-select');
        if (voiceSelect) {
            voiceSelect.addEventListener('change', (e) => {
                const name = e.target.value;
                try {
                    const voices = speechSynthesis.getVoices();
                    const pick = voices.find(v => (v.name + '|' + v.lang) === name);
                    if (pick) this.ttsVoice = pick;
                    localStorage.setItem('preferred_tts_voice', name);
                } catch (err) { console.warn('[TTS] voice select change failed', err); }
            });
        }
    }
    
    async startConversationMode(bookUuid) {
        console.log('[ConversationMode] Starting conversation mode for book:', bookUuid);
        
        try {
            // Show conversation modal
            const modal = document.getElementById('conversation-modal');
            if (!modal) {
                throw new Error('Conversation modal not found in DOM');
            }
            
            // Initialize waveform canvases
            const userWaveform = document.getElementById('user-waveform');
            const aiWaveform = document.getElementById('ai-waveform');
            
            if (!userWaveform || !aiWaveform) {
                throw new Error('Waveform canvases not found in DOM');
            }
            
            this.userWaveformCtx = userWaveform.getContext('2d');
            this.aiWaveformCtx = aiWaveform.getContext('2d');
            
            // Clear any previous messages
            const conversationBody = document.getElementById('conversation-body');
            if (conversationBody) {
                conversationBody.innerHTML = '';
                // Add welcome message only when showing AI text
                if (this.showAIText) {
                    this.addMessage('ai', 'Welcome to conversation mode! Click the microphone button when you want to speak.');
                }
            }
            
            // Show the modal
            // Ensure toggle reflects current setting
            const showTextToggle = document.getElementById('show-ai-text-toggle');
            if (showTextToggle) showTextToggle.checked = !!this.showAIText;

            modal.style.display = 'flex';

            // Auto-start listening immediately when modal opens (user already clicked to open)
            try {
                this.startListening();
            } catch (e) {
                console.warn('[ConversationMode] Auto start listening failed:', e);
            }

            // wire progress bar element
            try {
                const pb = document.getElementById('response-progress');
                if (pb) {
                    this.progressFill = pb.querySelector('.fill');
                    if (this.progressFill) this.progressFill.style.width = '0%';
                    pb.style.display = 'none';
                }
            } catch (e) { }
            
            // Reset status indicator
            const statusIndicator = document.getElementById('status-indicator');
            if (statusIndicator) {
                statusIndicator.textContent = '🎤 Click microphone to start speaking';
                statusIndicator.style.color = '#666';
            }
            
            // Generate a unique conversation ID
            const conversationId = 'conv_' + Date.now();
            console.log('[ConversationMode] Generated conversation ID:', conversationId);
            
            // Initialize WebSocket connection
            const wsUrl = 'ws://' + window.location.host + '/ws/conversation/' + conversationId + '?book_uuid=' + bookUuid;
            console.log('[ConversationMode] Connecting to WebSocket:', wsUrl);
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('[WebSocket] Received message type:', data.type);
                    // append raw debug if enabled
                    if (this.showDebug) {
                        try {
                            const panel = document.getElementById('debug-panel');
                            if (panel) {
                                const t = new Date().toISOString();
                                const pre = document.createElement('div');
                                pre.textContent = `[${t}] RX: ${event.data}`;
                                panel.appendChild(pre);
                                panel.scrollTop = panel.scrollHeight;
                            }
                        } catch (e) { /* ignore debug errors */ }
                    }
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('[WebSocket] Failed to parse message:', error);
                    this.showError('Failed to process server response');
                }
            };
            
            this.ws.onopen = () => {
                console.log('[WebSocket] Connection established');
                // Ensure TTS voice selection is ready
                try { this.selectPreferredVoice(); } catch (e) { /* ignore */ }
                // Don't auto-start recording, wait for user to click the mic button
                console.log('[ConversationMode] Ready for user interaction');
            };
        } catch (error) {
            console.error('[ConversationMode] Failed to start:', error);
            this.showError('Failed to start conversation mode');
            this.exitConversationMode();
        }
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Connection error. Please try again.');
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket connection closed');
            this.exitConversationMode();
        };
    }
    
    toggleRecording() {
        console.log('[Recording] Toggle recording, current state:', this.isRecording);
        const micButton = document.getElementById('mic-button');
        const statusIndicator = document.getElementById('status-indicator');
        
        // If AI is speaking, clicking mic should interrupt and immediately start recording
        if (!this.isRecording) {
            // If AI is speaking, interrupt first and then start listening
            if ((this.currentSpeaker === 'ai' || (window.speechSynthesis && speechSynthesis.speaking)) && this.ws && this.ws.readyState === WebSocket.OPEN) {
                try {
                    console.log('[Recording] Interrupting AI response and starting recording');
                    this.ws.send(JSON.stringify({ type: 'interrupt' }));
                } catch (e) {
                    console.warn('[Recording] Failed to send interrupt:', e);
                }
                try { if (window.speechSynthesis) speechSynthesis.cancel(); } catch (e) { /* ignore */ }
                this.shouldStop = true;
                this.stopWaveformAnimation();
            }

            // Start recognition via helper (so we can also auto-start when modal opens)
            this.startListening();
        } else {
            // Stop recognition (user finished speaking). The final transcript will be handled in onresult.
            console.log('[Recording] User stopped recording (manual)');
            this.stopListening();
        }
    }
    
    async handleTranscript(transcript) {
        console.log('[Transcript] Processing transcript:', transcript);
        
        if (!transcript.trim()) {
            console.log('[Transcript] Empty transcript, ignoring');
            return;
        }
        
        if (!this.ws) {
            console.error('[Transcript] WebSocket not connected');
            this.showError('Not connected to server');
            return;
        }
        
        try {
            console.log('[Transcript] Sending query to backend');
            // mark processing state until first chunk arrives
            this.isProcessing = true;
            this.receivedFirstChunk = false;

            // Send query to backend
            this.ws.send(JSON.stringify({ type: 'query', query: transcript }));

            // Add user message to conversation (if text display enabled)
            if (this.showAIText) this.addMessage('user', transcript);

            // Update status to show waiting state
            const statusIndicator = document.getElementById('status-indicator');
            statusIndicator.textContent = '⏳ Waiting for answer...';
            statusIndicator.style.color = '#6B7280';

            // Show progress bar (start small)
            try {
                const pb = document.getElementById('response-progress');
                if (pb) {
                    pb.style.display = 'block';
                    if (!this.progressFill) this.progressFill = pb.querySelector('.fill');
                    if (this.progressFill) this.progressFill.style.width = '5%';
                }
            } catch (e) { }

            // Disable mic visually while waiting to avoid duplicate submissions
            const micButton = document.getElementById('mic-button');
            if (micButton) {
                micButton.disabled = true;
                micButton.style.opacity = '0.6';
            }

            // set current speaker to ai in anticipation
            this.currentSpeaker = 'ai';

            console.log('[Transcript] Successfully sent transcript, waiting for streaming chunks');
        } catch (error) {
            console.error('[Transcript] Error processing transcript:', error);
            this.showError('Failed to send message');
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'chunk':
                this.handleResponseChunk(data.content);
                break;
            case 'interrupt_acknowledged':
                this.handleInterrupt();
                break;
            case 'error':
                this.showError(data.message);
                break;
            case 'done':
                this.handleResponseComplete();
                break;
        }
    }
    
    handleResponseChunk(chunk) {
        if (this.shouldStop) return;
        // On first chunk we consider processing finished and enable mic for interrupts
        if (this.isProcessing && !this.receivedFirstChunk) {
            this.receivedFirstChunk = true;
            this.isProcessing = false;
            // Re-enable mic to allow interruptions while AI speaks
            const micButton = document.getElementById('mic-button');
            if (micButton) {
                micButton.disabled = false;
                micButton.style.opacity = '1';
            }
        }

        // Aggregate chunks into a response buffer instead of speaking immediately
        try {
            this.responseTextBuffer += (this.responseTextBuffer ? ' ' : '') + chunk;
        } catch (e) {
            console.error('[Response] Failed to append chunk to buffer:', e);
        }

        // Optionally show AI text incrementally if requested
        if (this.showAIText) {
            this.addOrUpdateAIMessage(chunk);
        }

        // Also write to hidden aggregated div so speech synthesis (or external tooling) can read it
        try {
            const agg = document.getElementById('aggregated-ai-response');
            if (agg) agg.textContent = this.responseTextBuffer;
        } catch (e) { /* ignore */ }

        // Progress heuristic: increase fill per chunk up to 80%
        try {
            if (this.progressFill) {
                const cur = parseFloat(this.progressFill.style.width) || 0;
                const incr = 6 + Math.random()*4; // 6-10% per chunk
                const next = Math.min(80, cur + incr);
                this.progressFill.style.width = next + '%';
            }
        } catch (e) { }
    }

    speakChunk(text) {
        // Buffer small chunks and flush after a short window to reduce choppy TTS
        if (!text || !text.trim()) return;

        // If speech synthesis is not available, fall back to showing text
        if (!window.speechSynthesis) {
            console.warn('[TTS] speechSynthesis not supported in this browser');
            if (this.showAIText) this.addOrUpdateAIMessage(text);
            return;
        }

        // Append to TTS buffer and schedule a flush
        this.ttsBuffer += (this.ttsBuffer ? ' ' : '') + text;
        if (this.ttsTimer) clearTimeout(this.ttsTimer);
        this.ttsTimer = setTimeout(() => this.flushTTSBuffer(), this.ttsFlushMs);
    }

    // Combine buffered chunks and speak once
    flushTTSBuffer() {
        if (!this.ttsBuffer || !this.ttsBuffer.trim()) return;

        const textToSpeak = this.ttsBuffer.trim();
        this.ttsBuffer = '';
        if (this.ttsTimer) {
            clearTimeout(this.ttsTimer);
            this.ttsTimer = null;
        }

        // Debug: record flush event
        try {
            if (this.showDebug) {
                const panel = document.getElementById('debug-panel');
                if (panel) {
                    const pre = document.createElement('div');
                    pre.textContent = `[${new Date().toISOString()}] TTS flush: ${textToSpeak.slice(0,120)}`;
                    panel.appendChild(pre);
                    panel.scrollTop = panel.scrollHeight;
                }
            }
        } catch (e) { }

        // If recognition is active, stop it temporarily to avoid capturing TTS
        this.wasRecordingBeforeTTS = !!this.isRecording;
        if (this.wasRecordingBeforeTTS && this.recognition) {
            try { this.recognition.stop(); } catch (e) { /* ignore */ }
            this.isRecording = false;
            const micButton = document.getElementById('mic-button');
            if (micButton) micButton.classList.remove('recording');
        }

        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        if (this.ttsVoice) utterance.voice = this.ttsVoice;

        utterance.onstart = () => {
            try { if (this.aiWaveformCtx) this.animateWaveform(this.aiWaveformCtx, '#50c878'); } catch (e) { }
            const statusIndicator = document.getElementById('status-indicator');
            if (statusIndicator) {
                statusIndicator.textContent = '🔊 AI speaking...';
                statusIndicator.style.color = '#9C27B0';
            }
            // robot animation
            const robot = document.querySelector('.robot-anim');
            if (robot) robot.classList.add('speaking');
            // Debug: utterance start
            try { if (this.showDebug) { const panel = document.getElementById('debug-panel'); if (panel) { const pre = document.createElement('div'); pre.textContent = `[${new Date().toISOString()}] TTS start`; panel.appendChild(pre); panel.scrollTop = panel.scrollHeight; } } } catch (e) {}
        };

        utterance.onend = () => {
            try { this.stopWaveformAnimation(); } catch (e) { }
            // Restore recognition if it was active before
            if (this.wasRecordingBeforeTTS && this.recognition) {
                try { this.recognition.start(); this.isRecording = true; const micButton = document.getElementById('mic-button'); if (micButton) micButton.classList.add('recording'); } catch (e) { console.warn('[TTS] Failed to restart recognition after TTS:', e); }
            }
            const statusIndicator = document.getElementById('status-indicator');
            if (statusIndicator) {
                statusIndicator.textContent = '🎤 Click microphone to speak';
                statusIndicator.style.color = '#666';
            }
            const robot = document.querySelector('.robot-anim');
            if (robot) robot.classList.remove('speaking');
            // Debug: utterance end
            try { if (this.showDebug) { const panel = document.getElementById('debug-panel'); if (panel) { const pre = document.createElement('div'); pre.textContent = `[${new Date().toISOString()}] TTS end`; panel.appendChild(pre); panel.scrollTop = panel.scrollHeight; } } } catch (e) {}
        };

        try {
            speechSynthesis.speak(utterance);
        } catch (e) {
            console.error('[TTS] speak failed:', e);
        }
    }

    // Choose a preferred voice from available voices (try en-US)
    selectPreferredVoice() {
        if (!window.speechSynthesis) return;
        const choose = () => {
            const voices = speechSynthesis.getVoices();
            if (!voices || voices.length === 0) return;
            // prefer en-US voices and known good names
            let pick = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('en-us'))
                    || voices.find(v => v.name && /google/i.test(v.name))
                    || voices[0];
            this.ttsVoice = pick;
            this.voicesReady = true;
            console.log('[TTS] Selected voice:', pick && (pick.name + ' ' + pick.lang));
            // Populate voice-select dropdown if present
            try {
                const sel = document.getElementById('voice-select');
                if (sel) {
                    sel.innerHTML = '';
                    const preferred = localStorage.getItem('preferred_tts_voice');
                    voices.forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v.name + '|' + v.lang;
                        opt.textContent = `${v.name} — ${v.lang}`;
                        if (preferred && preferred === opt.value) opt.selected = true;
                        sel.appendChild(opt);
                    });
                    // If preferred stored, set ttsVoice accordingly
                    if (preferred) {
                        const pick2 = voices.find(v => (v.name + '|' + v.lang) === preferred);
                        if (pick2) this.ttsVoice = pick2;
                    } else if (pick) {
                        // select pick in the UI
                        const value = pick.name + '|' + pick.lang;
                        const optionToSelect = Array.from(sel.options).find(o => o.value === value);
                        if (optionToSelect) optionToSelect.selected = true;
                    }
                }
            } catch (e) { console.warn('[TTS] populate voice-select failed', e); }
        };

        // populate immediately if voices available, otherwise bind event
        const voices = speechSynthesis.getVoices();
        if (voices && voices.length > 0) choose();
        else window.speechSynthesis.onvoiceschanged = choose;
    }

    // Fallback: request server-side TTS audio and play it (returns audio blob)
    async fallbackPlayAudio(text) {
        if (!text) return;
        try {
            if (this.showDebug) {
                const panel = document.getElementById('debug-panel');
                if (panel) {
                    const pre = document.createElement('div');
                    pre.textContent = `[${new Date().toISOString()}] Fallback TTS request...`;
                    panel.appendChild(pre);
                    panel.scrollTop = panel.scrollHeight;
                }
            }

            const resp = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!resp.ok) {
                const txt = await resp.text();
                console.warn('[TTS] server tts failed:', resp.status, txt);
                return;
            }

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);

            // Animate while playing
            audio.onplay = () => {
                try { if (this.aiWaveformCtx) this.animateWaveform(this.aiWaveformCtx, '#50c878'); } catch (e) { }
                const robot = document.querySelector('.robot-anim'); if (robot) robot.classList.add('speaking');
                if (this.showDebug) { const panel = document.getElementById('debug-panel'); if (panel) { const pre = document.createElement('div'); pre.textContent = `[${new Date().toISOString()}] Fallback audio play start`; panel.appendChild(pre); panel.scrollTop = panel.scrollHeight; } }
            };
            audio.onended = () => {
                try { this.stopWaveformAnimation(); } catch (e) { }
                const robot = document.querySelector('.robot-anim'); if (robot) robot.classList.remove('speaking');
                if (this.showDebug) { const panel = document.getElementById('debug-panel'); if (panel) { const pre = document.createElement('div'); pre.textContent = `[${new Date().toISOString()}] Fallback audio play end`; panel.appendChild(pre); panel.scrollTop = panel.scrollHeight; } }
                URL.revokeObjectURL(url);
            };

            audio.play().catch(err => console.warn('[TTS] audio.play failed', err));

        } catch (e) {
            console.error('[TTS] fallbackPlayAudio error', e);
        }
    }
    
    handleInterrupt() {
        this.shouldStop = true;
        // Stop any ongoing TTS playback
        try { if (window.speechSynthesis) speechSynthesis.cancel(); } catch (e) { /* ignore */ }
        // Clear any buffered TTS
        try { if (this.ttsTimer) { clearTimeout(this.ttsTimer); this.ttsTimer = null; } this.ttsBuffer = ''; } catch (e) { }
        this.stopWaveformAnimation();
        
        // Show interrupt animation
        const aiWaveform = document.getElementById('ai-waveform');
        aiWaveform.classList.add('interrupt-animation');
        setTimeout(() => {
            aiWaveform.classList.remove('interrupt-animation');
        }, 500);
    }
    
    handleResponseComplete() {
        this.shouldStop = false;
        this.stopWaveformAnimation();
        this.currentSpeaker = null;

        // Reset processing flags and UI
        this.isProcessing = false;
        this.receivedFirstChunk = false;

        // Clear any TTS buffer/timers
        try {
            if (this.ttsTimer) { clearTimeout(this.ttsTimer); this.ttsTimer = null; }
            this.ttsBuffer = '';
        } catch (e) { }

        // Finalize progress bar
        try {
            if (this.progressFill) this.progressFill.style.width = '100%';
            const pb = document.getElementById('response-progress');
            if (pb) setTimeout(() => { pb.style.display = 'none'; if (this.progressFill) this.progressFill.style.width = '0%'; }, 700);
        } catch (e) { }

        const micButton = document.getElementById('mic-button');
        if (micButton) {
            micButton.disabled = false;
            micButton.style.opacity = '1';
        }

        const statusIndicator = document.getElementById('status-indicator');
        if (statusIndicator) {
            statusIndicator.textContent = '🎤 Click mic to ask again';
            statusIndicator.style.color = '#666';
        }

        // If we have aggregated response text, speak it as a single utterance
        const text = (this.responseTextBuffer || '').trim();
        if (!text) {
            this.responseTextBuffer = '';
            try { const agg = document.getElementById('aggregated-ai-response'); if (agg) agg.textContent = ''; } catch (e) {}
            return;
        }

        // Put text into the hidden aggregated div for accessibility/debug
        try { const agg = document.getElementById('aggregated-ai-response'); if (agg) agg.textContent = text; } catch (e) { }

        // Stop recognition temporarily if active
        this.wasRecordingBeforeTTS = !!this.isRecording;
        try { if (this.wasRecordingBeforeTTS && this.recognition) this.recognition.stop(); } catch (e) { }

        // Speak using Web Speech API if available, otherwise fallback to server-side TTS
        if (window.speechSynthesis) {
            try {
                const u = new SpeechSynthesisUtterance(text);
                if (this.ttsVoice) u.voice = this.ttsVoice;
                u.rate = 1.0; u.pitch = 1.0;
                u.onstart = () => {
                    try { if (this.aiWaveformCtx) this.animateWaveform(this.aiWaveformCtx, '#50c878'); } catch (e) { }
                    const statusIndicator = document.getElementById('status-indicator');
                    if (statusIndicator) { statusIndicator.textContent = '🔊 AI speaking...'; statusIndicator.style.color = '#9C27B0'; }
                    const robot = document.querySelector('.robot-anim'); if (robot) robot.classList.add('speaking');
                    if (this.showDebug) { const panel = document.getElementById('debug-panel'); if (panel) { const pre = document.createElement('div'); pre.textContent = `[${new Date().toISOString()}] TTS (aggregated) start`; panel.appendChild(pre); panel.scrollTop = panel.scrollHeight; } }
                };
                u.onend = () => {
                    try { this.stopWaveformAnimation(); } catch (e) { }
                    const robot = document.querySelector('.robot-anim'); if (robot) robot.classList.remove('speaking');
                    // restart recognition if it was active before
                    if (this.wasRecordingBeforeTTS && this.recognition) {
                        try { this.recognition.start(); this.isRecording = true; const micButton = document.getElementById('mic-button'); if (micButton) micButton.classList.add('recording'); } catch (e) { }
                    }
                    const statusIndicator = document.getElementById('status-indicator');
                    if (statusIndicator) { statusIndicator.textContent = '🎤 Click mic to speak'; statusIndicator.style.color = '#666'; }
                    if (this.showDebug) { const panel = document.getElementById('debug-panel'); if (panel) { const pre = document.createElement('div'); pre.textContent = `[${new Date().toISOString()}] TTS (aggregated) end`; panel.appendChild(pre); panel.scrollTop = panel.scrollHeight; } }
                };
                speechSynthesis.speak(u);
            } catch (e) {
                console.error('[TTS] speak failed on complete:', e);
                try { this.fallbackPlayAudio(text); } catch (err) { console.warn('[TTS] fallbackPlayAudio failed', err); }
            }
        } else {
            try { this.fallbackPlayAudio(text); } catch (e) { console.warn('[TTS] fallbackPlayAudio failed', e); }
        }

        // Clear buffer now (we've scheduled speak or fallback)
        this.responseTextBuffer = '';
        try { const agg = document.getElementById('aggregated-ai-response'); if (agg) agg.textContent = ''; } catch (e) {}
    }
    
    animateWaveform(ctx, color) {
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        
        const drawFrame = () => {
            if (!this.shouldStop) {
                const time = Date.now() / 1000;
                ctx.clearRect(0, 0, width, height);
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                
                for (let x = 0; x < width; x++) {
                    // Create a more complex waveform using multiple sine waves
                    const y = height/2 + 
                            Math.sin(x/50 + time*4) * 20 + 
                            Math.sin(x/30 + time*2) * 10 +
                            Math.sin(x/20 + time*6) * 5;
                    
                    if (x === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                
                ctx.stroke();
                this.animationFrameId = requestAnimationFrame(drawFrame);
            }
        };
        
        drawFrame();
    }
    
    stopWaveformAnimation() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        // Clear both canvases (guard against null contexts)
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
        console.log('[UI] Adding message:', type, content.substring(0, 50) + '...');
        const conversationBody = document.getElementById('conversation-body');
        if (!conversationBody) {
            console.error('[UI] Conversation body element not found');
            return;
        }
        // If AI text display is disabled, skip adding AI messages
        if (type === 'ai' && !this.showAIText) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + type + '-message';
        messageDiv.textContent = content;

        if (type === 'ai') {
            messageDiv.id = 'current-ai-message';
        }

        conversationBody.appendChild(messageDiv);
        conversationBody.scrollTop = conversationBody.scrollHeight;
    }
    
    addOrUpdateAIMessage(content) {
        const currentMessage = document.getElementById('current-ai-message');
        if (currentMessage) {
            currentMessage.textContent += content;
        } else {
            this.addMessage('ai', content);
        }
        
        // Scroll to bottom
        const conversationBody = document.getElementById('conversation-body');
        conversationBody.scrollTop = conversationBody.scrollHeight;
    }
    
    showError(message) {
        const statusIndicator = document.getElementById('status-indicator');
        statusIndicator.textContent = 'Error: ' + message;
        statusIndicator.style.color = '#ff6b6b';
        
        setTimeout(() => {
            statusIndicator.style.color = '#666';
            statusIndicator.textContent = 'Click microphone to speak';
        }, 3000);
    }
    
    exitConversationMode() {
        console.log('[ConversationMode] Exiting conversation mode');
        // Stop recording if active
        if (this.isRecording) {
            this.recognition.stop();
            this.isRecording = false;
        }
        
        // Stop animations
        this.stopWaveformAnimation();
        
        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        // Hide modal
        const modal = document.getElementById('conversation-modal');
        if (modal) {
            modal.style.display = 'none';
        }
        
        // Reset state
        this.currentSpeaker = null;
        this.shouldStop = false;
    }
}

// Initialize conversation mode when document is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('[ConversationMode] Setting up conversation mode...');
    // Only initialize if we're on the user page
    const modal = document.getElementById('conversation-modal');
    const button = document.getElementById('conversational-mode-btn');
    
    if (modal && button) {
        console.log('[ConversationMode] Found modal and button, initializing...');
        window.conversationMode = new ConversationMode();
        
        // Add click handler to conversation mode button
        button.addEventListener('click', () => {
            console.log('[ConversationMode] Button clicked');
            
            const classSelect = document.getElementById('class-select');
            const subjectSelect = document.getElementById('subject-select');
            
            if (!classSelect.value || classSelect.value === 'Select...') {
                alert('Please select a class first.');
                return;
            }
            if (!subjectSelect.value || subjectSelect.value === 'Select...') {
                alert('Please select a subject first.');
                return;
            }
            
            const selectedBook = window.selectedBook;
            console.log('[ConversationMode] Checking selected book:', selectedBook);
            
            if (!selectedBook) {
                console.log('[ConversationMode] No book selected');
                alert('Please select a book first to start conversational mode.');
                return;
            }
            
            // Additional validation
            if (!selectedBook.filename || !selectedBook.id) {
                console.error('[ConversationMode] Invalid book data:', selectedBook);
                alert('Book data is not complete. Please try reloading the book.');
                return;
            }
            
            console.log('[ConversationMode] Starting conversation mode with book:', selectedBook.id);
            window.conversationMode.startConversationMode(selectedBook.id);
        });
    } else {
        console.error('[ConversationMode] Required elements not found:', {
            modal: !!modal,
            button: !!button
        });
    }
});