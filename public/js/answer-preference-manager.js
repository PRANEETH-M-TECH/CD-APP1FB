/**
 * answer-preference-manager.js
 * ─────────────────────────────────────────────────────────────────────────────
 * AnswerPreferenceManager — Mode state management, UI visibility logic,
 * and preference-mic input handling.
 *
 * Modes:
 *   text_text   — 📖 Reading Mode   : text in, text out
 *   text_audio  — 🎧 Tutor Mode     : text in, text + audio out
 *   audio_text  — 🎙️ Voice Query Mode: mic in,  text out
 *   audio_audio — 🧑🏫 AI Voice Mode : mic in,  text + audio out
 *
 * Separated from tts-streaming-manager.js so UI / mic logic can evolve
 * independently of the audio delivery pipeline.
 *
 * Usage (auto-initialized on DOMContentLoaded):
 *   window.answerPreferenceManager.currentMode   — active mode string
 * ─────────────────────────────────────────────────────────────────────────────
 */

class AnswerPreferenceManager {
    constructor() {
        // ── Constants ────────────────────────────────────────────────────────
        this.MODES = {
            text_text:   { input: 'text', output: 'text'  },
            text_audio:  { input: 'text', output: 'audio' },
            audio_text:  { input: 'audio', output: 'text' },
            audio_audio: { input: 'audio', output: 'audio'}
        };
        this.STORAGE_KEY = 'answerPreference';
        this.DEFAULT_MODE = 'text_text';

        // ── State ─────────────────────────────────────────────────────────────
        this.currentMode = this.DEFAULT_MODE;
        this.recognition = null;
        this.isListening = false;
        this._micSupported = false;

        // ── DOM refs (resolved in init) ───────────────────────────────────────
        this.dropdown     = null;  // #answer-preference-select
        this.queryText    = null;  // #query-text
        this.sendBtn      = null;  // #submit-query-btn
        this.voiceBtn     = null;  // #voice-search-btn  (existing simple mic)
        this.prefMicBtn   = null;  // #pref-mic-btn      (new preference mic)
        this.prefMicWrap  = null;  // #pref-mic-wrapper
        this.micStatus    = null;  // #pref-mic-status

        console.log('[MODE] AnswerPreferenceManager constructing...');
    }

    // ── Initialization ────────────────────────────────────────────────────────

    init() {
        // Resolve DOM elements
        this.dropdown    = document.getElementById('answer-preference-select');
        this.queryText   = document.getElementById('query-text');
        this.sendBtn     = document.getElementById('submit-query-btn');
        this.voiceBtn    = document.getElementById('voice-search-btn');
        this.prefMicBtn  = document.getElementById('pref-mic-btn');
        this.prefMicWrap = document.getElementById('pref-mic-wrapper');
        this.micStatus   = document.getElementById('pref-mic-status');

        if (!this.dropdown) {
            console.warn('[MODE] #answer-preference-select not found — AnswerPreferenceManager inactive.');
            return;
        }

        // Adjustment 5: Restore persisted mode from localStorage
        const saved = localStorage.getItem(this.STORAGE_KEY);
        if (saved && this.MODES[saved]) {
            this.currentMode = saved;
            this.dropdown.value = saved;
            console.log(`[MODE] Restored Mode: ${saved}`);
        } else {
            this.currentMode = this.DEFAULT_MODE;
            this.dropdown.value = this.DEFAULT_MODE;
        }

        // Adjustment 4: Initialize SpeechRecognition with fallback
        this._initMic();

        // Apply initial UI state
        this._applyModeUI(this.currentMode);

        // Bind dropdown change listener (guard against duplicates)
        if (!this.dropdown.dataset.prefListenerAttached) {
            this.dropdown.addEventListener('change', (e) => {
                this._onModeChange(e.target.value);
            });
            this.dropdown.dataset.prefListenerAttached = 'true';
        }

        // Bind pref-mic button listener
        if (this.prefMicBtn && !this.prefMicBtn.dataset.prefListenerAttached) {
            this.prefMicBtn.addEventListener('click', () => {
                this._handleMicButtonClick();
            });
            this.prefMicBtn.dataset.prefListenerAttached = 'true';
        }

        // Bind flagship voice interaction panel mic button listener
        const voicePanelMic = document.getElementById('voice-panel-mic');
        if (voicePanelMic && !voicePanelMic.dataset.prefListenerAttached) {
            voicePanelMic.addEventListener('click', () => {
                this._handleMicButtonClick();
            });
            voicePanelMic.dataset.prefListenerAttached = 'true';
        }

        console.log(`[MODE] AnswerPreferenceManager initialized | active mode: ${this.currentMode}`);
    }

    // ── Mode Change ───────────────────────────────────────────────────────────

    _onModeChange(newMode) {
        if (!this.MODES[newMode]) {
            console.error(`[ERROR] Unknown mode: ${newMode}`);
            return;
        }

        // Stop mic if switching away from an audio-input mode
        if (this.isListening) {
            this._stopMic();
        }

        // Stop any running audio pipeline
        if (window.ttsPipeline && this.MODES[this.currentMode].output === 'audio') {
            window.ttsPipeline.stop();
        }

        this.currentMode = newMode;

        // Adjustment 5: Persist to localStorage
        localStorage.setItem(this.STORAGE_KEY, newMode);
        console.log(`[MODE] Selected Mode: ${newMode}`);

        this._applyModeUI(newMode);
    }

    // ── UI Visibility Logic ───────────────────────────────────────────────────

    _applyModeUI(mode) {
        const config = this.MODES[mode];
        if (!config) return;

        const isTextInput  = config.input  === 'text';
        const isAudioInput = config.input  === 'audio';

        // ── Input controls ────────────────────────────────────────────────
        // Standard textarea and Send button are ALWAYS visible for modern UI consistency
        if (this.queryText) {
            this.queryText.style.display = '';
            if (isAudioInput) {
                this.queryText.placeholder = "Tap the mic to ask by voice, or type your question";
            } else {
                this.queryText.placeholder = "Ask a question about the selected book...";
            }
        }
        if (this.sendBtn) {
            this.sendBtn.style.display = '';
        }

        // Toggle simple voice search button (visible only in text-input modes, hidden completely in Reading and Tutor Modes)
        if (this.voiceBtn) {
            this.voiceBtn.style.display = (isTextInput && mode !== 'text_text' && mode !== 'text_audio') ? '' : 'none';
        }

        // Preference mic button (visible only in audio-input modes)
        if (this.prefMicBtn) {
            this.prefMicBtn.style.display = isAudioInput ? '' : 'none';
        }

        // Preference mic status wrapper (contains the status text/warning)
        if (this.prefMicWrap) {
            this.prefMicWrap.style.display = isAudioInput ? 'flex' : 'none';
        }

        // Toggle speak buttons on all existing cards in the chat history (hidden completely in Reading and Voice Query Modes)
        const speakButtons = document.querySelectorAll('.speak-btn');
        speakButtons.forEach(btn => {
            btn.style.display = (mode === 'text_text' || mode === 'audio_text') ? 'none' : '';
        });

        // ── Pref-mic button state ─────────────────────────────────────────
        if (this.prefMicBtn) {
            if (isAudioInput && !this._micSupported) {
                this.prefMicBtn.disabled = true;
                this.prefMicBtn.title    = 'Voice input not supported in this browser';
                this._showMicWarning('⚠️ Voice input not supported in this browser.');
            } else if (isAudioInput) {
                this.prefMicBtn.disabled = false;
                this.prefMicBtn.title = 'Click to speak your question';
                this._hideMicWarning();
            }
        }

        // ── AI Voice Mode Flagship UI Toggle ──────────────────────────────
        const chatInput = document.getElementById('chat-input-container');
        const voicePanel = document.getElementById('voice-interaction-panel');
        if (mode === 'audio_audio') {
            if (chatInput) chatInput.style.display = 'none';
            if (voicePanel) {
                voicePanel.style.display = 'block';
                this.setVoicePanelState('idle');
            }
        } else {
            if (chatInput) chatInput.style.display = '';
            if (voicePanel) voicePanel.style.display = 'none';
        }

        // ── Mode indicator badge ──────────────────────────────────────────
        const badge = document.getElementById('active-mode-badge');
        if (badge) {
            const labels = {
                text_text:   '📖 Reading',
                text_audio:  '🎧 Tutor',
                audio_text:  '🎙️ Voice Query',
                audio_audio: '🧑🏫 AI Voice'
            };
            badge.textContent = labels[mode] || mode;
        }

        console.log(`[MODE] UI applied for mode: ${mode} | input=${config.input} | output=${config.output}`);
    }

    // ── SpeechRecognition (Adjustment 4) ──────────────────────────────────────

    _initMic() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            // Adjustment 4: Graceful fallback
            console.error('[ERROR] SpeechRecognition not supported in this browser');
            this._micSupported = false;
            if (this.prefMicBtn) {
                this.prefMicBtn.disabled = true;
                this.prefMicBtn.title    = 'Voice input not supported in this browser';
            }
            return;
        }

        this._micSupported = true;

        this.recognition = new SpeechRecognition();
        this.recognition.continuous      = false;
        this.recognition.interimResults  = true;
        this.recognition.lang            = 'en-US';

        let interimTranscript = '';
        let finalTranscript   = '';

        this.recognition.onstart = () => {
            this.isListening   = true;
            interimTranscript  = '';
            finalTranscript    = '';
            this._setMicState('listening');

            // AI Voice Mode adjustments
            if (this.currentMode === 'audio_audio') {
                this.setVoicePanelState('listening');
                const overlay = document.getElementById('live-transcript-overlay');
                const textEl = document.getElementById('live-transcript-text');
                if (overlay && textEl) {
                    textEl.textContent = 'Start speaking to ask a question...';
                    overlay.classList.remove('opacity-0', 'pointer-events-none');
                }
            }
            console.log('[MODE] Preference mic started listening');
        };

        this.recognition.onresult = (event) => {
            interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) {
                    finalTranscript += result[0].transcript;
                } else {
                    interimTranscript += result[0].transcript;
                }
            }

            // Show live transcript feedback
            if (this.micStatus) {
                this.micStatus.textContent = finalTranscript + interimTranscript || '🎤 Listening...';
            }

            // Update live transcript overlay in AI Voice Mode
            if (this.currentMode === 'audio_audio') {
                const textEl = document.getElementById('live-transcript-text');
                if (textEl) {
                    textEl.textContent = `"${finalTranscript + interimTranscript || 'Listening...'}"`;
                }
            }
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this._setMicState('idle');

            if (finalTranscript.trim()) {
                console.log(`[MODE] Preference mic transcript: "${finalTranscript.trim()}"`);
                
                // Transition voice panel state to processing if we got a transcript
                if (this.currentMode === 'audio_audio') {
                    this.setVoicePanelState('processing');
                }
                
                this._dispatchQuery(finalTranscript.trim());
            } else {
                if (this.currentMode === 'audio_audio') {
                    this.setVoicePanelState('idle');
                }
            }

            if (this.micStatus) {
                this.micStatus.textContent = '';
            }

            // Fade out overlay in AI Voice Mode
            if (this.currentMode === 'audio_audio') {
                const overlay = document.getElementById('live-transcript-overlay');
                if (overlay) {
                    overlay.classList.add('opacity-0', 'pointer-events-none');
                }
            }
        };

        this.recognition.onerror = (event) => {
            console.error(`[ERROR] Preference mic recognition error: ${event.error}`);
            this.isListening = false;
            this._setMicState('idle');

            if (this.currentMode === 'audio_audio') {
                this.setVoicePanelState('idle');
                const overlay = document.getElementById('live-transcript-overlay');
                if (overlay) {
                    overlay.classList.add('opacity-0', 'pointer-events-none');
                }
            }

            if (this.micStatus) {
                this.micStatus.textContent = `⚠️ Mic error: ${event.error}`;
                setTimeout(() => {
                    if (this.micStatus) this.micStatus.textContent = '';
                }, 3000);
            }
        };

        console.log('[MODE] SpeechRecognition initialized for preference mic');
    }

    _handleMicButtonClick() {
        if (!this._micSupported) return;

        if (this.isListening) {
            this._stopMic();
        } else {
            this._startMic();
        }
    }

    _startMic() {
        if (!this.recognition || this.isListening) return;

        try {
            this.recognition.start();
        } catch (e) {
            console.error('[ERROR] Could not start preference mic:', e);
            this._setMicState('idle');
        }
    }

    _stopMic() {
        if (!this.recognition) return;
        try {
            this.recognition.stop();
        } catch (e) {
            // Recognition may already be stopped
        }
        this.isListening = false;
        this._setMicState('idle');
    }

    // ── Mic Button Visual State ───────────────────────────────────────────────

    _setMicState(state) {
        if (!this.prefMicBtn) return;

        this.prefMicBtn.classList.remove('pref-mic-listening', 'pref-mic-processing');

        const iconEl = this.prefMicBtn.querySelector('.pref-mic-icon');

        switch (state) {
            case 'listening':
                this.prefMicBtn.classList.add('pref-mic-listening');
                this.prefMicBtn.title = 'Listening... Click to stop';
                if (iconEl) iconEl.textContent = '🔴';
                break;
            case 'processing':
                this.prefMicBtn.classList.add('pref-mic-processing');
                this.prefMicBtn.title = 'Processing...';
                if (iconEl) iconEl.textContent = '⏳';
                break;
            case 'idle':
            default:
                this.prefMicBtn.title = 'Click to speak your question';
                if (iconEl) iconEl.textContent = '🎙️';
                break;
        }
    }

    // ── Query Dispatch ────────────────────────────────────────────────────────

    /**
     * Dispatch a query from mic input to the existing submitSmartQuery pipeline.
     * Uses the global bridge function exposed by script.js.
     */
    _dispatchQuery(transcript) {
        this._setMicState('processing');

        if (typeof window.submitSmartQueryFromMic === 'function') {
            window.submitSmartQueryFromMic(transcript);
        } else {
            // Fallback: place text in query input and trigger form submit
            const queryTextEl = document.getElementById('query-text');
            const form        = document.getElementById('user-query-form');
            if (queryTextEl && form) {
                queryTextEl.value = transcript;
                const event = new Event('submit', { bubbles: true, cancelable: true });
                form.dispatchEvent(event);
            } else {
                console.error('[ERROR] Cannot dispatch mic query — form not found');
            }
        }

        // Reset mic state after a delay (query will take over)
        setTimeout(() => this._setMicState('idle'), 500);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    _showMicWarning(message) {
        let warning = document.getElementById('pref-mic-warning');
        if (!warning) {
            warning = document.createElement('div');
            warning.id = 'pref-mic-warning';
            warning.className = 'pref-mic-warning-msg';
            const form = document.getElementById('chat-input-container');
            if (form) form.appendChild(warning);
        }
        warning.textContent = message;
        warning.style.display = 'block';
    }

    _hideMicWarning() {
        const warning = document.getElementById('pref-mic-warning');
        if (warning) warning.style.display = 'none';
    }

    // ── Public helpers used by script.js ──────────────────────────────────────

    isAudioOutputMode() {
        return this.currentMode === 'text_audio' || this.currentMode === 'audio_audio';
    }

    isAudioInputMode() {
        return this.currentMode === 'audio_text' || this.currentMode === 'audio_audio';
    }

    // ── Voice Panel State Management ───────────────────────────────────────────
    setVoicePanelState(state) {
        const panel = document.getElementById('voice-interaction-panel');
        if (!panel) return;

        const micBtn   = document.getElementById('voice-panel-mic');
        const iconEl   = document.getElementById('voice-panel-icon');
        const statusEl = document.getElementById('voice-panel-status');
        const subEl    = document.getElementById('voice-panel-subtitle');

        if (!micBtn || !iconEl || !statusEl || !subEl) return;

        // Clear existing state classes
        micBtn.classList.remove('listening', 'processing', 'speaking', 'paused');

        switch (state) {
            case 'listening':
                micBtn.classList.add('listening');
                iconEl.textContent = '🎤';
                statusEl.textContent = '🎤 Listening...';
                subEl.textContent = 'Tap again to stop recording';
                subEl.style.display = '';
                break;
            case 'processing':
                micBtn.classList.add('processing');
                iconEl.textContent = '🤔';
                statusEl.textContent = '🤔 Understanding your question...';
                subEl.textContent = '';
                subEl.style.display = 'none';
                break;
            case 'generating':
            case 'speaking':
                micBtn.classList.add('speaking');
                iconEl.textContent = '🗣';
                statusEl.textContent = '🗣 Explaining...';
                subEl.textContent = '';
                subEl.style.display = 'none';
                break;
            case 'paused':
                micBtn.classList.add('paused');
                iconEl.textContent = '⏸';
                statusEl.textContent = '⏸ Paused';
                subEl.textContent = 'Narration is suspended';
                subEl.style.display = '';
                break;
            case 'idle':
            default:
                iconEl.textContent = '🎤';
                statusEl.textContent = '🎤 Ask by Voice';
                subEl.textContent = 'Ask anything from this textbook';
                subEl.style.display = '';
                break;
        }
        console.log(`[VOICE PANEL] State transition to: ${state}`);
    }
}

// ── Auto-initialize on DOMContentLoaded ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    window.answerPreferenceManager = new AnswerPreferenceManager();
    window.answerPreferenceManager.init();
    console.log('[MODE] window.answerPreferenceManager ready.');
});
