/**
 * tts-manager.js
 * ──────────────────────────────────────────────────────────────────────────
 * Centralized TTS (Text-to-Speech) controller.
 *
 * Handles model selection, voice selection, audio playback, and graceful
 * fallback to the browser's built-in SpeechSynthesis if a cloud call fails.
 *
 * Usage (auto-initialized on DOMContentLoaded):
 *   window.ttsManager.speak(text)      — speak text with current model/voice
 *   window.ttsManager.stop()           — stop current playback
 *   window.ttsManager.setModel(model)  — 'sarvam' | 'browser'
 *   window.ttsManager.setVoice(voice)  — 'meera' | 'aditya' | ...
 *   window.ttsManager.setLanguage(lang)— 'en-IN' | 'te-IN' | ...
 * ──────────────────────────────────────────────────────────────────────────
 */

class TTSManager {
    constructor() {
        // ── Defaults ────────────────────────────────────────────────────
        this.model    = 'sarvam';   // active TTS model
        this.voice    = 'anushka';  // active voice/speaker
        this.language = 'en-IN';    // active language code

        // ── State ────────────────────────────────────────────────────────
        this.isSpeaking  = false;
        this.currentAudio = null;   // HTMLAudioElement for cloud TTS playback
        this._activeBtn  = null;    // the 🔊 button that triggered speak()
        
        // Dual-Queue System
        this.fetchQueue = [];       // holds text waiting to be fetched
        this.isFetching = false;    // true if fetch loop is active
        this.playbackQueue = [];    // holds { text, url } ready for playback
        this.isPlayingQueue = false;// true if playback loop is active

        // Load saved preferences from localStorage
        this._loadPreferences();

        console.log(`[TTSManager] Initialized — model=${this.model}, voice=${this.voice}, lang=${this.language}`);
    }

    // ── Persistence ───────────────────────────────────────────────────────

    _loadPreferences() {
        try {
            const saved = JSON.parse(localStorage.getItem('tts_preferences') || '{}');
            if (saved.model)    this.model    = saved.model;
            if (saved.voice)    this.voice    = saved.voice;
            if (saved.language) this.language = saved.language;
        } catch (e) {
            console.warn('[TTSManager] Could not load preferences:', e);
        }
    }

    _savePreferences() {
        try {
            localStorage.setItem('tts_preferences', JSON.stringify({
                model:    this.model,
                voice:    this.voice,
                language: this.language,
            }));
        } catch (e) { /* ignore */ }
    }

    // ── Setters ───────────────────────────────────────────────────────────

    setModel(model) {
        this.model = model;
        this._savePreferences();
        console.log(`[TTSManager] Model changed → ${model}`);
    }

    setVoice(voice) {
        this.voice = voice;
        this._savePreferences();
        console.log(`[TTSManager] Voice changed → ${voice}`);
    }

    setLanguage(lang) {
        this.language = lang;
        this._savePreferences();
        console.log(`[TTSManager] Language changed → ${lang}`);
    }

    // ── Core: speak() ─────────────────────────────────────────────────────

    /**
     * Speak text using the active model.
     * @param {string} text  — The text to speak.
     * @param {HTMLElement|null} btn — Optional 🔊 button (for icon updates).
     */
    async speak(text, btn = null) {
        if (!text || !text.trim()) return;

        this._activeBtn = btn;
        this.fetchQueue.push(text);

        if (!this.isFetching) {
            this._processFetchQueue();
        }
    }

    async _processFetchQueue() {
        if (this.fetchQueue.length === 0) {
            this.isFetching = false;
            return;
        }

        this.isFetching = true;
        const text = this.fetchQueue.shift();

        let audioUrl = null;

        try {
            if (this.model === 'sarvam') {
                audioUrl = await this._fetchCloudAudio(text);
            }
        } catch (err) {
            console.error('[TTSManager] Cloud fetch failed:', err);
        }

        // Push successfully fetched URL (or null for browser fallback) to playback queue
        this.playbackQueue.push({ text, url: audioUrl });

        if (!this.isPlayingQueue) {
            this._processPlaybackQueue();
        }

        // Process next item in fetch queue
        this._processFetchQueue();
    }

    async _processPlaybackQueue() {
        if (this.playbackQueue.length === 0) {
            this.isPlayingQueue = false;
            this.isSpeaking = false;
            // Notify PlaybackController that static narration is done
            if (window.playbackController && window.playbackController.currentEngine === 'manager') {
                window.playbackController.setState({
                    isPlaying: false,
                    isPaused: false,
                    isStopped: true,
                    currentNarrationId: null,
                    currentEngine: null,
                    playbackStatus: 'idle'
                });
            }
            return;
        }

        this.isPlayingQueue = true;
        this.isSpeaking = true;

        const item = this.playbackQueue.shift();

        try {
            if (item.url) {
                await this._playAudioUrl(item.url);
            } else {
                await this._speakBrowser(item.text);
            }
        } catch (err) {
            console.error('[TTSManager] Playback error:', err);
        }

        // Play next item in queue
        this._processPlaybackQueue();
    }

    // ── Core: stop() ──────────────────────────────────────────────────────

    stop() {
        this.fetchQueue = [];
        this.playbackQueue = [];
        this.isFetching = false;
        this.isPlayingQueue = false;

        // Stop cloud audio
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }

        // Stop browser TTS
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }

        this.isSpeaking = false;
        this._setButtonState('idle');
    }

    // ── Internal: Cloud TTS (Sarvam) ──────────────────────────────────────

    async _fetchCloudAudio(text) {
        console.log(`[TTSManager] Fetching /api/tts — model=${this.model}, voice=${this.voice}, chars=${text.length}`);

        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text:     text,
                model:    this.model,
                language: this.language,
                speaker:  this.voice,
            }),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || response.status);
        }

        const data = await response.json();
        const { audio_base64, format } = data;

        if (!audio_base64) {
            throw new Error('No audio data returned from TTS API.');
        }

        // Decode base64 → Blob → Object URL
        const byteChars = atob(audio_base64);
        const byteArr   = new Uint8Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) {
            byteArr[i] = byteChars.charCodeAt(i);
        }

        const mimeType = format === 'mp3' ? 'audio/mpeg' : 'audio/wav';
        const blob      = new Blob([byteArr], { type: mimeType });
        return URL.createObjectURL(blob);
    }

    async _playAudioUrl(url) {
        return new Promise(async (resolve) => {
            const audio = new Audio(url);
            this.currentAudio = audio;

            audio.onended = () => {
                URL.revokeObjectURL(url);
                this.currentAudio = null;
                console.log('[TTSManager] Playback finished.');
                resolve();
            };

            audio.onerror = (e) => {
                URL.revokeObjectURL(url);
                this.currentAudio = null;
                console.error('[TTSManager] Audio playback error:', e);
                resolve();
            };

            try {
                await audio.play();
                console.log('[TTSManager] Cloud audio playing…');
            } catch (err) {
                console.error('[TTSManager] Exception during play:', err);
                resolve(); 
            }
        });
    }

    // ── Internal: Browser TTS (fallback) ─────────────────────────────────

    _speakBrowser(text) {
        return new Promise((resolve) => {
            if (!window.speechSynthesis) {
                console.warn('[TTSManager] Browser TTS not available.');
                resolve();
                return;
            }

            const utterance = new SpeechSynthesisUtterance(text);

            utterance.onend = () => {
                resolve();
            };

            utterance.onerror = () => {
                resolve();
            };

            window.speechSynthesis.speak(utterance);
        });
    }

    // ── Internal: Button State ────────────────────────────────────────────

    _setButtonState(state) {
        // No-op: all button visuals are controlled by PlaybackController subscribers
    }

    // ── Fetch voices from backend ─────────────────────────────────────────

    async fetchVoices(model = null) {
        const m = model || this.model;
        try {
            const res   = await fetch(`/api/tts/voices?model=${m}`);
            const data  = await res.json();
            return data.voices || [];
        } catch (e) {
            console.warn('[TTSManager] Could not fetch voices:', e);
            return [];
        }
    }
}

// ── Auto-initialize on DOMContentLoaded ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    window.ttsManager = new TTSManager();
    console.log('[TTSManager] window.ttsManager ready.');
});
