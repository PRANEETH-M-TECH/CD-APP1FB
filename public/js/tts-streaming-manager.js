/**
 * tts-streaming-manager.js
 * ─────────────────────────────────────────────────────────────────────────────
 * StreamingAudioPipeline — Chunked streaming audio delivery for Tutor Mode
 * and AI Voice Mode.
 *
 * Responsibilities:
 *   - Accumulate Gemini SSE tokens into delivery chunks
 *   - Send each chunk to Sarvam TTS (/api/tts)
 *   - Maintain a strictly ordered playback queue (Chunk N+1 never plays before N)
 *   - Track and log [LATENCY] for every chunk
 *   - Synchronize text display with audio playback
 *
 * Separated from answer-preference-manager.js so TTS providers (Sarvam, Azure,
 * EdgeTTS, Teacher Mode) can be swapped without touching UI / mic logic.
 *
 * Usage:
 *   window.ttsPipeline = new StreamingAudioPipeline();
 *   window.ttsPipeline.start();
 *   window.ttsPipeline.pushToken("some text ");
 *   window.ttsPipeline.flush();   // called when stream ends
 *   window.ttsPipeline.stop();    // abort mid-stream
 * ─────────────────────────────────────────────────────────────────────────────
 */

class StreamingAudioPipeline {
    constructor(options = {}) {
        // ── Configuration ────────────────────────────────────────────────────
        // Flush when this many complete sentences are buffered
        this.sentenceThreshold = options.sentenceThreshold || 2;
        // Flush when buffer reaches this many characters (Adjustment 2)
        this.charThreshold = options.charThreshold || 300;
        // If true, skip real Sarvam calls (credit protection during testing)
        this.dryRun = options.dryRun || false;

        // ── Internal State ───────────────────────────────────────────────────
        this.chunkIdCounter = 0;
        this.textBuffer = '';
        this.deliveryQueue = [];       // { chunk_id, text_chunk, audio_blob_url, status }
        this.fetchQueue = [];          // chunks waiting for TTS fetch
        this.isProcessingPlayback = false;
        this.isFetchingTTS = false;
        this.isActive = false;
        this.currentAudio = null;
        this.abortController = null;

        // ── Play/Pause Control & State ───────────────────────────────────────
        this.isPaused = false;
        this.hasStartedAudio = false;
        this._activeBtn = null;

        // ── Display callback (set by external code) ──────────────────────────
        // Called with (text_chunk, chunk_id) when a chunk is ready to display
        this.onDisplayChunk = options.onDisplayChunk || null;
        // Called when all chunks are done
        this.onComplete = options.onComplete || null;

        // ── Regex for sentence detection ─────────────────────────────────────
        this._sentenceRe = /[.!?।]+\s/g;

        console.log(`[STREAM] StreamingAudioPipeline initialized | sentenceThreshold=${this.sentenceThreshold} | charThreshold=${this.charThreshold} | dryRun=${this.dryRun}`);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Reset state and begin accepting tokens.
     */
    start() {
        this.chunkIdCounter = 0;
        this.textBuffer = '';
        this.deliveryQueue = [];
        this.fetchQueue = [];
        this.isProcessingPlayback = false;
        this.isFetchingTTS = false;
        this.isActive = true;
        this.isPaused = false;
        this.hasStartedAudio = false;
        this._activeBtn = null;
        this._stopCurrentAudio();
        console.log('[STREAM] Gemini Stream Started');
    }

    /**
     * Push a raw token from the Gemini SSE stream.
     * @param {string} token
     */
    pushToken(token) {
        if (!this.isActive) return;
        this.textBuffer += token;
        this._tryFlush();
    }

    /**
     * Force-flush remaining buffer as the final chunk.
     * Call when the Gemini stream emits [DONE].
     */
    flush() {
        if (!this.isActive) return;
        const remaining = this.textBuffer.trim();
        if (remaining) {
            this._createChunk(remaining);
            this.textBuffer = '';
        }
        console.log('[STREAM] Gemini Stream Ended — final flush complete');
    }

    /**
     * Abort all queued chunks and stop current audio.
     */
    stop() {
        this.isActive = false;
        this.isPaused = false;
        this.hasStartedAudio = false;
        this.fetchQueue = [];
        this.deliveryQueue = [];
        this.textBuffer = '';
        this.isProcessingPlayback = false;
        this.isFetchingTTS = false;
        this._stopCurrentAudio();
        if (window.answerPreferenceManager && window.answerPreferenceManager.currentMode === 'audio_audio') {
            window.answerPreferenceManager.setVoicePanelState('idle');
        }
        console.log('[STREAM] Pipeline stopped (abort)');
    }

    // ── Play/Pause API ────────────────────────────────────────────────────────

    pause() {
        if (!this.isActive || this.isPaused) return;
        this.isPaused = true;
        
        // Pause active audio engine
        if (this.currentAudio) {
            this.currentAudio.pause();
        } else if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.pause();
        }

        // Update button icon to Play (▶)
        const btn = this._activeBtn || this._findActiveButton();
        if (btn) {
            btn.textContent = '▶';
            btn.title = 'Resume narration';
            this._activeBtn = btn;
        }

        if (window.answerPreferenceManager && window.answerPreferenceManager.currentMode === 'audio_audio') {
            window.answerPreferenceManager.setVoicePanelState('paused');
        }
        console.log('[STREAM] Audio playback paused');
    }

    resume() {
        if (!this.isActive || !this.isPaused) return;
        this.isPaused = false;

        // Resume active audio engine
        if (this.currentAudio) {
            this.currentAudio.play().catch(err => {
                console.error('[STREAM] Error resuming cloud audio:', err);
            });
        } else if (window.speechSynthesis && window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
        }

        // Update button icon to Pause (⏸)
        const btn = this._activeBtn || this._findActiveButton();
        if (btn) {
            btn.textContent = '⏸';
            btn.title = 'Pause narration';
            this._activeBtn = btn;
        }

        if (window.answerPreferenceManager && window.answerPreferenceManager.currentMode === 'audio_audio') {
            window.answerPreferenceManager.setVoicePanelState('speaking');
        }
        console.log('[STREAM] Audio playback resumed');
    }

    togglePlayPause(button) {
        if (!this.isActive) return;
        this._activeBtn = button;
        if (this.isPaused) {
            this.resume();
        } else {
            this.pause();
        }
    }

    _waitForResume() {
        return new Promise((resolve) => {
            const check = () => {
                if (!this.isActive) { resolve(); return; }
                if (!this.isPaused) { resolve(); return; }
                setTimeout(check, 50);
            };
            check();
        });
    }

    _findActiveButton() {
        const cards = document.querySelectorAll('.ai-card');
        if (cards.length > 0) {
            const latestCard = cards[cards.length - 1];
            return latestCard.querySelector('.speak-btn');
        }
        return null;
    }

    _speakBrowser(text) {
        return new Promise((resolve) => {
            if (!window.speechSynthesis) {
                console.warn('[STREAM] Browser TTS not available.');
                resolve();
                return;
            }

            const utterance = new SpeechSynthesisUtterance(text);
            
            // Set language if available from ttsManager
            if (window.ttsManager && window.ttsManager.language) {
                utterance.lang = window.ttsManager.language;
            }

            utterance.onend = () => {
                resolve();
            };

            utterance.onerror = (e) => {
                console.error('[STREAM] Browser TTS error:', e);
                resolve();
            };

            window.speechSynthesis.speak(utterance);
        });
    }

    // ── Internal: Buffer Management ───────────────────────────────────────────

    /**
     * Check flush conditions (Adjustment 2):
     *   A. Complete sentences >= sentenceThreshold
     *   OR
     *   B. Buffer length >= charThreshold
     *
     * Splits cleanly at sentence boundary when condition A fires.
     */
    _tryFlush() {
        // Condition B: character count threshold
        if (this.textBuffer.length >= this.charThreshold) {
            // Try to split at last sentence boundary within the buffer
            const lastBoundary = this._findLastSentenceBoundary(this.textBuffer);
            if (lastBoundary > 0) {
                const chunk = this.textBuffer.slice(0, lastBoundary).trim();
                this.textBuffer = this.textBuffer.slice(lastBoundary).trim();
                this._createChunk(chunk);
                console.log(`[BUFFER] Char threshold (${this.charThreshold}) triggered flush`);
                return;
            }
        }

        // Condition A: sentence count threshold
        const sentenceCount = this._countCompleteSentences(this.textBuffer);
        if (sentenceCount >= this.sentenceThreshold) {
            const { chunk, remainder } = this._extractSentences(this.textBuffer, this.sentenceThreshold);
            if (chunk) {
                this.textBuffer = remainder;
                this._createChunk(chunk);
                console.log(`[BUFFER] Sentence threshold (${this.sentenceThreshold}) triggered flush`);
            }
        }
    }

    _countCompleteSentences(text) {
        const matches = text.match(/[.!?।]+\s/g);
        return matches ? matches.length : 0;
    }

    _findLastSentenceBoundary(text) {
        let lastPos = -1;
        const re = /[.!?।]+\s/g;
        let m;
        while ((m = re.exec(text)) !== null) {
            lastPos = m.index + m[0].length;
        }
        return lastPos;
    }

    _extractSentences(text, count) {
        const re = /[.!?।]+\s/g;
        let lastPos = 0;
        let found = 0;
        let m;
        while ((m = re.exec(text)) !== null) {
            found++;
            lastPos = m.index + m[0].length;
            if (found >= count) break;
        }
        if (found >= count) {
            return {
                chunk: text.slice(0, lastPos).trim(),
                remainder: text.slice(lastPos).trim()
            };
        }
        return { chunk: '', remainder: text };
    }

    _sanitizeForTTS(text) {
        if (!text) return '';
        
        let clean = text;
        
        // Remove bold/italic markers
        clean = clean.replace(/\*\*/g, '');
        clean = clean.replace(/\*/g, '');
        clean = clean.replace(/__/g, '');
        clean = clean.replace(/_/g, '');
        
        // Remove headers
        clean = clean.replace(/^\s*#+\s+/gm, '');
        
        // Remove markdown bullets (e.g., * list item, - list item) at start of lines
        clean = clean.replace(/^\s*[\*\-\•]\s+/gm, '');
        
        // Convert trailing colons or colons inside header-like structures to periods
        clean = clean.replace(/:\s*$/gm, '.');
        clean = clean.replace(/(\w+):\s/g, '$1. ');

        // Normalize multiple spaces
        clean = clean.replace(/\s+/g, ' ').trim();
        
        return clean;
    }

    _createChunk(text) {
        this.chunkIdCounter++;
        const chunk_id = this.chunkIdCounter;
        const sanitized = this._sanitizeForTTS(text);
        const chunk = {
            chunk_id,
            text_chunk: text,
            sanitized_text: sanitized,
            audio_blob_url: null,
            status: 'pending',        // pending → fetching → ready → playing → done
            _createTime: performance.now()
        };
        this.fetchQueue.push(chunk);
        this.deliveryQueue.push(chunk);

        console.log(`[BUFFER] Chunk #${chunk_id} Created | "${text.slice(0, 60)}..."`);

        // Display the chunk in the UI immediately as it is created, so text
        // streams without being blocked by play/pause states.
        if (typeof this.onDisplayChunk === 'function') {
            this.onDisplayChunk(text, chunk_id);
        }

        // Kick off TTS fetch if not already running
        if (!this.isFetchingTTS) {
            this._processFetchQueue();
        }
    }

    // ── Internal: TTS Fetching ────────────────────────────────────────────────

    async _processFetchQueue() {
        if (this.fetchQueue.length === 0) {
            this.isFetchingTTS = false;
            // All fetches done — kick off playback if not started
            if (!this.isProcessingPlayback) {
                this._processPlaybackQueue();
            }
            return;
        }

        this.isFetchingTTS = true;
        const chunk = this.fetchQueue.shift();
        chunk.status = 'fetching';

        await this._fetchTTS(chunk);

        // Process next item
        this._processFetchQueue();
    }

    async _fetchTTS(chunk) {
        const { chunk_id, text_chunk } = chunk;
        const fetchStart = performance.now();

        // Get current TTS settings from ttsManager if available
        const model = (window.ttsManager && window.ttsManager.model) || 'sarvam';

        if (model === 'browser') {
            console.log(`[TTS] Using Browser TTS for Chunk #${chunk_id}`);
            chunk.status = 'ready';
            chunk.isBrowserTTS = true;
            chunk.audio_blob_url = null;
            this._onChunkReady(chunk);
            return;
        }

        if (this.dryRun) {
            console.log(`[TTS] DRY RUN — skipping Sarvam call for Chunk #${chunk_id}`);
            // Simulate a short delay to mimic network latency
            await new Promise(r => setTimeout(r, 200));
            chunk.audio_blob_url = null;
            chunk.status = 'ready';
            const elapsed = Math.round(performance.now() - fetchStart);
            console.log(`[LATENCY] Chunk #${chunk_id} TTS Generation (dry run): ${elapsed}ms`);
            this._onChunkReady(chunk);
            return;
        }

        console.log(`[TTS] Sending Chunk #${chunk_id} To Sarvam | ${chunk.sanitized_text.length} chars (sanitized)`);

        try {
            const speaker = (window.ttsManager && window.ttsManager.voice) || 'anushka';
            const language = (window.ttsManager && window.ttsManager.language) || 'en-IN';

            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: chunk.sanitized_text,
                    model: model,
                    language: language,
                    speaker: speaker
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            const elapsed = Math.round(performance.now() - fetchStart);
            console.log(`[TTS] Audio Received For Chunk #${chunk_id}`);
            console.log(`[LATENCY] Chunk #${chunk_id} TTS Generation: ${elapsed}ms`);

            // Decode base64 → Blob → Object URL
            const { audio_base64, format } = data;
            if (audio_base64) {
                const byteChars = atob(audio_base64);
                const byteArr = new Uint8Array(byteChars.length);
                for (let i = 0; i < byteChars.length; i++) {
                    byteArr[i] = byteChars.charCodeAt(i);
                }
                const mimeType = format === 'mp3' ? 'audio/mpeg' : 'audio/wav';
                const blob = new Blob([byteArr], { type: mimeType });
                chunk.audio_blob_url = URL.createObjectURL(blob);
            }

            chunk.status = 'ready';
            chunk._ttsTime = elapsed;
            this._onChunkReady(chunk);

        } catch (err) {
            console.error(`[ERROR] TTS fetch failed for Chunk #${chunk_id}:`, err);
            chunk.status = 'ready';   // continue without audio for this chunk
            chunk.audio_blob_url = null;
            this._onChunkReady(chunk);
        }
    }

    _onChunkReady(chunk) {
        console.log(`[QUEUE] Chunk #${chunk.chunk_id} Added`);
        // Kick off playback if not already running
        if (!this.isProcessingPlayback) {
            this._processPlaybackQueue();
        }
    }

    // ── Internal: Playback Queue (Strict Ordering) ────────────────────────────

    /**
     * Process the deliveryQueue in strict chunk_id order.
     * Chunk N+1 NEVER starts before Chunk N completes.
     */
    async _processPlaybackQueue() {
        if (this.isProcessingPlayback) return;
        this.isProcessingPlayback = true;

        while (this.deliveryQueue.length > 0) {
            const chunk = this.deliveryQueue[0];

            // Wait until this chunk is ready (TTS might still be fetching)
            if (chunk.status !== 'ready' && chunk.status !== 'done') {
                const queueWaitStart = performance.now();
                await this._waitForChunkReady(chunk);
                const queueWait = Math.round(performance.now() - queueWaitStart);
                if (queueWait > 10) {
                    console.log(`[LATENCY] Chunk #${chunk.chunk_id} Queue Wait: ${queueWait}ms`);
                    const total = (chunk._ttsTime || 0) + queueWait;
                    console.log(`[LATENCY] Total Audio Delay (Chunk #${chunk.chunk_id}): ${total}ms`);
                }
            }

            // Dequeue and play
            this.deliveryQueue.shift();
            await this._playChunk(chunk);
        }

        this.isProcessingPlayback = false;

        // Notify caller that all chunks are done
        if (typeof this.onComplete === 'function') {
            this.onComplete();
        }
    }

    /**
     * Poll until chunk.status === 'ready' (TTS arrived) or pipeline stopped.
     */
    _waitForChunkReady(chunk) {
        return new Promise((resolve) => {
            const check = () => {
                if (!this.isActive) { resolve(); return; }
                if (chunk.status === 'ready' || chunk.status === 'done') { resolve(); return; }
                setTimeout(check, 30);
            };
            check();
        });
    }

    /**
     * Display text chunk, then play audio for that chunk.
     * @param {Object} chunk
     */
    async _playChunk(chunk) {
        const { chunk_id, text_chunk, audio_blob_url } = chunk;

        // Wait if playback is paused before playing audio
        if (this.isPaused) {
            await this._waitForResume();
        }

        chunk.status = 'playing';
        const playStart = performance.now();

        console.log(`[PLAYBACK] Playing Chunk #${chunk_id}`);

        // 1. Display of text chunk was already handled immediately in _createChunk
        // to prevent display block when audio is paused.

        // 2. Play audio (if available)
        if (audio_blob_url || chunk.isBrowserTTS) {
            // Once first audio chunk starts, show the play/pause button and set to pause icon (⏸)
            if (!this.hasStartedAudio) {
                this.hasStartedAudio = true;
                const btn = this._activeBtn || this._findActiveButton();
                if (btn) {
                    btn.style.display = '';
                    btn.textContent = '⏸';
                    btn.title = 'Pause narration';
                    this._activeBtn = btn;
                }
            }

            if (window.answerPreferenceManager && window.answerPreferenceManager.currentMode === 'audio_audio') {
                window.answerPreferenceManager.setVoicePanelState('speaking');
            }

            if (this.isPaused) {
                await this._waitForResume();
            }

            if (audio_blob_url) {
                await this._playAudioUrl(audio_blob_url);
            } else if (chunk.isBrowserTTS) {
                await this._speakBrowser(chunk.sanitized_text);
            }
        } else if (this.dryRun) {
            // Silence simulation
            await new Promise(r => setTimeout(r, 300));
        }
        // If no audio_blob_url and not dryRun, text was already displayed — just continue

        const elapsed = Math.round(performance.now() - playStart);
        console.log(`[PLAYBACK] Chunk #${chunk_id} finished in ${elapsed}ms`);

        chunk.status = 'done';
    }

    _playAudioUrl(url) {
        return new Promise((resolve) => {
            const audio = new Audio(url);
            this.currentAudio = audio;

            audio.onended = () => {
                URL.revokeObjectURL(url);
                this.currentAudio = null;
                resolve();
            };

            audio.onerror = (e) => {
                console.error('[ERROR] Audio playback error:', e);
                URL.revokeObjectURL(url);
                this.currentAudio = null;
                resolve();
            };

            audio.play().catch((err) => {
                console.error('[ERROR] Audio play() rejected:', err);
                resolve();
            });
        });
    }

    _stopCurrentAudio() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            try { URL.revokeObjectURL(this.currentAudio.src); } catch (_) {}
            this.currentAudio = null;
        }
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }
    }
}

// ── Auto-initialize on DOMContentLoaded ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    window.ttsPipeline = new StreamingAudioPipeline({
        sentenceThreshold: 2,
        charThreshold: 300,
        dryRun: false
    });
    console.log('[STREAM] window.ttsPipeline ready.');
});
