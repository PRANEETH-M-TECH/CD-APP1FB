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

class PlaybackController {
    constructor() {
        this.isPlaying = false;
        this.isPaused = false;
        this.isStopped = true;
        this.currentNarrationId = null; // AI card button element
        this.currentEngine = null; // 'pipeline' | 'manager'
        this.playbackStatus = 'idle'; // 'idle' | 'speaking' | 'paused' | 'stopped'
        this.subscribers = [];
    }

    subscribe(callback) {
        if (typeof callback === 'function') {
            this.subscribers.push(callback);
        }
    }

    unsubscribe(callback) {
        this.subscribers = this.subscribers.filter(sub => sub !== callback);
    }

    notify() {
        const state = {
            isPlaying: this.isPlaying,
            isPaused: this.isPaused,
            isStopped: this.isStopped,
            currentNarrationId: this.currentNarrationId,
            currentEngine: this.currentEngine,
            playbackStatus: this.playbackStatus
        };
        console.log('[PLAYBACK CONTROLLER] State changed:', state);
        this.subscribers.forEach(sub => {
            try {
                sub(state);
            } catch (err) {
                console.error('[PLAYBACK CONTROLLER] Error notifying subscriber:', err);
            }
        });
    }

    setState(newState) {
        let changed = false;
        for (let key in newState) {
            if (this[key] !== newState[key]) {
                this[key] = newState[key];
                changed = true;
            }
        }
        if (changed) {
            this.notify();
        }
    }

    stopAll() {
        console.log('[PLAYBACK CONTROLLER] stopAll triggered');
        
        // Stop Streaming Pipeline if active
        if (window.ttsPipeline && window.ttsPipeline.isActive) {
            window.ttsPipeline._stopCurrentAudio();
            window.ttsPipeline.isActive = false;
            window.ttsPipeline.isPaused = false;
            window.ttsPipeline.hasStartedAudio = false;
            window.ttsPipeline.fetchQueue = [];
            window.ttsPipeline.deliveryQueue = [];
            window.ttsPipeline.renderQueue = [];
            window.ttsPipeline.textBuffer = '';
            window.ttsPipeline.isProcessingPlayback = false;
            window.ttsPipeline.isProcessingRender = false;
            window.ttsPipeline.isFetchingTTS = false;
        }

        // Stop Static TTS Manager if active
        if (window.ttsManager) {
            window.ttsManager.fetchQueue = [];
            window.ttsManager.playbackQueue = [];
            window.ttsManager.isFetching = false;
            window.ttsManager.isPlayingQueue = false;
            if (window.ttsManager.currentAudio) {
                window.ttsManager.currentAudio.pause();
                window.ttsManager.currentAudio.currentTime = 0;
                window.ttsManager.currentAudio = null;
            }
            window.ttsManager.isSpeaking = false;
        }

        // Direct SpeechSynthesis cancellation
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }

        this.isPlaying = false;
        this.isPaused = false;
        this.isStopped = true;
        this.currentNarrationId = null;
        this.currentEngine = null;
        this.playbackStatus = 'idle';
    }

    startPipeline(button = null) {
        this.stopAll();
        this.setState({
            isPlaying: true,
            isPaused: false,
            isStopped: false,
            currentNarrationId: button,
            currentEngine: 'pipeline',
            playbackStatus: 'idle'
        });
        if (window.ttsPipeline) {
            window.ttsPipeline._activeBtn = button;
            window.ttsPipeline.start();
        }
    }

    pausePipeline() {
        if (this.currentEngine !== 'pipeline' || this.isPaused) return;
        this.setState({
            isPaused: true,
            playbackStatus: 'paused'
        });
        if (window.ttsPipeline) {
            window.ttsPipeline.pause();
        }
    }

    resumePipeline() {
        if (this.currentEngine !== 'pipeline' || !this.isPaused) return;
        this.setState({
            isPaused: false,
            playbackStatus: 'speaking'
        });
        if (window.ttsPipeline) {
            window.ttsPipeline.resume();
        }
    }

    stopPipeline() {
        if (this.currentEngine !== 'pipeline') return;
        if (window.ttsPipeline) {
            window.ttsPipeline.stop();
        }
        this.setState({
            isPlaying: false,
            isPaused: false,
            isStopped: true,
            currentNarrationId: null,
            currentEngine: null,
            playbackStatus: 'idle'
        });
    }

    startManager(text, button) {
        this.stopAll();
        this.setState({
            isPlaying: true,
            isPaused: false,
            isStopped: false,
            currentNarrationId: button,
            currentEngine: 'manager',
            playbackStatus: 'speaking'
        });
        if (window.ttsManager) {
            window.ttsManager.speak(text, button);
        }
    }

    stopManager() {
        if (this.currentEngine !== 'manager') return;
        if (window.ttsManager) {
            window.ttsManager.stop();
        }
        this.setState({
            isPlaying: false,
            isPaused: false,
            isStopped: true,
            currentNarrationId: null,
            currentEngine: null,
            playbackStatus: 'idle'
        });
    }

    pauseManager() {
        if (this.currentEngine !== 'manager' || this.isPaused) return;
        this.setState({
            isPaused: true,
            playbackStatus: 'paused'
        });
        if (window.ttsManager && window.ttsManager.currentAudio) {
            window.ttsManager.currentAudio.pause();
        } else if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.pause();
        }
    }

    resumeManager() {
        if (this.currentEngine !== 'manager' || !this.isPaused) return;
        this.setState({
            isPaused: false,
            playbackStatus: 'speaking'
        });
        if (window.ttsManager && window.ttsManager.currentAudio) {
            window.ttsManager.currentAudio.play().catch(err => {
                console.error('[PLAYBACK CONTROLLER] Error resuming static audio:', err);
            });
        } else if (window.speechSynthesis && window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
        }
    }
}

class StreamingAudioPipeline {
    constructor(options = {}) {
        // ── Configuration ────────────────────────────────────────────────────
        // Flush when this many complete sentences are buffered
        this.sentenceThreshold = options.sentenceThreshold || 1;
        // Flush when buffer reaches this many characters (Adjustment 2)
        this.charThreshold = options.charThreshold || 300;
        // If true, skip real Sarvam calls (credit protection during testing)
        this.dryRun = options.dryRun || false;

        // ── Internal State ───────────────────────────────────────────────────
        this.chunkIdCounter = 0;
        this.textBuffer = '';
        this.deliveryQueue = [];       // { chunk_id, text_chunk, audio_blob_url, status }
        this.renderQueue = [];         // text chunks waiting to be rendered
        this.fetchQueue = [];          // chunks waiting for TTS fetch
        this.isProcessingPlayback = false;
        this.isProcessingRender = false;
        this.isFetchingTTS = false;
        this.isActive = false;
        this.currentAudio = null;
        this.abortController = null;
        this.streamCompleted = false;

        // ── Play/Pause Control & State ───────────────────────────────────────
        this.isPaused = false;
        this.hasStartedAudio = false;
        this._activeBtn = null;

        // ── Display callback (set by external code) ──────────────────────────
        // Called with (text_chunk, chunk_id) when a chunk is ready to display
        this.onDisplayChunk = options.onDisplayChunk || null;
        // Called when all text rendering is complete
        this.onRenderComplete = options.onRenderComplete || null;
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
        this.renderQueue = [];
        this.fetchQueue = [];
        this.isProcessingPlayback = false;
        this.isProcessingRender = false;
        this.isFetchingTTS = false;
        this.isActive = true;
        this.isPaused = false;
        this.hasStartedAudio = false;
        this.streamCompleted = false;
        this._activeBtn = null;
        this._stopCurrentAudio();
        console.log('[STREAM] Gemini Stream Started');

        // Watchdog: If no chunks render within 8 seconds, force-display them to prevent "..." stall.
        if (this._watchdogTimer) clearTimeout(this._watchdogTimer);
        this._watchdogTimer = setTimeout(() => {
            if (this.isActive && this.renderQueue.length > 0) {
                const firstChunk = this.renderQueue[0];
                if (firstChunk && !firstChunk.text_displayed) {
                    console.warn('[WATCHDOG] 8-second watchdog fired! No text was rendered. Forcing all chunks to display.');
                    this.renderQueue.forEach(c => c.display_allowed = true);
                    this._processRenderQueue();
                }
            }
        }, 8000);
    }

    /**
     * Push a raw token from the Gemini SSE stream.
     * @param {string} token
     */
    pushToken(token) {
        if (!this.isActive) {
            console.warn('[STREAM] pushToken called but pipeline was inactive! Force restarting pipeline.');
            this.isActive = true;
        }
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
        this.streamCompleted = true;
        console.log('[STREAM] Gemini Stream Ended — final flush complete');
        
        // Trigger render queue process just in case
        this._processRenderQueue();
    }

    stop() {
        if (this._watchdogTimer) {
            clearTimeout(this._watchdogTimer);
            this._watchdogTimer = null;
        }
        // If there are any chunks in renderQueue that haven't been displayed,
        // render them immediately so the full text is not lost!
        while (this.renderQueue && this.renderQueue.length > 0) {
            const chunk = this.renderQueue.shift();
            if (chunk && !chunk.text_displayed) {
                if (typeof this.onDisplayChunk === 'function') {
                    this.onDisplayChunk(chunk.text_chunk, chunk.chunk_id);
                }
                chunk.text_displayed = true;
            }
        }
        // If there are any chunks in deliveryQueue that haven't been displayed,
        // render them immediately so the full text is not lost!
        while (this.deliveryQueue && this.deliveryQueue.length > 0) {
            const chunk = this.deliveryQueue.shift();
            if (chunk && !chunk.text_displayed) {
                if (typeof this.onDisplayChunk === 'function') {
                    this.onDisplayChunk(chunk.text_chunk, chunk.chunk_id);
                }
                chunk.text_displayed = true;
            }
        }
        // If there is any remaining text in textBuffer, render it too
        const remaining = this.textBuffer.trim();
        if (remaining) {
            if (typeof this.onDisplayChunk === 'function') {
                this.onDisplayChunk(remaining, this.chunkIdCounter + 1);
            }
            this.textBuffer = '';
        }

        this.isActive = false;
        this.isPaused = false;
        this.hasStartedAudio = false;
        this.fetchQueue = [];
        this.deliveryQueue = [];
        this.renderQueue = [];
        this.textBuffer = '';
        this.isProcessingPlayback = false;
        this.isProcessingRender = false;
        this.isFetchingTTS = false;
        this.streamCompleted = true;
        this._stopCurrentAudio();

        // Trigger render-completion callback instantly
        if (typeof this.onRenderComplete === 'function') {
            this.onRenderComplete();
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

        // Trigger rendering loop so it immediately flushes any queued chunks when paused
        this._processRenderQueue();
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

    _speakBrowser(text, chunk = null) {
        return new Promise((resolve) => {
            if (!window.speechSynthesis) {
                console.warn('[STREAM] Browser TTS not available.');
                resolve();
                return;
            }

            const utterance = (chunk && chunk.browserUtterance) ? chunk.browserUtterance : new SpeechSynthesisUtterance(text);
            
            // Set language if available from ttsManager
            if (!utterance.lang && window.ttsManager && window.ttsManager.language) {
                utterance.lang = window.ttsManager.language;
            }

            const charCount = text ? text.length : 0;
            const timeoutDuration = Math.min(12000, Math.max(3000, charCount * 45));
            let timer = setTimeout(() => {
                console.warn(`[STREAM] Browser TTS speaking timeout (${timeoutDuration}ms) for text: "${text.slice(0, 40)}..."`);
                if (window.speechSynthesis) {
                    window.speechSynthesis.cancel();
                }
                resolve();
            }, timeoutDuration);

            utterance.onend = () => {
                clearTimeout(timer);
                resolve();
            };

            utterance.onerror = (e) => {
                clearTimeout(timer);
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
            text_displayed: false,
            display_allowed: false,
            _createTime: performance.now()
        };
        this.fetchQueue.push(chunk);
        this.deliveryQueue.push(chunk);
        this.renderQueue.push(chunk);

        console.log(`[BUFFER] Chunk #${chunk_id} Created | "${text.slice(0, 60)}..."`);
        console.log(`[STREAM] chunk received | Chunk #${chunk_id}`);

        // Re-couple rendering loop
        this._processRenderQueue();

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
        try {
            const chunk = this.fetchQueue.shift();
            if (chunk) {
                chunk.status = 'fetching';
                await this._fetchTTS(chunk);
            }
        } catch (err) {
            console.error('[STREAM ERROR] Error in fetch queue processing:', err);
        } finally {
            // Process next item
            this._processFetchQueue();
        }
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
            // Pre-warm browser utterance
            try {
                const utterance = new SpeechSynthesisUtterance(chunk.sanitized_text);
                if (window.ttsManager && window.ttsManager.language) {
                    utterance.lang = window.ttsManager.language;
                }
                chunk.browserUtterance = utterance;
                console.log(`[TTS] SpeechSynthesisUtterance pre-warmed for Chunk #${chunk_id}`);
            } catch (e) {
                console.warn(`[TTS] Failed to pre-warm Browser TTS for Chunk #${chunk_id}:`, e);
            }
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

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.warn(`[TTS] Fetch timeout (6s) reached for Chunk #${chunk_id}. Aborting request.`);
            controller.abort();
        }, 6000);

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
                }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

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

                // Pre-warm / Pre-load the Audio object
                try {
                    chunk.audioElement = new Audio(chunk.audio_blob_url);
                    chunk.audioElement.load();
                    console.log(`[TTS] AudioElement pre-warmed for Chunk #${chunk_id}`);
                } catch (e) {
                    console.warn(`[TTS] Failed to pre-warm AudioElement for Chunk #${chunk_id}:`, e);
                }
            }

            chunk.status = 'ready';
            chunk._ttsTime = elapsed;
            this._onChunkReady(chunk);

        } catch (err) {
            clearTimeout(timeoutId);
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

    async _processRenderQueue() {
        if (this.isProcessingRender) return;
        this.isProcessingRender = true;

        try {
            while (this.renderQueue.length > 0) {
                const chunk = this.renderQueue[0];
                
                // If we are paused, we allow displaying all remaining chunks immediately (flush mode)
                // Otherwise, we only display the chunk if display_allowed is true
                if (!this.isPaused && !chunk.display_allowed) {
                    break;
                }

                // Dequeue
                this.renderQueue.shift();

                if (!chunk.text_displayed) {
                    if (this._watchdogTimer) {
                        clearTimeout(this._watchdogTimer);
                        this._watchdogTimer = null;
                    }
                    if (typeof this.onDisplayChunk === 'function') {
                        this.onDisplayChunk(chunk.text_chunk, chunk.chunk_id);
                    }
                    chunk.text_displayed = true;
                }
            }
        } catch (err) {
            console.error('[STREAM ERROR] Error in render queue:', err);
        } finally {
            this.isProcessingRender = false;
        }
        
        // If the entire stream is completed and all rendering is complete, notify
        if (this.streamCompleted && this.renderQueue.length === 0) {
            if (typeof this.onRenderComplete === 'function') {
                try {
                    this.onRenderComplete();
                } catch (err) {
                    console.error('[STREAM ERROR] Error in onRenderComplete callback:', err);
                }
            }
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

        try {
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
        } catch (err) {
            console.error('[STREAM ERROR] Error in playback queue:', err);
        } finally {
            this.isProcessingPlayback = false;
        }

        // Notify caller that all chunks are done
        if (typeof this.onComplete === 'function') {
            try {
                this.onComplete();
            } catch (err) {
                console.error('[STREAM ERROR] Error in onComplete callback:', err);
            }
        }
    }

    /**
     * Poll until chunk.status === 'ready' (TTS arrived) or pipeline stopped.
     */
    _waitForChunkReady(chunk) {
        const startTime = performance.now();
        return new Promise((resolve) => {
            const check = () => {
                if (!this.isActive) { resolve(); return; }
                if (chunk.status === 'ready' || chunk.status === 'done') { resolve(); return; }
                // Timeout after 5 seconds to prevent blocking rendering and playback queues
                if (performance.now() - startTime > 5000) {
                    console.warn(`[STREAM WARNING] Timeout waiting for Chunk #${chunk.chunk_id} to become ready. Status was: ${chunk.status}. Forcing to ready.`);
                    chunk.status = 'ready';
                    chunk.audio_blob_url = null;
                    resolve();
                    return;
                }
                setTimeout(check, 30);
            };
            check();
        });
    }

    async _playChunk(chunk) {
        if (!chunk) return;
        const { chunk_id, text_chunk, audio_blob_url } = chunk;

        try {
            // Wait if playback is paused before playing audio
            if (this.isPaused) {
                await this._waitForResume();
            }

            chunk.status = 'playing';
            const playStart = performance.now();

            console.log(`[PLAYBACK] Playing Chunk #${chunk_id}`);
            console.log(`[AUDIO] chunk playing | Chunk #${chunk_id}`);

            // Allow independent text rendering loop to display this chunk
            chunk.display_allowed = true;
            this._processRenderQueue();

            if (!this.hasStartedAudio) {
                this.hasStartedAudio = true;
                if (window.playbackController && window.playbackController.currentEngine === 'pipeline') {
                    window.playbackController.setState({ playbackStatus: 'speaking' });
                }
            }

            // 2. Play audio (if available)
            if (audio_blob_url || chunk.isBrowserTTS) {
                if (this.isPaused) {
                    await this._waitForResume();
                }

                if (audio_blob_url) {
                    await this._playAudioUrl(audio_blob_url, chunk);
                } else if (chunk.isBrowserTTS) {
                    await this._speakBrowser(chunk.sanitized_text, chunk);
                }
            } else if (this.dryRun) {
                // Silence simulation
                await new Promise(r => setTimeout(r, 300));
            }

            const elapsed = Math.round(performance.now() - playStart);
            console.log(`[PLAYBACK] Chunk #${chunk_id} finished in ${elapsed}ms`);
            console.log(`[AUDIO] chunk completed | Chunk #${chunk_id}`);
        } catch (err) {
            console.error(`[STREAM ERROR] Error playing Chunk #${chunk_id}:`, err);
        } finally {
            chunk.status = 'done';
        }
    }

    _playAudioUrl(url, chunk = null) {
        return new Promise((resolve) => {
            const audio = (chunk && chunk.audioElement) ? chunk.audioElement : new Audio(url);
            this.currentAudio = audio;

            const timer = setTimeout(() => {
                console.warn('[ERROR] Audio playback timeout (30s) reached.');
                cleanup();
                resolve();
            }, 30000);

            const cleanup = () => {
                clearTimeout(timer);
                if (audio) {
                    audio.onended = null;
                    audio.onerror = null;
                    audio.pause();
                }
                try {
                    URL.revokeObjectURL(url);
                } catch (_) {}
                if (this.currentAudio === audio) {
                    this.currentAudio = null;
                }
            };

            audio.onended = () => {
                cleanup();
                resolve();
            };

            audio.onerror = (e) => {
                console.error('[ERROR] Audio playback error:', e);
                cleanup();
                resolve();
            };

            try {
                const playPromise = audio.play();
                if (playPromise !== undefined && typeof playPromise.catch === 'function') {
                    playPromise.catch((err) => {
                        console.error('[ERROR] Audio play() rejected:', err);
                        cleanup();
                        resolve();
                    });
                } else {
                    // Older/mock browser fallback
                    console.log('[AUDIO] play() did not return a promise (sync play).');
                }
            } catch (err) {
                console.error('[ERROR] Sync exception in audio.play():', err);
                cleanup();
                resolve();
            }
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
    window.playbackController = new PlaybackController();

    window.ttsPipeline = new StreamingAudioPipeline({
        sentenceThreshold: 1,
        charThreshold: 300,
        dryRun: false
    });
    console.log('[STREAM] window.ttsPipeline and window.playbackController ready.');
});
