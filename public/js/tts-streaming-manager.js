/**
 * tts-streaming-manager.js
 * ─────────────────────────────────────────────────────────────────────────────
 * PlaybackController + StreamingAudioPipeline
 *
 * PlaybackController — Single source of truth for all playback state.
 *   No other component may independently change button icons or voice panel.
 *
 * StreamingAudioPipeline — Chunked streaming audio delivery for Tutor Mode
 *   and AI Voice Mode.
 *
 * Architecture:
 *   Gemini SSE tokens → textBuffer → _createChunk() →
 *     renderQueue (gated by display_allowed / isPaused)
 *     fetchQueue  (TTS fetch)
 *     deliveryQueue (strict-order playback)
 *
 *   Text renders IN SYNC with audio (live teacher experience).
 *   Pause only pauses audio — text continues rendering, follow-ups appear.
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ═══════════════════════════════════════════════════════════════════════════════
// PlaybackController — SINGLE SOURCE OF TRUTH
// ═══════════════════════════════════════════════════════════════════════════════

class PlaybackController {
    constructor() {
        this.isPlaying = false;
        this.isPaused = false;
        this.isStopped = true;
        this.currentNarrationId = null; // AI card button element
        this.currentEngine = null;      // 'pipeline' | 'manager'
        this.playbackStatus = 'idle';   // 'idle' | 'speaking' | 'paused'
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
        console.log('[PLAYBACK CONTROLLER] State changed:', state.playbackStatus);
        this.subscribers.forEach(sub => {
            try {
                sub(state);
            } catch (err) {
                console.error('[PLAYBACK CONTROLLER] Subscriber error:', err);
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

    // ── Stop everything ──────────────────────────────────────────────────────

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
        this.notify();
    }

    // ── Pipeline (streaming) controls ────────────────────────────────────────

    startPipeline(button = null) {
        this.stopAll();
        this.setState({
            isPlaying: true,
            isPaused: false,
            isStopped: false,
            currentNarrationId: button,
            currentEngine: 'pipeline',
            playbackStatus: 'speaking'
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

    // ── Manager (static card read-aloud) controls ────────────────────────────

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
                console.error('[PLAYBACK CONTROLLER] Resume error:', err);
            });
        } else if (window.speechSynthesis && window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
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
}

// ═══════════════════════════════════════════════════════════════════════════════
// StreamingAudioPipeline — Live Teacher Experience
// ═══════════════════════════════════════════════════════════════════════════════

class StreamingAudioPipeline {
    constructor(options = {}) {
        // ── Configuration ────────────────────────────────────────────────────
        this.sentenceThreshold = options.sentenceThreshold || 2;
        this.charThreshold = options.charThreshold || 300;
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
        this.streamCompleted = false;

        // ── Play/Pause Control & State ───────────────────────────────────────
        this.isPaused = false;
        this.hasStartedAudio = false;
        this._activeBtn = null;

        // ── Display callbacks (set by external code) ─────────────────────────
        this.onDisplayChunk = options.onDisplayChunk || null;
        this.onRenderComplete = options.onRenderComplete || null;
        this.onComplete = options.onComplete || null;

        // ── Regex for sentence detection ─────────────────────────────────────
        this._sentenceRe = /[.!?।]+\s/g;

        console.log(`[STREAM] StreamingAudioPipeline initialized | sentenceThreshold=${this.sentenceThreshold} | charThreshold=${this.charThreshold} | dryRun=${this.dryRun}`);
    }

    // ── Public API ────────────────────────────────────────────────────────────

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
        console.log('[STREAM] Pipeline started');
    }

    pushToken(token) {
        if (!this.isActive) return;
        this.textBuffer += token;
        this._tryFlush();
    }

    flush() {
        if (!this.isActive) return;
        const remaining = this.textBuffer.trim();
        if (remaining) {
            this._createChunk(remaining);
            this.textBuffer = '';
        }
        this.streamCompleted = true;
        console.log('[STREAM] Final flush — stream complete');

        // Trigger render queue in case it needs to check completion
        this._processRenderQueue();
    }

    stop() {
        // Flush all unrendered text to screen so the answer is never lost
        while (this.renderQueue && this.renderQueue.length > 0) {
            const chunk = this.renderQueue.shift();
            if (chunk && !chunk.text_displayed) {
                if (typeof this.onDisplayChunk === 'function') {
                    this.onDisplayChunk(chunk.text_chunk, chunk.chunk_id);
                }
                chunk.text_displayed = true;
            }
        }
        while (this.deliveryQueue && this.deliveryQueue.length > 0) {
            const chunk = this.deliveryQueue.shift();
            if (chunk && !chunk.text_displayed) {
                if (typeof this.onDisplayChunk === 'function') {
                    this.onDisplayChunk(chunk.text_chunk, chunk.chunk_id);
                }
                chunk.text_displayed = true;
            }
        }
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

        // Pause active audio engine only
        if (this.currentAudio) {
            this.currentAudio.pause();
        } else if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.pause();
        }

        // Trigger rendering loop so it immediately flushes queued chunks when paused
        this._processRenderQueue();
        console.log('[STREAM] Audio paused');
    }

    resume() {
        if (!this.isActive || !this.isPaused) return;
        this.isPaused = false;

        // Resume active audio engine
        if (this.currentAudio) {
            this.currentAudio.play().catch(err => {
                console.error('[STREAM] Error resuming audio:', err);
            });
        } else if (window.speechSynthesis && window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
        }
        console.log('[STREAM] Audio resumed');
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

            if (window.ttsManager && window.ttsManager.language) {
                utterance.lang = window.ttsManager.language;
            }

            let resolved = false;
            const safeResolve = () => {
                if (!resolved) {
                    resolved = true;
                    clearTimeout(timeoutId);
                    resolve();
                }
            };

            utterance.onend = () => { safeResolve(); };
            utterance.onerror = (e) => {
                console.error('[STREAM] Browser TTS error:', e);
                safeResolve();
            };

            // Safety timeout: 80ms per character safety limit, minimum 5 seconds
            const duration = Math.max(5000, text.length * 80);
            const timeoutId = setTimeout(() => {
                console.warn('[STREAM] Browser TTS safety timeout triggered for chunk');
                safeResolve();
            }, duration);

            window.speechSynthesis.speak(utterance);
        });
    }

    // ── Internal: Buffer Management ───────────────────────────────────────────

    _tryFlush() {
        // Condition B: character count threshold
        if (this.textBuffer.length >= this.charThreshold) {
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
        clean = clean.replace(/\*\*/g, '');
        clean = clean.replace(/\*/g, '');
        clean = clean.replace(/__/g, '');
        clean = clean.replace(/_/g, '');
        clean = clean.replace(/^\s*#+\s+/gm, '');
        clean = clean.replace(/^\s*[\*\-\•]\s+/gm, '');
        clean = clean.replace(/:\s*$/gm, '.');
        clean = clean.replace(/(\w+):\s/g, '$1. ');
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

        // Kick off independent rendering loop
        this._processRenderQueue();

        // Kick off TTS fetch if not already running
        if (!this.isFetchingTTS) {
            this._processFetchQueue();
        }
    }

    // ── Internal: Render Queue (Gated Text Display) ──────────────────────────

    async _processRenderQueue() {
        if (this.isProcessingRender) return;
        this.isProcessingRender = true;

        while (this.isActive && this.renderQueue.length > 0) {
            const chunk = this.renderQueue[0];

            // Render immediately if paused (text must keep flowing),
            // or wait until audio playback sets display_allowed
            if (this.isPaused || chunk.display_allowed) {
                this.renderQueue.shift();
                if (!chunk.text_displayed) {
                    if (typeof this.onDisplayChunk === 'function') {
                        this.onDisplayChunk(chunk.text_chunk, chunk.chunk_id);
                    }
                    chunk.text_displayed = true;
                }
            } else {
                // Wait a short bit and check again
                await new Promise(resolve => setTimeout(resolve, 30));
            }
        }

        this.isProcessingRender = false;

        // Check if rendering is completely finished and stream is completed
        if (this.isActive && this.streamCompleted && this.renderQueue.length === 0 && this.textBuffer.length === 0) {
            if (typeof this.onRenderComplete === 'function') {
                this.onRenderComplete();
            }
        }
    }

    // ── Internal: TTS Fetching ────────────────────────────────────────────────

    async _processFetchQueue() {
        if (this.fetchQueue.length === 0) {
            this.isFetchingTTS = false;
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
        const { chunk_id } = chunk;
        const fetchStart = performance.now();

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
            console.log(`[TTS] DRY RUN — skipping Sarvam for Chunk #${chunk_id}`);
            await new Promise(r => setTimeout(r, 200));
            chunk.audio_blob_url = null;
            chunk.status = 'ready';
            const elapsed = Math.round(performance.now() - fetchStart);
            console.log(`[LATENCY] Chunk #${chunk_id} TTS (dry run): ${elapsed}ms`);
            this._onChunkReady(chunk);
            return;
        }

        console.log(`[TTS] Sending Chunk #${chunk_id} To Sarvam | ${chunk.sanitized_text.length} chars`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.warn(`[TTS] Fetch timeout triggered for Chunk #${chunk_id}`);
            controller.abort();
        }, 6000); // 6 seconds timeout

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
            console.log(`[LATENCY] Chunk #${chunk_id} TTS: ${elapsed}ms`);

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
            clearTimeout(timeoutId);
            console.error(`[ERROR] TTS fetch failed for Chunk #${chunk_id}:`, err);
            chunk.status = 'ready';
            chunk.audio_blob_url = null;
            this._onChunkReady(chunk);
        }
    }

    _onChunkReady(chunk) {
        console.log(`[QUEUE] Chunk #${chunk.chunk_id} Ready`);
        if (!this.isProcessingPlayback) {
            this._processPlaybackQueue();
        }
    }

    // ── Internal: Playback Queue (Strict Ordering) ────────────────────────────

    async _processPlaybackQueue() {
        if (this.isProcessingPlayback) return;
        this.isProcessingPlayback = true;

        while (this.deliveryQueue.length > 0) {
            const chunk = this.deliveryQueue[0];

            if (chunk.status !== 'ready' && chunk.status !== 'done') {
                const queueWaitStart = performance.now();
                await this._waitForChunkReady(chunk);
                const queueWait = Math.round(performance.now() - queueWaitStart);
                if (queueWait > 10) {
                    console.log(`[LATENCY] Chunk #${chunk.chunk_id} Queue Wait: ${queueWait}ms`);
                }
            }

            this.deliveryQueue.shift();
            await this._playChunk(chunk);
        }

        this.isProcessingPlayback = false;

        // Notify caller that all audio chunks are done
        if (typeof this.onComplete === 'function') {
            this.onComplete();
        }
    }

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

    async _playChunk(chunk) {
        const { chunk_id, audio_blob_url } = chunk;

        // Wait if playback is paused before playing this chunk's audio
        if (this.isPaused) {
            await this._waitForResume();
        }

        chunk.status = 'playing';
        const playStart = performance.now();

        console.log(`[PLAYBACK] Playing Chunk #${chunk_id}`);

        // Allow the render queue to display this chunk's text
        chunk.display_allowed = true;

        // Trigger rendering process in case it is waiting
        this._processRenderQueue();

        // Play audio (if available)
        if (audio_blob_url || chunk.isBrowserTTS) {
            if (!this.hasStartedAudio) {
                this.hasStartedAudio = true;
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
            await new Promise(r => setTimeout(r, 300));
        }

        const elapsed = Math.round(performance.now() - playStart);
        console.log(`[PLAYBACK] Chunk #${chunk_id} finished in ${elapsed}ms`);

        chunk.status = 'done';
    }

    _playAudioUrl(url) {
        return new Promise((resolve) => {
            const audio = new Audio(url);
            this.currentAudio = audio;

            let resolved = false;
            const safeResolve = () => {
                if (!resolved) {
                    resolved = true;
                    clearTimeout(timeoutId);
                    URL.revokeObjectURL(url);
                    this.currentAudio = null;
                    resolve();
                }
            };

            audio.onended = () => {
                safeResolve();
            };

            audio.onerror = (e) => {
                console.error('[ERROR] Audio playback error:', e);
                safeResolve();
            };

            // Safety timeout: 30 seconds
            const timeoutId = setTimeout(() => {
                console.warn('[STREAM] Audio play safety timeout triggered');
                safeResolve();
            }, 30000);

            audio.play().catch((err) => {
                console.error('[ERROR] Audio play() rejected:', err);
                safeResolve();
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
    window.playbackController = new PlaybackController();

    window.ttsPipeline = new StreamingAudioPipeline({
        sentenceThreshold: 2,
        charThreshold: 300,
        dryRun: false
    });
    console.log('[STREAM] window.ttsPipeline and window.playbackController ready.');
});
