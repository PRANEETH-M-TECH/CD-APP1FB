/**
 * Visual Learning Mode Controller for CHADUVU-GURU (V2 Continuous Canvas-Driven Engine)
 * Orchestrates storyboard creation, scene asset rendering (GSAP + Lottie),
 * audio playback synchronization, aspect ratio scaling, player states,
 * virtual camera pans/zooms, and dynamic SVG connectors drawing.
 * Enriched with detailed debug logging.
 */

const LOTTIE_PRESETS = {
    water_drops: 'https://assets2.lottiefiles.com/packages/lf20_0yupmguv.json',
    clouds: 'https://assets5.lottiefiles.com/packages/lf20_yqyw4b5c.json',
    arrows: 'https://assets5.lottiefiles.com/packages/lf20_96b5ltgj.json'
};

function loadLottieAsset(container, presetName) {
    const url = LOTTIE_PRESETS[presetName] || LOTTIE_PRESETS['clouds'];
    console.log(`[Lottie LOG] Attempting to load Lottie preset: "${presetName}" from URL: ${url}`);
    try {
        const anim = lottie.loadAnimation({
            container: container,
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: url
        });
        
        anim.addEventListener('DOMLoaded', () => {
            console.log(`[Lottie LOG] Successfully loaded Lottie JSON for: "${presetName}"`);
        });

        anim.addEventListener('data_failed', () => {
            console.error(`[Lottie ERROR] Failed to load JSON data for preset: "${presetName}". Triggering SVG fallback.`);
            renderLottieFallback(container, presetName);
        });
        
        return anim;
    } catch (e) {
        console.error(`[Lottie EXCEPTION] Error loading preset "${presetName}":`, e);
        renderLottieFallback(container, presetName);
        return null;
    }
}

function renderLottieFallback(container, presetName) {
    console.warn(`[Lottie LOG] Rendering static SVG animation fallback for preset: "${presetName}"`);
    container.innerHTML = '';
    let svg = '';
    if (presetName === 'water_drops') {
        svg = `<svg viewBox="0 0 24 24" width="100%" height="100%" fill="#60a5fa" class="animate-pulse" style="animation-duration: 2s;">
            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>
        </svg>`;
    } else if (presetName === 'arrows') {
        svg = `<svg viewBox="0 0 24 24" width="100%" height="100%" fill="#a78bfa" stroke="#a78bfa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <polyline points="19 12 12 19 5 12"></polyline>
        </svg>`;
    } else { // clouds
        svg = `<svg viewBox="0 0 24 24" width="100%" height="100%" fill="#94a3b8" class="animate-bounce" style="animation-duration: 4s;">
            <path d="M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z"/>
        </svg>`;
    }
    container.innerHTML = svg;
}

class VisualLearningController {
    constructor() {
        console.log("[VisualLearning LOG] Initializing VisualLearningController...");
        this.lessonPackage = null;
        this.currentSlideIndex = 0; // index of the active scene
        this.isPlaying = false;
        this.audio = null;
        this.timeline = null;
        this.lottieInstancesMap = {};
        this.renderedAssets = {};
        
        // Sync flags
        this.timelineFinished = false;
        this.audioFinished = false;
        this.isSeeking = false;
        
        this.preloadedImages = {};
        this.preloadedAudios = {};
        this.clipStartTimes = [];
        this.totalDuration = 0;
        
        this.activeProgressSteps = [
            'understanding_topic',
            'designing_lesson',
            'generating_visuals',
            'creating_narration',
            'hyperframes_engine',
            'launching_lesson'
        ];
        
        // DOM Cache
        this.container = null;
        this.loadingScreen = null;
        this.playerUI = null;
        this.canvas = null;
        this.notesOverlay = null;
        this.notesBody = null;
        this.progressBarFill = null;
        this.progressText = null;
        this.slidesList = null;
        this.playPauseBtn = null;
        this.prevBtn = null;
        this.nextBtn = null;
        this.notesBtn = null;
        
        this.initDOMElements();
        
        // Listen for IPC events from Hyperframes Engine inside iframe
        window.addEventListener('message', (e) => this.handleIframeEvent(e));

        // Dynamic Canvas Scaling
        window.addEventListener('resize', () => {
            console.log("[VisualLearning LOG] Window resized. Re-calculating canvas scale.");
            this.scaleIframe();
        });
    }

    initDOMElements() {
        console.log("[VisualLearning LOG] Binding DOM elements and click listeners.");
        this.container = document.getElementById('visual-learning-container');
        this.loadingScreen = document.getElementById('vl-loading-screen');
        this.playerUI = document.getElementById('vl-player');
        this.canvas = document.getElementById('vl-canvas');
        this.notesOverlay = document.getElementById('vl-notes-overlay');
        this.notesBody = document.getElementById('vl-notes-body');
        this.progressBarFill = document.getElementById('vl-progress-bar-fill');
        this.progressText = document.getElementById('vl-progress-text');
        this.slidesList = document.getElementById('vl-slides-list');
        
        this.playPauseBtn = document.getElementById('vl-play-pause-btn');
        this.prevBtn = document.getElementById('vl-prev-btn');
        this.nextBtn = document.getElementById('vl-next-btn');
        this.notesBtn = document.getElementById('vl-notes-btn');
        
        if (this.playPauseBtn) this.playPauseBtn.onclick = () => this.togglePlay();
        if (this.prevBtn) this.prevBtn.onclick = () => this.previousSlide();
        if (this.nextBtn) this.nextBtn.onclick = () => this.nextSlide();
        
        const closeNotes = document.getElementById('vl-close-notes-btn');
        if (closeNotes) closeNotes.onclick = () => this.toggleNotes(false);
        if (this.notesBtn) this.notesBtn.onclick = () => this.toggleNotes();
        
        const notesToggleBtn = document.getElementById('vl-notes-toggle-btn');
        if (notesToggleBtn) notesToggleBtn.onclick = () => this.toggleNotes();

        const exitBtn = document.getElementById('vl-exit-btn');
        if (exitBtn) exitBtn.onclick = () => this.destroyLesson();
        
        const errorExitBtn = document.getElementById('vl-error-exit-btn');
        if (errorExitBtn) errorExitBtn.onclick = () => this.destroyLesson();

        // Controls: Replay, Mute, Speed, Fullscreen
        const replayBtn = document.getElementById('vl-replay-btn');
        if (replayBtn) replayBtn.onclick = () => {
            console.log("[VisualLearning LOG] Replay lesson triggered.");
            this.sendCommandToIframe('RESTART');
        };

        this.isMuted = false;
        const muteBtn = document.getElementById('vl-mute-btn');
        const muteIcon = document.getElementById('vl-mute-icon');
        if (muteBtn) muteBtn.onclick = () => {
            this.isMuted = !this.isMuted;
            this.sendCommandToIframe('TOGGLE_MUTE', { isMuted: this.isMuted });
            if (muteIcon) muteIcon.textContent = this.isMuted ? '🔇' : '🔊';
        };

        this.playbackSpeeds = [1.0, 1.25, 1.5, 2.0, 0.75];
        this.speedIndex = 0;
        const speedBtn = document.getElementById('vl-speed-btn');
        if (speedBtn) speedBtn.onclick = () => {
            this.speedIndex = (this.speedIndex + 1) % this.playbackSpeeds.length;
            const speed = this.playbackSpeeds[this.speedIndex];
            speedBtn.innerHTML = `<span>${speed}x</span>`;
            this.sendCommandToIframe('SET_PLAYBACK_RATE', { rate: speed });
        };

        const fullscreenBtn = document.getElementById('vl-fullscreen-btn');
        if (fullscreenBtn) fullscreenBtn.onclick = () => {
            const player = document.getElementById('vl-player') || document.getElementById('vl-video-container');
            if (!document.fullscreenElement) {
                if (player.requestFullscreen) player.requestFullscreen();
                else if (player.webkitRequestFullscreen) player.webkitRequestFullscreen();
            } else {
                if (document.exitFullscreen) document.exitFullscreen();
            }
        };

        // Wire AI Tool Chips
        const chips = document.querySelectorAll('.vl-ai-chip');
        chips.forEach(chip => {
            chip.onclick = () => {
                const label = chip.textContent.trim();
                console.log(`[VisualLearning LOG] AI Tool Chip clicked: ${label}`);
                this.showAIToolToast(label);
            };
        });

        const tabScriptBtn = document.getElementById('vl-tab-script');
        const tabOutlineBtn = document.getElementById('vl-tab-outline');
        const scriptContent = document.getElementById('vl-drawer-script-content');
        const outlineContent = document.getElementById('vl-drawer-outline-content');

        if (tabScriptBtn && tabOutlineBtn && scriptContent && outlineContent) {
            tabScriptBtn.onclick = () => {
                tabScriptBtn.classList.add('active');
                tabOutlineBtn.classList.remove('active');
                scriptContent.style.display = 'block';
                outlineContent.style.display = 'none';
            };
            tabOutlineBtn.onclick = () => {
                tabOutlineBtn.classList.add('active');
                tabScriptBtn.classList.remove('active');
                scriptContent.style.display = 'none';
                outlineContent.style.display = 'block';
            };
        }

        // Wire seek bar container click
        const progressBarContainer = document.querySelector('.vl-progress-bar-container');
        if (progressBarContainer) {
            progressBarContainer.onclick = (e) => this.handleProgressBarClick(e);
            progressBarContainer.style.cursor = 'pointer';
        }
    }

    resizeCanvas() {
        const container = document.getElementById('vl-main');
        if (!this.canvas || !container) {
            console.warn("[VisualLearning WARNING] Canvas or Main Container element missing during resize.");
            return;
        }
        
        this.canvas.style.width = '800px';
        this.canvas.style.height = '450px';
        
        // If a lesson is active, re-focus camera to the current scene's camera coordinates
        if (this.lessonPackage && this.lessonPackage.scenes) {
            const scene = this.lessonPackage.scenes[this.currentSlideIndex];
            if (scene) {
                const cam = scene.camera || { focus_x: 50, focus_y: 50, zoom: 1.0 };
                this.focusCamera(cam.focus_x, cam.focus_y, cam.zoom, 0); // instant update
            }
        } else {
            // Default center
            this.focusCamera(50, 50, 1.0, 0);
        }
        console.log("[VisualLearning LOG] Canvas resized and refocused.");
    }

    async startLesson(query) {
        console.log(`[VisualLearning LOG] startLesson triggered for question: "${query}"`);
        
        if (!window.selectedBook) {
            console.error("[VisualLearning ERROR] Lesson start aborted: No book selected.");
            alert("Please select a Class and Subject from the dropdown list first!");
            return;
        }
        
        const errContainer = document.getElementById('vl-error-container');
        const errMessage = document.getElementById('vl-error-message');
        const loadingTitle = document.getElementById('vl-loading-title');
        const loadingSpinner = document.getElementById('vl-loading-spinner');
        const loadingSteps = document.getElementById('vl-loading-steps-container');
        
        if (errContainer) errContainer.classList.add('hidden');
        if (loadingTitle) loadingTitle.style.display = '';
        if (loadingSpinner) loadingSpinner.style.display = '';
        if (loadingSteps) loadingSteps.style.display = '';

        this.resetProgressUI();
        
        if (this.container) this.container.style.display = 'flex';
        if (this.loadingScreen) this.loadingScreen.style.display = 'flex';
        if (this.playerUI) this.playerUI.style.display = 'none';
        
        const vlLanding = document.getElementById('vl-landing-view');
        if (vlLanding) vlLanding.style.display = 'none';
        
        const chatContainer = document.getElementById('chat-container');
        const followupPanel = document.getElementById('followup-sticky-panel');
        const chatInput = document.getElementById('chat-input-container');
        if (chatContainer) chatContainer.style.display = 'none';
        if (followupPanel) followupPanel.style.display = 'none';
        if (chatInput) chatInput.style.display = 'none';

        try {
            const requestPayload = {
                query: query,
                book_uuid: window.selectedBook.id,
                class_name: String(window.currentUserClass || "8"),
                subject: window.selectedBook.subject
            };
            
            console.log("[VisualLearning LOG] POSTing to /api/visual_learning with payload: ", requestPayload);
            const response = await fetch('/api/visual_learning', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload)
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server returned ${response.status}: ${errText || response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    const cleanLine = line.trim();
                    if (cleanLine.startsWith("data:")) {
                        const dataStr = cleanLine.substring(5).trim();
                        if (dataStr === "[DONE]") {
                            console.log("[VisualLearning LOG] SSE Stream complete ([DONE]).");
                            break;
                        }
                        
                        try {
                            const eventData = JSON.parse(dataStr);
                            await this.handleSSEEvent(eventData);
                        } catch (err) {
                            console.error("[VisualLearning ERROR] Failed to parse SSE payload line: ", cleanLine, err);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("[VisualLearning ERROR] Lesson pipeline execution failed: ", error);
            
            if (loadingTitle) loadingTitle.style.display = 'none';
            if (loadingSpinner) loadingSpinner.style.display = 'none';
            if (loadingSteps) loadingSteps.style.display = 'none';
            
            if (errContainer && errMessage) {
                errMessage.textContent = error.message || "An unexpected error occurred during lesson generation.";
                errContainer.classList.remove('hidden');
            } else {
                alert(`Error: ${error.message}`);
                this.destroyLesson();
            }
        }
    }

    async handleSSEEvent(event) {
        console.log("[VisualLearning LOG] Received SSE Event: ", event.type, event.step || "");
        if (event.type === 'progress') {
            const stepId = event.step;
            const status = event.status;
            
            const stepIndex = this.activeProgressSteps.indexOf(stepId);
            for (let i = 0; i < stepIndex; i++) {
                const prevStep = document.getElementById(`vl-step-${this.activeProgressSteps[i]}`);
                if (prevStep) {
                    prevStep.className = 'vl-loading-step completed';
                    const check = prevStep.querySelector('.vl-step-check');
                    if (check) check.textContent = '✓';
                }
            }

            const activeStepEl = document.getElementById(`vl-step-${stepId}`);
            if (activeStepEl) {
                if (status === 'in_progress') {
                    activeStepEl.className = 'vl-loading-step active';
                } else if (status === 'complete') {
                    activeStepEl.className = 'vl-loading-step completed';
                    const check = activeStepEl.querySelector('.vl-step-check');
                    if (check) check.textContent = '✓';
                }
            }

            const logTextEl = document.getElementById('vl-engine-log-text');
            if (logTextEl && event.message) {
                logTextEl.textContent = event.message;
            }
        } else if (event.type === 'lesson_ready') {
            console.log("[VisualLearning LOG] Lesson package is ready. Starting asset preloading sequence.", event.lesson);
            
            const finalStep = document.getElementById('vl-step-launching_lesson');
            if (finalStep) finalStep.className = 'vl-loading-step active';

            this.lessonPackage = event.lesson;
            await this.preloadLessonAssets();
            this.launchPlayer();
        } else if (event.type === 'error') {
            console.error("[VisualLearning ERROR] Server reported pipeline error: ", event.message);
            throw new Error(event.message);
        }
    }

    resetProgressUI() {
        console.log("[VisualLearning LOG] Resetting loader progress steps.");
        this.activeProgressSteps.forEach((step, idx) => {
            const stepEl = document.getElementById(`vl-step-${step}`);
            if (stepEl) {
                stepEl.className = 'vl-loading-step';
                const check = stepEl.querySelector('.vl-step-check');
                if (check) check.textContent = (idx + 1).toString();
            }
        });
    }

    async preloadLessonAssets() {
        if (this.lessonPackage && (this.lessonPackage.html_url || this.lessonPackage.video_url)) {
            console.log("[VisualLearning LOG] Fast-path active for Hyperframes player composition. Skipping client audio preloading.");
            return;
        }

        const scenes = this.lessonPackage.scenes;
        console.log(`[VisualLearning LOG] Preloading assets for ${scenes.length} scenes.`);
        this.preloadedImages = {};
        this.preloadedAudios = {};

        // 1. Preload global assets
        const globalAssets = this.lessonPackage.global_assets || [];
        const globalImagePromises = globalAssets.map(asset => {
            if (asset.type === 'image' && asset.asset_url) {
                console.log(`[VisualLearning LOG] Preload global image asset: id="${asset.id}" | URL: ${asset.asset_url}`);
                return new Promise((resolve) => {
                    const img = new Image();
                    img.onload = () => {
                        this.preloadedImages[asset.id] = img;
                        resolve();
                    };
                    img.onerror = () => {
                        console.error(`[VisualLearning ERROR] Preload global image failed: id="${asset.id}"`);
                        resolve();
                    };
                    img.src = asset.asset_url;
                });
            }
            return Promise.resolve();
        });

        // 2. Preload local clip assets
        const localImagePromises = [];
        scenes.forEach(scene => {
            const assets = scene.local_assets || scene.assets || [];
            assets.forEach(asset => {
                if (asset.type === 'image' && asset.asset_url) {
                    console.log(`[VisualLearning LOG] Preload local image asset: id="${asset.id}" | URL: ${asset.asset_url}`);
                    const promise = new Promise((resolve) => {
                        const img = new Image();
                        img.onload = () => {
                            this.preloadedImages[asset.id] = img;
                            resolve();
                        };
                        img.onerror = () => {
                            console.error(`[VisualLearning ERROR] Preload local image failed: id="${asset.id}"`);
                            resolve();
                        };
                        img.src = asset.asset_url;
                    });
                    localImagePromises.push(promise);
                }
            });
        });

        // 3. Preload narration audio and extract durations
        const audioPromises = scenes.map((scene, index) => {
            const sceneNo = scene.scene_no || scene.clip_no || (index + 1);
            console.log(`[VisualLearning LOG] Preload audio asset: scene_no=${sceneNo} | URL: ${scene.audio_url}`);
            return new Promise((resolve) => {
                const aud = new Audio();
                aud.src = scene.audio_url;
                
                const onCanPlay = () => {
                    this.preloadedAudios[sceneNo] = aud;
                    console.log(`[VisualLearning LOG] Audio preloaded successfully: scene_no=${sceneNo}, duration=${aud.duration}`);
                    cleanup();
                    resolve();
                };

                const onError = (e) => {
                    console.error(`[VisualLearning ERROR] Preload narration audio failed: scene_no=${sceneNo} | URL: ${scene.audio_url}`, e);
                    cleanup();
                    resolve();
                };

                const cleanup = () => {
                    aud.removeEventListener('canplaythrough', onCanPlay);
                    aud.removeEventListener('error', onError);
                };

                aud.addEventListener('canplaythrough', onCanPlay);
                aud.addEventListener('error', onError);
                
                // Safety timeout
                setTimeout(() => {
                    console.warn(`[VisualLearning WARNING] Preload timeout triggered for audio of scene ${sceneNo}`);
                    cleanup();
                    resolve();
                }, 8000);
            });
        });

        await Promise.all([...globalImagePromises, ...localImagePromises, ...audioPromises]);

        // 4. Calculate cumulative durations and start times
        this.clipStartTimes = [];
        let cumulativeTime = 0;
        scenes.forEach((scene, index) => {
            const sceneNo = scene.scene_no || scene.clip_no || (index + 1);
            const aud = this.preloadedAudios[sceneNo];
            let duration = 0;
            if (aud && !isNaN(aud.duration) && aud.duration > 0) {
                duration = aud.duration;
            } else {
                // Fallback duration calculation
                const script = scene.teacher_script || "";
                duration = Math.max(5, Math.ceil(script.length * 0.08));
                console.warn(`[VisualLearning WARNING] Scene #${sceneNo} audio duration unavailable. Using estimated duration: ${duration}s`);
                
                // Create mock audio element to prevent downstream player crashes
                if (!this.preloadedAudios[sceneNo]) {
                    this.preloadedAudios[sceneNo] = {
                        duration: duration,
                        currentTime: 0,
                        play: async () => {},
                        pause: () => {},
                        addEventListener: () => {},
                        removeEventListener: () => {}
                    };
                }
            }
            this.clipStartTimes.push(cumulativeTime);
            cumulativeTime += duration;
        });
        this.totalDuration = cumulativeTime;
        console.log(`[VisualLearning LOG] Total continuous lesson duration: ${this.totalDuration.toFixed(2)}s. Start times:`, this.clipStartTimes);
    }

    sendCommandToIframe(command, payload = {}) {
        const iframe = document.getElementById('vl-html-iframe');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({
                target: 'HYPERFRAMES_ENGINE',
                command,
                ...payload
            }, '*');
        }
    }

    handleIframeEvent(e) {
        const data = e.data;
        if (!data || data.source !== 'HYPERFRAMES_ENGINE') return;
        
        switch (data.type) {
            case 'READY':
                console.log("⚡ [VisualLearning Parent] Received READY from Hyperframes Engine:", data);
                this.iframeReady = true;
                this.totalDuration = data.duration || 0;
                this.totalScenes = data.totalScenes || 1;
                this.lessonScenes = data.scenes || [];
                
                this.renderSidebarFromMetadata(data.scenes || []);
                this.updateHeaderTitle(data.lessonTitle);
                this.updateProgressTimeDisplay(0);

                // Autoplay by default upon engine ready
                console.log("▶ [VisualLearning Parent] Triggering automatic playback by default.");
                this.sendCommandToIframe('PLAY');
                this.isPlaying = true;
                this.updatePlayPauseBtnUI(true);
                break;

            case 'PLAYING':
                this.isPlaying = true;
                this.updatePlayPauseBtnUI(true);
                break;

            case 'PAUSED':
                this.isPlaying = false;
                this.updatePlayPauseBtnUI(false);
                break;

            case 'CURRENT_TIME':
                this.updatePlaybackTimeUI(data.currentTime, data.duration);
                break;

            case 'SCENE_CHANGED':
                this.currentSlideIndex = (data.currentScene || 1) - 1;
                this.updateSceneStateUI(data.currentScene, data.totalScenes, data.title, data.script);
                break;

            case 'SUBTITLE_CHANGED':
                this.updateSubtitleText(data.script);
                break;

            case 'TIMELINE_FINISHED':
                this.isPlaying = false;
                this.updatePlayPauseBtnUI(false);
                if (this.progressBarFill) this.progressBarFill.style.width = '100%';
                break;
        }
    }

    updatePlayPauseBtnUI(isPlaying) {
        if (!this.playPauseBtn) return;
        if (isPlaying) {
            this.playPauseBtn.innerHTML = '<span>⏸</span> Pause';
            this.playPauseBtn.className = 'vl-btn vl-btn-primary';
        } else {
            this.playPauseBtn.innerHTML = '<span>▶</span> Play';
            this.playPauseBtn.className = 'vl-btn vl-btn-primary';
        }
    }

    updatePlaybackTimeUI(currentTime, duration) {
        const dur = duration || this.totalDuration || 1;
        const pct = Math.max(0, Math.min(100, (currentTime / dur) * 100));
        if (this.progressBarFill) {
            this.progressBarFill.style.width = `${pct.toFixed(2)}%`;
        }
        this.updateProgressTimeDisplay(currentTime);
    }

    updateSceneStateUI(currentSceneNo, totalScenes, title, script) {
        const total = totalScenes || (this.lessonPackage && this.lessonPackage.scenes ? this.lessonPackage.scenes.length : 1);
        if (this.progressText) {
            const curTimeStr = this.formatSeconds(this.lastCurrentTime || 0);
            const totTimeStr = this.formatSeconds(this.totalDuration || 0);
            this.progressText.textContent = `Scene ${currentSceneNo} / ${total} • ${curTimeStr} / ${totTimeStr}`;
        }

        // Update Subtitle Panel
        this.updateSubtitleText(script);

        // Update Notes Drawer
        if (this.notesBody && script) {
            this.notesBody.innerHTML = `<p>${script}</p>`;
        }

        // Highlight Active Scene in Sidebar
        this.updateSidebarHighlight(currentSceneNo);
    }

    updateSubtitleText(scriptText) {
        const subText = document.getElementById('vl-subtitle-text');
        if (subText && scriptText) {
            subText.textContent = scriptText;
        }
    }

    updateHeaderTitle(titleText) {
        const headerTitle = document.getElementById('vl-header-title');
        if (headerTitle && titleText) {
            headerTitle.textContent = titleText;
        }
    }

    formatSeconds(time) {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }

    renderSidebarFromMetadata(scenes) {
        if (!this.slidesList) return;
        this.slidesList.innerHTML = '';
        scenes.forEach((scene, index) => {
            const sceneNo = scene.scene_no || (index + 1);
            const item = document.createElement('div');
            item.id = `vl-slide-item-${sceneNo}`;
            item.className = index === 0 ? 'vl-slide-item current' : 'vl-slide-item upcoming';
            item.onclick = () => {
                console.log(`[VisualLearning Parent UI] Clicked scene item #${sceneNo}`);
                this.sendCommandToIframe('JUMP_SCENE', { sceneNo });
            };
            item.innerHTML = `
                <div class="vl-slide-status-icon">${sceneNo}</div>
                <div class="vl-slide-details">
                    <span class="vl-slide-title">${scene.title || `Scene ${sceneNo}`}</span>
                </div>
            `;
            this.slidesList.appendChild(item);
        });
    }

    updateSidebarHighlight(activeSceneNo) {
        if (!this.lessonScenes) return;
        this.lessonScenes.forEach((scene, index) => {
            const sceneNo = scene.scene_no || (index + 1);
            const item = document.getElementById(`vl-slide-item-${sceneNo}`);
            if (!item) return;
            const icon = item.querySelector('.vl-slide-status-icon');
            if (sceneNo < activeSceneNo) {
                item.className = 'vl-slide-item completed';
                if (icon) icon.textContent = '✓';
            } else if (sceneNo === activeSceneNo) {
                item.className = 'vl-slide-item current';
                if (icon) icon.textContent = sceneNo;
            } else {
                item.className = 'vl-slide-item upcoming';
                if (icon) icon.textContent = sceneNo;
            }
        });
    }

    showAIToolToast(toolName) {
        let toast = document.getElementById('vl-ai-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'vl-ai-toast';
            toast.style.cssText = `
                position: fixed; bottom: 24px; right: 24px; z-index: 9999;
                background: rgba(15, 23, 42, 0.94); border: 1px solid rgba(99, 102, 241, 0.4);
                color: #f1f5f9; padding: 12px 20px; border-radius: 12px; font-size: 13px; font-weight: 600;
                backdrop-filter: blur(12px); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                transition: all 0.3s ease; transform: translateY(10px); opacity: 0;
            `;
            document.body.appendChild(toast);
        }
        toast.innerHTML = `✨ <strong>${toolName}</strong> — Feature initialized! Connected to AI pipeline.`;
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
        }, 3200);
    }

    launchPlayer() {
        console.log("[VisualLearning LOG] launchPlayer triggered.");
        console.log("1. lessonPackage.video_url:", this.lessonPackage ? this.lessonPackage.video_url : undefined);
        console.log("2. lessonPackage.html_url:", this.lessonPackage ? this.lessonPackage.html_url : undefined);

        if (this.loadingScreen) this.loadingScreen.style.display = 'none';
        if (this.playerUI) this.playerUI.style.display = 'flex';
        
        if (this.lessonPackage && this.lessonPackage.video_url && !this.lessonPackage.video_url.endsWith('.html')) {
            this.mountVideoPlayer(this.lessonPackage.video_url);
            return;
        }

        if (this.lessonPackage && (this.lessonPackage.html_url || (this.lessonPackage.video_url && this.lessonPackage.video_url.endsWith('.html')))) {
            const htmlUrl = this.lessonPackage.html_url || this.lessonPackage.video_url;
            if (this.canvas) {
                if (window.gsap) gsap.killTweensOf(this.canvas);
                this.canvas.style.transform = '';
                this.canvas.style.width = '';
                this.canvas.style.height = '';
            }
            this.mountIframePlayer(htmlUrl);
            return;
        }

        this.currentSlideIndex = 0;
        this.isPlaying = false;
        
        this.resizeCanvas();
        this.renderSidebarFromMetadata(this.lessonPackage ? this.lessonPackage.scenes : []);
        this.playSlide(1);
    }

    mountIframePlayer(htmlUrl) {
        console.log("5. Inside mountIframePlayer():");
        console.log("   - iframe created");
        console.log("   - iframe.src:", htmlUrl);

        this.canvas.innerHTML = `
            <div class="hyperframes-iframe-wrapper" style="
                width: 100%; height: 100%;
                position: relative;
                background: #090d16;
                border-radius: 12px;
                overflow: hidden;
            ">
                <iframe id="vl-html-iframe" src="${htmlUrl}"
                    style="
                        position: absolute;
                        top: 0; left: 0;
                        width: 1280px; height: 720px;
                        border: none; outline: none;
                        background: #090d16;
                        transform-origin: top left;
                    "
                    allow="autoplay">
                </iframe>
            </div>
        `;

        const iframe = document.getElementById('vl-html-iframe');
        if (iframe) {
            iframe.onerror = () => {
                console.warn("[VisualLearning] iframe failed to load. Falling back to client-side slide player.");
                if (this.lessonPackage) {
                    this.lessonPackage.html_url = null;
                    this.launchPlayer();
                }
            };
            iframe.onload = () => {
                console.log("6. iframe.onload fired");
                requestAnimationFrame(() => this.scaleIframe());
                
                // Hide inside-iframe subtitles container so subtitles ONLY render in dedicated Subtitle Panel below video
                try {
                    const doc = iframe.contentDocument || iframe.contentWindow.document;
                    if (doc) {
                        // If Vercel redirected to root page (405 / error page), fall back to client-side player
                        if (doc.title && (doc.title.includes('404') || doc.title.includes('Error') || doc.location.pathname === '/')) {
                            console.warn("[VisualLearning] iframe loaded redirect/error page. Falling back to client-side slide player.");
                            if (this.lessonPackage) {
                                this.lessonPackage.html_url = null;
                                this.launchPlayer();
                            }
                            return;
                        }
                        const style = doc.createElement('style');
                        style.textContent = '.subtitles-container { display: none !important; }';
                        doc.head.appendChild(style);
                    }
                } catch(e) {
                    console.warn("[VisualLearning] Could not inject iframe subtitle override:", e);
                }
            };
        }

        this.updateHeaderAndSubtitles();
        this.renderSidebar();
        requestAnimationFrame(() => this.scaleIframe());
        window.addEventListener('resize', () => this.scaleIframe());
    }

    scaleIframe() {
        const iframe = document.getElementById('vl-html-iframe');
        const wrapper = iframe && iframe.parentElement;
        if (!iframe || !wrapper) return;

        // Read dimensions from the wrapper (the immediate parent of the
        // iframe) rather than this.canvas to avoid any stale canvas size.
        const containerWidth  = wrapper.clientWidth  || this.canvas.clientWidth  || 800;
        const containerHeight = wrapper.clientHeight || this.canvas.clientHeight || 450;
        if (!containerWidth || !containerHeight) return;

        // Uniform scale that fits the 1280×720 content into the container
        // while preserving the aspect ratio (letterbox / pillarbox if needed).
        const scale = Math.min(containerWidth / 1280, containerHeight / 720);
        if (scale <= 0) return;

        // Centre the scaled iframe within the wrapper using translate.
        // transform-origin is top-left so the maths are straightforward:
        //   scaledW = 1280 * scale,  offsetX = (containerW - scaledW) / 2
        const offsetX = (containerWidth  - 1280 * scale) / 2;
        const offsetY = (containerHeight - 720  * scale) / 2;

        iframe.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
        console.log(`[scaleIframe] container=${containerWidth}x${containerHeight} scale=${scale.toFixed(4)} offset=(${offsetX.toFixed(1)},${offsetY.toFixed(1)})`);
    }

    mountVideoPlayer(videoUrl) {
        this.canvas.innerHTML = `
            <div class="hyperframes-video-wrapper" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: #000; position: relative; overflow: hidden; border-radius: 12px;">
                <video id="vl-mp4-video" src="${videoUrl}" 
                       style="width: 100%; height: 100%; object-fit: contain; outline: none; background: #000;" 
                       playsinline controls autoplay>
                </video>
            </div>
        `;

        this.renderSidebar();

        const videoEl = document.getElementById('vl-mp4-video');
        if (videoEl) {
            videoEl.onplay = () => {
                this.isPlaying = true;
                if (this.playPauseBtn) this.playPauseBtn.innerHTML = '❚❚';
            };
            videoEl.onpause = () => {
                this.isPlaying = false;
                if (this.playPauseBtn) this.playPauseBtn.innerHTML = '▶';
            };
            videoEl.ontimeupdate = () => {
                if (videoEl.duration) {
                    const pct = (videoEl.currentTime / videoEl.duration) * 100;
                    if (this.progressBarFill) this.progressBarFill.style.width = pct + '%';
                    if (this.progressText) {
                        const cur = Math.floor(videoEl.currentTime);
                        const tot = Math.floor(videoEl.duration);
                        const curStr = `${Math.floor(cur / 60)}:${(cur % 60).toString().padStart(2, '0')}`;
                        const totStr = `${Math.floor(tot / 60)}:${(tot % 60).toString().padStart(2, '0')}`;
                        this.progressText.textContent = `${curStr} / ${totStr}`;
                    }
                }
            };
        }

        if (this.playPauseBtn) {
            this.playPauseBtn.onclick = () => {
                if (videoEl) {
                    if (videoEl.paused) videoEl.play();
                    else videoEl.pause();
                }
            };
        }
    }

    renderSidebar() {
        this.slidesList.innerHTML = '';
        this.lessonPackage.scenes.forEach((scene, index) => {
            const sceneNo = scene.scene_no || scene.clip_no || (index + 1);
            const item = document.createElement('div');
            item.id = `vl-slide-item-${sceneNo}`;
            item.className = 'vl-slide-item upcoming';
            item.onclick = () => {
                console.log(`[VisualLearning LOG] User clicked sidebar item: Scene ${sceneNo}`);
                const startTime = this.clipStartTimes[sceneNo - 1] || 0;
                this.seekTo(startTime);
            };

            item.innerHTML = `
                <div class="vl-slide-status-icon">${sceneNo}</div>
                <div class="vl-slide-details">
                    <span class="vl-slide-title">${scene.title || `Scene ${sceneNo}`}</span>
                </div>
            `;
            this.slidesList.appendChild(item);
        });
    }

    updateSidebarStates() {
        this.lessonPackage.scenes.forEach((scene, index) => {
            const sceneNo = scene.scene_no || scene.clip_no || (index + 1);
            const item = document.getElementById(`vl-slide-item-${sceneNo}`);
            if (!item) return;

            const icon = item.querySelector('.vl-slide-status-icon');
            
            if (sceneNo < this.currentSlideIndex + 1) {
                item.className = 'vl-slide-item completed';
                if (icon) icon.textContent = '✓';
            } else if (sceneNo === this.currentSlideIndex + 1) {
                item.className = 'vl-slide-item current';
                if (icon) icon.textContent = sceneNo;
            } else {
                item.className = 'vl-slide-item upcoming';
                if (icon) icon.textContent = sceneNo;
            }
        });
    }

    updateCanvasTheme(scene) {
        const mainContainer = document.getElementById('vl-main');
        if (!mainContainer) return;
        
        // Reset active glow classes
        mainContainer.classList.remove('glow-indigo', 'glow-gold', 'glow-emerald', 'glow-rose');
        
        const titleLower = (scene.title || "").toLowerCase();
        const assetsKeywords = (scene.local_assets || scene.assets || []).map(a => (a.search_query || "").toLowerCase()).join(" ");
        const merged = `${titleLower} ${assetsKeywords}`;
        
        let theme = 'indigo';
        let radialGradient = 'radial-gradient(circle at center, #1b153f 0%, #0b0c16 100%)';
        
        if (merged.match(/sun|fire|heat|gold|yellow|solar|combustion/)) {
            theme = 'gold';
            radialGradient = 'radial-gradient(circle at center, #2e1e05 0%, #0c0802 100%)';
        } else if (merged.match(/plant|leaf|root|forest|tree|green|photosynthesis|stomata/)) {
            theme = 'emerald';
            radialGradient = 'radial-gradient(circle at center, #072517 0%, #020b07 100%)';
        } else if (merged.match(/atom|electron|science|chemical|energy|red|volcano|magma|explosion/)) {
            theme = 'rose';
            radialGradient = 'radial-gradient(circle at center, #2b0b14 0%, #0e0306 100%)';
        }
        
        mainContainer.classList.add(`glow-${theme}`);
        mainContainer.style.background = radialGradient;
        
        if (this.canvas) {
            this.canvas.style.background = 'transparent';
        }
        console.log(`[VisualLearning LOG] Applied Canvas theme to main wrapper: "${theme}"`);
    }

    playSlide(sceneNo, startOffset = 0) {
        console.log(`[VisualLearning LOG] playSlide initiated for Scene #${sceneNo} | startOffset = ${startOffset}s`);

        const sceneIndex = sceneNo - 1;
        this.currentSlideIndex = sceneIndex;
        const scene = this.lessonPackage.scenes[sceneIndex];

        this.updateHeaderAndSubtitles();

        // 1. Stop active audio
        if (this.audio) {
            console.log("[VisualLearning LOG] Pausing and resetting active audio.");
            this.audio.pause();
            this.audio.currentTime = 0;
            this.audio.onended = null;
            this.audio.ontimeupdate = null;
        }

        // 2. Kill current GSAP timeline
        if (this.timeline) {
            console.log("[VisualLearning LOG] Killing active GSAP timeline.");
            this.timeline.kill();
            this.timeline = null;
        }

        // Apply theme color & gradients dynamically based on scene content
        this.updateCanvasTheme(scene);

        // Reset sync statuses
        this.timelineFinished = false;
        this.audioFinished = false;

        // Ensure canvas has the overlay container (SVG)
        let svgOverlay = document.getElementById('vl-connections-svg');
        if (!svgOverlay) {
            svgOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svgOverlay.setAttribute('id', 'vl-connections-svg');
            svgOverlay.style.position = 'absolute';
            svgOverlay.style.top = '0';
            svgOverlay.style.left = '0';
            svgOverlay.style.width = '100%';
            svgOverlay.style.height = '100%';
            svgOverlay.style.pointerEvents = 'none';
            svgOverlay.style.zIndex = '1';
            this.canvas.appendChild(svgOverlay);
        }

        // Render Scene Banner (Title + Script Subtitles)
        let sceneBanner = document.getElementById('vl-scene-banner');
        if (!sceneBanner) {
            sceneBanner = document.createElement('div');
            sceneBanner.id = 'vl-scene-banner';
            sceneBanner.style.position = 'absolute';
            sceneBanner.style.top = '24px';
            sceneBanner.style.left = '50%';
            sceneBanner.style.transform = 'translateX(-50%)';
            sceneBanner.style.zIndex = '10';
            sceneBanner.style.textAlign = 'center';
            sceneBanner.style.maxWidth = '85%';
            sceneBanner.style.pointerEvents = 'none';
            this.canvas.appendChild(sceneBanner);
        }
        sceneBanner.innerHTML = `
            <div style="background: rgba(15, 23, 42, 0.82); backdrop-filter: blur(12px); border: 1px solid rgba(99, 102, 241, 0.4); padding: 14px 28px; border-radius: 20px; box-shadow: 0 12px 36px rgba(0,0,0,0.6);">
                <div style="color: #a5b4fc; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;">Scene ${sceneNo} of ${this.lessonPackage.scenes.length}</div>
                <h3 style="color: #ffffff; font-size: 20px; font-weight: 800; margin: 0 0 6px 0; font-family: system-ui, sans-serif;">${scene.title || `Concept #${sceneNo}`}</h3>
                <p style="color: #cbd5e1; font-size: 13px; line-height: 1.5; margin: 0; font-family: system-ui, sans-serif;">${scene.teacher_script || ''}</p>
            </div>
        `;

        // Render/Diff Assets
        const globalAssets = this.lessonPackage.global_assets || [];
        const localAssets = scene.local_assets || scene.assets || [];
        const activeAssets = [...globalAssets, ...localAssets];
        const activeAssetIds = activeAssets.map(a => a.id);

        if (!this.renderedAssets) {
            this.renderedAssets = {};
        }

        // Remove assets not in active list
        Object.keys(this.renderedAssets).forEach(assetId => {
            if (!activeAssetIds.includes(assetId)) {
                const el = document.getElementById(`asset-${assetId}`);
                if (el) {
                    console.log(`[VisualLearning LOG] Fading out and removing asset: "${assetId}"`);
                    gsap.to(el, {
                        opacity: 0,
                        duration: 0.5,
                        onComplete: () => {
                            el.remove();
                            // Destroy Lottie if any
                            if (this.lottieInstancesMap[assetId]) {
                                try {
                                    this.lottieInstancesMap[assetId].destroy();
                                } catch (e) {
                                    console.error(e);
                                }
                                delete this.lottieInstancesMap[assetId];
                            }
                        }
                    });
                }
                delete this.renderedAssets[assetId];
            }
        });

        // Add/update active assets
        activeAssets.forEach(asset => {
            let el = document.getElementById(`asset-${asset.id}`);
            if (!el) {
                console.log(`[VisualLearning LOG] Creating new asset: "${asset.id}"`);
                el = document.createElement('div');
                el.id = `asset-${asset.id}`;
                el.className = 'vl-asset';
                el.style.zIndex = '2'; // render above SVG lines

                const layout = asset.layout || {};
                el.style.top = `${layout.top}%`;
                el.style.left = `${layout.left}%`;
                el.style.width = `${layout.width}%`;
                if (layout.height) {
                    el.style.height = `${layout.height}%`;
                }

                const contentWrapper = document.createElement('div');
                contentWrapper.className = 'vl-asset-content idle-floating';
                contentWrapper.style.width = '100%';
                contentWrapper.style.height = '100%';
                contentWrapper.style.position = 'relative';
                el.appendChild(contentWrapper);

                if (asset.type === 'image' || asset.type === 'icon') {
                    if (asset.asset_url) {
                        const img = document.createElement('img');
                        img.src = asset.asset_url;
                        img.alt = asset.id;
                        img.onerror = () => {
                            console.warn(`[VisualLearning WARNING] Image failed to load: ${img.src}. Rendering fallback card.`);
                            contentWrapper.innerHTML = '';
                            const fallback = document.createElement('div');
                            fallback.className = 'vl-fallback-card';
                            fallback.innerHTML = `
                                <div class="icon">✨</div>
                                <div class="label">${asset.search_query || asset.id}</div>
                            `;
                            contentWrapper.appendChild(fallback);
                        };
                        contentWrapper.appendChild(img);
                    } else {
                        // Image search failed - show glassmorphic fallback card
                        const fallback = document.createElement('div');
                        fallback.className = 'vl-fallback-card';
                        fallback.innerHTML = `
                            <div class="icon">✨</div>
                            <div class="label">${asset.search_query || asset.id}</div>
                        `;
                        contentWrapper.appendChild(fallback);
                    }
                } else if (asset.type === 'text') {
                    const textDiv = document.createElement('div');
                    textDiv.className = 'vl-asset-text';
                    textDiv.textContent = asset.text_content || asset.id || '';
                    contentWrapper.appendChild(textDiv);
                } else if (asset.type === 'lottie') {
                    const lottieWrapper = document.createElement('div');
                    lottieWrapper.className = 'vl-asset-lottie';
                    lottieWrapper.style.width = '100%';
                    lottieWrapper.style.height = '100%';
                    contentWrapper.appendChild(lottieWrapper);

                    const instance = loadLottieAsset(lottieWrapper, asset.search_query);
                    if (instance) {
                        this.lottieInstancesMap[asset.id] = instance;
                    }
                }

                // Initial state for entrance animation
                el.style.opacity = '0';
                this.canvas.appendChild(el);
                this.renderedAssets[asset.id] = true;
            } else {
                // Asset is already on canvas, make sure it is visible
                el.style.opacity = '1';
            }
        });

        // 4. Initialize GSAP Scene Timeline
        console.log("[VisualLearning LOG] Instantiating new GSAP Timeline.");
        this.timeline = gsap.timeline({ paused: true });

        let hasTimelineAnimations = false;

        localAssets.forEach(asset => {
            const el = document.getElementById(`asset-${asset.id}`);
            if (!el) return;

            const anims = asset.animations || [];
            anims.forEach(anim => {
                const type = anim.type;
                const duration = anim.duration || 1.0;
                const delay = anim.start_time || anim.delay || 0.0;

                console.log(`[VisualLearning LOG] Animation Config for "${asset.id}": Type="${type}", Duration=${duration}s, Delay=${delay}s`);

                if (type === 'rotate' && duration >= 4.0) {
                    console.log(`[VisualLearning LOG] Spinning asset "${asset.id}" continuously (repeat: -1) using independent GSAP tween.`);
                    gsap.to(el, { rotation: 360, duration: duration, repeat: -1, ease: "none" });
                } else {
                    hasTimelineAnimations = true;
                    switch (type) {
                        case 'fade_in':
                            this.timeline.fromTo(el, { opacity: 0 }, { opacity: 1, duration, ease: "power1.out" }, delay);
                            break;
                        case 'fade_out':
                            this.timeline.to(el, { opacity: 0, duration, ease: "power1.in" }, delay);
                            break;
                        case 'move_up':
                        case 'slide_in_bottom':
                            this.timeline.fromTo(el, { y: 150, opacity: 0 }, { y: 0, opacity: 1, duration, ease: "power2.out" }, delay);
                            break;
                        case 'move_down':
                        case 'slide_in_top':
                            this.timeline.fromTo(el, { y: -150, opacity: 0 }, { y: 0, opacity: 1, duration, ease: "power2.out" }, delay);
                            break;
                        case 'move_left':
                        case 'slide_in_right':
                            this.timeline.fromTo(el, { x: 150, opacity: 0 }, { x: 0, opacity: 1, duration, ease: "power2.out" }, delay);
                            break;
                        case 'move_right':
                        case 'slide_in_left':
                            this.timeline.fromTo(el, { x: -150, opacity: 0 }, { x: 0, opacity: 1, duration, ease: "power2.out" }, delay);
                            break;
                        case 'scale_up':
                            this.timeline.fromTo(el, { scale: 0, opacity: 0 }, { scale: 1, opacity: 1, duration, ease: "back.out(1.5)" }, delay);
                            break;
                        case 'scale_down':
                            this.timeline.to(el, { scale: 0, opacity: 0, duration, ease: "power2.in" }, delay);
                            break;
                        case 'spin':
                        case 'rotate':
                            this.timeline.to(el, { rotation: 360, duration, ease: "none" }, delay);
                            break;
                        case 'appear':
                            this.timeline.set(el, { opacity: 1, visibility: 'visible' }, delay);
                            break;
                        case 'disappear':
                            this.timeline.set(el, { opacity: 0, visibility: 'hidden' }, delay);
                            break;
                        default:
                            this.timeline.fromTo(el, { opacity: 0 }, { opacity: 1, duration, ease: "power1.out" }, delay);
                    }
                }
            });
        });

        // Setup timeline complete check
        if (hasTimelineAnimations) {
            this.timeline.eventCallback("onComplete", () => {
                console.log(`[VisualLearning LOG] GSAP Timeline complete for Scene #${sceneNo}`);
                this.timelineFinished = true;
                this.checkAutoAdvance();
            });
        } else {
            console.log(`[VisualLearning LOG] Scene #${sceneNo} has no timeline transitions. Marking timeline complete instantly.`);
            this.timelineFinished = true;
        }

        // 5. Draw connections
        this.drawSVGConnections(scene);

        // 6. Camera Glides and Focus
        const cam = scene.camera || { focus_x: 50, focus_y: 50, zoom: 1.0, transition_duration: 1.5 };
        const cameraDuration = this.isSeeking ? 0.2 : (cam.transition_duration || 1.5);
        this.focusCamera(cam.focus_x, cam.focus_y, cam.zoom, cameraDuration);

        // Update script notes
        if (this.notesBody) {
            this.notesBody.innerHTML = `<p>${scene.teacher_script}</p>`;
        }

        // Navigation state updates
        this.updateSidebarStates();

        // Setup Narration audio
        const cachedAudio = this.preloadedAudios[sceneNo];
        if (cachedAudio) {
            console.log(`[VisualLearning LOG] Playing preloaded audio for Scene #${sceneNo}`);
            this.audio = cachedAudio;
        } else {
            console.warn(`[VisualLearning WARNING] Audio for Scene #${sceneNo} was not cached. Initiating raw load.`);
            this.audio = new Audio(scene.audio_url);
        }

        // Set playback offset if seeking
        this.audio.currentTime = startOffset;
        if (this.timeline && hasTimelineAnimations) {
            this.timeline.seek(startOffset);
        }

        this.audio.onended = () => {
            console.log(`[VisualLearning LOG] Narration Audio ended for Scene #${sceneNo}`);
            this.audioFinished = true;
            this.checkAutoAdvance();
        };

        this.audio.ontimeupdate = () => {
            if (this.audio && this.audio.duration && !this.isSeeking) {
                const currentSceneTime = this.audio.currentTime;
                const globalCurrentTime = this.clipStartTimes[sceneIndex] + currentSceneTime;
                
                if (this.progressBarFill && this.totalDuration > 0) {
                    const pct = (globalCurrentTime / this.totalDuration) * 100;
                    this.progressBarFill.style.width = `${pct}%`;
                }

                this.updateProgressTimeDisplay(globalCurrentTime);
            }
        };

        // Resume/Start playback
        if (this.isPlaying) {
            console.log(`[VisualLearning LOG] Triggering play state for audio & GSAP timelines on Scene #${sceneNo}`);
            this.audio.play().catch(err => {
                console.error("[VisualLearning ERROR] Playback failed/blocked:", err);
                this.pause();
            });
            this.timeline.play();
            Object.values(this.lottieInstancesMap).forEach(inst => {
                try { inst.play(); } catch(e) {}
            });
        } else {
            console.log("[VisualLearning LOG] Launching first play triggered.");
            this.play();
        }
        
        this.resizeCanvas();
    }

    focusCamera(x, y, zoom, duration) {
        console.log(`[VisualLearning LOG] focusCamera: x=${x}, y=${y}, zoom=${zoom}, duration=${duration}`);
        if (!this.canvas) return;

        const container = document.getElementById('vl-main');
        if (!container) return;

        const W = container.clientWidth;
        const H = container.clientHeight;
        const C_w = 800;
        const C_h = 450;

        // Calculate base scale to fit container (contain mode)
        const baseScale = Math.min(W / C_w, H / C_h);
        const totalScale = baseScale * zoom;

        // Calculate translation in pixels to center the coordinate (x%, y%) in the container
        const tx = W / 2 - totalScale * (x / 100) * C_w;
        const ty = H / 2 - totalScale * (y / 100) * C_h;

        console.log(`[VisualLearning LOG] Camera Glide: coord=(${x}%, ${y}%), zoom=${zoom}, baseScale=${baseScale.toFixed(4)}, totalScale=${totalScale.toFixed(4)}, tx=${tx.toFixed(1)}px, ty=${ty.toFixed(1)}px`);

        gsap.to(this.canvas, {
            transform: `translate(${tx}px, ${ty}px) scale(${totalScale})`,
            transformOrigin: "0 0",
            duration: duration,
            ease: "power2.out"
        });
    }

    drawSVGConnections(scene) {
        const svgOverlay = document.getElementById('vl-connections-svg');
        if (!svgOverlay) return;

        svgOverlay.innerHTML = '';
        
        const mainContainer = document.getElementById('vl-main');
        const themeClass = Array.from(mainContainer.classList).find(c => c.startsWith('glow-')) || 'glow-indigo';
        const theme = themeClass.replace('glow-', '');
        
        const strokeColors = {
            indigo: '#6366f1',
            gold: '#f59e0b',
            emerald: '#10b981',
            rose: '#ef4444'
        };
        const strokeColor = strokeColors[theme] || '#6366f1';

        svgOverlay.innerHTML = `
          <defs>
            <marker id="vl-arrow-indigo" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="${strokeColors.indigo}" />
            </marker>
            <marker id="vl-arrow-gold" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="${strokeColors.gold}" />
            </marker>
            <marker id="vl-arrow-emerald" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="${strokeColors.emerald}" />
            </marker>
            <marker id="vl-arrow-rose" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="${strokeColors.rose}" />
            </marker>
          </defs>
        `;

        const connections = this.lessonPackage.connections || [];
        console.log(`[VisualLearning LOG] Drawing ${connections.length} connection paths.`);

        connections.forEach(conn => {
            const fromId = conn.from;
            const toId = conn.to;

            const fromEl = document.getElementById(`asset-${fromId}`);
            const toEl = document.getElementById(`asset-${toId}`);

            if (fromEl && toEl) {
                const fromLayout = this.getAssetLayout(fromId);
                const toLayout = this.getAssetLayout(toId);

                if (fromLayout && toLayout) {
                    const x1 = fromLayout.left + fromLayout.width / 2;
                    const y1 = fromLayout.top + fromLayout.height / 2;
                    const x2 = toLayout.left + toLayout.width / 2;
                    const y2 = toLayout.top + toLayout.height / 2;

                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', `${x1}%`);
                    line.setAttribute('y1', `${y1}%`);
                    line.setAttribute('x2', `${x1}%`);
                    line.setAttribute('y2', `${y1}%`);
                    line.setAttribute('stroke', strokeColor);
                    line.setAttribute('stroke-width', '3');
                    line.setAttribute('stroke-linecap', 'round');
                    
                    if (conn.style === 'dashed') {
                        line.setAttribute('stroke-dasharray', '6,6');
                    } else {
                        line.setAttribute('stroke-dasharray', 'none');
                    }

                    if (conn.type === 'arrow' || conn.type === 'connector') {
                        line.setAttribute('marker-end', `url(#vl-arrow-${theme})`);
                    }

                    svgOverlay.appendChild(line);

                    const transitionDuration = this.isSeeking ? 0.2 : 1.2;
                    gsap.to(line, {
                        attr: { x2: `${x2}%`, y2: `${y2}%` },
                        duration: transitionDuration,
                        ease: "power1.inOut"
                    });
                }
            }
        });
    }

    getAssetLayout(assetId) {
        if (!this.lessonPackage) return null;
        if (this.lessonPackage.global_assets) {
            const ga = this.lessonPackage.global_assets.find(a => a.id === assetId);
            if (ga) return ga.layout;
        }
        for (const scene of this.lessonPackage.scenes) {
            const la = (scene.local_assets || scene.assets || []).find(a => a.id === assetId);
            if (la) return la.layout;
        }
        return null;
    }

    updateProgressTimeDisplay(globalTime) {
        if (!this.progressText) return;
        this.lastCurrentTime = globalTime;
        const currentScene = this.currentSlideIndex + 1;
        const totalScenes = this.totalScenes || (this.lessonPackage && this.lessonPackage.scenes ? this.lessonPackage.scenes.length : 1);
        const currentStr = this.formatSeconds(globalTime);
        const totalStr = this.formatSeconds(this.totalDuration);
        this.progressText.textContent = `Scene ${currentScene} / ${totalScenes} • ${currentStr} / ${totalStr}`;
    }

    handleProgressBarClick(e) {
        const progressBarContainer = document.querySelector('.vl-progress-bar-container');
        if (!progressBarContainer || this.totalDuration <= 0) return;

        const rect = progressBarContainer.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const pct = Math.max(0, Math.min(1, offsetX / rect.width));
        const targetTime = pct * this.totalDuration;
        
        console.log(`[VisualLearning Parent UI] User seek bar click: pct=${(pct*100).toFixed(1)}%, targetTime=${targetTime.toFixed(2)}s`);
        this.sendCommandToIframe('SEEK', { targetTime });
    }

    togglePlay() {
        console.log("[VisualLearning Parent UI] togglePlay triggered. isPlaying=", this.isPlaying);
        this.sendCommandToIframe(this.isPlaying ? 'PAUSE' : 'PLAY');
    }

    nextSlide() {
        const nextNo = this.currentSlideIndex + 2;
        console.log(`[VisualLearning Parent UI] Next button clicked. Requesting scene #${nextNo}`);
        this.sendCommandToIframe('JUMP_SCENE', { sceneNo: nextNo });
    }

    previousSlide() {
        const prevNo = this.currentSlideIndex;
        console.log(`[VisualLearning Parent UI] Prev button clicked. Requesting scene #${prevNo}`);
        this.sendCommandToIframe('JUMP_SCENE', { sceneNo: prevNo });
    }

    jumpToSlide(sceneNo) {
        console.log(`[VisualLearning LOG] JumpToSlide called for scene #${sceneNo}`);
        const startTime = this.clipStartTimes[sceneNo - 1] || 0;
        this.seekTo(startTime);
    }

    toggleNotes(forceOpen) {
        if (!this.notesOverlay) return;
        
        if (forceOpen === undefined) {
            this.notesOverlay.classList.toggle('open');
        } else if (forceOpen) {
            this.notesOverlay.classList.add('open');
        } else {
            this.notesOverlay.classList.remove('open');
        }
        
        console.log(`[VisualLearning LOG] Toggle script panel: open=${this.notesOverlay.classList.contains('open')}`);

        if (this.notesOverlay.classList.contains('open')) {
            const tabScriptBtn = document.getElementById('vl-tab-script');
            if (tabScriptBtn) tabScriptBtn.click();
        }
    }

    destroyLesson() {
        console.log("[VisualLearning LOG] destroyLesson triggered. Cleaning memory and resetting containers.");
        
        if (this.audio) {
            this.audio.pause();
            this.audio.currentTime = 0;
            this.audio.onended = null;
            this.audio.ontimeupdate = null;
        }
        this.audio = null;

        if (this.timeline) {
            this.timeline.kill();
            this.timeline = null;
        }

        if (this.lottieInstancesMap) {
            Object.values(this.lottieInstancesMap).forEach(instance => {
                if (instance && typeof instance.destroy === 'function') {
                    instance.destroy();
                }
            });
        }
        this.lottieInstancesMap = {};
        this.renderedAssets = {};
        
        this.lessonPackage = null;
        this.preloadedImages = {};
        this.preloadedAudios = {};
        this.clipStartTimes = [];
        this.totalDuration = 0;
        this.isPlaying = false;
        
        if (this.canvas) {
            this.canvas.innerHTML = '';
        }
        
        if (this.container) this.container.style.display = 'none';
        if (this.notesOverlay) this.notesOverlay.classList.remove('open');
        
        const errContainer = document.getElementById('vl-error-container');
        if (errContainer) errContainer.classList.add('hidden');
        
        const chatInput = document.getElementById('chat-input-container');
        if (chatInput) chatInput.style.display = '';

        const currentMode = window.answerPreferenceManager ? window.answerPreferenceManager.currentMode : '';
        const vlLanding = document.getElementById('vl-landing-view');
        
        if (currentMode === 'visual_learning') {
            if (this.container) this.container.style.display = 'flex';
            if (vlLanding) vlLanding.style.display = 'flex';
        } else {
            const chatContainer = document.getElementById('chat-container');
            const followupPanel = document.getElementById('followup-sticky-panel');
            if (chatContainer) chatContainer.style.display = 'flex';
            if (followupPanel) followupPanel.style.display = 'block';
            
            document.body.classList.remove('left-pane-collapsed');
            const toggleLeftBtn = document.getElementById('toggle-left-pane-btn');
            if (toggleLeftBtn) toggleLeftBtn.style.display = '';
        }
    }
}

window.VisualLearningRenderer = new VisualLearningController();
