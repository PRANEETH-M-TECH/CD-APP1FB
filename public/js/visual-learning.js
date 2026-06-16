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
        
        // Dynamic Canvas Scaling
        window.addEventListener('resize', () => {
            console.log("[VisualLearning LOG] Window resized. Re-calculating canvas scale.");
            this.resizeCanvas();
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
        
        const exitBtn = document.getElementById('vl-exit-btn');
        if (exitBtn) exitBtn.onclick = () => this.destroyLesson();
        
        const errorExitBtn = document.getElementById('vl-error-exit-btn');
        if (errorExitBtn) errorExitBtn.onclick = () => this.destroyLesson();

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

    launchPlayer() {
        console.log("[VisualLearning LOG] Launching player UI.");
        this.loadingScreen.style.display = 'none';
        this.playerUI.style.display = 'flex';
        this.currentSlideIndex = 0;
        this.isPlaying = false;
        
        this.resizeCanvas();
        this.renderSidebar();
        this.playSlide(1);
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
        const formatTime = (time) => {
            const mins = Math.floor(time / 60);
            const secs = Math.floor(time % 60);
            return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
        };
        const currentScene = this.currentSlideIndex + 1;
        const totalScenes = this.lessonPackage.scenes.length;
        const currentStr = formatTime(globalTime);
        const totalStr = formatTime(this.totalDuration);
        this.progressText.textContent = `Scene ${currentScene} / ${totalScenes} | ${currentStr} / ${totalStr}`;
    }

    handleProgressBarClick(e) {
        const progressBarContainer = document.querySelector('.vl-progress-bar-container');
        if (!progressBarContainer || !this.lessonPackage || this.totalDuration <= 0) return;

        const rect = progressBarContainer.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const pct = Math.max(0, Math.min(1, offsetX / rect.width));
        const targetTime = pct * this.totalDuration;
        
        console.log(`[VisualLearning LOG] User seek bar click: pct=${(pct*100).toFixed(1)}%, targetTime=${targetTime.toFixed(2)}s`);
        this.seekTo(targetTime);
    }

    seekTo(targetTime) {
        if (!this.lessonPackage || this.totalDuration <= 0) return;

        this.isSeeking = true;

        // Find active scene index
        let targetSceneIndex = 0;
        for (let i = 0; i < this.lessonPackage.scenes.length; i++) {
            const startTime = this.clipStartTimes[i];
            const endTime = this.clipStartTimes[i + 1] || Infinity;
            if (targetTime >= startTime && targetTime < endTime) {
                targetSceneIndex = i;
                break;
            }
        }

        const sceneNo = targetSceneIndex + 1;
        const startOffset = targetTime - this.clipStartTimes[targetSceneIndex];

        console.log(`[VisualLearning LOG] Seek targetTime maps to Scene #${sceneNo} at offset ${startOffset.toFixed(2)}s`);

        if (this.currentSlideIndex === targetSceneIndex && this.audio) {
            // Same scene, jump audio and timeline offset
            this.audio.currentTime = startOffset;
            if (this.timeline) {
                this.timeline.seek(startOffset);
            }
            if (this.isPlaying) {
                this.audio.play().catch(e => console.error(e));
                this.timeline.play();
                Object.values(this.lottieInstancesMap).forEach(inst => {
                    try { inst.play(); } catch(e) {}
                });
            } else {
                this.audio.pause();
                this.timeline.pause();
                Object.values(this.lottieInstancesMap).forEach(inst => {
                    try { inst.pause(); } catch(e) {}
                });
            }
            
            // Force GUI updates
            if (this.progressBarFill) {
                const pct = (targetTime / this.totalDuration) * 100;
                this.progressBarFill.style.width = `${pct}%`;
            }
            this.updateProgressTimeDisplay(targetTime);
        } else {
            // Scene changed, perform slide transition jump
            this.playSlide(sceneNo, startOffset);
        }

        this.isSeeking = false;
    }

    checkAutoAdvance() {
        console.log(`[VisualLearning LOG] checkAutoAdvance state: timelineFinished=${this.timelineFinished} | audioFinished=${this.audioFinished}`);
        if (this.timelineFinished && this.audioFinished) {
            const nextNo = this.currentSlideIndex + 2;
            if (nextNo <= this.lessonPackage.scenes.length) {
                console.log(`[VisualLearning LOG] Auto-advancing to Scene #${nextNo}`);
                this.playSlide(nextNo);
            } else {
                console.log("[VisualLearning LOG] Storyboard Lesson complete. Closing timeline play.");
                this.pause();
                if (this.progressBarFill) this.progressBarFill.style.width = '100%';
            }
        }
    }

    play() {
        if (!this.audio) {
            console.warn("[VisualLearning WARNING] Play command issued, but no audio exists.");
            return;
        }
        console.log("[VisualLearning LOG] Player resumed/played.");
        this.isPlaying = true;
        if (this.playPauseBtn) {
            this.playPauseBtn.innerHTML = '<span>⏸</span> Pause';
            this.playPauseBtn.className = 'vl-btn vl-btn-primary';
        }
        this.audio.play().catch(err => {
            console.error("[VisualLearning ERROR] Playback failed:", err);
            this.pause();
        });
        if (this.timeline) this.timeline.play();
        Object.values(this.lottieInstancesMap).forEach(inst => {
            try { inst.play(); } catch(e) {}
        });
    }

    pause() {
        if (!this.audio) {
            console.warn("[VisualLearning WARNING] Pause command issued, but no audio exists.");
            return;
        }
        console.log("[VisualLearning LOG] Player paused.");
        this.isPlaying = false;
        if (this.playPauseBtn) {
            this.playPauseBtn.innerHTML = '<span>▶</span> Resume';
            this.playPauseBtn.className = 'vl-btn vl-btn-primary';
        }
        this.audio.pause();
        if (this.timeline) this.timeline.pause();
        Object.values(this.lottieInstancesMap).forEach(inst => {
            try { inst.pause(); } catch(e) {}
        });
    }

    togglePlay() {
        console.log("[VisualLearning LOG] TogglePlay triggered. Current state: isPlaying=", this.isPlaying);
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    nextSlide() {
        if (this.currentSlideIndex < this.lessonPackage.scenes.length - 1) {
            console.log(`[VisualLearning LOG] Next button clicked. Scene index shifting up.`);
            const nextSceneIndex = this.currentSlideIndex + 1;
            const startTime = this.clipStartTimes[nextSceneIndex] || 0;
            this.seekTo(startTime);
        }
    }

    previousSlide() {
        if (this.currentSlideIndex > 0) {
            console.log(`[VisualLearning LOG] Prev button clicked. Scene index shifting down.`);
            const prevSceneIndex = this.currentSlideIndex - 1;
            const startTime = this.clipStartTimes[prevSceneIndex] || 0;
            this.seekTo(startTime);
        }
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
