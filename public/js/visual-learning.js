/**
 * Visual Learning Mode Controller for CHADUVU-GURU (V1 Scene-Based)
 * Orchestrates storyboard creation, scene asset rendering (GSAP + Lottie),
 * audio playback synchronization, aspect ratio scaling, and player states.
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
        this.lottieInstances = [];
        
        // Sync flags
        this.timelineFinished = false;
        this.audioFinished = false;
        
        this.preloadedImages = {};
        this.preloadedAudios = {};
        
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
    }

    resizeCanvas() {
        const container = document.getElementById('vl-main');
        if (!this.canvas || !container) {
            console.warn("[VisualLearning WARNING] Canvas or Main Container element missing during resize.");
            return;
        }
        
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const ratio = 800 / 450;
        
        let scale = 1;
        if (cw / ch > ratio) {
            scale = ch / 450;
        } else {
            scale = cw / 800;
        }
        
        scale *= 0.95; // Breathing room
        
        this.canvas.style.transform = `scale(${scale})`;
        this.canvas.style.transformOrigin = 'center center';
        console.log(`[VisualLearning LOG] Rescaled 16:9 Canvas to factor: ${scale.toFixed(4)}`);
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

        const imagePromises = [];
        scenes.forEach(scene => {
            const assets = scene.assets || [];
            assets.forEach(asset => {
                if (asset.type === 'image' && asset.asset_url) {
                    console.log(`[VisualLearning LOG] Preload image asset: id="${asset.id}" | URL: ${asset.asset_url}`);
                    const promise = new Promise((resolve) => {
                        const img = new Image();
                        img.onload = () => {
                            this.preloadedImages[asset.id] = img;
                            console.log(`[VisualLearning LOG] Image preloaded successfully: id="${asset.id}"`);
                            resolve();
                        };
                        img.onerror = (e) => {
                            console.error(`[VisualLearning ERROR] Preload image failed: id="${asset.id}" | URL: ${asset.asset_url}`, e);
                            resolve(); // resolve to avoid blocking UI loader loop
                        };
                        img.src = asset.asset_url;
                    });
                    imagePromises.push(promise);
                }
            });
        });

        const audioPromises = scenes.map(scene => {
            console.log(`[VisualLearning LOG] Preload audio asset: scene_no=${scene.scene_no} | URL: ${scene.audio_url}`);
            return new Promise((resolve) => {
                const aud = new Audio();
                aud.src = scene.audio_url;
                
                const onCanPlay = () => {
                    this.preloadedAudios[scene.scene_no] = aud;
                    console.log(`[VisualLearning LOG] Audio preloaded successfully: scene_no=${scene.scene_no}`);
                    cleanup();
                    resolve();
                };

                const onError = (e) => {
                    console.error(`[VisualLearning ERROR] Preload narration audio failed: scene_no=${scene.scene_no} | URL: ${scene.audio_url}`, e);
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
                    console.warn(`[VisualLearning WARNING] Preload timeout triggered for audio of scene ${scene.scene_no}`);
                    cleanup();
                    resolve();
                }, 8000);
            });
        });

        await Promise.all([...imagePromises, ...audioPromises]);
        console.log("[VisualLearning LOG] Media preloading phase completed.");
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
        this.lessonPackage.scenes.forEach((scene) => {
            const item = document.createElement('div');
            item.id = `vl-slide-item-${scene.scene_no}`;
            item.className = 'vl-slide-item upcoming';
            item.onclick = () => {
                console.log(`[VisualLearning LOG] User clicked sidebar item: Scene ${scene.scene_no}`);
                this.jumpToSlide(scene.scene_no);
            };

            item.innerHTML = `
                <div class="vl-slide-status-icon">${scene.scene_no}</div>
                <div class="vl-slide-details">
                    <span class="vl-slide-title">${scene.title}</span>
                </div>
            `;
            this.slidesList.appendChild(item);
        });
    }

    updateSidebarStates() {
        this.lessonPackage.scenes.forEach((scene) => {
            const item = document.getElementById(`vl-slide-item-${scene.scene_no}`);
            if (!item) return;

            const icon = item.querySelector('.vl-slide-status-icon');
            
            if (scene.scene_no < this.currentSlideIndex + 1) {
                item.className = 'vl-slide-item completed';
                if (icon) icon.textContent = '✓';
            } else if (scene.scene_no === this.currentSlideIndex + 1) {
                item.className = 'vl-slide-item current';
                if (icon) icon.textContent = scene.scene_no;
            } else {
                item.className = 'vl-slide-item upcoming';
                if (icon) icon.textContent = scene.scene_no;
            }
        });
    }

    updateCanvasTheme(scene) {
        if (!this.canvas) return;
        
        // Reset active glow classes
        this.canvas.classList.remove('glow-indigo', 'glow-gold', 'glow-emerald', 'glow-rose');
        
        const titleLower = (scene.title || "").toLowerCase();
        const assetsKeywords = (scene.assets || []).map(a => (a.search_query || "").toLowerCase()).join(" ");
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
        
        this.canvas.classList.add(`glow-${theme}`);
        this.canvas.style.background = radialGradient;
        console.log(`[VisualLearning LOG] Applied Canvas Theme: "${theme}"`);
    }

    playSlide(sceneNo) {
        console.log(`[VisualLearning LOG] PlaySlide initiated for Scene #${sceneNo}`);

        // Stop active audio
        if (this.audio) {
            console.log("[VisualLearning LOG] Pausing and resetting active audio.");
            this.audio.pause();
            this.audio.currentTime = 0;
            this.audio.onended = null;
            this.audio.ontimeupdate = null;
        }

        // Kill active GSAP timeline
        if (this.timeline) {
            console.log("[VisualLearning LOG] Killing active GSAP timeline.");
            this.timeline.kill();
            this.timeline = null;
        }

        // Destroy active Lottie animations
        if (this.lottieInstances) {
            console.log(`[VisualLearning LOG] Cleaning up ${this.lottieInstances.length} Lottie instances.`);
            this.lottieInstances.forEach(instance => {
                if (instance && typeof instance.destroy === 'function') {
                    instance.destroy();
                }
            });
        }
        this.lottieInstances = [];

        // Clear Canvas elements
        if (this.canvas) {
            this.canvas.innerHTML = '';
            console.log("[VisualLearning LOG] Canvas cleared.");
        }

        const sceneIndex = sceneNo - 1;
        this.currentSlideIndex = sceneIndex;
        const scene = this.lessonPackage.scenes[sceneIndex];

        // Apply theme color & gradients dynamically based on scene content
        this.updateCanvasTheme(scene);

        // Reset sync statuses
        this.timelineFinished = false;
        this.audioFinished = false;

        // Render Scene Title Overlay
        const titleCard = document.createElement('div');
        titleCard.style.cssText = `
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(8px);
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
            color: #e2e8f0;
            font-size: 14px;
            font-weight: 600;
            z-index: 10;
        `;
        titleCard.textContent = scene.title;
        this.canvas.appendChild(titleCard);

        // Render Assets onto 16:9 Canvas
        const assets = scene.assets || [];
        console.log(`[VisualLearning LOG] Scene #${sceneNo} has ${assets.length} assets. Rendering...`);
        
        assets.forEach(asset => {
            const el = document.createElement('div');
            el.id = `asset-${asset.id}`;
            el.className = 'vl-asset';
            
            const layout = asset.layout || {};
            el.style.top = layout.top || '0%';
            el.style.left = layout.left || '0%';
            el.style.width = layout.width || 'auto';
            if (layout.height) el.style.height = layout.height;
            
            // Nested content wrapper to apply floating animation separate from GSAP
            const contentWrapper = document.createElement('div');
            contentWrapper.className = 'vl-asset-content idle-floating';
            contentWrapper.style.width = '100%';
            contentWrapper.style.height = '100%';
            contentWrapper.style.position = 'relative';
            el.appendChild(contentWrapper);
            
            console.log(`[VisualLearning LOG] Render asset: "${asset.id}" | Type: ${asset.type} | Top: ${layout.top}, Left: ${layout.left}, Width: ${layout.width}`);
            
            if (asset.type === 'image') {
                const img = document.createElement('img');
                img.src = asset.asset_url || '/static/favicon.svg';
                img.alt = asset.id;
                contentWrapper.appendChild(img);
            } else if (asset.type === 'lottie') {
                const lottieWrapper = document.createElement('div');
                lottieWrapper.className = 'vl-asset-lottie';
                lottieWrapper.style.width = '100%';
                lottieWrapper.style.height = '100%';
                contentWrapper.appendChild(lottieWrapper);
                
                const instance = loadLottieAsset(lottieWrapper, asset.search_query);
                if (instance) {
                    this.lottieInstances.push(instance);
                }
            }
            this.canvas.appendChild(el);
        });

        // Initialize GSAP Scene Timeline
        console.log("[VisualLearning LOG] Instantiating new GSAP Timeline.");
        this.timeline = gsap.timeline({ paused: true });
        
        let hasTimelineAnimations = false;
        
        assets.forEach(asset => {
            const el = document.getElementById(`asset-${asset.id}`);
            if (!el) return;
            
            const anims = asset.animations || [];
            anims.forEach(anim => {
                const type = anim.type;
                const duration = anim.duration || 1.0;
                const delay = anim.delay || 0.0;
                
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
                            this.timeline.fromTo(el, { y: 150 }, { y: 0, duration, ease: "power2.out" }, delay);
                            break;
                        case 'move_down':
                            this.timeline.fromTo(el, { y: -150 }, { y: 0, duration, ease: "power2.out" }, delay);
                            break;
                        case 'move_left':
                            this.timeline.fromTo(el, { x: 150 }, { x: 0, duration, ease: "power2.out" }, delay);
                            break;
                        case 'move_right':
                            this.timeline.fromTo(el, { x: -150 }, { x: 0, duration, ease: "power2.out" }, delay);
                            break;
                        case 'scale_up':
                            this.timeline.fromTo(el, { scale: 0 }, { scale: 1, duration, ease: "back.out(1.5)" }, delay);
                            break;
                        case 'scale_down':
                            this.timeline.to(el, { scale: 0, duration, ease: "power2.in" }, delay);
                            break;
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
                            console.error(`[VisualLearning ERROR] Unsupported animation token bypassed: "${type}"`);
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

        // Update script notes
        if (this.notesBody) {
            this.notesBody.innerHTML = `<p>${scene.teacher_script}</p>`;
        }

        // Navigation state updates
        if (this.progressText) {
            this.progressText.textContent = `Scene ${sceneNo} / ${this.lessonPackage.scenes.length}`;
        }
        if (this.prevBtn) this.prevBtn.disabled = (sceneNo === 1);
        if (this.nextBtn) this.nextBtn.disabled = (sceneNo === this.lessonPackage.scenes.length);
        this.updateSidebarStates();
        
        if (this.progressBarFill) this.progressBarFill.style.width = '0%';

        // Setup Narration audio
        const cachedAudio = this.preloadedAudios[sceneNo];
        if (cachedAudio) {
            console.log(`[VisualLearning LOG] Playing preloaded audio for Scene #${sceneNo}`);
            this.audio = cachedAudio;
            this.audio.currentTime = 0;
        } else {
            console.warn(`[VisualLearning WARNING] Audio for Scene #${sceneNo} was not cached. Initiating raw load.`);
            this.audio = new Audio(scene.audio_url);
        }

        this.audio.onended = () => {
            console.log(`[VisualLearning LOG] Narration Audio ended for Scene #${sceneNo}`);
            this.audioFinished = true;
            this.checkAutoAdvance();
        };

        this.audio.ontimeupdate = () => {
            if (this.audio && this.audio.duration && this.progressBarFill) {
                const pct = (this.audio.currentTime / this.audio.duration) * 100;
                this.progressBarFill.style.width = `${pct}%`;
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
            this.lottieInstances.forEach(inst => inst.play());
        } else {
            console.log("[VisualLearning LOG] Launching first play triggered.");
            this.play();
        }
        
        this.resizeCanvas();
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
        this.lottieInstances.forEach(inst => inst.play());
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
        this.lottieInstances.forEach(inst => inst.pause());
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
            this.playSlide(this.currentSlideIndex + 2);
        }
    }

    previousSlide() {
        if (this.currentSlideIndex > 0) {
            console.log(`[VisualLearning LOG] Prev button clicked. Scene index shifting down.`);
            this.playSlide(this.currentSlideIndex);
        }
    }

    jumpToSlide(sceneNo) {
        console.log(`[VisualLearning LOG] JumpToSlide called for scene #${sceneNo}`);
        this.playSlide(sceneNo);
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

        if (this.lottieInstances) {
            this.lottieInstances.forEach(instance => {
                if (instance && typeof instance.destroy === 'function') {
                    instance.destroy();
                }
            });
        }
        this.lottieInstances = [];
        
        this.lessonPackage = null;
        this.preloadedImages = {};
        this.preloadedAudios = {};
        this.isPlaying = false;
        
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
