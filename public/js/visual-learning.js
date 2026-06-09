/**
 * Visual Learning Mode Controller for CHADUVU-GURU
 * Handles lesson creation flow, SSE progress streaming, asset pre-loading,
 * slide display, audio playback synchronization, and player controls.
 */

class VisualLearningController {
    constructor() {
        this.lessonPackage = null;
        this.currentSlideIndex = 0;
        this.isPlaying = false;
        this.audio = null;
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
        this.slideImage = null;
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
    }

    initDOMElements() {
        // Cache DOM elements if they exist
        this.container = document.getElementById('visual-learning-container');
        this.loadingScreen = document.getElementById('vl-loading-screen');
        this.playerUI = document.getElementById('vl-player');
        this.slideImage = document.getElementById('vl-slide-image');
        this.notesOverlay = document.getElementById('vl-notes-overlay');
        this.notesBody = document.getElementById('vl-notes-body');
        this.progressBarFill = document.getElementById('vl-progress-bar-fill');
        this.progressText = document.getElementById('vl-progress-text');
        this.slidesList = document.getElementById('vl-slides-list');
        
        // Buttons
        this.playPauseBtn = document.getElementById('vl-play-pause-btn');
        this.prevBtn = document.getElementById('vl-prev-btn');
        this.nextBtn = document.getElementById('vl-next-btn');
        this.notesBtn = document.getElementById('vl-notes-btn');
        
        // Bind UI events
        if (this.playPauseBtn) this.playPauseBtn.onclick = () => this.togglePlay();
        if (this.prevBtn) this.prevBtn.onclick = () => this.previousSlide();
        if (this.nextBtn) this.nextBtn.onclick = () => this.nextSlide();
        
        const closeNotes = document.getElementById('vl-close-notes-btn');
        if (closeNotes) closeNotes.onclick = () => this.toggleNotes(false);
        if (this.notesBtn) this.notesBtn.onclick = () => this.toggleNotes();
        
        const exitBtn = document.getElementById('vl-exit-btn');
        if (exitBtn) exitBtn.onclick = () => this.destroyLesson();
        
        // Bind error exit button
        const errorExitBtn = document.getElementById('vl-error-exit-btn');
        if (errorExitBtn) errorExitBtn.onclick = () => this.destroyLesson();

        // Bind drawer tabs
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

    async startLesson(query) {
        console.log(`[VisualLearning] startLesson triggered for query: "${query}"`);
        
        if (!window.selectedBook) {
            console.warn("[VisualLearning] No book selected. Prompting user to select a class/subject.");
            alert("Please select a Class and Subject from the dropdown list first!");
            return;
        }
        
        // Reset and hide error state
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
        
        // Show container and loading screen
        if (this.container) this.container.style.display = 'flex';
        if (this.loadingScreen) this.loadingScreen.style.display = 'flex';
        if (this.playerUI) this.playerUI.style.display = 'none';
        
        // Hide landing view inside container
        const vlLanding = document.getElementById('vl-landing-view');
        if (vlLanding) vlLanding.style.display = 'none';
        
        // Hide normal chat containers and bottom input box
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
            
            console.log("[VisualLearning] Fetching from /api/visual_learning with payload:", requestPayload);

            const response = await fetch('/api/visual_learning', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
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
                buffer = lines.pop(); // Keep last incomplete line in buffer

                for (const line of lines) {
                    const cleanLine = line.trim();
                    if (cleanLine.startsWith("data:")) {
                        const dataStr = cleanLine.substring(5).trim();
                        if (dataStr === "[DONE]") {
                            break;
                        }
                        
                        try {
                            const eventData = JSON.parse(dataStr);
                            await this.handleSSEEvent(eventData);
                        } catch (err) {
                            console.error("[VisualLearning] Error parsing SSE payload:", err);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("[VisualLearning] Stream error during generation:", error);
            
            // Hide progress steps, spinners and titles
            if (loadingTitle) loadingTitle.style.display = 'none';
            if (loadingSpinner) loadingSpinner.style.display = 'none';
            if (loadingSteps) loadingSteps.style.display = 'none';
            
            // Show inline error alert inside the loading screen
            if (errContainer && errMessage) {
                errMessage.textContent = error.message || "An unexpected error occurred during lesson generation.";
                errContainer.classList.remove('hidden');
            } else {
                alert(`Error creating visual lesson: ${error.message}`);
                this.destroyLesson();
            }
        }
    }

    async handleSSEEvent(event) {
        if (event.type === 'progress') {
            const stepId = event.step;
            const status = event.status;
            
            // Mark previous steps as complete
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
            console.log("[VisualLearning] Lesson blueprint ready. Preloading media assets...", event.lesson);
            
            // Update loading screen to step 5
            const finalStep = document.getElementById('vl-step-launching_lesson');
            if (finalStep) {
                finalStep.className = 'vl-loading-step active';
            }

            this.lessonPackage = event.lesson;
            await this.preloadLessonAssets();
            this.launchPlayer();
        } else if (event.type === 'error') {
            throw new Error(event.message);
        }
    }

    resetProgressUI() {
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
        const slides = this.lessonPackage.slides;
        this.preloadedImages = {};
        this.preloadedAudios = {};

        const imagePromises = slides.map(slide => {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    this.preloadedImages[slide.slide_no] = img;
                    resolve();
                };
                img.onerror = () => {
                    console.error(`[VisualLearning] Failed to preload image: ${slide.image_url}`);
                    resolve(); // Resolve anyway so loader doesn't hang
                };
                img.src = slide.image_url;
            });
        });

        const audioPromises = slides.map(slide => {
            return new Promise((resolve) => {
                const aud = new Audio();
                aud.src = slide.audio_url;
                
                const onCanPlay = () => {
                    this.preloadedAudios[slide.slide_no] = aud;
                    cleanup();
                    resolve();
                };

                const onError = (e) => {
                    console.error(`[VisualLearning] Failed to preload audio: ${slide.audio_url}`, e);
                    cleanup();
                    resolve();
                };

                const cleanup = () => {
                    aud.removeEventListener('canplaythrough', onCanPlay);
                    aud.removeEventListener('error', onError);
                };

                aud.addEventListener('canplaythrough', onCanPlay);
                aud.addEventListener('error', onError);
                
                // Backup timer to prevent hangs
                setTimeout(() => {
                    cleanup();
                    resolve();
                }, 8000);
            });
        });

        await Promise.all([...imagePromises, ...audioPromises]);
        console.log("[VisualLearning] Preloading complete.");
    }

    launchPlayer() {
        this.loadingScreen.style.display = 'none';
        this.playerUI.style.display = 'flex';
        this.currentSlideIndex = 0;
        this.isPlaying = false;
        
        this.renderSidebar();
        this.playSlide(1);
    }

    renderSidebar() {
        this.slidesList.innerHTML = '';
        this.lessonPackage.slides.forEach((slide) => {
            const item = document.createElement('div');
            item.id = `vl-slide-item-${slide.slide_no}`;
            item.className = 'vl-slide-item upcoming';
            item.onclick = () => this.jumpToSlide(slide.slide_no);

            item.innerHTML = `
                <div class="vl-slide-status-icon">${slide.slide_no}</div>
                <div class="vl-slide-details">
                    <span class="vl-slide-title">${slide.title}</span>
                </div>
            `;
            this.slidesList.appendChild(item);
        });
    }

    updateSidebarStates() {
        this.lessonPackage.slides.forEach((slide) => {
            const item = document.getElementById(`vl-slide-item-${slide.slide_no}`);
            if (!item) return;

            const icon = item.querySelector('.vl-slide-status-icon');
            
            if (slide.slide_no < this.currentSlideIndex + 1) {
                item.className = 'vl-slide-item completed';
                if (icon) icon.textContent = '✓';
            } else if (slide.slide_no === this.currentSlideIndex + 1) {
                item.className = 'vl-slide-item current';
                if (icon) icon.textContent = slide.slide_no;
            } else {
                item.className = 'vl-slide-item upcoming';
                if (icon) icon.textContent = slide.slide_no;
            }
        });
    }

    playSlide(slideNo) {
        // Stop current audio if any
        if (this.audio) {
            this.audio.pause();
            this.audio.currentTime = 0;
            this.audio.onended = null;
            this.audio.ontimeupdate = null;
        }

        const slideIndex = slideNo - 1;
        this.currentSlideIndex = slideIndex;
        const slide = this.lessonPackage.slides[slideIndex];

        // Update image
        this.slideImage.style.opacity = 0;
        setTimeout(() => {
            this.slideImage.src = slide.image_url;
            
            // Adjust object-fit based on slide type (SVG vs JPEG)
            // If the slide is an SVG (e.g. diagram), use contain to avoid cropping labels.
            // If it is a JPG/PNG (e.g. AI generated photo), use cover to eliminate black bars.
            if (slide.image_url.toLowerCase().includes('.svg')) {
                this.slideImage.style.objectFit = 'contain';
            } else {
                this.slideImage.style.objectFit = 'cover';
            }
            
            this.slideImage.style.opacity = 1;
        }, 150);

        // Update notes
        this.notesBody.innerHTML = `<p>${slide.teacher_script}</p>`;

        // Update navigation and counts
        this.progressText.textContent = `Slide ${slideNo} / ${this.lessonPackage.slides.length}`;
        this.prevBtn.disabled = (slideNo === 1);
        this.nextBtn.disabled = (slideNo === this.lessonPackage.slides.length);
        this.updateSidebarStates();
        
        // Reset progress bar
        if (this.progressBarFill) this.progressBarFill.style.width = '0%';

        // Setup audio
        const cachedAudio = this.preloadedAudios[slideNo];
        if (cachedAudio) {
            this.audio = cachedAudio;
            this.audio.currentTime = 0;
        } else {
            this.audio = new Audio(slide.audio_url);
        }

        this.audio.onended = () => {
            if (slideNo < this.lessonPackage.slides.length) {
                this.nextSlide();
            } else {
                this.pause();
                if (this.progressBarFill) this.progressBarFill.style.width = '100%';
            }
        };

        this.audio.ontimeupdate = () => {
            if (this.audio && this.audio.duration && this.progressBarFill) {
                const pct = (this.audio.currentTime / this.audio.duration) * 100;
                this.progressBarFill.style.width = `${pct}%`;
            }
        };

        // Auto-play if state was playing
        if (this.isPlaying) {
            this.audio.play().catch(err => {
                console.error("[VisualLearning] Playback blocked by browser:", err);
                this.pause();
            });
        } else {
            // First slide loads in paused state, play it automatically
            this.play();
        }
    }

    play() {
        if (!this.audio) return;
        this.isPlaying = true;
        this.playPauseBtn.innerHTML = '<span>⏸</span> Pause';
        this.playPauseBtn.className = 'vl-btn vl-btn-primary';
        this.audio.play().catch(err => {
            console.error("[VisualLearning] Playback blocked:", err);
            this.pause();
        });
    }

    pause() {
        if (!this.audio) return;
        this.isPlaying = false;
        this.playPauseBtn.innerHTML = '<span>▶</span> Resume';
        this.playPauseBtn.className = 'vl-btn vl-btn-primary';
        this.audio.pause();
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    nextSlide() {
        if (this.currentSlideIndex < this.lessonPackage.slides.length - 1) {
            const wasPlaying = this.isPlaying;
            this.isPlaying = wasPlaying; // preserve play state
            this.playSlide(this.currentSlideIndex + 2);
        }
    }

    previousSlide() {
        if (this.currentSlideIndex > 0) {
            const wasPlaying = this.isPlaying;
            this.isPlaying = wasPlaying; // preserve play state
            this.playSlide(this.currentSlideIndex);
        }
    }

    jumpToSlide(slideNo) {
        const wasPlaying = this.isPlaying;
        this.isPlaying = wasPlaying; // preserve play state
        this.playSlide(slideNo);
    }

    toggleNotes(forceOpen) {
        if (forceOpen === undefined) {
            this.notesOverlay.classList.toggle('open');
        } else if (forceOpen) {
            this.notesOverlay.classList.add('open');
        } else {
            this.notesOverlay.classList.remove('open');
        }

        // Reset to Script tab when opened
        if (this.notesOverlay.classList.contains('open')) {
            const tabScriptBtn = document.getElementById('vl-tab-script');
            if (tabScriptBtn) tabScriptBtn.click();
        }
    }

    destroyLesson() {
        console.log("[VisualLearning] Destroying current lesson player...");
        // Stop audio
        if (this.audio) {
            this.audio.pause();
            this.audio.currentTime = 0;
            this.audio.onended = null;
            this.audio.ontimeupdate = null;
        }
        
        this.audio = null;
        this.lessonPackage = null;
        this.preloadedImages = {};
        this.preloadedAudios = {};
        this.isPlaying = false;
        
        // Hide player UI
        if (this.container) this.container.style.display = 'none';
        if (this.notesOverlay) this.notesOverlay.classList.remove('open');
        
        // Hide error container
        const errContainer = document.getElementById('vl-error-container');
        if (errContainer) errContainer.classList.add('hidden');
        
        // Restore bottom input box container
        const chatInput = document.getElementById('chat-input-container');
        if (chatInput) chatInput.style.display = '';

        // Handle layout display restoring based on active mode
        const currentMode = window.answerPreferenceManager ? window.answerPreferenceManager.currentMode : '';
        const vlLanding = document.getElementById('vl-landing-view');
        if (currentMode === 'visual_learning') {
            if (this.container) this.container.style.display = 'flex';
            if (vlLanding) vlLanding.style.display = 'flex';
        } else {
            // Restore normal chat elements and left pane if we switched away
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

// Instantiate and expose globally
window.VisualLearningRenderer = new VisualLearningController();
