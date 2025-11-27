document.addEventListener('DOMContentLoaded', () => {
    // Check which page we are on and run the appropriate setup function
    if (document.getElementById('admin-form')) {
        setupAdminPage();
    } else if (document.getElementById('chapters-form')) {
        setupChaptersPage();
    } else if (document.getElementById('user-query-form')) {
        setupUserPage();
    }

    // Global visibility change handler to stop TTS
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && window.speechSynthesis) {
            console.log('[Global] Tab hidden, stopping TTS.');
            window.speechSynthesis.cancel();
            // Reset all speak buttons
            document.querySelectorAll('.speak-btn').forEach(btn => btn.textContent = '🔊');
        }
    });
});

/**
 * Sets up the main admin page (uploading class, subject, and PDF).
 */
function setupAdminPage() {
    const adminForm = document.getElementById('admin-form');

    adminForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        showStatus('Uploading PDF...', 'info');

        const pdfFile = document.getElementById('pdf-file').files[0];
        const className = document.getElementById('class').value;
        const subject = document.getElementById('subject').value;

        if (!pdfFile || !className || !subject) {
            showStatus('Please fill out all fields and select a PDF.', 'error');
            return;
        }

        const uploadFormData = new FormData();
        uploadFormData.append('file', pdfFile);

        try {
            // Step 1: Upload the file
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: uploadFormData,
            });
            const uploadResult = await response.json();
            if (!response.ok) {
                throw new Error(uploadResult.detail || 'Failed to upload file.');
            }

            // Step 2: Redirect to the chapters page with data in URL
            const queryParams = new URLSearchParams({
                filename: uploadResult.filename,
                class_name: className,
                subject: subject
            });
            window.location.href = `/chapters?${queryParams.toString()}`;

        } catch (error) {
            showStatus(`Upload failed: ${error.message}`, 'error');
        }
    });
}

/**
 * Sets up the chapters definition page (PDF viewer and chapter form).
 */
function setupChaptersPage() {
    const params = new URLSearchParams(window.location.search);
    const filename = params.get('filename');
    const className = params.get('class_name'); // Changed to 'class_name'
    const subject = params.get('subject');

    if (!filename) {
        document.body.innerHTML = '<h1 style="color: red; text-align: center;">Error: No PDF file specified. Please go back to the admin page and upload a file.</h1>';
        return;
    }

    const pdfUrl = `/uploads/${filename}`;
    const chaptersForm = document.getElementById('chapters-form');

    // PDF.js state
    let pdfDoc = null;
    let pageNum = 1;
    let pageRendering = false;
    let pageNumPending = null;
    const scale = 1.5;
    const canvas = document.getElementById('pdf-canvas');
    const ctx = canvas.getContext('2d');

    /**
     * Get page info from document, resize canvas accordingly, and render page.
     */
    function renderPage(num) {
        pageRendering = true;
        document.getElementById('pdf-loading-message').style.display = 'block';

        // Using promise to fetch the page
        pdfDoc.getPage(num).then(function (page) {
            const container = document.getElementById('pdf-render-area');
            const unscaledViewport = page.getViewport({ scale: 1 });

            // Dynamically calculate scale to fit container width
            const scale = container.clientWidth / unscaledViewport.width;
            const viewport = page.getViewport({ scale: scale });

            canvas.height = viewport.height;
            canvas.width = viewport.width;

            // Render PDF page into canvas context
            const renderContext = {
                canvasContext: ctx,
                viewport: viewport
            };
            const renderTask = page.render(renderContext);

            // Wait for rendering to finish
            renderTask.promise.then(function () {
                pageRendering = false;
                document.getElementById('pdf-loading-message').style.display = 'none';
                if (pageNumPending !== null) {
                    // New page rendering is pending
                    renderPage(pageNumPending);
                    pageNumPending = null;
                }
            });
        });

        // Update page counters
        document.getElementById('page-num').textContent = num;
    }

    /**
     * If another page rendering in progress, waits until the rendering is
     * finished. Otherwise, executes rendering immediately.
     */
    function queueRenderPage(num) {
        if (pageRendering) {
            pageNumPending = num;
        } else {
            renderPage(num);
        }
    }

    // Load the PDF
    pdfjsLib.getDocument(pdfUrl).promise.then(function (pdfDoc_) {
        pdfDoc = pdfDoc_;
        document.getElementById('page-count').textContent = pdfDoc.numPages;
        renderPage(pageNum);
    }).catch(err => {
        showStatus(`Error loading PDF: ${err.message}`, 'error');
        document.getElementById('pdf-loading-message').textContent = 'Error loading PDF.';
    });

    // Button events
    document.getElementById('prev-page').addEventListener('click', () => {
        if (pageNum <= 1) return;
        pageNum--;
        queueRenderPage(pageNum);
    });

    document.getElementById('next-page').addEventListener('click', () => {
        if (pageNum >= pdfDoc.numPages) return;
        pageNum++;
        queueRenderPage(pageNum);
    });

    // Chapter input generation
    const numChaptersInput = document.getElementById('num-chapters');
    const chaptersTableBody = document.getElementById('chapters-table-body');

    function createChapterRow() {
        const row = document.createElement('tr');
        row.classList.add('chapter-entry');
        row.innerHTML = `
            <td><input type="text" class="chapter-name" placeholder="e.g., Introduction" required></td>
            <td><input type="number" class="start-page" placeholder="e.g., 1" min="1" required></td>
            <td><input type="number" class="end-page" placeholder="e.g., 10" min="1" required></td>
            <td><button type="button" class="remove-chapter-btn">Remove</button></td>
        `;

        row.querySelector('.remove-chapter-btn').addEventListener('click', () => {
            row.remove();
        });

        return row;
    }

    numChaptersInput.addEventListener('input', () => {
        const count = parseInt(numChaptersInput.value, 10);
        chaptersTableBody.innerHTML = ''; // Clear existing rows

        if (count > 0) {
            for (let i = 0; i < count; i++) {
                chaptersTableBody.appendChild(createChapterRow());
            }
        }
    });


    // Final form submission
    chaptersForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Clear previous errors
        document.querySelectorAll('#chapters-table-body .input-error').forEach(el => el.classList.remove('input-error'));

        const chapterEntries = document.querySelectorAll('#chapters-table-body tr');
        const chapters = [];
        let validationError = false;

        if (chapterEntries.length === 0) {
            showStatus('Please add at least one chapter.', 'error');
            return;
        }

        chapterEntries.forEach(entry => {
            const nameInput = entry.querySelector('.chapter-name');
            const startPageInput = entry.querySelector('.start-page');
            const endPageInput = entry.querySelector('.end-page');

            const name = nameInput.value;
            const start_page = parseInt(startPageInput.value, 10);
            const end_page = parseInt(endPageInput.value, 10);

            let hasRowError = false;
            if (!name) {
                nameInput.classList.add('input-error');
                hasRowError = true;
            }
            if (isNaN(start_page) || start_page <= 0) {
                startPageInput.classList.add('input-error');
                hasRowError = true;
            }
            if (isNaN(end_page) || end_page < start_page) {
                endPageInput.classList.add('input-error');
                hasRowError = true;
            }

            if (hasRowError) {
                validationError = true;
            } else {
                // Send chapter pages - backend will calculate PDF pages using offset
                chapters.push({
                    chapter_name: name,
                    chpstpage: start_page,  // Chapter start page
                    chpendpage: end_page    // Chapter end page
                });
            }
        });

        if (validationError) {
            showStatus('Please fix the errors in the highlighted fields.', 'error');
            return;
        }

        showStatus('Processing book and chapters...', 'info');

        const finalData = {
            class_name: className,
            subject: subject,
            filename: filename,
            chapters: chapters
        };

        try {
            const response = await fetch('/api/books', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(finalData)
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Failed to process book.');

            const finalMessage = "Processing started in the background. You can now safely leave this page. The book will be available in a few minutes.";
            showStatus(finalMessage, 'success');

            chaptersForm.reset();
            chaptersTableBody.innerHTML = '';
            numChaptersInput.value = ''; // Clear the number input

        } catch (error) {
            showStatus(`Error: ${error.message}`, 'error');
        }
    });

    // Add event listeners to clear errors on input
    chaptersTableBody.addEventListener('input', (e) => {
        if (e.target.classList.contains('input-error')) {
            e.target.classList.remove('input-error');
        }
    });
}

/**
 * Sets up the main user query page wizard.
 */
function setupUserPage() {
    // --- Element Selectors ---
    const classSelect = document.getElementById('class-select');
    const subjectSelect = document.getElementById('subject-select');
    const viewerPlaceholder = document.getElementById('viewer-placeholder');
    const pdfLoadingIndicator = document.getElementById('pdf-loading-user');
    const pdfCanvas = document.getElementById('pdf-canvas-user');
    const pdfHeader = document.getElementById('pdf-viewer-header-user');
    const pageNumEl = document.getElementById('page-num-user');
    const pageCountEl = document.getElementById('page-count-user');
    const prevPageBtn = document.getElementById('prev-page-user');
    const nextPageBtn = document.getElementById('next-page-user');
    const chatHistory = document.getElementById('chat-history');
    const queryForm = document.getElementById('user-query-form');
    const queryText = document.getElementById('query-text');
    const submitButton = document.getElementById('submit-query-btn');
    const listChaptersBtn = document.getElementById('list-chapters-btn');
    const conversationalModeBtn = document.getElementById('conversational-mode-btn');
    const voiceSearchBtn = document.getElementById('voice-search-btn');
    const voiceStatus = document.getElementById('voice-status');
    const voiceVisualizer = document.getElementById('voice-visualizer');
    let hideVoiceVisualizerTimeout = null;
    const ctx = pdfCanvas.getContext('2d');

    // --- App State ---
    let selectedBook = null;
    let pdfDoc = null;
    let pageNum = 1;
    let pageRendering = false;
    let pageNumPending = null;
    let isFirstQuery = true;
    // Removed isSpeakingStream and sentenceQueue

    // --- Smart Conversational Context State ---
    let currentSessionId = null;
    let turnCount = 0;
    let currentFollowUps = [];

    // --- Voice Search State (simple mode) ---
    let simpleRecognition;
    let isSimpleRecording = false;

    // --- Initialization ---
    setupSimpleVoiceSearch();
    createFollowupVoiceOverlay();

    // --- Event Listeners ---
    classSelect.addEventListener('change', () => {
        const selectedClass = classSelect.value;
        if (selectedClass) {
            populateSubjects(selectedClass);
        } else {
            subjectSelect.innerHTML = '<option value="">Select Subject</option>';
            subjectSelect.disabled = true;
        }
        resetUI();
    });

    subjectSelect.addEventListener('change', () => loadBook());
    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handleQuerySubmit();
    });
    listChaptersBtn.addEventListener('click', () => handleListChapters());
    prevPageBtn.addEventListener('click', () => {
        if (pageNum <= 1) return;
        pageNum--;
        queueRenderPage(pageNum);
    });
    nextPageBtn.addEventListener('click', () => {
        if (pdfDoc && pageNum >= pdfDoc.numPages) return;
        pageNum++;
        queueRenderPage(pageNum);
    });
    queryText.addEventListener('input', () => {
        queryText.style.height = 'auto';
        queryText.style.height = (queryText.scrollHeight) + 'px';
    });

    // --- Voice Search Setup ---
    function setupSimpleVoiceSearch() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            voiceSearchBtn.disabled = true;
            voiceSearchBtn.title = "Voice search not supported";
            return;
        }

        simpleRecognition = new SpeechRecognition();
        simpleRecognition.interimResults = true;
        simpleRecognition.lang = 'en-US';

        const scheduleFrame = typeof window !== 'undefined' && window.requestAnimationFrame
            ? window.requestAnimationFrame.bind(window)
            : (cb) => setTimeout(cb, 16);

        function showVoiceVisualizer() {
            if (!voiceVisualizer) return;
            if (hideVoiceVisualizerTimeout) {
                clearTimeout(hideVoiceVisualizerTimeout);
                hideVoiceVisualizerTimeout = null;
            }
            voiceVisualizer.classList.remove('hidden');
            scheduleFrame(() => voiceVisualizer.classList.add('active'));
        }

        function hideVoiceVisualizer() {
            if (!voiceVisualizer) return;
            voiceVisualizer.classList.remove('active');
            hideVoiceVisualizerTimeout = window.setTimeout(() => {
                voiceVisualizer.classList.add('hidden');
            }, 220);
        }

        simpleRecognition.onstart = () => {
            isSimpleRecording = true;
            voiceStatus.textContent = 'Recording...';
            voiceStatus.classList.remove('hidden');
            voiceSearchBtn.classList.remove('bg-gray-200', 'hover:bg-gray-300');
            voiceSearchBtn.classList.add('bg-red-500', 'hover:bg-red-600');
            voiceSearchBtn.classList.add('recording');
            showVoiceVisualizer();
        };

        simpleRecognition.onend = () => {
            isSimpleRecording = false;
            voiceStatus.classList.add('hidden');
            voiceSearchBtn.classList.remove('bg-red-500', 'hover:bg-red-600');
            voiceSearchBtn.classList.add('bg-gray-200', 'hover:bg-gray-300');
            voiceSearchBtn.classList.remove('recording');
            hideVoiceVisualizer();
        };

        simpleRecognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            voiceStatus.textContent = `Error: ${event.error}`;
            isSimpleRecording = false;
            voiceStatus.classList.remove('hidden');
            voiceSearchBtn.classList.remove('bg-red-500', 'hover:bg-red-600');
            voiceSearchBtn.classList.add('bg-gray-200', 'hover:bg-gray-300');
            voiceSearchBtn.classList.remove('recording');
            hideVoiceVisualizer();
        };

        simpleRecognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            // Update the queryText with both final and interim results for live feedback
            queryText.value = finalTranscript + interimTranscript;
        };

        voiceSearchBtn.addEventListener('click', () => {
            if (isSimpleRecording) {
                simpleRecognition.stop();
            } else {
                // Prevent conflict with conversational mode
                if (window.conversationMode && window.conversationMode.isRecording) {
                    alert("Please stop the conversational mode first.");
                    return;
                }
                try {
                    simpleRecognition.start();
                } catch (e) {
                    console.error("Could not start recognition:", e);
                    voiceStatus.textContent = "Mic error.";
                    voiceStatus.classList.remove('hidden');
                }
            }
        });
    }

    // --- Core Functions ---
    async function populateSubjects(className) {
        try {
            const response = await fetch(`/api/books?class_name=${className}`);
            if (!response.ok) throw new Error('Failed to fetch subjects');
            const books = await response.json();
            const subjects = [...new Set(books.map(book => book.subject))];
            subjectSelect.innerHTML = '<option value="">Select Subject</option>';
            subjects.forEach(subject => {
                const option = document.createElement('option');
                option.value = subject;
                option.textContent = subject;
                subjectSelect.appendChild(option);
            });
            subjectSelect.disabled = false;
        } catch (error) {
            console.error('Error fetching subjects:', error);
        }
    }

    function resetUI() {
        pdfDoc = null;
        selectedBook = null;
        pageNum = 1;
        pdfCanvas.style.display = 'none';
        pdfHeader.style.display = 'none';
        viewerPlaceholder.style.display = 'flex';
        pdfLoadingIndicator.style.display = 'none';
        queryText.disabled = true;
        submitButton.disabled = true;
        listChaptersBtn.classList.add('hidden');
        if (conversationalModeBtn) {
            conversationalModeBtn.disabled = true;
            conversationalModeBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
        queryText.placeholder = 'Ask a question about the selected book...';
    }

    async function loadBook() {
        const className = classSelect.value;
        const subject = subjectSelect.value;
        if (!className || !subject) return;

        resetUI();
        viewerPlaceholder.style.display = 'none';
        pdfLoadingIndicator.style.display = 'flex';

        try {
            const response = await fetch(`/api/books?class_name=${className}&subject=${subject}`);
            if (!response.ok) throw new Error('Book not found.');
            const books = await response.json();
            if (books.length === 0) throw new Error('Book not found for this selection.');

            selectedBook = books[0];
            window.selectedBook = selectedBook;

            queryText.disabled = false;
            submitButton.disabled = false;
            listChaptersBtn.classList.remove('hidden');
            if (conversationalModeBtn) {
                conversationalModeBtn.disabled = false;
                conversationalModeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }

            const pdfUrl = `/uploads/${selectedBook.filename}`;
            pdfDoc = await pdfjsLib.getDocument(pdfUrl).promise;

            pdfLoadingIndicator.style.display = 'none';
            pdfCanvas.style.display = 'block';
            pdfHeader.style.display = 'flex';
            pageCountEl.textContent = pdfDoc.numPages;
            renderPage(pageNum);

            chatHistory.innerHTML = '';
            isFirstQuery = false;

            // Reset session when book changes
            currentSessionId = null;
            turnCount = 0;
            currentFollowUps = [];

            // appendAIResponse now handles speech based on isSpeechEnabledByDefault
            appendAIResponse(`Book "${selectedBook.subject}" loaded. You can now ask questions about it.`, `Book ${selectedBook.subject} loaded. You can now ask questions about it.`);

        } catch (error) {
            pdfLoadingIndicator.style.display = 'none';
            viewerPlaceholder.style.display = 'flex';
            viewerPlaceholder.innerHTML = `<p class="error-message">${error.message}</p>`;
            console.error(error);
        }
    }

    async function renderPage(num) {
        pageRendering = true;
        pdfLoadingIndicator.style.display = 'flex';
        const page = await pdfDoc.getPage(num);
        const container = document.getElementById('pdf-render-area-user');
        const viewport = page.getViewport({ scale: container.clientWidth / page.getViewport({ scale: 1 }).width });
        const outputScale = window.devicePixelRatio || 1;

        pdfCanvas.width = Math.floor(viewport.width * outputScale);
        pdfCanvas.height = Math.floor(viewport.height * outputScale);
        pdfCanvas.style.width = Math.floor(viewport.width) + 'px';
        pdfCanvas.style.height = Math.floor(viewport.height) + 'px';

        const renderContext = {
            canvasContext: ctx,
            viewport: viewport,
            transform: [outputScale, 0, 0, outputScale, 0, 0]
        };
        const renderTask = page.render(renderContext);
        await renderTask.promise;
        pageRendering = false;
        pdfLoadingIndicator.style.display = 'none';
        if (pageNumPending !== null) {
            renderPage(pageNumPending);
            pageNumPending = null;
        }
        pageNumEl.textContent = num;
    }

    /**
     * If another page rendering in progress, waits until the rendering is
     * finished. Otherwise, executes rendering immediately.
     */
    function queueRenderPage(num) {
        if (pageRendering) {
            pageNumPending = num;
        } else {
            renderPage(num);
        }
    }

    function addUserMessage(text) {
        const messageEl = document.createElement('div');
        messageEl.className = 'user-message p-3 bg-blue-100 rounded-lg self-end max-w-xl fade-in';
        messageEl.textContent = text;
        chatHistory.appendChild(messageEl);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function handleQuerySubmit() {
        const query = queryText.value.trim();
        if (!query || !selectedBook) return;

        // Use the new smart query system
        await submitSmartQuery(query, false);

        queryText.value = '';
        queryText.style.height = 'auto'; // Reset height
    }

    /**
     * Smart Query Submission with Conversational Context
     * Connects to /api/smart_query endpoint with session management
     */
    async function submitSmartQuery(query, isClickedFollowup = false) {
        if (!selectedBook) return;

        if (isFirstQuery) {
            chatHistory.innerHTML = '';
            isFirstQuery = false;
        }

        // Add user message
        addUserMessage(query);
        submitButton.setAttribute('disabled', 'true');
        listChaptersBtn.classList.add('hidden');

        // Create AI message card with loading state
        let intentType = 'independent';
        let followups = [];
        let fullResponse = "";
        let fullReadText = "";

        const thinkingCard = createAIMessageCard(turnCount + 1, 'loading');
        chatHistory.appendChild(thinkingCard);
        const contentDiv = thinkingCard.querySelector('.markdown-content');
        contentDiv.innerHTML = marked.parse('...');

        // Build request URL
        const params = new URLSearchParams({
            book_uuid: selectedBook.id,
            query: query,
            class_name: selectedBook.class_name,
            subject: selectedBook.subject,
            is_clicked_followup: isClickedFollowup.toString()
        });

        if (currentSessionId) {
            params.append('session_id', currentSessionId);
        }

        const source = new EventSource(`/api/smart_query?${params.toString()}`);

        source.onmessage = function (event) {
            if (event.data === "[DONE]") {
                source.close();
                submitButton.removeAttribute('disabled');
                listChaptersBtn.classList.remove('hidden');

                // Update turn counter and UI
                turnCount++;
                const headerEl = thinkingCard.querySelector('.ai-card-header');
                if (headerEl) {
                    const turnIndicator = headerEl.querySelector('.turn-indicator');
                    if (turnIndicator) {
                        turnIndicator.textContent = `Turn ${turnCount} of ${turnCount}`;
                    }
                }

                // Speak the response
                if (fullReadText) {
                    const utterance = new SpeechSynthesisUtterance(fullReadText);
                    speechSynthesis.cancel();
                    speechSynthesis.speak(utterance);
                }

                chatHistory.scrollTop = chatHistory.scrollHeight;
                return;
            }

            try {
                const data = JSON.parse(event.data);

                if (data.type === 'intent') {
                    const intentType = data.intent || 'unknown';
                    updateIntentBadge(thinkingCard, intentType);
                }

                if (data.type === 'followups') {
                    followups = data.followups || [];
                    currentFollowUps = followups;
                    // Add follow-up UI to sticky panel (NO auto-scroll)
                    addFollowUpsUI(thinkingCard, followups);
                }

                if (data.type === 'metadata') {
                    if (data.session_id) {
                        currentSessionId = data.session_id;
                    }
                    if (data.turn) {
                        turnCount = data.turn;
                    }
                }

                if (data.display_text) {
                    fullResponse += data.display_text;
                    contentDiv.innerHTML = marked.parse(fullResponse);

                    // Only scroll if displaying content (NOT for follow-ups)
                    // Check if user is near bottom before scrolling
                    const isNearBottom = (chatHistory.scrollHeight - chatHistory.scrollTop - chatHistory.clientHeight) < 100;
                    if (isNearBottom) {
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    }
                }

                if (data.read_text) {
                    fullReadText += data.read_text;
                }

                if (data.error) {
                    contentDiv.innerHTML = `<p class="error-message">Error: ${data.error}</p>`;
                    source.close();
                    submitButton.removeAttribute('disabled');
                    listChaptersBtn.classList.remove('hidden');
                }

            } catch (e) {
                console.error('Error parsing SSE data:', e, event.data);
            }
        };

        source.onerror = function (error) {
            console.error('EventSource failed:', error);
            contentDiv.innerHTML = `<p class="error-message">Connection error. Please try again.</p>`;
            source.close();
            submitButton.removeAttribute('disabled');
            listChaptersBtn.classList.remove('hidden');
        };
    }

    /**
     * Create AI Message Card with Turn Counter and Intent Badge
     */
    function createAIMessageCard(turnNumber, initialIntent = 'loading') {
        const messageDiv = document.createElement("div");
        messageDiv.className = "ai-card fade-in";

        const header = `
            <div class="ai-card-header">
                <div class="flex items-center gap-2">
                    <h2 class="font-semibold text-gray-700">🤖 AI Response</h2>
                    <span class="intent-badge ${initialIntent}" style="display: none;"></span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="turn-indicator">Turn ${turnNumber}</span>
                    <button class="copy-btn" onclick="copyMessage(this)">📋</button>
                    <button class="speak-btn" onclick="speakMessage(this)">🔊</button>
                </div>
            </div>
            <div class="markdown-content"></div>
            <div class="followup-section" style="display: none;"></div>
        `;

        messageDiv.innerHTML = header;
        return messageDiv;
    }

    /**
     * Update Intent Badge when intent is received from backend
     */
    function updateIntentBadge(cardElement, intentType) {
        const badge = cardElement.querySelector('.intent-badge');
        if (!badge) return;

        badge.style.display = 'inline-flex';
        badge.className = `intent-badge ${intentType}`;

        if (intentType === 'followup') {
            badge.innerHTML = '🔄 Follow-up';
        } else if (intentType === 'independent') {
            badge.innerHTML = '✨ New Topic';
        }
    }

    /**
     * Add Follow-up Suggestions UI to AI Card
     */
    function addFollowUpsUI(cardElement, followups) {
        if (!followups || followups.length === 0) return;

        const followupSection = cardElement.querySelector('.followup-section');
        if (!followupSection) return;

        // Show section
        followupSection.style.display = 'block';

        // Fun emoji array
        const emojis = ['💡', '🤔', '🔍', '⭐', '🎯', '💭', '🌟', '✨'];

        let html = `
            <div class="followup-header">
                <span class="icon">💡</span>
                <h4>Quick Follow-ups</h4>
            </div>
            <div class="followup-chips-container">
        `;

        // Add follow-up chips
        followups.forEach((followup, index) => {
            const escapedFollowup = followup.replace(/'/g, "\\'");
            const emoji = emojis[index % emojis.length];
            html += `
                <button class="followup-chip" onclick="handleFollowupClick('${escapedFollowup}')">
                    <span class="followup-chip-icon">${emoji}</span>
                    <span class="followup-chip-text">${followup}</span>
                </button>
            `;
        });

        html += `</div>`;
        
        // Removed custom input field as per user request (redundant with main chat)

        followupSection.innerHTML = html;

        // Hide sticky panel if it exists (cleanup)
        const stickyPanel = document.getElementById('followup-sticky-panel');
        if (stickyPanel) {
            stickyPanel.classList.add('hidden');
        }
    }

    // Handle sticky panel input
    window.handleStickyFollowup = function (input) {
        const question = input.value.trim();
        if (!question) return;

        const queryText = document.getElementById('query-text');
        if (!queryText) return;

        queryText.value = question;
        input.value = '';

        // Trigger form submission
        const form = document.getElementById('user-query-form');
        if (form) {
            const event = new Event('submit', { bubbles: true, cancelable: true });
            form.dispatchEvent(event);
        }
    };

    // Toggle sticky panel collapsed state
    window.toggleFollowupPanel = function () {
        const panel = document.getElementById('followup-sticky-panel');
        if (panel) {
            panel.classList.toggle('collapsed');
        }
    };


    async function handleListChapters() {
        if (!selectedBook) return;

        if (isFirstQuery) {
            chatHistory.innerHTML = '';
            isFirstQuery = false;
        }

        addUserMessage('List all chapters');
        const thinkingMessage = appendAIResponse('Fetching chapters...', 'Fetching chapters'); // Pass initial read text as well

        submitButton.setAttribute('disabled', 'true');
        listChaptersBtn.classList.add('hidden');

        try {
            const className = classSelect.value;
            const subject = subjectSelect.value;
            const response = await fetch(`/api/list-chapters?class_name=${className}&subject=${subject}`);

            if (!response.ok) {
                const errorResult = await response.json();
                throw new Error(errorResult.detail || 'Failed to get chapters.');
            }

            const result = await response.json();
            let chapters = result.chapters;

            if (!chapters || chapters.length === 0) {
                throw new Error("No chapters were found for this book in the database.");
            }

            chapters.sort((a, b) => a.start_page - b.start_page);

            let tableMd = `
| S.No. | Chapter Name | Pages |
|---|---|---|
`;
            chapters.forEach((chapter, index) => {
                tableMd += `| ${index + 1} | ${chapter.name} | ${chapter.start_page} - ${chapter.end_page} |\n`;
            });

            const formatted = marked.parse(tableMd);
            thinkingMessage.querySelector('.markdown-content').innerHTML = formatted;
            // Always speak the chapters list
            speechSynthesis.cancel();
            speechSynthesis.speak(new SpeechSynthesisUtterance("Here are the chapters."));

        } catch (error) {
            thinkingMessage.querySelector('.markdown-content').innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}</p>`;
            // Always speak the error message
            speechSynthesis.cancel();
            speechSynthesis.speak(new SpeechSynthesisUtterance(`Sorry, an error occurred: ${error.message}`));
        } finally {
            submitButton.removeAttribute('disabled');
            listChaptersBtn.classList.remove('hidden');
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }
}

/**
 * Global Helper Functions for Smart Follow-ups
 */

// Toggle follow-up suggestions panel
window.toggleFollowups = function (header) {
    const chips = header.nextElementSibling;
    const icon = header.querySelector('.toggle-icon');

    if (chips.style.display === 'none') {
        chips.style.display = 'flex';
        icon.textContent = '▼';
        icon.classList.remove('collapsed');
    } else {
        chips.style.display = 'none';
        icon.textContent = '▶';
        icon.classList.add('collapsed');
    }
};

// Handle follow-up chip click
window.handleFollowupClick = async function (question) {
    const queryText = document.getElementById('query-text');
    if (!queryText) return;

    // Get the submitSmartQuery function from the setupUserPage scope
    // We need to trigger a smart query with isClickedFollowup=true
    queryText.value = question;

    // Find the user page's submit handler
    const form = document.getElementById('user-query-form');
    if (form) {
        // Programmatically trigger the form submission
        // which calls handleQuerySubmit -> submitSmartQuery
        const event = new Event('submit', { bubbles: true, cancelable: true });
        form.dispatchEvent(event);
    }
};

// Handle inline follow-up input
window.handleInlineFollowup = function (input) {
    const question = input.value.trim();
    if (!question) return;

    const queryText = document.getElementById('query-text');
    if (!queryText) return;

    queryText.value = question;
    input.value = '';

    // Trigger form submission
    const form = document.getElementById('user-query-form');
    if (form) {
        const event = new Event('submit', { bubbles: true, cancelable: true });
        form.dispatchEvent(event);
    }
};


/**
 * Utility to show status messages to the user.
 * This function is kept for other pages but is not used in the new user page wizard.
 * A more integrated status/error display is used instead.
 */
function showStatus(message, type) {
    const statusContainer = document.getElementById('status-container');
    if (statusContainer) {
        statusContainer.textContent = message;
        statusContainer.className = `status-message ${type}`;
        statusContainer.style.display = 'block';
    }
}

// Modified appendAIResponse to take both display and read text, and handle speech
function appendAIResponse(displayText, readText = '') {
    const chatHistory = document.getElementById("chat-history");
    const messageDiv = document.createElement("div");
    messageDiv.className = "ai-card fade-in";

    const header = `
        <div class="flex justify-between items-center mb-2">
          <h2 class="font-semibold text-gray-700">🤖 AI Response</h2>
          <div>
            <button class="copy-btn" onclick="copyMessage(this)">📋</button>
            <button class="speak-btn" onclick="speakMessage(this)">🔊</button>
          </div>
        </div>`;

    const formatted = marked.parse(displayText);
    messageDiv.innerHTML = header + `<div class="markdown-content">${formatted}</div>`;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Auto-read by default
    if (readText) {
        speechSynthesis.cancel(); // Stop any previous speech
        const utterance = new SpeechSynthesisUtterance(readText);
        speechSynthesis.speak(utterance);
    }

    return messageDiv; // Return the element
}

function copyMessage(btn) {
    const text = btn.closest(".ai-card").querySelector(".markdown-content").innerText;
    navigator.clipboard.writeText(text);
    btn.textContent = "✅";
    setTimeout(() => (btn.textContent = "📋"), 1200);
}




// Modified window.speakMessage to use the global mute functionality and correctly get content
window.speakMessage = function (button) {
    // Determine the content to speak. If this is a re-read, get the display text.
    const card = button.closest('.ai-card');
    const content = card ? card.querySelector('.markdown-content').innerText : ''; // Fallback for auto-read

    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
        // Reset all speak buttons to '🔊'
        document.querySelectorAll('.speak-btn').forEach(btn => btn.textContent = '🔊');
    } else {
        const utterance = new SpeechSynthesisUtterance(content);
        utterance.onstart = () => { if (button) button.textContent = '🔇'; };
        utterance.onend = () => { if (button) button.textContent = '🔊'; };
        speechSynthesis.speak(utterance);
    }
}

/**
 * =====================================================
 * VOICE INPUT FOR FOLLOW-UP QUESTIONS
 * =====================================================
 */

// Global state for followup voice recognition
let followupVoiceRecognition = null;
let followupVoiceTranscript = '';
let isFollowupVoiceActive = false;

// Create voice overlay for follow-up recording
function createFollowupVoiceOverlay() {
    const overlay = document.createElement('div');
    overlay.id = 'followup-voice-overlay';
    overlay.className = 'followup-voice-overlay';
    overlay.innerHTML = `
        <div class="followup-voice-content">
            <h3>🎤 Speak Your Follow-up Question</h3>
            <div class="followup-voice-animation">
                <div class="followup-voice-bar"></div>
                <div class="followup-voice-bar"></div>
                <div class="followup-voice-bar"></div>
                <div class="followup-voice-bar"></div>
                <div class="followup-voice-bar"></div>
            </div>
            <div class="followup-voice-transcript" id="followup-voice-transcript">
                Listening...
            </div>
            <div class="followup-voice-actions">
                <button class="followup-voice-cancel" onclick="cancelFollowupVoice()">
                    ✕ Cancel
                </button>
                <button class="followup-voice-submit" id="followup-voice-submit" onclick="submitFollowupVoice()" disabled>
                    ✓ Submit
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
}

// Initialize speech recognition for follow-ups
function initFollowupVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert('Voice recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
        return null;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        console.log('[FollowupVoice] Recognition started');
        isFollowupVoiceActive = true;
    };

    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
            } else {
                interimTranscript += transcript;
            }
        }

        // Update global transcript
        if (finalTranscript) {
            followupVoiceTranscript = (followupVoiceTranscript + ' ' + finalTranscript).trim();
        }

        // Display transcript
        const transcriptEl = document.getElementById('followup-voice-transcript');
        if (transcriptEl) {
            const displayText = followupVoiceTranscript + (interimTranscript ? ' ' + interimTranscript : '');
            transcriptEl.textContent = displayText || 'Listening...';

            // Enable submit button if we have text
            const submitBtn = document.getElementById('followup-voice-submit');
            if (submitBtn) {
                submitBtn.disabled = !followupVoiceTranscript.trim();
            }
        }
    };

    recognition.onerror = (event) => {
        console.error('[FollowupVoice] Recognition error:', event.error);
        const transcriptEl = document.getElementById('followup-voice-transcript');
        if (transcriptEl) {
            transcriptEl.textContent = `Error: ${event.error}. Please try again.`;
            transcriptEl.style.color = '#ef4444';
        }
    };

    recognition.onend = () => {
        console.log('[FollowupVoice] Recognition ended');
        isFollowupVoiceActive = false;
    };

    return recognition;
}

// Handle voice button click for custom follow-up input
window.handleCustomFollowupVoice = function () {
    console.log('[FollowupVoice] Starting voice input for custom follow-up');

    // Reset transcript
    followupVoiceTranscript = '';

    // Initialize recognition if needed
    if (!followupVoiceRecognition) {
        followupVoiceRecognition = initFollowupVoiceRecognition();
        if (!followupVoiceRecognition) return; // Not supported
    }

    // Show overlay
    const overlay = document.getElementById('followup-voice-overlay');
    if (overlay) {
        overlay.classList.add('active');

        // Reset UI
        const transcriptEl = document.getElementById('followup-voice-transcript');
        if (transcriptEl) {
            transcriptEl.textContent = 'Listening...';
            transcriptEl.style.color = '#374151';
        }

        const submitBtn = document.getElementById('followup-voice-submit');
        if (submitBtn) {
            submitBtn.disabled = true;
        }
    }

    // Start recognition
    try {
        followupVoiceRecognition.start();
    } catch (e) {
        console.error('[FollowupVoice] Failed to start recognition:', e);
        // If already running, stop and restart
        followupVoiceRecognition.stop();
        setTimeout(() => {
            try {
                followupVoiceRecognition.start();
            } catch (err) {
                console.error('[FollowupVoice] Failed to restart recognition:', err);
                alert('Could not start voice recognition. Please try again.');
                closeFollowupVoiceOverlay();
            }
        }, 300);
    }
};

// Cancel voice input
window.cancelFollowupVoice = function () {
    console.log('[FollowupVoice] Canceling voice input');

    if (followupVoiceRecognition && isFollowupVoiceActive) {
        followupVoiceRecognition.stop();
    }

    closeFollowupVoiceOverlay();
    followupVoiceTranscript = '';
};

// Submit voice input as follow-up query
window.submitFollowupVoice = function () {
    console.log('[FollowupVoice] Submitting voice input:', followupVoiceTranscript);

    if (!followupVoiceTranscript.trim()) {
        alert('No speech detected. Please try again.');
        return;
    }

    // Stop recognition
    if (followupVoiceRecognition && isFollowupVoiceActive) {
        followupVoiceRecognition.stop();
    }

    // Close overlay
    closeFollowupVoiceOverlay();

    // Submit as follow-up query
    const queryText = document.getElementById('query-text');
    if (queryText) {
        queryText.value = followupVoiceTranscript;

        // Trigger form submission
        const form = document.getElementById('user-query-form');
        if (form) {
            const event = new Event('submit', { bubbles: true, cancelable: true });
            form.dispatchEvent(event);
        }
    }

    // Reset transcript
    followupVoiceTranscript = '';
};

// Helper to close overlay
function closeFollowupVoiceOverlay() {
    const overlay = document.getElementById('followup-voice-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}
