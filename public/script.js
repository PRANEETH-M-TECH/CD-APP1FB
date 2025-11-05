document.addEventListener('DOMContentLoaded', () => {
    // Check which page we are on and run the appropriate setup function
    if (document.getElementById('admin-form')) {
        setupAdminPage();
    } else if (document.getElementById('chapters-form')) {
        setupChaptersPage();
    } else if (document.getElementById('user-query-form')) {
        setupUserPage();
    }
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
        pdfDoc.getPage(num).then(function(page) {
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
            renderTask.promise.then(function() {
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
    pdfjsLib.getDocument(pdfUrl).promise.then(function(pdfDoc_) {
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
                chapters.push({ chapter_name: name, pdf_startpg: start_page, pdf_endpg: end_page });
            }
        });

        if (validationError) {
            showStatus('Please fix the errors in the highlighted fields.', 'error');
            return;
        }
        
        showStatus('Processing book and chapters...', 'info');

        const finalFormData = new FormData();
        finalFormData.append('class_name', className);
        finalFormData.append('subject', subject);
        finalFormData.append('filename', filename);

        try {
            const response = await fetch('/api/books', { method: 'POST', body: finalFormData });
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
    const voiceSearchBtn = document.getElementById('voice-search-btn');
    const voiceModal = document.getElementById('voice-modal');
    const micButton = document.getElementById('mic-button');
    const voiceStatus = document.getElementById('voice-status');
    // Left Pane Elements
    const classSelect = document.getElementById('class-select');
    const subjectSelect = document.getElementById('subject-select');
    const viewerPlaceholder = document.getElementById('viewer-placeholder');
    const pdfLoadingIndicator = document.getElementById('pdf-loading-user');
    const pdfCanvas = document.getElementById('pdf-canvas-user');
    const ctx = pdfCanvas.getContext('2d');

    // Right Pane Elements
    const chatHistory = document.getElementById('chat-history');
    const queryForm = document.getElementById('user-query-form');
    const queryText = document.getElementById('query-text');
    const submitButton = document.getElementById('submit-query-btn');
    const listChaptersBtn = document.getElementById('list-chapters-btn');

    // App State
    let selectedBook = null;
    let pdfDoc = null;
    let pageNum = 1; // Current page number
    let pageRendering = false;
    let pageNumPending = null;
    let isFirstQuery = true; // To handle the welcome message

    let sentenceQueue = '';
    let isSpeakingStream = false;

    // --- Event Listeners ---

    classSelect.addEventListener('change', () => {
        subjectSelect.disabled = false;
        subjectSelect.value = '';
        resetUI();
    });

    subjectSelect.addEventListener('change', () => {
        loadBook();
    });

    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        isSpeakingStream = true; // Enable streaming speech for new query
        handleQuerySubmit();
    });

    listChaptersBtn.addEventListener('click', () => {
        handleListChapters();
    });

    // Auto-resize textarea
    queryText.addEventListener('input', () => {
        queryText.style.height = 'auto';
        queryText.style.height = (queryText.scrollHeight) + 'px';
    });

    // PDF Navigation Buttons
    document.getElementById('prev-page-user').addEventListener('click', () => {
        if (pageNum <= 1) return;
        pageNum--;
        queueRenderPage(pageNum);
    });

    document.getElementById('next-page-user').addEventListener('click', () => {
        if (pdfDoc && pageNum >= pdfDoc.numPages) return;
        pageNum++;
        queueRenderPage(pageNum);
    });

    const conversationalModal = document.getElementById('conversational-modal');
    const exitConversationalBtn = document.getElementById('exit-conversational-btn');

    const conversationalModeBtn = document.getElementById('conversational-mode-btn');
    const voiceWaveformCanvas = conversationalModal.querySelector('#voice-waveform-modal'); // Correctly select canvas inside modal
    const waveformCtx = voiceWaveformCanvas.getContext('2d');
    let animationFrameId;

    let isRecording = false;
    let isConversationalMode = false;

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    // Event listener for the voice search modal
    recognition.addEventListener('result', e => {
        const transcript = Array.from(e.results)
            .map(result => result[0])
            .map(result => result.transcript)
            .join('');
        if (isConversationalMode) {
            if (e.results[0].isFinal) {
                handleTranscription(transcript);
            }
        } else {
            queryText.value = transcript;
        }
    });

    recognition.addEventListener('end', () => {
        if (isConversationalMode) {
            if (speechSynthesis.speaking) {
                // If AI is speaking, don't restart recognition yet.
                // It will be restarted after AI finishes speaking.
            } else {
                recognition.start();
            }
        } else {
            micButton.textContent = 'Start Recording';
            isRecording = false;
        }
    });

    voiceSearchBtn.addEventListener('click', () => {
        voiceModal.classList.remove('hidden');
    });

    micButton.addEventListener('click', () => {
        if (isRecording) {
            recognition.stop();
        } else {
            recognition.start();
            micButton.textContent = 'Stop Recording';
            voiceStatus.textContent = 'Recording...';
            isRecording = true;
        }
    });

    voiceModal.addEventListener('click', (e) => {
        if (e.target === voiceModal) {
            voiceModal.classList.add('hidden');
        }
    });

    conversationalModeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!selectedBook) {
            alert('Please select a book first to start conversational mode.');
            return;
        }
        startConversationalMode();
    });

    exitConversationalBtn.addEventListener('click', () => {
        stopConversationalMode();
    });

    function startConversationalMode() {
        isConversationalMode = true;
        conversationalModal.classList.remove('hidden');
        recognition.start();
    }

    function stopConversationalMode() {
        isConversationalMode = false;
        conversationalModal.classList.add('hidden');
        recognition.stop();
        speechSynthesis.cancel();
        cancelAnimationFrame(animationFrameId);
        waveformCtx.clearRect(0, 0, voiceWaveformCanvas.width, voiceWaveformCanvas.height);
    }

    async function handleTranscription(transcript) {
        if (!transcript.trim() || !selectedBook) return;

        console.log(`--- Transcribed Text: ${transcript} ---`);

        const source = new EventSource(`/api/query?book_uuid=${selectedBook.id}&query=${transcript}`);
        let fullReadText = "";

        source.onmessage = function(event) {
            if (event.data === "[DONE]") {
                source.close();
                if (fullReadText) {
                    const utterance = new SpeechSynthesisUtterance(fullReadText);
                    utterance.onstart = () => {
                        animateWaveform();
                    };
                    utterance.onend = () => {
                        if (isConversationalMode) {
                            recognition.start();
                        }
                    };
                    speechSynthesis.speak(utterance);
                }
                return;
            }
            const data = JSON.parse(event.data);
            if (data.read_text) {
                fullReadText += data.read_text;
            }
        };

        source.onerror = function(error) {
            console.error('EventSource failed:', error);
            source.close();
        };
    }

    function animateWaveform() {
        if (!isConversationalMode || !speechSynthesis.speaking) {
            cancelAnimationFrame(animationFrameId);
            waveformCtx.clearRect(0, 0, voiceWaveformCanvas.width, voiceWaveformCanvas.height);
            return;
        }

        const width = voiceWaveformCanvas.width;
        const height = voiceWaveformCanvas.height;
        const time = Date.now();

        waveformCtx.clearRect(0, 0, width, height);
        waveformCtx.lineWidth = 2;
        waveformCtx.strokeStyle = '#3b82f6';

        waveformCtx.beginPath();

        const sliceWidth = width * 1.0 / 100;
        let x = 0;

        for (let i = 0; i < 100; i++) {
            const v = 1.5 * Math.sin(i / 2 + time / 100);
            const y = v * height / 2 + height / 2;

            if (i === 0) {
                waveformCtx.moveTo(x, y);
            } else {
                waveformCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        waveformCtx.lineTo(width, height / 2);
        waveformCtx.stroke();

        animationFrameId = requestAnimationFrame(animateWaveform);
    }

    // --- Core Functions ---

    function resetUI() {
        pdfDoc = null;
        selectedBook = null;
        pageNum = 1; // Reset page number
        pdfCanvas.style.display = 'none';
        document.getElementById('pdf-viewer-header-user').style.display = 'none'; // Hide header
        viewerPlaceholder.style.display = 'flex';
        pdfLoadingIndicator.style.display = 'none';
        queryText.setAttribute('disabled', 'true');
        submitButton.setAttribute('disabled', 'true');
        listChaptersBtn.classList.add('hidden'); // Hide the button
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
            // Step 1: Fetch book metadata (including filename)
            const response = await fetch(`/api/books?class_name=${className}&subject=${subject}`);
            if (!response.ok) throw new Error('Book not found.');
            
            const books = await response.json();
            if (books.length === 0) throw new Error('Book not found for this selection.');
            
            selectedBook = books[0]; // Assume the first book is the correct one

            // Step 2: Load the PDF document
            const pdfUrl = `/uploads/${selectedBook.filename}`;
            pdfDoc = await pdfjsLib.getDocument(pdfUrl).promise;
            
            pdfLoadingIndicator.style.display = 'none';
            pdfCanvas.style.display = 'block';
            document.getElementById('pdf-viewer-header-user').style.display = 'flex'; // Show header
            
            document.getElementById('page-count-user').textContent = pdfDoc.numPages;
            renderPage(pageNum); // Render the first page

            // Enable chat
            queryText.removeAttribute('disabled');
            submitButton.removeAttribute('disabled');
            listChaptersBtn.classList.remove('hidden'); // Show the button
            
            // Clear welcome and add loaded message
            chatHistory.innerHTML = '';
            isFirstQuery = false;
            appendAIResponse(`Book "${selectedBook.subject}" loaded. You can now ask questions about it.`);

        } catch (error) {
            pdfLoadingIndicator.style.display = 'none';
            viewerPlaceholder.style.display = 'flex';
            viewerPlaceholder.innerHTML = `<p class="error-message">${error.message}</p>`;
            console.error(error);
        }
    }

    /**
     * Get page info from document, resize canvas accordingly, and render page.
     */
    async function renderPage(num) {
        pageRendering = true;
        pdfLoadingIndicator.style.display = 'flex'; // Show loading indicator

        // Using promise to fetch the page
        const page = await pdfDoc.getPage(num);
        const container = document.getElementById('pdf-render-area-user');
        const unscaledViewport = page.getViewport({ scale: 1 });
        
        // Dynamically calculate scale to fit container width
        const scale = container.clientWidth / unscaledViewport.width;
        
        const viewport = page.getViewport({ scale: scale });

        const outputScale = window.devicePixelRatio || 1;

        pdfCanvas.height = Math.floor(viewport.height * outputScale);
        pdfCanvas.width = Math.floor(viewport.width * outputScale);
        pdfCanvas.style.width = Math.floor(viewport.width) + 'px';
        pdfCanvas.style.height = Math.floor(viewport.height) + 'px';

        // Render PDF page into canvas context
        const renderContext = {
            canvasContext: ctx,
            viewport: viewport,
            transform: [outputScale, 0, 0, outputScale, 0, 0]
        };
        const renderTask = page.render(renderContext);

        // Wait for rendering to finish
        await renderTask.promise;
        pageRendering = false;
        pdfLoadingIndicator.style.display = 'none'; // Hide loading indicator
        if (pageNumPending !== null) {
            // New page rendering is pending
            renderPage(pageNumPending);
            pageNumPending = null;
        }

        // Update page counters
        document.getElementById('page-num-user').textContent = num;
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

        if (isFirstQuery) {
            chatHistory.innerHTML = '';
            isFirstQuery = false;
        }

        addUserMessage(query);
        queryText.value = '';
        queryText.style.height = 'auto'; // Reset height
        submitButton.setAttribute('disabled', 'true');
        listChaptersBtn.classList.add('hidden');

        const thinkingMessage = appendAIResponse('...');
        const contentDiv = thinkingMessage.querySelector('.markdown-content');
        let fullResponse = "";
        contentDiv.innerHTML = '';

        const source = new EventSource(`/api/query?book_uuid=${selectedBook.id}&query=${query}`);

        let fullReadText = "";
        source.onmessage = function(event) {
            if (event.data === "[DONE]") {
                source.close();
                if (fullReadText && isSpeakingStream) {
                    const utterance = new SpeechSynthesisUtterance(fullReadText);
                    speechSynthesis.speak(utterance);
                }
                isSpeakingStream = false;
                submitButton.removeAttribute('disabled');
                listChaptersBtn.classList.remove('hidden');
                return;
            }
            const data = JSON.parse(event.data);
            if (data.display_text) {
                fullResponse += data.display_text;
                contentDiv.innerHTML = marked.parse(fullResponse);
            }
            if (data.read_text) {
                fullReadText += data.read_text;
            }
            chatHistory.scrollTop = chatHistory.scrollHeight;
        };

        source.onerror = function(error) {
            console.error('EventSource failed:', error);
            contentDiv.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
            source.close();
        };

        source.onopen = function() {
            // This is called when the connection is established
        };

        source.onend = function() {
            submitButton.removeAttribute('disabled');
            listChaptersBtn.classList.remove('hidden');
            isSpeakingStream = false; // Disable streaming speech when done
            // Speak any remaining text in the queue
            if (sentenceQueue.trim()) {
                const utterance = new SpeechSynthesisUtterance(sentenceQueue.trim());
                speechSynthesis.speak(utterance);
                sentenceQueue = '';
            }
        };
    }
    async function handleListChapters() {
        if (!selectedBook) return;
        
        if (isFirstQuery) {
            chatHistory.innerHTML = '';
            isFirstQuery = false;
        }

        addUserMessage('List all chapters');
        const thinkingMessage = appendAIResponse('Fetching chapters...');

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

        } catch (error) {
            thinkingMessage.querySelector('.markdown-content').innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}</p>`;
        } finally {
            submitButton.removeAttribute('disabled');
            listChaptersBtn.classList.remove('hidden');
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }
}

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

function appendAIResponse(markdownText) {
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

    const formatted = marked.parse(markdownText);
    messageDiv.innerHTML = header + `<div class="markdown-content">${formatted}</div>`;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return messageDiv; // Return the element
}

function copyMessage(btn) {
    const text = btn.closest(".ai-card").querySelector(".markdown-content").innerText;
    navigator.clipboard.writeText(text);
    btn.textContent = "✅";
    setTimeout(() => (btn.textContent = "📋"), 1200);
}

function speakStream(chunk) {
    if (!isSpeakingStream) return; // Only speak if streaming is enabled

    sentenceQueue += chunk;

    // Use a simple regex to detect sentence endings
    const sentenceEndings = /[.!?。？！]/;
    const parts = sentenceQueue.split(sentenceEndings);

    // If the last part is not a complete sentence, keep it in the queue
    if (parts.length > 1 && sentenceEndings.test(sentenceQueue.slice(-1))) {
        for (let i = 0; i < parts.length - 1; i++) {
            const sentence = parts[i].trim();
            if (sentence) {
                const utterance = new SpeechSynthesisUtterance(sentence);
                speechSynthesis.speak(utterance);
            }
        }
        sentenceQueue = parts[parts.length - 1]; // Keep the incomplete sentence
    }
}


// Call window.speakMessage to use an updated version.
window.speakMessage = function(button) {
    if (speechSynthesis.speaking || isSpeakingStream) {
        speechSynthesis.cancel();
        isSpeakingStream = false;
        button.textContent = '🔊'; // Reset icon
    } else {
        // If no speech is active, start speaking the full message content
        const card = button.closest('.ai-card');
        const content = card.querySelector('.markdown-content').innerText;
        const utterance = new SpeechSynthesisUtterance(content);
        utterance.onstart = () => { button.textContent = '🔇'; };
        utterance.onend = () => { button.textContent = '🔊'; };
        speechSynthesis.speak(utterance);
    }
}
