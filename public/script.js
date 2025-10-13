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
        const className = document.getElementById('class-name').value;
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
                className: className,
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
    const className = params.get('className');
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
                chapters.push({ name, start_page, end_page });
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
        finalFormData.append('chapters', JSON.stringify(chapters));
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
            addMessage('ai', `Book "${selectedBook.subject}" loaded. You can now ask questions about it.`);

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

    async function handleQuerySubmit() {
        const query = queryText.value.trim();
        if (!query || !selectedBook) return;

        addMessage('user', query);
        queryText.value = '';
        queryText.style.height = 'auto'; // Reset height
        submitButton.setAttribute('disabled', 'true');
        listChaptersBtn.classList.add('hidden');

        const thinkingMessage = addMessage('ai', '...');

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    book_uuid: selectedBook.id,
                    query: query
                })
            });

            if (!response.ok) {
                const errorResult = await response.json();
                throw new Error(errorResult.detail || 'Failed to get answer.');
            }

            const result = await response.json();
            
            let answer = result.answer;
            
            thinkingMessage.querySelector('.message-content').innerHTML = `<p>${answer.replace(/\n/g, '<br>')}</p>`;

        } catch (error) {
            thinkingMessage.querySelector('.message-content').innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
        } finally {
            submitButton.removeAttribute('disabled');
            listChaptersBtn.classList.remove('hidden');
        }
    }

    async function handleListChapters() {
        if (!selectedBook) return;

        addMessage('user', 'List all chapters');
        appendAIResponse('Fetching chapters...');

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

            // Sort chapters by start_page numerically
            chapters.sort((a, b) => a.start_page - b.start_page);

            // Build an HTML table string
            let tableHtml = `
                <h3 class="text-lg font-bold mb-2">Chapters in this book:</h3>
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr>
                            <th class="border-b-2 p-2">S.No.</th>
                            <th class="border-b-2 p-2">Chapter Name</th>
                            <th class="border-b-2 p-2">Pages</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            chapters.forEach((chapter, index) => {
                const sno = index + 1;
                tableHtml += `
                        <tr>
                            <td class="border-b p-2">${sno}</td>
                            <td class="border-b p-2">${chapter.name}</td>
                            <td class="border-b p-2">${chapter.start_page} - ${chapter.end_page}</td>
                        </tr>
                `;
            });

            tableHtml += `
                    </tbody>
                </table>
            `;

            const chatHistory = document.getElementById('chat-history');
            const placeholderCard = chatHistory.lastChild;
            
            // Directly set the innerHTML with the new table
            placeholderCard.querySelector('.markdown-content').innerHTML = tableHtml;

        } catch (error) {
            const chatHistory = document.getElementById('chat-history');
            const placeholderCard = chatHistory.lastChild;
            placeholderCard.querySelector('.markdown-content').innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}</p>`;
        } finally {
            submitButton.removeAttribute('disabled');
            listChaptersBtn.classList.remove('hidden');
        }
    }

    function addMessage(sender, text) {
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${sender}-message`;
        
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        contentEl.innerHTML = `<p>${text.replace(/\n/g, '<br>')}</p>`;
        
        messageEl.appendChild(contentEl);
        chatHistory.appendChild(messageEl);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return messageEl;
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
