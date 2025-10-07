

document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;

    if (path.includes('/admin')) {
        initAdminPage();
    } else if (path.includes('/user')) {
        initUserPage();
    }
});

function initAdminPage() {
    const uploadForm = document.getElementById('upload-form');
    const bookDetailsSection = document.getElementById('book-details-section');
    const bookDetailsForm = document.getElementById('book-details-form');
    const addChapterBtn = document.getElementById('add-chapter-btn');
    const chaptersContainer = document.getElementById('chapters-container');
    const uploadStatus = document.getElementById('upload-status');
    const bookStatus = document.getElementById('book-status');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        uploadStatus.textContent = 'Uploading...';

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                uploadStatus.textContent = `File '${result.filename}' uploaded successfully!`;
                bookDetailsSection.style.display = 'block';
            } else {
                uploadStatus.textContent = 'Upload failed.';
            }
        } catch (error) {
            uploadStatus.textContent = 'An error occurred during upload.';
            console.error('Upload error:', error);
        }
    });

    addChapterBtn.addEventListener('click', () => {
        const chapterCount = chaptersContainer.children.length;
        const chapterGroup = document.createElement('div');
        chapterGroup.className = 'chapter-group';
        chapterGroup.innerHTML = `
            <input type="text" placeholder="Chapter ${chapterCount + 1} Name" class="chapter-name" required>
            <input type="number" placeholder="Start Page" class="start-page" required>
            <input type="number" placeholder="End Page" class="end-page" required>
        `;
        chaptersContainer.appendChild(chapterGroup);
    });

    bookDetailsForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const title = document.getElementById('book-title').value;
        const className = document.getElementById('class-name').value;
        const subject = document.getElementById('subject').value;

        const chapters = Array.from(chaptersContainer.children).map(group => ({
            name: group.querySelector('.chapter-name').value,
            start_page: parseInt(group.querySelector('.start-page').value),
            end_page: parseInt(group.querySelector('.end-page').value),
        }));

        if (chapters.some(c => !c.name || !c.start_page || !c.end_page)) {
            bookStatus.textContent = 'Please fill in all chapter details.';
            return;
        }

        const formData = new FormData();
        formData.append('title', title);
        formData.append('class_name', className);
        formData.append('subject', subject);
        formData.append('chapters', JSON.stringify(chapters));

        bookStatus.textContent = 'Saving book...';

        try {
            const response = await fetch('/api/books', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                bookStatus.textContent = 'Book saved successfully!';
                bookDetailsForm.reset();
                chaptersContainer.innerHTML = '';
            } else {
                const error = await response.json();
                bookStatus.textContent = `Failed to save book: ${error.detail || 'Unknown error'}`;
            }
        } catch (error) {
            bookStatus.textContent = 'An error occurred while saving the book.';
            console.error('Save book error:', error);
        }
    });
}

function initUserPage() {
    const bookSelect = document.getElementById('book-select');
    const queryForm = document.getElementById('query-form');
    const queryText = document.getElementById('query-text');
    const answerContainer = document.getElementById('answer-container');
    const answerText = document.getElementById('answer-text');
    const sourcesContainer = document.getElementById('sources-container');
    const loadingIndicator = document.getElementById('loading-indicator');

    async function loadBooks() {
        try {
            const response = await fetch('/api/books');
            if (response.ok) {
                const books = await response.json();
                bookSelect.innerHTML = '<option value="">-- Select a Book --</option>';
                books.forEach(book => {
                    const option = document.createElement('option');
                    option.value = book;
                    option.textContent = book;
                    bookSelect.appendChild(option);
                });
            } else {
                bookSelect.innerHTML = '<option value="">Failed to load books</option>';
            }
        } catch (error) {
            bookSelect.innerHTML = '<option value="">Error loading books</option>';
            console.error('Load books error:', error);
        }
    }

    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const bookTitle = bookSelect.value;
        const query = queryText.value;

        if (!bookTitle) {
            alert('Please select a book first.');
            return;
        }

        loadingIndicator.style.display = 'block';
        answerText.textContent = '';
        sourcesContainer.innerHTML = '';

        try {
            const params = new URLSearchParams({
                book_title: bookTitle,
                query: query
            });

            const response = await fetch(`/api/query?${params}` , { method: 'POST' });

            if (response.ok) {
                const result = await response.json();
                answerText.textContent = result.answer;
                displaySources(result.sources);
            } else {
                answerText.textContent = 'Failed to get an answer.';
            }
        } catch (error) {
            answerText.textContent = 'An error occurred.';
            console.error('Query error:', error);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });

    function displaySources(sources) {
        sourcesContainer.innerHTML = '';
        if (!sources || sources.length === 0) {
            sourcesContainer.innerHTML = '<p>No sources found.</p>';
            return;
        }

        sources.forEach(source => {
            const sourceEl = document.createElement('div');
            sourceEl.className = 'source';
            sourceEl.innerHTML = `
                <p><strong>Chapter:</strong> ${source.payload.chapter_name}</p>
                <p>"${source.payload.text.substring(0, 150)}..."</p>
            `;
            sourcesContainer.appendChild(sourceEl);
        });
    }

    loadBooks();
}
