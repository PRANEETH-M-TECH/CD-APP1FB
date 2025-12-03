class InlineNotebookManager {
    constructor() {
        this.isOpen = false;
        this.currentNotebook = null;
        this.quill = null;
        this.pages = [];
        this.currentPageIndex = 0;
        this.autoSaveTimer = null;
        this.uid = null;
        this.isFullscreen = false;

        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        // Wait for Firebase Auth
        firebase.auth().onAuthStateChanged(user => {
            if (user) {
                this.uid = user.uid;
            }
        });

        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only activate shortcuts when notebook is open
            if (!this.isOpen) return;

            const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
            const modKey = isMac ? e.metaKey : e.ctrlKey;

            // Cmd/Ctrl + S: Save
            if (modKey && e.key === 's') {
                e.preventDefault();
                this.savePage();
                return;
            }

            // Cmd/Ctrl + N: New Page
            if (modKey && e.key === 'n') {
                e.preventDefault();
                this.addPage();
                return;
            }

            // Cmd/Ctrl + Backspace: Delete Page
            if (modKey && e.key === 'Backspace') {
                e.preventDefault();
                this.deletePage();
                return;
            }

            // Escape: Close if not in fullscreen, exit fullscreen if in fullscreen
            if (e.key === 'Escape') {
                if (this.isFullscreen) {
                    this.toggleFullscreen();
                } else {
                    this.close();
                }
                return;
            }

            // Cmd/Ctrl + Left Arrow: Previous Page
            if (modKey && e.key === 'ArrowLeft') {
                e.preventDefault();
                this.prevPage();
                return;
            }

            // Cmd/Ctrl + Right Arrow: Next Page
            if (modKey && e.key === 'ArrowRight') {
                e.preventDefault();
                this.nextPage();
                return;
            }
        });
    }

    async open(notebookId, highlightText = null) {
        if (!this.uid) {
            alert('Please login first');
            return;
        }

        console.log('[INLINE NOTEBOOK] Opening notebook:', notebookId);

        // Swap Views - Hide the LEFT PANE (PDF viewer) and show the notebook container
        const leftPane = document.getElementById('left-pane');
        const notebookContainer = document.getElementById('inline-notebook-container');

        if (leftPane && notebookContainer) {
            leftPane.style.display = 'none';
            notebookContainer.style.display = 'flex';
            this.isOpen = true;
        } else {
            console.error('Could not find viewer containers', { leftPane, notebookContainer });
            return;
        }

        // Initialize Quill if needed
        if (!this.quill) {
            this.initQuill();
        }

        // Load Notebook Data
        await this.loadNotebook(notebookId);

        // If there's text to highlight/insert
        if (highlightText) {
            this.insertAndHighlight(highlightText);
        }
    }

    close() {
        console.log('[INLINE NOTEBOOK] Closing editor');

        // Save before closing
        this.savePage(true);

        // Swap Views Back - Show the LEFT PANE (PDF viewer) and hide the notebook container
        const leftPane = document.getElementById('left-pane');
        const notebookContainer = document.getElementById('inline-notebook-container');

        if (leftPane && notebookContainer) {
            notebookContainer.style.display = 'none';
            leftPane.style.display = 'flex'; // Restore flex layout
            this.isOpen = false;
        }
    }

    initQuill() {
        this.quill = new Quill('#inline-editor', {
            theme: 'snow',
            modules: {
                toolbar: [
                    [{ 'header': [1, 2, false] }],
                    ['bold', 'italic', 'underline', 'strike'],
                    [{ 'color': [] }, { 'background': [] }],
                    [{ 'list': 'ordered' }, { 'list': 'bullet' }],
                    ['clean']
                ]
            },
            placeholder: 'Start writing your notes...'
        });

        // Auto-save listener
        this.quill.on('text-change', () => {
            document.getElementById('save-status').textContent = 'Unsaved changes...';
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = setTimeout(() => this.savePage(true), 2000);
        });
    }

    async loadNotebook(notebookId) {
        try {
            // Load Metadata
            const doc = await firebase.firestore()
                .collection('notebooks')
                .doc(notebookId)
                .get();

            if (!doc.exists) {
                alert('Notebook not found');
                this.close();
                return;
            }

            this.currentNotebook = { id: doc.id, ...doc.data() };
            document.getElementById('inline-notebook-title').textContent = this.currentNotebook.name;

            // Load notebook switcher
            await this.loadNotebookSwitcher();

            // Load Pages
            const snapshot = await firebase.firestore()
                .collection('notebooks')
                .doc(notebookId)
                .collection('pages')
                .orderBy('pageNumber', 'asc')
                .get();

            this.pages = snapshot.docs.map(d => ({ id: d.id, ...d.data() }));

            if (this.pages.length === 0) {
                await this.createFirstPage();
            } else {
                this.loadPage(0);
            }

            this.updateNavigation();

        } catch (error) {
            console.error('Error loading notebook:', error);
            alert('Failed to load notebook');
        }
    }

    // Load notebook switcher dropdown
    async loadNotebookSwitcher() {
        if (!this.uid) return;

        try {
            const response = await fetch(`/api/bag/notebooks?uid=${this.uid}`);
            const data = await response.json();
            const notebooks = data.notebooks || [];

            // Add switcher to title section if not exists
            const titleSection = document.querySelector('.notebook-title-section');
            if (!titleSection) return;

            let switcher = document.getElementById('notebook-switcher');
            if (!switcher) {
                switcher = document.createElement('select');
                switcher.id = 'notebook-switcher';
                switcher.className = 'notebook-switcher';
                switcher.onchange = (e) => this.switchNotebook(e.target.value);

                // Insert after title
                const emojiSpan = titleSection.querySelector('.notebook-emoji');
                if (emojiSpan) {
                    emojiSpan.after(switcher);
                } else {
                    titleSection.prepend(switcher);
                }
            }

            // Populate dropdown
            switcher.innerHTML = notebooks.map(nb =>
                `<option value="${nb.notebook_id}" ${nb.notebook_id === this.currentNotebook.id ? 'selected' : ''}>
                    ${nb.name}
                </option>`
            ).join('');

        } catch (error) {
            console.error('Error loading notebook switcher:', error);
        }
    }

    // Switch to a different notebook
    async switchNotebook(notebookId) {
        if (notebookId === this.currentNotebook?.id) return;

        // Save current page before switching
        await this.savePage(true);

        // Load new notebook
        await this.loadNotebook(notebookId);
    }

    async createFirstPage() {
        const newPage = {
            pageNumber: 1,
            content: '',
            createdAt: firebase.firestore.FieldValue.serverTimestamp(),
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        };

        const docRef = await firebase.firestore()
            .collection('notebooks')
            .doc(this.currentNotebook.id)
            .collection('pages')
            .add(newPage);

        this.pages = [{ id: docRef.id, ...newPage }];
        this.loadPage(0);
    }

    loadPage(index) {
        this.currentPageIndex = index;
        const page = this.pages[index];

        if (page && page.content) {
            try {
                this.quill.setContents(JSON.parse(page.content));
            } catch (e) {
                this.quill.setText(page.content); // Fallback for plain text
            }
        } else {
            this.quill.setText('');
        }

        this.updateNavigation();
    }

    async savePage(silent = false) {
        if (!this.currentNotebook || this.pages.length === 0) return;

        const page = this.pages[this.currentPageIndex];
        const content = JSON.stringify(this.quill.getContents());

        document.getElementById('save-status').textContent = 'Saving...';

        try {
            await firebase.firestore()
                .collection('notebooks')
                .doc(this.currentNotebook.id)
                .collection('pages')
                .doc(page.id)
                .update({
                    content: content,
                    updatedAt: firebase.firestore.FieldValue.serverTimestamp()
                });

            page.content = content;
            document.getElementById('save-status').textContent = 'All changes saved';

            if (!silent) {
                // Visual feedback
            }
        } catch (error) {
            console.error('Error saving:', error);
            document.getElementById('save-status').textContent = 'Error saving!';
        }
    }

    async insertAndHighlight(text) {
        // Append text to current page
        const length = this.quill.getLength();
        this.quill.insertText(length, `\n${text}\n`, 'api');

        // Scroll to bottom
        this.quill.setSelection(length, text.length);
        this.quill.scrollIntoView();

        // Save immediately
        await this.savePage(true);
    }

    nextPage() {
        if (this.currentPageIndex < this.pages.length - 1) {
            this.savePage(true);
            this.loadPage(this.currentPageIndex + 1);
        }
    }

    prevPage() {
        if (this.currentPageIndex > 0) {
            this.savePage(true);
            this.loadPage(this.currentPageIndex - 1);
        }
    }

    async addPage() {
        this.savePage(true);

        const newPageNumber = this.pages.length + 1;
        const newPage = {
            pageNumber: newPageNumber,
            content: '',
            createdAt: firebase.firestore.FieldValue.serverTimestamp(),
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        };

        try {
            const docRef = await firebase.firestore()
                .collection('notebooks')
                .doc(this.currentNotebook.id)
                .collection('pages')
                .add(newPage);

            this.pages.push({ id: docRef.id, ...newPage });
            this.loadPage(this.pages.length - 1);
        } catch (error) {
            console.error('Error adding page:', error);
        }
    }

    updateNavigation() {
        // Update header page numbers
        document.getElementById('current-page-num-header').textContent = this.currentPageIndex + 1;
        document.getElementById('total-pages-num-header').textContent = this.pages.length;

        // Update header navigation buttons
        const prevBtn = document.getElementById('prev-page-btn-header');
        const nextBtn = document.getElementById('next-page-btn-header');

        if (prevBtn) prevBtn.disabled = this.currentPageIndex === 0;
        if (nextBtn) nextBtn.disabled = this.currentPageIndex === this.pages.length - 1;
    }

    // Toggle Fullscreen Mode
    toggleFullscreen() {
        const container = document.getElementById('inline-notebook-container');
        const rightPane = document.getElementById('right-pane');
        const toggleBtn = document.getElementById('fullscreen-toggle-btn');

        this.isFullscreen = !this.isFullscreen;

        if (this.isFullscreen) {
            // Hide chat pane and expand notebook
            if (rightPane) rightPane.style.display = 'none';
            container.classList.add('fullscreen');
            if (toggleBtn) toggleBtn.title = 'Exit fullscreen';
        } else {
            // Restore chat pane
            if (rightPane) rightPane.style.display = 'flex';
            container.classList.remove('fullscreen');
            if (toggleBtn) toggleBtn.title = 'Expand notebook';
        }
    }

    // Delete Current Page
    async deletePage() {
        if (this.pages.length <= 1) {
            alert('Cannot delete the last page. A notebook must have at least one page.');
            return;
        }

        const confirmed = confirm(`Delete page ${this.currentPageIndex + 1}? This action cannot be undone.`);
        if (!confirmed) return;

        try {
            const page = this.pages[this.currentPageIndex];

            // Delete from Firestore
            await firebase.firestore()
                .collection('notebooks')
                .doc(this.currentNotebook.id)
                .collection('pages')
                .doc(page.id)
                .delete();

            // Remove from local array
            this.pages.splice(this.currentPageIndex, 1);

            // Navigate to previous page or stay on same index if it was the last page
            const newIndex = Math.min(this.currentPageIndex, this.pages.length - 1);
            this.loadPage(newIndex);

            console.log('[INLINE NOTEBOOK] Page deleted successfully');
        } catch (error) {
            console.error('Error deleting page:', error);
            alert('Failed to delete page. Please try again.');
        }
    }

    // Open notebook in full My Bag editor
    openInMyBag() {
        if (!this.currentNotebook) return;
        // Save current page before navigating
        this.savePage(true);
        // Navigate to full editor (with /static prefix)
        window.location.href = `/static/notebook-editor.html?id=${this.currentNotebook.id}`;
    }
}

// Global Instance
window.inlineNotebook = new InlineNotebookManager();

// Global Helpers
window.closeInlineNotebook = () => window.inlineNotebook.close();
window.saveInlinePage = () => window.inlineNotebook.savePage();
window.prevInlinePage = () => window.inlineNotebook.prevPage();
window.nextInlinePage = () => window.inlineNotebook.nextPage();
window.addInlinePage = () => window.inlineNotebook.addPage();
window.deleteInlinePage = () => window.inlineNotebook.deletePage();
window.toggleFullscreen = () => window.inlineNotebook.toggleFullscreen();
window.openInMyBag = () => window.inlineNotebook.openInMyBag();
