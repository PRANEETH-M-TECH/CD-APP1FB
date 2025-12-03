/**
 * My Bag - Premium Animated Feature
 * Works with existing HTML modal structure
 */

class MyBag {
    constructor() {
        this.isOpen = false;
        this.currentNotebook = null;
        this.currentNotebookId = null;
        this.uid = null;
        this.selectedColor = '#6366f1';

        this.init();
    }

    init() {
        // Get auth state
        if (window.firebase) {
            firebase.auth().onAuthStateChanged(user => {
                if (user) {
                    this.uid = user.uid;
                }
            });
        }

        // Wire up create notebook form if it exists
        this.setupModalHandlers();
    }

    setupModalHandlers() {
        // Add event listener for color selection
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('color-btn')) {
                document.querySelectorAll('.color-btn').forEach(btn => btn.classList.remove('selected'));
                e.target.classList.add('selected');
                this.selectedColor = e.target.dataset.color;
                const hiddenInput = document.getElementById('notebook-color');
                if (hiddenInput) hiddenInput.value = this.selectedColor;
            }
        });
    }

    open() {
        // Show My Bag overlay and sidebar
        const overlay = document.getElementById('my-bag-overlay');
        const sidebar = document.getElementById('my-bag-sidebar');

        if (overlay) overlay.classList.add('active');
        if (sidebar) sidebar.classList.add('active');

        // Load notebooks
        this.loadNotebooks();
    }

    close() {
        const overlay = document.getElementById('my-bag-overlay');
        const sidebar = document.getElementById('my-bag-sidebar');

        if (overlay) overlay.classList.remove('active');
        if (sidebar) sidebar.classList.remove('active');
    }

    async loadNotebooks() {
        if (!this.uid) {
            console.error('No user ID available');
            return;
        }

        try {
            const response = await fetch(`/api/bag/notebooks?uid=${this.uid}`);
            const notebooks = await response.json();

            const grid = document.getElementById('notebooks-grid');
            if (!grid) return;

            grid.innerHTML = '';

            if (notebooks.length === 0) {
                grid.innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: rgba(255,255,255,0.5);">
                        <p>No notebooks yet!</p>
                        <p style="font-size: 0.9rem; margin-top: 10px;">Click "Create New Notebook" to get started</p>
                    </div>
                `;
                return;
            }

            notebooks.forEach(nb => {
                const card = document.createElement('div');
                card.className = 'notebook-card';
                card.style.setProperty('--color', nb.color || '#6366f1');
                card.onclick = () => this.openNotebook(nb);

                card.innerHTML = `
                    <div class="notebook-cover" style="background: linear-gradient(135deg, ${nb.color || '#6366f1'}, #1e1b4b);">
                        <span style="font-size: 3rem;">📓</span>
                    </div>
                    <div class="notebook-title">${nb.name}</div>
                    <div class="notebook-meta">${nb.item_count || 0} items • ${nb.subject}</div>
                `;
                grid.appendChild(card);
            });

        } catch (error) {
            console.error('Failed to load notebooks:', error);
        }
    }

    showCreateModal() {
        const overlay = document.getElementById('create-notebook-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.classList.add('active');

            // Reset form
            const nameInput = document.getElementById('notebook-name');
            if (nameInput) nameInput.value = '';
            this.selectedColor = '#6366f1';
        }
    }

    hideCreateModal() {
        const overlay = document.getElementById('create-notebook-overlay');
        if (overlay) {
            overlay.style.display = 'none';
            overlay.classList.remove('active');
        }
    }

    async createNotebook() {
        const name = document.getElementById('notebook-name')?.value;
        const subject = document.getElementById('notebook-subject')?.value || 'general';
        const color = this.selectedColor || '#6366f1';

        if (!name || !name.trim()) {
            alert('Please enter a notebook name');
            return;
        }

        if (!this.uid) {
            alert('Please login first');
            return;
        }

        try {
            const response = await fetch('/api/bag/notebooks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    uid: this.uid,
                    name: name.trim(),
                    subject: subject,
                    color: color
                })
            });

            if (response.ok) {
                this.hideCreateModal();
                // Reload notebooks to show the new one
                await this.loadNotebooks();
            } else {
                alert('Failed to create notebook');
            }
        } catch (error) {
            console.error('Failed to create notebook:', error);
            alert('Error creating notebook');
        }
    }

    async openNotebook(notebook) {
        this.currentNotebook = notebook;
        this.currentNotebookId = notebook.notebook_id;

        // For now, just show an alert. You can implement a full editor view later
        console.log('Opening notebook:', notebook);
        alert(`Notebook "${notebook.name}" opened!\n\nEditor view coming soon...`);

        // TODO: Implement editor overlay similar to the injected HTML version
        // For now, this is a placeholder
    }

    // Helper to save from chat
    async saveFromChat(content) {
        if (!this.uid) {
            alert("Please login first!");
            return;
        }

        try {
            // Get notebooks
            const response = await fetch(`/api/bag/notebooks?uid=${this.uid}`);
            const notebooks = await response.json();

            if (notebooks.length === 0) {
                // Create a default notebook first
                await fetch('/api/bag/notebooks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        uid: this.uid,
                        name: 'General Notes',
                        subject: 'general',
                        color: '#6366f1'
                    })
                });
                // Re-fetch
                return this.saveFromChat(content);
            }

            // Save to the first notebook
            const targetNotebook = notebooks[0];

            if (confirm(`Save this to "${targetNotebook.name}"?`)) {
                await fetch('/api/bag/items', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        uid: this.uid,
                        notebook_id: targetNotebook.notebook_id,
                        content: content,
                        title: content.substring(0, 30) + '...'
                    })
                });
                alert("✅ Saved to bag!");
            }
        } catch (error) {
            console.error('Failed to save to bag:', error);
            alert('Error saving to bag');
        }
    }
}

// Initialize
const myBag = new MyBag();
window.myBag = myBag;

// Expose global functions
window.openBag = () => myBag.open();
window.closeBag = () => myBag.close();
window.showCreateNotebookModal = () => myBag.showCreateModal();
window.hideCreateNotebookModal = () => myBag.hideCreateModal();
window.createNotebook = () => myBag.createNotebook();
window.selectColor = (color) => { myBag.selectedColor = color; };
window.showBagTab = (tabName) => {
    // Switch tabs
    document.querySelectorAll('.bag-tab').forEach(tab => tab.classList.remove('active'));
    document.getElementById(`${tabName}-tab`)?.classList.add('active');

    // Switch views
    if (tabName === 'notebooks') {
        document.getElementById('notebooks-view').style.display = 'block';
        document.getElementById('saved-view').style.display = 'none';
    } else {
        document.getElementById('notebooks-view').style.display = 'none';
        document.getElementById('saved-view').style.display = 'block';
    }
};

console.log('[MY BAG] Initialized successfully');
