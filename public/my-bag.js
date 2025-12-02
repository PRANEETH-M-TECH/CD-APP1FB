// My Bag Logic - Premium Redesign

// State
let currentNotebooks = [];
let bagItems = [];
let selectedColor = '#6366f1';

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize if user is logged in
    if (typeof firebase !== 'undefined' && firebase.auth()) {
        firebase.auth().onAuthStateChanged(user => {
            if (user) {
                loadNotebooks();
            }
        });
    }
});

// --- Open/Close Logic ---
window.openBag = function () {
    console.log('Opening My Bag...');
    const overlay = document.getElementById('my-bag-overlay');
    const sidebar = document.getElementById('my-bag-sidebar');

    if (overlay && sidebar) {
        overlay.classList.add('active');
        sidebar.classList.add('active');
        loadNotebooks();
    } else {
        console.error('My Bag elements not found!');
    }
}

window.closeBag = function () {
    document.getElementById('my-bag-overlay').classList.remove('active');
    document.getElementById('my-bag-sidebar').classList.remove('active');
}

// --- Tab Switching ---
window.showBagTab = function (tabName) {
    // Update buttons
    document.querySelectorAll('.bag-tab').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Update views
    document.getElementById('notebooks-view').style.display = tabName === 'notebooks' ? 'block' : 'none';
    document.getElementById('saved-view').style.display = tabName === 'saved' ? 'block' : 'none';

    // Load items if switching to saved tab
    if (tabName === 'saved') {
        loadBagItems();
    }
}

// --- Notebook Management ---
async function loadNotebooks() {
    const user = firebase.auth().currentUser;
    if (!user) return;

    try {
        const snapshot = await firebase.firestore()
            .collection('users')
            .doc(user.uid)
            .collection('notebooks')
            .orderBy('createdAt', 'desc')
            .get();

        currentNotebooks = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
        renderNotebooks();
        updateNotebookFilter();
    } catch (error) {
        console.error('Error loading notebooks:', error);
    }
}

function renderNotebooks() {
    const grid = document.getElementById('notebooks-grid');
    if (!grid) return;

    if (currentNotebooks.length === 0) {
        grid.innerHTML = `
            <div class="text-center py-8 opacity-50">
                <div class="text-4xl mb-2">📓</div>
                <p>No notebooks yet. Create one to get started!</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = currentNotebooks.map(nb => `
        <div class="notebook-card" onclick="openNotebook('${nb.id}')" style="--notebook-color: ${nb.color || '#6366f1'}">
            <div class="notebook-icon" style="color: ${nb.color || '#6366f1'}">
                ${getSubjectIcon(nb.subject)}
            </div>
            <div class="notebook-info">
                <h4>${nb.name}</h4>
                <p>${nb.subject} • ${nb.itemCount || 0} items</p>
            </div>
        </div>
    `).join('');
}

// --- Modal Handling ---
window.showCreateNotebookModal = function () {
    const overlay = document.getElementById('create-notebook-overlay');
    if (overlay) {
        // Reset form
        document.getElementById('notebook-name').value = '';
        document.getElementById('notebook-subject').value = 'science';
        selectColor('#6366f1'); // Reset to default color
        overlay.classList.add('active');
    }
}

window.hideCreateNotebookModal = function () {
    const overlay = document.getElementById('create-notebook-overlay');
    if (overlay) overlay.classList.remove('active');
}

window.selectColor = function (color) {
    selectedColor = color;
    document.getElementById('notebook-color').value = color;

    // Update UI
    document.querySelectorAll('.color-btn').forEach(btn => {
        if (btn.dataset.color === color) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });
}

window.createNotebook = async function () {
    const name = document.getElementById('notebook-name').value.trim();
    const subject = document.getElementById('notebook-subject').value;
    const color = document.getElementById('notebook-color').value;
    const user = firebase.auth().currentUser;

    // Validation
    if (!name) {
        alert('Please enter a notebook name');
        return;
    }

    if (!user) {
        alert('You must be logged in to create notebooks');
        return;
    }

    console.log('[MY BAG] Creating notebook:', { name, subject, color, userId: user.uid });

    try {
        // Create notebook in Firestore
        const docRef = await firebase.firestore()
            .collection('users')
            .doc(user.uid)
            .collection('notebooks')
            .add({
                name,
                subject,
                color,
                createdAt: firebase.firestore.FieldValue.serverTimestamp(),
                itemCount: 0
            });

        console.log('[MY BAG] Notebook created successfully:', docRef.id);

        // Hide modal first
        hideCreateNotebookModal();

        // Show success message
        const successMsg = document.createElement('div');
        successMsg.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-[9999] animate-bounce-in';
        successMsg.textContent = `✓ Notebook "${name}" created successfully!`;
        document.body.appendChild(successMsg);
        setTimeout(() => successMsg.remove(), 3000);

        // Reload notebooks
        await loadNotebooks();

        // Success Animation
        showFlyingAnimation('📓', window.innerWidth / 2, window.innerHeight / 2);

    } catch (error) {
        console.error('[MY BAG] Error creating notebook:', error);
        console.error('[MY BAG] Error details:', error.message, error.code);

        // Show user-friendly error message
        let errorMessage = 'Failed to create notebook. ';
        if (error.code === 'permission-denied') {
            errorMessage += 'Please check your permissions and try again.';
        } else if (error.code === 'unavailable') {
            errorMessage += 'Network error. Please check your connection.';
        } else {
            errorMessage += error.message || 'Unknown error occurred.';
        }

        alert(errorMessage);
    }
}

// --- Note Modal Handling ---
window.showCreateNoteModal = async function () {
    const modal = document.getElementById('create-note-modal');
    if (modal) {
        // Load notebooks for the dropdown
        await loadNotebooks();

        // Populate notebook dropdown
        const select = document.getElementById('note-notebook');
        if (select && currentNotebooks.length > 0) {
            select.innerHTML = currentNotebooks.map(nb =>
                `<option value="${nb.id}">${nb.name}</option>`
            ).join('');
        }

        modal.style.display = 'flex';
    }
}

window.hideCreateNoteModal = function () {
    const modal = document.getElementById('create-note-modal');
    if (modal) modal.style.display = 'none';
}

window.saveNote = async function () {
    const notebookId = document.getElementById('note-notebook').value;
    const title = document.getElementById('note-title').value;
    const content = document.getElementById('note-content').value;
    const user = firebase.auth().currentUser;

    if (!notebookId || !title || !content || !user) return;

    try {
        await firebase.firestore()
            .collection('users')
            .doc(user.uid)
            .collection('notebooks')
            .doc(notebookId)
            .collection('items')
            .add({
                type: 'note',
                title,
                content,
                createdAt: firebase.firestore.FieldValue.serverTimestamp()
            });

        // Update item count
        await firebase.firestore()
            .collection('users')
            .doc(user.uid)
            .collection('notebooks')
            .doc(notebookId)
            .update({
                itemCount: firebase.firestore.FieldValue.increment(1)
            });
        hideCreateNoteModal();
        loadNotebooks(); // Refresh counts

        // Success Animation
        showFlyingAnimation('📝', window.innerWidth / 2, window.innerHeight / 2);

    } catch (error) {
        console.error('Error saving note:', error);
        alert('Failed to save note');
    }
}

// --- Animation Logic ---
function showFlyingAnimation(emoji, startX, startY) {
    const flyer = document.createElement('div');
    flyer.className = 'flying-item';
    flyer.textContent = emoji;

    // Set start position
    flyer.style.left = `${startX}px`;
    flyer.style.top = `${startY}px`;

    // Calculate end position (My Bag icon in sidebar)
    // Assuming sidebar is open, bag icon is top left of sidebar
    const endX = window.innerWidth - 400; // Approx sidebar left
    const endY = 50; // Approx header height

    flyer.style.setProperty('--mid-x', `${(endX - startX) / 2}px`);
    flyer.style.setProperty('--mid-y', `${(endY - startY) / 2 - 100}px`); // Arc up
    flyer.style.setProperty('--end-x', `${endX - startX}px`);
    flyer.style.setProperty('--end-y', `${endY - startY}px`);

    document.body.appendChild(flyer);

    // Cleanup
    setTimeout(() => flyer.remove(), 1000);
}

// --- Helpers ---
function getSubjectIcon(subject) {
    const icons = {
        science: '🔬',
        maths: '🔢',
        social: '🌍',
        english: '📖',
        general: '📚'
    };
    return icons[subject?.toLowerCase()] || '📚';
}

function updateNotebookFilter() {
    const filter = document.getElementById('notebook-filter');
    if (filter) {
        filter.innerHTML = '<option value="">All Notebooks</option>' +
            currentNotebooks.map(nb => `<option value="${nb.id}">${nb.name}</option>`).join('');
    }
}

// --- Items Management ---
async function loadBagItems(notebookId = null) {
    const user = firebase.auth().currentUser;
    if (!user) return;

    const list = document.getElementById('items-list');
    if (!list) return;

    list.innerHTML = '<div class="flex justify-center py-8"><div class="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div></div>';

    try {
        let items = [];

        let notebooksToFetch = currentNotebooks;
        if (notebookId) {
            notebooksToFetch = currentNotebooks.filter(nb => nb.id === notebookId);
        }

        // Fetch items from each notebook (parallel requests)
        const promises = notebooksToFetch.map(async nb => {
            const snapshot = await firebase.firestore()
                .collection('users')
                .doc(user.uid)
                .collection('notebooks')
                .doc(nb.id)
                .collection('items')
                .orderBy('createdAt', 'desc')
                .limit(20) // Limit per notebook to avoid overload
                .get();

            return snapshot.docs.map(doc => ({
                id: doc.id,
                notebookId: nb.id,
                notebookName: nb.name,
                notebookColor: nb.color,
                ...doc.data()
            }));
        });

        const results = await Promise.all(promises);
        items = results.flat();

        // Sort combined results by createdAt
        items.sort((a, b) => {
            const dateA = a.createdAt ? a.createdAt.toDate() : new Date(0);
            const dateB = b.createdAt ? b.createdAt.toDate() : new Date(0);
            return dateB - dateA;
        });

        renderItems(items);

    } catch (error) {
        console.error('Error loading items:', error);
        list.innerHTML = '<p class="text-center text-red-500 py-4">Error loading items</p>';
    }
}

function renderItems(items) {
    const list = document.getElementById('items-list');
    if (!list) return;

    if (items.length === 0) {
        list.innerHTML = `
            <div class="text-center py-8 opacity-50">
                <div class="text-4xl mb-2">📝</div>
                <p>No notes found.</p>
            </div>
        `;
        return;
    }

    list.innerHTML = items.map(item => `
        <div class="notebook-card item-card" style="--notebook-color: ${item.notebookColor || '#6366f1'}">
            <div class="notebook-icon" style="color: ${item.notebookColor || '#6366f1'}">
                ${item.type === 'note' ? '📝' : '📄'}
            </div>
            <div class="notebook-info">
                <h4 class="font-bold text-gray-800">${item.title}</h4>
                <p class="text-xs text-gray-500 mb-1 flex items-center gap-1">
                    <span class="w-2 h-2 rounded-full" style="background: ${item.notebookColor}"></span>
                    ${item.notebookName}
                </p>
                <p class="text-sm text-gray-600 line-clamp-2">${item.content}</p>
            </div>
        </div>
    `).join('');
}

window.filterItems = function () {
    const notebookId = document.getElementById('notebook-filter').value;
    loadBagItems(notebookId);
}

window.openNotebook = function (id) {
    // Switch to saved tab
    showBagTab('saved');
    // Set filter
    const filter = document.getElementById('notebook-filter');
    if (filter) {
        filter.value = id;
        loadBagItems(id);
    }
}
