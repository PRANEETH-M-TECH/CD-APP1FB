/**
 * Text Selection Save to Bag Feature
 * Allows users to select text from chat responses and save them to notebooks
 */

let selectionTooltip = null;
let selectedTextContent = '';

// Initialize text selection feature
function initTextSelection() {
    console.log('[TEXT SELECTION] Initializing...');

    // Create tooltip element
    createSelectionTooltip();

    // Add event listeners to chat history
    const chatHistory = document.getElementById('chat-history');
    if (chatHistory) {
        chatHistory.addEventListener('mouseup', handleTextSelection);
        document.addEventListener('mousedown', handleClickOutside);
    }
}

function createSelectionTooltip() {
    if (selectionTooltip) return;

    selectionTooltip = document.createElement('div');
    selectionTooltip.id = 'selection-tooltip';
    selectionTooltip.className = 'selection-tooltip hidden';
    selectionTooltip.innerHTML = `
        <button class="save-selection-btn" onclick="showNotebookSelector()">
            <span>🎒</span> Save to Bag
        </button>
    `;
    document.body.appendChild(selectionTooltip);
}

function handleTextSelection(event) {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    // Hide tooltip if no text selected
    if (!selectedText || selectedText.length < 3) {
        hideSelectionTooltip();
        return;
    }

    // Check if selection is within chat history
    const chatHistory = document.getElementById('chat-history');
    if (!chatHistory.contains(selection.anchorNode) && !chatHistory.contains(selection.focusNode)) {
        hideSelectionTooltip();
        return;
    }

    // Store selected text
    selectedTextContent = selectedText;

    // Position and show tooltip
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    showSelectionTooltip(rect);
}

function showSelectionTooltip(rect) {
    if (!selectionTooltip) return;

    const tooltipWidth = 180;
    const tooltipHeight = 50;

    // Position above the selection, centered
    let left = rect.left + (rect.width / 2) - (tooltipWidth / 2);
    let top = rect.top - tooltipHeight - 10;

    // Ensure tooltip stays within viewport
    if (left < 10) left = 10;
    if (left + tooltipWidth > window.innerWidth - 10) {
        left = window.innerWidth - tooltipWidth - 10;
    }
    if (top < 10) {
        top = rect.bottom + 10; // Show below if not enough space above
    }

    selectionTooltip.style.left = `${left + window.scrollX}px`;
    selectionTooltip.style.top = `${top + window.scrollY}px`;
    selectionTooltip.classList.remove('hidden');
}

function hideSelectionTooltip() {
    if (selectionTooltip) {
        selectionTooltip.classList.add('hidden');
    }
}

function handleClickOutside(event) {
    if (selectionTooltip && !selectionTooltip.contains(event.target)) {
        // Small delay to let the save button click register
        setTimeout(() => {
            const selection = window.getSelection();
            if (!selection.toString().trim()) {
                hideSelectionTooltip();
            }
        }, 100);
    }
}

window.showNotebookSelector = async function () {
    // Load notebooks if not already loaded
    // Fix: Ensure window.myBag exists
    if (!window.myBag) {
        console.error('MyBag not initialized');
        return;
    }

    if (window.myBag.notebooks.length === 0) {
        await window.myBag.loadNotebooks();
    }

    if (window.myBag.notebooks.length === 0) {
        alert('Please create a notebook first before saving selections.');
        hideSelectionTooltip();
        return;
    }

    // Create and show notebook selector modal
    const modal = document.createElement('div');
    modal.id = 'notebook-selector-modal';
    modal.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm z-[9999] flex items-center justify-center';
    modal.innerHTML = `
        <div class="bg-white rounded-2xl p-6 w-full max-w-md shadow-2xl m-4 animate-scale-in">
            <h3 class="text-xl font-bold mb-4 text-gray-800 flex items-center gap-2">
                <span>📓</span> Save Selection
            </h3>
            <div class="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200 max-h-32 overflow-y-auto">
                <p class="text-sm text-gray-700 italic">"${truncateText(selectedTextContent, 150)}"</p>
            </div>
            
            <div class="mb-4">
                <label class="flex items-center gap-2 text-sm font-medium text-gray-700 cursor-pointer select-none">
                    <input type="checkbox" id="open-after-save" checked class="w-4 h-4 text-purple-600 rounded border-gray-300 focus:ring-purple-500">
                    Open in editor after saving
                </label>
            </div>

            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">Choose Notebook:</label>
                <div class="space-y-2" id="notebook-selector-list">
                    ${window.myBag.notebooks.map(nb => `
                        <button onclick="saveSelectionToNotebook('${nb.notebook_id}')" 
                                class="w-full flex items-center gap-3 p-3 border-2 border-gray-200 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition group">
                            <div class="w-10 h-10 rounded-lg flex items-center justify-center text-xl" 
                                 style="background: ${nb.color || '#6366f1'}20; color: ${nb.color || '#6366f1'}">
                                📓
                            </div>
                            <div class="flex-1 text-left">
                                <div class="font-semibold text-gray-800">${nb.name}</div>
                                <div class="text-xs text-gray-500">${nb.subject} • ${nb.item_count || 0} items</div>
                            </div>
                            <svg class="w-5 h-5 text-gray-400 group-hover:text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                            </svg>
                        </button>
                    `).join('')}
                </div>
            </div>
            <button onclick="closeNotebookSelector()" 
                    class="w-full bg-gray-100 text-gray-700 py-2 rounded-lg font-semibold hover:bg-gray-200 transition">
                Cancel
            </button>
        </div>
    `;

    document.body.appendChild(modal);
    hideSelectionTooltip();
}

window.saveSelectionToNotebook = async function (notebookId) {
    const user = firebase.auth().currentUser;
    if (!user || !selectedTextContent) {
        closeNotebookSelector();
        return;
    }

    try {
        // Check if user wants to open editor
        const shouldOpenEditor = document.getElementById('open-after-save')?.checked;

        // If opening editor, we can let the inline editor handle the saving/insertion!
        // This is much cleaner than duplicating logic.
        if (shouldOpenEditor && window.inlineNotebook) {
            closeNotebookSelector();
            window.inlineNotebook.open(notebookId, selectedTextContent);
            return;
        }

        // Otherwise, save to Firestore manually (background save)
        const pagesSnapshot = await firebase.firestore()
            .collection('notebooks')
            .doc(notebookId)
            .collection('pages')
            .orderBy('pageNumber', 'asc')
            .limit(1)
            .get();

        let pageId;

        if (pagesSnapshot.empty) {
            // Create first page
            const newPage = {
                pageNumber: 1,
                content: JSON.stringify({
                    ops: [{ insert: selectedTextContent + '\n' }]
                }),
                createdAt: firebase.firestore.FieldValue.serverTimestamp(),
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            };
            const pageRef = await firebase.firestore()
                .collection('notebooks')
                .doc(notebookId)
                .collection('pages')
                .add(newPage);
            pageId = pageRef.id;
        } else {
            // Append to first page
            const firstPage = pagesSnapshot.docs[0];
            pageId = firstPage.id;
            const existingContent = firstPage.data().content;

            // Parse existing Quill Delta content
            let delta;
            try {
                delta = JSON.parse(existingContent);
            } catch (e) {
                delta = { ops: [] };
            }

            // Append new content with highlighting
            delta.ops.push({ insert: '\n\n' });
            delta.ops.push({ insert: selectedTextContent, attributes: { background: '#d1fae5' } });
            delta.ops.push({ insert: '\n' });

            await firebase.firestore()
                .collection('notebooks')
                .doc(notebookId)
                .collection('pages')
                .doc(pageId)
                .update({
                    content: JSON.stringify(delta),
                    updatedAt: firebase.firestore.FieldValue.serverTimestamp()
                });
        }

        // Update item count (optional, if backend doesn't do it)
        // Note: Backend usually handles this via triggers, but we can do it optimistically if needed.

        // Close modal
        closeNotebookSelector();

        // Show success message
        showSuccessMessage('✓ Saved to notebook');

        // Clear selection
        window.getSelection().removeAllRanges();
        selectedTextContent = '';

    } catch (error) {
        console.error('[TEXT SELECTION] Error saving:', error);
        alert('Failed to save selection. Please try again.');
        closeNotebookSelector();
    }
}

window.closeNotebookSelector = function () {
    const modal = document.getElementById('notebook-selector-modal');
    if (modal) modal.remove();
}

function showSuccessMessage(message) {
    const msg = document.createElement('div');
    msg.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-[9999] animate-bounce-in';
    msg.textContent = message;
    document.body.appendChild(msg);
    setTimeout(() => msg.remove(), 3000);
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTextSelection);
} else {
    initTextSelection();
}

console.log('[TEXT SELECTION] Module loaded');
