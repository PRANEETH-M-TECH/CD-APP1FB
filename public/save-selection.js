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
    if (currentNotebooks.length === 0) {
        await loadNotebooks();
    }

    if (currentNotebooks.length === 0) {
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
                <label class="block text-sm font-medium text-gray-700 mb-2">Choose Notebook:</label>
                <div class="space-y-2" id="notebook-selector-list">
                    ${currentNotebooks.map(nb => `
                        <button onclick="saveSelectionToNotebook('${nb.id}')" 
                                class="w-full flex items-center gap-3 p-3 border-2 border-gray-200 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition group">
                            <div class="w-10 h-10 rounded-lg flex items-center justify-center text-xl" 
                                 style="background: ${nb.color}20; color: ${nb.color}">
                                ${getSubjectIcon(nb.subject)}
                            </div>
                            <div class="flex-1 text-left">
                                <div class="font-semibold text-gray-800">${nb.name}</div>
                                <div class="text-xs text-gray-500">${nb.subject} • ${nb.itemCount || 0} items</div>
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
        // Save to Firestore
        await firebase.firestore()
            .collection('users')
            .doc(user.uid)
            .collection('notebooks')
            .doc(notebookId)
            .collection('items')
            .add({
                type: 'selection',
                content: selectedTextContent,
                source: 'chat_response',
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

        // Close modal
        closeNotebookSelector();

        // Show success message
        showSuccessMessage('✓ Saved to notebook!');

        // Flying animation
        showFlyingAnimation('📝', window.innerWidth / 2, window.innerHeight / 2);

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
