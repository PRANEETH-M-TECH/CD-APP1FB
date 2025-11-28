/**
 * Authentication Modal Functions
 * Handles student login, class selection, and CTA auth checks
 */

let selectedClassNum = null;
let pendingRedirect = null;

// Show student login modal
function showStudentLoginModal(redirectTo = null) {
    pendingRedirect = redirectTo;
    document.getElementById('student-login-modal').style.display = 'flex';
}

// Close student login modal
function closeStudentLoginModal() {
    document.getElementById('student-login-modal').style.display = 'none';
    document.getElementById('login-error').style.display = 'none';
    document.getElementById('student-login-form').reset();
}

// Handle student login form submission
async function handleStudentLogin(event) {
    event.preventDefault();

    const email = document.getElementById('student-email').value;
    const password = document.getElementById('student-password').value;
    const errorDiv = document.getElementById('login-error');

    // Hide previous errors
    errorDiv.style.display = 'none';

    // Attempt login
    const result = await authManager.login(email, password);

    if (result.success) {
        console.log('[AUTH] Login successful');
        closeStudentLoginModal();

        // Wait for userData to load
        setTimeout(() => {
            // Check if user needs to select class
            if (authManager.needsClassSelection()) {
                showClassSelectionModal();
            } else {
                // Redirect to mode selection page (NEW FLOW!)
                window.location.href = '/mode-selection';
            }
        }, 500);
    } else {
        errorDiv.textContent = result.error;
        errorDiv.style.display = 'block';
    }
}

// Show class selection modal
function showClassSelectionModal() {
    document.getElementById('class-selection-modal').style.display = 'flex';
}

// Close class selection modal
function closeClassSelectionModal() {
    document.getElementById('class-selection-modal').style.display = 'none';
    selectedClassNum = null;

    // Deselect all buttons
    document.querySelectorAll('.class-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
}

// Select a class
function selectClass(classNum) {
    selectedClassNum = classNum;

    // Update button states
    document.querySelectorAll('.class-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    event.target.classList.add('selected');

    // Enable save button
    document.getElementById('save-class-btn').disabled = false;
}

// Save class selection
async function saveClass() {
    if (!selectedClassNum) return;

    const section = document.getElementById('student-section').value.trim();

    // Update user class in Firestore
    const result = await authManager.updateUserClass(selectedClassNum);

    if (result.success) {
        console.log('[AUTH] Class saved:', selectedClassNum);
        closeClassSelectionModal();

        // Go to mode selection page
        window.location.href = '/mode-selection';
    } else {
        alert('Error saving class: ' + result.error);
    }
}

// Check auth before accessing features
function checkAuthAndProceed() {
    if (authManager.isAuthenticated()) {
        // Already logged in - go to mode selection
        window.location.href = '/mode-selection';
    } else {
        // Not logged in - show login modal
        showStudentLoginModal();
    }
}

// Initialize auth check listeners on page load
document.addEventListener('DOMContentLoaded', () => {
    // "Start Learning" button (NEW!)
    const startLearningBtn = document.getElementById('start-learning-btn');
    if (startLearningBtn) {
        startLearningBtn.addEventListener('click', (e) => {
            e.preventDefault();
            checkAuthAndProceed();
        });
    }
});

console.log('[AUTH] Modal functions loaded');
