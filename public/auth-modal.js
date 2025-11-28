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
            // Check if user needs to complete profile (class or avatar)
            if (authManager.needsProfileSetup()) {
                console.log('[AUTH] User needs profile setup');
                showClassSelectionModal();
            } else {
                console.log('[AUTH] Profile complete, redirecting to mode selection');
                // Redirect to mode selection page
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

    // Check if user already has a class (existing user without avatar)
    const userData = authManager.userData;
    const hasClass = userData && userData.class;

    // Populate avatars if not already done
    const avatarGrid = document.getElementById('avatar-grid');
    if (avatarGrid && avatarGrid.children.length === 0) {
        avatarGrid.innerHTML = STUDENT_AVATARS.map(avatar => `
            <div class="avatar-option" data-avatar-id="${avatar.id}" style="
                text-align: center;
                padding: 1rem 0.5rem;
                border: 3px solid #e5e7eb;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.2s;
            " onclick="selectAvatarInModal('${avatar.id}')">
                <div style="font-size: 3rem;">${avatar.emoji}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">${avatar.name}</div>
            </div>
        `).join('');
    }

    // If user already has a class, pre-select it and hide class step header
    if (hasClass) {
        console.log('[AUTH] User already has class:', userData.class);
        selectedClassNum = userData.class;

        // Hide class selection heading (just show buttons selected)
        const classStep = document.getElementById('class-step');
        if (classStep) {
            const heading = classStep.querySelector('h3');
            if (heading) heading.style.display = 'none';
        }

        // Pre-select the class button
        setTimeout(() => {
            const buttons = document.querySelectorAll('.class-btn');
            buttons.forEach(btn => {
                if (btn.textContent === String(userData.class)) {
                    btn.classList.add('selected');
                }
            });
        }, 100);
    }
}

// Close class selection modal
function closeClassSelectionModal() {
    document.getElementById('class-selection-modal').style.display = 'none';
    selectedClassNum = null;
    selectedAvatarId = null;

    // Deselect all buttons
    document.querySelectorAll('.class-btn').forEach(btn => {
        btn.classList.remove('selected');
    });

    // Deselect all avatars
    document.querySelectorAll('.avatar-option').forEach(opt => {
        opt.style.borderColor = '#e5e7eb';
        opt.style.background = 'white';
    });
}

// Select avatar in modal
function selectAvatarInModal(avatarId) {
    // Remove previous selection
    document.querySelectorAll('.avatar-option').forEach(opt => {
        opt.style.borderColor = '#e5e7eb';
        opt.style.background = 'white';
    });

    // Highlight selected
    const selected = document.querySelector(`[data-avatar-id="${avatarId}"]`);
    if (selected) {
        selected.style.borderColor = '#6b5cff';
        selected.style.background = 'rgba(107, 92, 255, 0.05)';
    }

    selectedAvatarId = avatarId;

    // Hide avatar error
    const avatarError = document.getElementById('avatar-error');
    if (avatarError) avatarError.style.display = 'none';

    // Enable button if class is also selected
    updateSaveButton();
}

// Select a class
function selectClass(classNum) {
    selectedClassNum = classNum;

    // Update button states
    document.querySelectorAll('.class-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    event.target.classList.add('selected');

    // Hide class error
    const classError = document.getElementById('class-error');
    if (classError) classError.style.display = 'none';

    // Enable button if avatar is also selected
    updateSaveButton();
}

function updateSaveButton() {
    const saveBtn = document.getElementById('save-class-btn');
    const userData = authManager.userData;
    const hasExistingClass = userData && userData.class;

    if (saveBtn) {
        if (selectedAvatarId && (selectedClassNum || hasExistingClass)) {
            saveBtn.disabled = false;
        } else {
            saveBtn.disabled = true;
        }
    }
}

// Save class AND avatar selection
async function saveClassAndAvatar() {
    // Validate avatar selection
    if (!selectedAvatarId) {
        const avatarError = document.getElementById('avatar-error');
        if (avatarError) {
            avatarError.style.display = 'block';
            avatarError.textContent = 'Please select an avatar';
        }
        return;
    }

    // Check if user already has a class
    const userData = authManager.userData;
    const hasExistingClass = userData && userData.class;

    // If no class selected and user doesn't have one, show error
    if (!selectedClassNum && !hasExistingClass) {
        const classError = document.getElementById('class-error');
        if (classError) {
            classError.style.display = 'block';
            classError.textContent = 'Please select your class';
        }
        return;
    }

    // Use existing class or newly selected class
    const finalClass = selectedClassNum || userData.class;

    // Find avatar details
    const selectedAvatar = STUDENT_AVATARS.find(a => a.id === selectedAvatarId);

    console.log('[AUTH] Saving profile - Class:', finalClass, 'Avatar:', selectedAvatar.name);

    // Update user profile in Firestore
    const result = await authManager.updateUserProfile({
        class: finalClass,
        avatar: selectedAvatar.emoji,
        avatarId: selectedAvatar.id,
        avatarName: selectedAvatar.name
    });

    if (result.success) {
        console.log('[AUTH] ✅ Profile updated successfully!');
        closeClassSelectionModal();

        // Go to mode selection page
        window.location.href = '/mode-selection';
    } else {
        alert('Error saving profile: ' + result.error);
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
