/**
 * Profile Dropdown Component
 * Creates a professional dropdown menu in the top-right corner
 * Shows user avatar, name, and options for Profile and Logout
 */

// Global state
let profileDropdownVisible = false;

/**
 * Initialize profile dropdown in the top-right corner
 * @param {Object} userData - User data containing name, avatar, class, etc.
 */
function initializeProfileDropdown(userData) {
    const avatarDisplay = userData.avatar || (userData.name || 'S').charAt(0).toUpperCase();
    const isEmoji = userData.avatar && userData.avatar.length <= 2;

    const dropdownHTML = `
        <div class="profile-dropdown-container">
            <button class="profile-dropdown-trigger" id="profile-trigger" onclick="toggleProfileDropdown(event)">
                <div class="profile-trigger-avatar" style="${isEmoji ? 'font-size: 1.25rem;' : ''}">${avatarDisplay}</div>
                <div class="profile-trigger-info">
                    <div class="profile-trigger-name">${userData.name || 'Student'}</div>
                    <div class="profile-trigger-class">Class ${userData.class || '-'}</div>
                </div>
                <svg class="profile-trigger-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 9l6 6 6-6" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>

            <div class="profile-dropdown-menu" id="profile-dropdown-menu">
                <div class="profile-dropdown-header">
                    <div class="profile-dropdown-avatar" style="${isEmoji ? 'font-size: 2.5rem;' : ''}">${avatarDisplay}</div>
                    <div class="profile-dropdown-user-info">
                        <div class="profile-dropdown-name">${userData.name || 'Student'}</div>
                        <div class="profile-dropdown-class">Class ${userData.class || '-'}</div>
                        ${userData.email ? `<div class="profile-dropdown-email">${userData.email}</div>` : ''}
                    </div>
                </div>
                <div class="profile-dropdown-divider"></div>
                <div class="profile-dropdown-items">
                    <div class="profile-dropdown-item" onclick="goToProfile(); closeProfileDropdown();">
                        <span class="profile-dropdown-item-icon">👤</span>
                        <span>My Profile</span>
                    </div>
                    <div class="profile-dropdown-item logout-item" onclick="handleLogout(); closeProfileDropdown();">
                        <span class="profile-dropdown-item-icon">🚪</span>
                        <span>Logout</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Find or create container in the top-right
    let container = document.getElementById('profile-dropdown-root');
    if (!container) {
        container = document.createElement('div');
        container.id = 'profile-dropdown-root';
        document.body.appendChild(container);
    }

    container.innerHTML = dropdownHTML;

    // Close dropdown when clicking outside
    document.addEventListener('click', handleOutsideClick);
}

function toggleProfileDropdown(event) {
    event.stopPropagation();
    const menu = document.getElementById('profile-dropdown-menu');
    const trigger = document.getElementById('profile-trigger');

    profileDropdownVisible = !profileDropdownVisible;

    if (profileDropdownVisible) {
        menu.classList.add('visible');
        trigger.classList.add('active');
    } else {
        menu.classList.remove('visible');
        trigger.classList.remove('active');
    }
}

function closeProfileDropdown() {
    profileDropdownVisible = false;
    const menu = document.getElementById('profile-dropdown-menu');
    const trigger = document.getElementById('profile-trigger');

    if (menu) menu.classList.remove('visible');
    if (trigger) trigger.classList.remove('active');
}

function handleOutsideClick(event) {
    const container = document.querySelector('.profile-dropdown-container');
    if (container && !container.contains(event.target) && profileDropdownVisible) {
        closeProfileDropdown();
    }
}

console.log('[PROFILE DROPDOWN] Component loaded');
