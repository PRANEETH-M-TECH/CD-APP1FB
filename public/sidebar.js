/**
 * Sidebar Component
 * Reusable sidebar with student profile, navigation, and mode switching
 */

function initializeSidebar(userData, options = {}) {
  const container = document.getElementById('sidebar-container');
  if (!container) return;

  const streak = userData.studyStreak || 0;
  const avatarDisplay = userData.avatar || (userData.name || 'S').charAt(0).toUpperCase();
  const isEmoji = userData.avatar && userData.avatar.length <= 2;

  const showSwitchMode = !options.hideSwitchMode;
  const showProfile = !options.hideProfile;
  const showAchievements = !options.hideAchievements;

  // Determine dashboard URL based on user role
  const dashboardUrl = userData.role === 'admin' ? '/admin-dashboard' : '/enhanced-dashboard';

  const sidebarHTML = `
    <!-- Hamburger Menu Button -->
    <button class="hamburger-menu" id="hamburger-menu" onclick="toggleSidebar()">
      <div class="hamburger-icon">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <span class="menu-text">Menu</span>
    </button>

    <!-- Overlay -->
    <div class="sidebar-overlay" onclick="closeSidebar()"></div>

    <!-- Sidebar -->
    <div class="sidebar" id="sidebar">
      <!-- Full Profile Section (Inside Sidebar) - Now clickable to close -->
      <div class="sidebar-profile" onclick="closeSidebar()">
        <div class="profile-avatar" style="${isEmoji ? 'font-size: 2rem;' : ''}">${avatarDisplay}</div>
        <div class="profile-details">
          <div class="profile-name">${userData.name || 'Student'}</div>
          <div class="profile-class">Class ${userData.class || '-'}</div>
          ${streak > 0 ? `<div class="profile-streak">🔥 ${streak} day streak!</div>` : ''}
        </div>
      </div>

      <!-- Navigation -->
      <nav class="sidebar-nav">
        <div class="nav-item" onclick="goToDashboard()">
          <span class="nav-icon">📊</span>
          <span class="nav-label">My Dashboard</span>
        </div>
        <div class="nav-item" onclick="window.openBag()">
          <span class="nav-icon">🎒</span>
          <span class="nav-label">My Bag</span>
        </div>
        ${showSwitchMode ? `
        <div class="nav-item" onclick="goToModeSelection()">
          <span class="nav-icon">🔄</span>
          <span class="nav-label">Switch Mode</span>
        </div>
        ` : ''}
        ${showProfile ? `
        <div class="nav-item" onclick="goToProfile()">
          <span class="nav-icon">👤</span>
          <span class="nav-label">My Profile</span>
        </div>
        ` : ''}
        ${showAchievements ? `
        <div class="nav-item" onclick="showAchievements()">
          <span class="nav-icon">🏆</span>
          <span class="nav-label">Achievements</span>
        </div>
        ` : ''}
      </nav>

      <!-- Home Button at bottom -->
      <div class="sidebar-footer">
        <div class="nav-item" onclick="goToHome()">
          <span class="nav-icon">🏠</span>
          <span class="nav-label">Home</span>
        </div>
      </div>
    </div>
  `;

  // Store dashboard URL for navigation function
  window.__dashboardUrl = dashboardUrl;

  container.innerHTML = sidebarHTML;

  // Also initialize the profile dropdown in the top-right corner
  // Check if the profile dropdown function exists
  if (typeof initializeProfileDropdown === 'function') {
    initializeProfileDropdown(userData);
  } else {
    console.warn('[SIDEBAR] Profile dropdown component not loaded. Include profile-dropdown.js');
  }
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.querySelector('.sidebar-overlay');
  const hamburger = document.getElementById('hamburger-menu');

  const isVisible = sidebar.classList.toggle('visible');
  overlay.classList.toggle('visible', isVisible);
  document.body.classList.toggle('sidebar-open', isVisible);
  if (hamburger) hamburger.classList.toggle('active', isVisible);
}

function closeSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.querySelector('.sidebar-overlay');
  const hamburger = document.getElementById('hamburger-menu');

  sidebar.classList.remove('visible');
  overlay.classList.remove('visible');
  document.body.classList.remove('sidebar-open');
  if (hamburger) hamburger.classList.remove('active');
}

// Make openBag globally accessible
// Make openBag globally accessible if not already defined
if (!window.openBag) {
  window.openBag = function () {
    console.warn('My Bag feature not loaded yet');
    alert('My Bag is loading... Please try again in a moment.');
  };
}

function goToDashboard() {
  window.location.href = window.__dashboardUrl || '/enhanced-dashboard';
}

function goToModeSelection() {
  window.location.href = '/mode-selection';
}

function goToProfile() {
  window.location.href = '/profile';
}

function showAchievements() {
  window.location.href = '/achievements';
}

function goToHome() {
  window.location.href = '/';
}

async function handleLogout() {
  const confirm = window.confirm('Are you sure you want to logout?');
  if (!confirm) return;

  try {
    await firebase.auth().signOut();
    localStorage.clear();
    window.location.href = '/';
  } catch (error) {
    console.error('[SIDEBAR] Logout error:', error);
    alert('Error logging out. Please try again.');
  }
}

console.log('[SIDEBAR] Component loaded');
