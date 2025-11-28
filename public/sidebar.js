/**
 * Sidebar Component
 * Reusable sidebar with student profile, navigation, and mode switching
 */

function initializeSidebar(userData, options = {}) {
  const container = document.getElementById('sidebar-container');
  if (!container) return;

  const streak = userData.studyStreak || 0;
  const avatarDisplay = userData.avatar || (userData.name || 'S').charAt(0).toUpperCase();
  const isEmoji = userData.avatar && userData.avatar.length <= 2; // Emoji detection

  // Options: { hideSwitchMode: true/false, hideProfile: true/false, hideAchievements: true/false }
  const showSwitchMode = !options.hideSwitchMode;
  const showProfile = !options.hideProfile;
  const showAchievements = !options.hideAchievements;

  const sidebarHTML = `
    <!-- Mobile Toggle -->
    <button class="sidebar-toggle" onclick="toggleSidebar()">
      ☰
    </button>

    <!-- Overlay -->
    <div class="sidebar-overlay" onclick="closeSidebar()"></div>

    <!-- Sidebar -->
    <div class="sidebar" id="sidebar">
      <!-- Profile Section -->
      <div class="sidebar-profile">
        <div class="profile-avatar" style="${isEmoji ? 'font-size: 3.5rem;' : ''}">${avatarDisplay}</div>
        <div class="profile-name">${userData.name || 'Student'}</div>
        <div class="profile-class">Class ${userData.class || '-'}</div>
        ${streak > 0 ? `<div class="profile-streak">🔥 ${streak} day streak!</div>` : ''}
      </div>

      <!-- Navigation -->
      <nav class="sidebar-nav">
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

      <!-- Logout Button -->
      <button class="logout-btn" onclick="handleLogout()">
        <span>🚪</span>
        <span>Logout</span>
      </button>
    </div>
  `;

  container.innerHTML = sidebarHTML;
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.querySelector('.sidebar-overlay');
  const isMobile = window.innerWidth <= 768;

  if (isMobile) {
    // Mobile: Toggle visible class
    sidebar.classList.toggle('visible');
    overlay.classList.toggle('visible');
    document.body.classList.toggle('sidebar-open');
  } else {
    // Desktop: Toggle collapsed class
    sidebar.classList.toggle('collapsed');
    document.body.classList.toggle('sidebar-collapsed');
  }
}

function closeSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.querySelector('.sidebar-overlay');
  const isMobile = window.innerWidth <= 768;

  if (isMobile) {
    sidebar.classList.remove('visible');
    overlay.classList.remove('visible');
    document.body.classList.remove('sidebar-open');
  } else {
    sidebar.classList.add('collapsed');
    document.body.classList.add('sidebar-collapsed');
  }
}

function goToModeSelection() {
  window.location.href = '/mode-selection';
}

function goToProfile() {
  alert('Profile page coming soon! 🚧');
  // TODO: Create profile page
}

function showAchievements() {
  alert('Achievements:\n🏆 First Login\n🔥 Study Streak Active\n\nMore badges coming soon!');
  // TODO: Create achievements modal
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
