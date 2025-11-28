// Avatar selection options for students
const STUDENT_AVATARS = [
    {
        id: 'boy_1',
        emoji: '👦🏻',
        name: 'Smart Boy',
        gender: 'boy'
    },
    {
        id: 'boy_2',
        emoji: '👦🏽',
        name: 'Cool Dude',
        gender: 'boy'
    },
    {
        id: 'boy_3',
        emoji: '🧑🏻',
        name: 'Scholar',
        gender: 'boy'
    },
    {
        id: 'girl_1',
        emoji: '👧🏻',
        name: 'Bright Girl',
        gender: 'girl'
    },
    {
        id: 'girl_2',
        emoji: '👧🏽',
        name: 'Smart Girl',
        gender: 'girl'
    },
    {
        id: 'girl_3',
        emoji: '👩🏻',
        name: 'Genius',
        gender: 'girl'
    },
    {
        id: 'student_1',
        emoji: '🧑‍🎓',
        name: 'Graduate',
        gender: 'neutral'
    },
    {
        id: 'student_2',
        emoji: '👨‍🎓',
        name: 'Topper',
        gender: 'neutral'
    }
];

/**
 * Show avatar selection modal
 */
function showAvatarSelection(callback) {
    const modal = document.createElement('div');
    modal.className = 'auth-modal';
    modal.style.zIndex = '10001';

    modal.innerHTML = `
    <div class="auth-modal-content" style="max-width: 600px;">
      <div class="auth-modal-header">
        <h2>Choose Your Avatar</h2>
        <p style="color: #6b7280; font-size: 0.9rem; margin-top: 0.5rem;">
          Pick an avatar that represents you!
        </p>
      </div>
      
      <div class="auth-modal-body">
        <div class="avatar-grid" style="
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 1rem;
          margin-bottom: 1.5rem;
        ">
          ${STUDENT_AVATARS.map(avatar => `
            <div class="avatar-option" data-avatar-id="${avatar.id}" style="
              text-align: center;
              padding: 1.5rem 1rem;
              border: 3px solid #e5e7eb;
              border-radius: 16px;
              cursor: pointer;
              transition: all 0.2s;
            " onclick="selectAvatar('${avatar.id}')">
              <div style="font-size: 4rem; margin-bottom: 0.5rem;">${avatar.emoji}</div>
              <div style="font-weight: 600; color: #374151; font-size: 0.85rem;">${avatar.name}</div>
            </div>
          `).join('')}
        </div>
        
        <button 
          class="btn primary full-width" 
          id="confirm-avatar-btn" 
          disabled
          style="opacity: 0.5; cursor: not-allowed;"
        >
          Continue
        </button>
      </div>
    </div>
  `;

    document.body.appendChild(modal);

    window.selectedAvatarId = null;
    window.avatarCallback = callback;
}

/**
 * Select an avatar
 */
function selectAvatar(avatarId) {
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

    // Enable confirm button
    const confirmBtn = document.getElementById('confirm-avatar-btn');
    if (confirmBtn) {
        confirmBtn.disabled = false;
        confirmBtn.style.opacity = '1';
        confirmBtn.style.cursor = 'pointer';
        confirmBtn.onclick = () => confirmAvatarSelection(avatarId);
    }

    window.selectedAvatarId = avatarId;
}

/**
 * Confirm avatar selection
 */
function confirmAvatarSelection(avatarId) {
    const avatar = STUDENT_AVATARS.find(a => a.id === avatarId);

    // Close modal
    const modal = document.querySelector('.auth-modal');
    if (modal) {
        modal.remove();
    }

    // Call callback if exists
    if (window.avatarCallback) {
        window.avatarCallback(avatar);
    }
}

console.log('[AVATAR] Avatar selection loaded');
