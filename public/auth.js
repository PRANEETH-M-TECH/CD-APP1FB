/**
 * Firebase Authentication Handler
 * Manages session persistence, login/logout, and user state
 */

class AuthManager {
    constructor() {
        this.currentUser = null;
        this.userData = null;
        this.init();
    }

    init() {
        // Listen for auth state changes (session persistence)
        firebase.auth().onAuthStateChanged(async (user) => {
            if (user) {
                console.log('[AUTH] User logged in:', user.email);
                this.currentUser = user;
                await this.loadUserData();
                this.updateUI();
            } else {
                console.log('[AUTH] User logged out');
                this.currentUser = null;
                this.userData = null;
                this.updateUI();
            }
        });
    }

    async loadUserData() {
        try {
            const userDoc = await db.collection('users').doc(this.currentUser.uid).get();
            if (userDoc.exists) {
                this.userData = userDoc.data();

                // Store in localStorage for quick access
                localStorage.setItem('userClass', this.userData.class || '');
                localStorage.setItem('userRole', this.userData.role || '');
                localStorage.setItem('userName', this.userData.name || '');

                console.log('[AUTH] User data loaded:', this.userData);
            } else {
                console.warn('[AUTH] No user document found in Firestore');
            }
        } catch (error) {
            console.error('[AUTH] Error loading user data:', error);
        }
    }

    updateUI() {
        const navCta = document.querySelector('.lg-nav-cta');
        if (!navCta) return;

        if (this.currentUser && this.userData) {
            // Show user profile
            navCta.innerHTML = `
                <div class="user-profile">
                    <div class="user-info">
                        <span class="user-icon">👤</span>
                        <span class="user-name">${this.userData.name}</span>
                        ${this.userData.class ? `<span class="user-class">(Class ${this.userData.class})</span>` : ''}
                    </div>
                    <button class="btn small outline" onclick="authManager.logout()">Logout</button>
                </div>
            `;
        } else {
            // Show admin login only
            navCta.innerHTML = `
                <a href="/admin-login.html" class="btn small outline">Admin Login</a>
            `;
        }

        // Auto-populate class dropdowns if logged in
        if (this.userData && this.userData.class) {
            document.querySelectorAll('.class-select').forEach(select => {
                select.value = this.userData.class;
                select.disabled = true;
            });
        }
    }

    async login(email, password) {
        try {
            const userCredential = await firebase.auth().signInWithEmailAndPassword(email, password);
            return { success: true, user: userCredential.user };
        } catch (error) {
            console.error('[AUTH] Login error:', error);
            return { success: false, error: error.message };
        }
    }

    async logout() {
        try {
            await firebase.auth().signOut();
            localStorage.clear();
            window.location.href = '/';
            return { success: true };
        } catch (error) {
            console.error('[AUTH] Logout error:', error);
            return { success: false, error: error.message };
        }
    }

    isAuthenticated() {
        return this.currentUser !== null;
    }

    hasRole(role) {
        return this.userData && this.userData.role === role;
    }

    async updateUserClass(classNum) {
        if (!this.currentUser) return { success: false, error: 'Not authenticated' };

        try {
            await db.collection('users').doc(this.currentUser.uid).update({
                class: classNum
            });
            this.userData.class = classNum;
            localStorage.setItem('userClass', classNum);
            this.updateUI();
            return { success: true };
        } catch (error) {
            console.error('[AUTH] Error updating class:', error);
            return { success: false, error: error.message };
        }
    }

    async updateUserProfile(data) {
        if (!this.currentUser) return { success: false, error: 'Not authenticated' };

        try {
            await db.collection('users').doc(this.currentUser.uid).update(data);
            // Update local userData
            Object.assign(this.userData, data);
            // Update localStorage
            if (data.class) localStorage.setItem('userClass', data.class);
            if (data.avatar) localStorage.setItem('userAvatar', data.avatar);
            this.updateUI();
            return { success: true };
        } catch (error) {
            console.error('[AUTH] Error updating profile:', error);
            return { success: false, error: error.message };
        }
    }

    // Check if user needs to complete profile (class or avatar missing)
    needsProfileSetup() {
        if (!this.userData || this.userData.role !== 'student') return false;
        return !this.userData.class || !this.userData.avatar;
    }

    // Legacy method - kept for compatibility
    needsClassSelection() {
        return this.needsProfileSetup();
    }
}

// Global auth manager instance
const authManager = new AuthManager();

// Add CSS for user profile
const userProfileStyles = `
<style>
.user-profile {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.user-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: white;
    font-size: 0.9rem;
}

.user-icon {
    font-size: 1.2rem;
}

.user-name {
    font-weight: 600;
}

.user-class {
    opacity: 0.8;
    font-size: 0.85rem;
}
</style>
`;

// Inject styles
if (!document.querySelector('#user-profile-styles')) {
    const styleEl = document.createElement('div');
    styleEl.id = 'user-profile-styles';
    styleEl.innerHTML = userProfileStyles;
    document.head.appendChild(styleEl);
}

console.log('[AUTH] Auth manager initialized');
