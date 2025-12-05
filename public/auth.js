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
            // Explicitly set current user and load their data to avoid race conditions.
            // onAuthStateChanged will also fire but this ensures data is ready immediately.
            this.currentUser = userCredential.user;
            await this.loadUserData();
            return { success: true, user: this.currentUser, userData: this.userData };
        } catch (error) {
            console.error('[AUTH] Login error:', error);
            // Clear user data on login failure
            this.currentUser = null;
            this.userData = null;
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

    /**
     * Enforce authentication on protected pages.
     * If not logged in, saves current URL and redirects to login.
     */
    requireAuth() {
        // Give Firebase a moment to restore session
        const unsubscribe = firebase.auth().onAuthStateChanged(user => {
            unsubscribe(); // Run once
            if (!user) {
                console.log('[AUTH] User not logged in, redirecting to login...');
                // Save current URL to return after login
                sessionStorage.setItem('redirect_after_login', window.location.href);
                // Redirect to landing page with login trigger
                window.location.href = '/?login=true';
            } else {
                console.log('[AUTH] User authenticated:', user.uid);
            }
        });
    }

    /**
     * Handle successful login redirection.
     * Checks for saved redirect URL, otherwise defaults to Dashboard.
     */
    handleLoginSuccess() {
        const redirectUrl = sessionStorage.getItem('redirect_after_login');
        if (redirectUrl) {
            console.log('[AUTH] Restoring saved session URL:', redirectUrl);
            sessionStorage.removeItem('redirect_after_login');
            window.location.href = redirectUrl;
        } else {
            console.log('[AUTH] No saved URL, going to Dashboard');
            window.location.href = '/enhanced-dashboard';
        }
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
