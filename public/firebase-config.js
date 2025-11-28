// Firebase Configuration
// Replace these values with your Firebase project settings from Firebase Console

const firebaseConfig = {
    apiKey: "AIzaSyDa4ull0e07rSXv96ZLDRBHotx887mo1qw",
    authDomain: "chaduvu-guru.firebaseapp.com",
    projectId: "chaduvu-guru",
    storageBucket: "chaduvu-guru.firebasestorage.app",
    messagingSenderId: "341712084918",
    appId: "1:341712084918:web:c2909c74616c881babeb17"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Initialize Firestore
const db = firebase.firestore();

console.log('[FIREBASE] Client initialized');
