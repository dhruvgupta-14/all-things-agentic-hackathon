import { initializeApp } from 'firebase/app'
import { getAuth } from 'firebase/auth'

/**
 * Firebase, for identity only.
 *
 * The `apiKey` here is a public project identifier, not a credential. It is
 * meant to ship in browser JavaScript, and every request it can make is still
 * checked by the backend, which verifies the ID token against this exact
 * project. Hiding it in an env var would only make deploys harder.
 *
 * `projectId` must match the backend's `FIREBASE_PROJECT_ID`. Verification
 * pins the token's audience to that value, so a mismatch rejects every token
 * with a 401 that looks like a login bug.
 *
 * Analytics is deliberately not initialised: it pulls a second SDK, brings a
 * cookie-consent question with it, and adds a failure mode to a recorded demo
 * for no benefit here.
 */
const firebaseConfig = {
  apiKey: 'AIzaSyDBoycPkygePPPkrzT3CoKSGLy7C5nLQ20',
  authDomain: 'research-companion-506013.firebaseapp.com',
  projectId: 'research-companion-506013',
  storageBucket: 'research-companion-506013.firebasestorage.app',
  messagingSenderId: '929850602194',
  appId: '1:929850602194:web:dc8b76513459942b75b30b',
}

export const app = initializeApp(firebaseConfig)
export const auth = getAuth(app)

/**
 * A fresh ID token, or null when nobody is signed in.
 *
 * Always asked for per request rather than cached: Firebase rotates ID tokens
 * roughly hourly, and the SDK refreshes transparently here. Storing the string
 * would serve a stale token the moment a session outlived an hour — which is
 * exactly the length of a demo rehearsal.
 */
export async function currentIdToken() {
  const user = auth.currentUser
  if (!user) return null
  try {
    return await user.getIdToken()
  } catch {
    return null
  }
}
