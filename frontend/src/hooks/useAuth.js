import { useCallback, useEffect, useState } from 'react'
import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
} from 'firebase/auth'

import { auth } from '../lib/firebase'

/**
 * Who is signed in, if anyone.
 *
 * `ready` is separate from `user` on purpose. Firebase resolves a persisted
 * session asynchronously on load, so for the first moment after a refresh
 * `currentUser` is null even for someone who is signed in. Rendering the login
 * screen during that gap would flash it at every returning reader.
 */
export function useAuth() {
  const [user, setUser] = useState(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => onAuthStateChanged(auth, (next) => {
    setUser(next)
    setReady(true)
  }), [])

  const signIn = useCallback(async (email, password) => {
    setError(null)
    try {
      await signInWithEmailAndPassword(auth, email, password)
      return true
    } catch (caught) {
      setError(messageFor(caught))
      return false
    }
  }, [])

  const signOut = useCallback(async () => {
    await firebaseSignOut(auth)
  }, [])

  return { user, ready, error, signIn, signOut }
}

/**
 * Firebase error codes, in words a reader can act on.
 *
 * Deliberately does not distinguish "no such account" from "wrong password":
 * telling an anonymous caller which emails exist is an account-enumeration
 * gift, and it does not help the person who simply mistyped.
 */
function messageFor(error) {
  switch (error?.code) {
    case 'auth/invalid-email':
      return 'That does not look like an email address.'
    case 'auth/invalid-credential':
    case 'auth/wrong-password':
    case 'auth/user-not-found':
      return 'That email and password do not match an account.'
    case 'auth/too-many-requests':
      return 'Too many attempts. Wait a moment and try again.'
    case 'auth/network-request-failed':
      return 'Could not reach the sign-in service. Check your connection.'
    case 'auth/operation-not-allowed':
      // Almost always the console step nobody remembers.
      return 'Email sign-in is not enabled for this project.'
    default:
      return 'Could not sign in. Please try again.'
  }
}
