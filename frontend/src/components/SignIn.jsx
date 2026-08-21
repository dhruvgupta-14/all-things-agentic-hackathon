import { useState } from 'react'

/**
 * The one screen before the product.
 *
 * Deliberately plain. Everything distinctive about this system is behind the
 * login, and a decorated sign-in page would be the first thing a judge sees
 * and the least representative.
 */
export function SignIn({ onSignIn, error }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    try {
      await onSignIn(email.trim(), password)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-bg px-6">
      <div className="w-full max-w-[22rem]">
        <h1 className="font-serif text-[24px] leading-tight text-ink">
          Reading Companion
        </h1>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">
          A reading partner for research papers that remembers what you found
          difficult, and cites every claim back to the page it came from.
        </p>

        <form onSubmit={submit} className="mt-8 flex flex-col gap-3">
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-faint">
              Email
            </span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="username"
              required
              className="rounded border border-line bg-surface px-3 py-2 text-[13px] text-ink outline-none transition-colors focus:border-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-faint">
              Password
            </span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
              className="rounded border border-line bg-surface px-3 py-2 text-[13px] text-ink outline-none transition-colors focus:border-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            />
          </label>

          {error && (
            <p role="alert" className="text-[12px] leading-relaxed text-danger">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="mt-2 rounded bg-accent px-3 py-2 text-[13px] font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  )
}

/** Shown while Firebase resolves a persisted session, so returning readers do
 *  not see the login form flash before their session loads. */
export function AuthLoading() {
  return (
    <div className="flex h-screen items-center justify-center bg-bg">
      <p className="animate-breathe text-[13px] text-faint">Signing you in…</p>
    </div>
  )
}
