/**
 * The HTTP surface, one function per endpoint.
 *
 * Requests go to a relative path. FastAPI serves this bundle from the same
 * origin as the API — in the container and on localhost alike — so the browser
 * never leaves that origin, the server needs no CORS middleware, and the SSE
 * stream is first-party. See app/spa.py.
 *
 * `user_id` is never sent. The server derives identity from the bearer token
 * and scopes every query itself.
 */

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Request failed (${status})`)
    this.status = status
    this.detail = detail
  }
}

/**
 * Where a bearer token comes from.
 *
 * Injected rather than imported so this module stays free of the identity
 * provider: the offline verification harnesses load it directly in Node, and
 * pulling the Firebase SDK in here would make the transport untestable without
 * a browser. `main.jsx` supplies the real provider at startup.
 *
 * The default returns nothing, which produces a request with no bearer
 * header — and a 401. There is no unauthenticated path to fall back on.
 */
let tokenProvider = async () => null

export function setTokenProvider(provider) {
  tokenProvider = provider ?? (async () => null)
}

/**
 * The bearer header for one request.
 *
 * Asked for per request rather than read from storage: Firebase rotates ID
 * tokens roughly hourly and refreshes them transparently, so a cached string
 * goes stale inside a single sitting.
 */
async function authHeaders() {
  const token = await tokenProvider()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : { 'Content-Type': 'application/json' }),
      ...(await authHeaders()),
      ...(options.headers || {}),
    },
  })

  if (!response.ok) {
    throw new ApiError(response.status, await readDetail(response))
  }
  if (response.status === 204) return null
  return response.json()
}

async function readDetail(response) {
  try {
    const body = await response.json()
    return typeof body.detail === 'string' ? body.detail : null
  } catch {
    return null
  }
}

export const api = {
  me: () => request('/api/me'),

  listPapers: () => request('/api/papers'),
  getPaper: (paperId) => request(`/api/papers/${paperId}`),
  uploadPaper: (file) => {
    const body = new FormData()
    body.append('file', file)
    return request('/api/papers', { method: 'POST', body })
  },

  listSessions: () => request('/api/sessions'),
  createSession: (paperId) =>
    request('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ paper_id: paperId ?? null }),
    }),
  getMessages: (sessionId) => request(`/api/sessions/${sessionId}/messages`),

  getCitation: (turnId, chunkId) => request(`/api/citations/${turnId}/${chunkId}`),
  // The durable answer to "what did this turn cite?". A reloaded transcript
  // rehydrates from here rather than from the localStorage cache, which does
  // not survive a different browser.
  getTurnCitations: (turnId) => request(`/api/turns/${turnId}/citations`),

  listConcepts: ({ onlyWeak = false } = {}) =>
    request(`/api/memory/concepts${onlyWeak ? '?only_weak=true' : ''}`),
  getConcept: (conceptId) => request(`/api/memory/concepts/${conceptId}`),
  correctConcept: (conceptId, understandingScore, note) =>
    request(`/api/memory/concepts/${conceptId}`, {
      method: 'PATCH',
      body: JSON.stringify({ understanding_score: understandingScore, note }),
    }),
  getGraph: () => request('/api/memory/graph'),
}

export { authHeaders }
