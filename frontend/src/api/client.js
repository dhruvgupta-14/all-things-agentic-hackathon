/**
 * The HTTP surface, one function per endpoint.
 *
 * Requests go to a relative path: Vite proxies `/api` to the backend, so the
 * browser stays same-origin and the server needs no CORS middleware.
 *
 * `user_id` is never sent. The server derives identity from the bearer token
 * (or, locally, from the dev bypass) and scopes every query itself.
 */

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Request failed (${status})`)
    this.status = status
    this.detail = detail
  }
}

function authHeaders() {
  // Absent in local development, where the backend's dev bypass authenticates
  // the request. Present once Firebase is wired in.
  const token = localStorage.getItem('authToken')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : { 'Content-Type': 'application/json' }),
      ...authHeaders(),
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
}

export { authHeaders }
