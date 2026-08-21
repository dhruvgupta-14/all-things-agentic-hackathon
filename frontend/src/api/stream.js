/**
 * Reading a turn's SSE stream.
 *
 * `EventSource` cannot be used: the turn is a POST with a JSON body, and
 * EventSource only issues GETs and cannot carry an Authorization header. So
 * the stream is read off `fetch`'s body directly and framed by hand.
 *
 * The framing mirrors `app/schemas/sse.py`: `event: <name>\n data: <json>`,
 * frames separated by a blank line. Network chunks land on arbitrary byte
 * boundaries, so a partial frame is held in the buffer until its terminator
 * arrives — splitting a chunk on `\n\n` without buffering would silently drop
 * whichever event happened to straddle the boundary.
 */

// Extension included so this module is loadable by bare node as well as Vite,
// which is what lets the framing be exercised outside a browser.
import { ApiError, authHeaders } from './client.js'

const FRAME_SEPARATOR = '\n\n'

function parseFrame(frame) {
  let name = null
  let data = null

  for (const line of frame.split('\n')) {
    if (line.startsWith('event: ')) name = line.slice(7).trim()
    else if (line.startsWith('data: ')) data = line.slice(6)
  }

  if (!name || data === null) return null
  try {
    return { event: name, ...JSON.parse(data) }
  } catch {
    return null
  }
}

/**
 * Yields decoded events in wire order:
 *   state* -> token* -> citations -> memory_used -> state -> done
 * `error` may replace the tail of any stream.
 */
export async function* streamTurn({ sessionId, message, signal }) {
  const response = await fetch(`/api/sessions/${sessionId}/turns`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
    body: JSON.stringify({ message }),
    signal,
  })

  if (!response.ok || !response.body) {
    let detail = null
    try {
      detail = (await response.json())?.detail ?? null
    } catch {
      /* the body may not be JSON */
    }
    throw new ApiError(response.status, detail)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      let boundary
      while ((boundary = buffer.indexOf(FRAME_SEPARATOR)) !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + FRAME_SEPARATOR.length)
        const event = parseFrame(frame)
        if (event) yield event
      }
    }

    // A final frame with no trailing blank line still counts.
    const tail = parseFrame(buffer)
    if (tail) yield tail
  } finally {
    reader.cancel().catch(() => {})
  }
}

export { parseFrame }
