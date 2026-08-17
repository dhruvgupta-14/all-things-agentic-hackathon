/**
 * Client-side cache of each turn's citation set, keyed by turn_id.
 *
 * `GET /api/sessions/{id}/messages` returns role, content and turn_id — it has
 * no citation payload, and there is no endpoint that lists a turn's citations
 * after the fact. So without this, reloading the page would leave every marker
 * in the transcript inert: the click-through needs a chunk_id, and only the
 * live `citations` event carries one.
 *
 * This is a convenience, not a source of truth. A marker with no cached entry
 * still renders — it is simply not clickable — so a cleared cache degrades to
 * the honest state rather than to a broken one.
 */

const KEY = 'citations:v1'
const MAX_TURNS = 200

function readAll() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) ?? {}
  } catch {
    return {}
  }
}

export function rememberCitations(turnId, citations) {
  if (!turnId || !citations?.length) return
  const all = readAll()
  all[turnId] = citations

  // Bounded, so a long-running browser profile cannot fill its quota.
  const keys = Object.keys(all)
  if (keys.length > MAX_TURNS) {
    for (const stale of keys.slice(0, keys.length - MAX_TURNS)) delete all[stale]
  }

  try {
    localStorage.setItem(KEY, JSON.stringify(all))
  } catch {
    /* quota exceeded: the cache is optional, so drop it silently */
  }
}

export function recallCitations(turnId) {
  if (!turnId) return []
  return readAll()[turnId] ?? []
}
