/**
 * Client-side cache of each turn's citation set, keyed by turn_id.
 *
 * **Fallback only.** `GET /api/turns/{turn_id}/citations` is the authoritative
 * source and is what a reloaded transcript reads; this covers the case where
 * that request fails, so a flaky network degrades to a stale-but-clickable
 * pill rather than an inert one.
 *
 * It used to be the only answer, back when the transcript endpoint carried no
 * citation payload and nothing listed a turn's citations after the fact. That
 * left markers inert on any machine that had not seen the turn live.
 *
 * Still not a source of truth. A marker with no entry anywhere still renders —
 * it is simply not clickable — so a cleared cache degrades to the honest state
 * rather than to a broken one.
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
