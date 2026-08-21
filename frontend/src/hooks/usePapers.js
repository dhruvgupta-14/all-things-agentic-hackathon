import { useCallback, useEffect, useRef, useState } from 'react'

import { api } from '../api/client'

// Ingestion is a background job: POST returns 202 with status `queued`, and
// the paper becomes searchable some seconds later. These are the states from
// which it will not move on its own.
const TERMINAL = new Set(['ready', 'partially_ready', 'failed'])
const POLL_MS = 2500

// After this long, a paper that is still `queued` is not slow — something has
// gone wrong that nothing will report. Cloud Tasks retries the ingestion push
// five times and then drops the task, and it has no way to tell the
// application it gave up: the row simply stays `queued` forever, with no error
// recorded anywhere. A spinner that never stops says nothing; this turns the
// silence into a statement.
//
// Three minutes is comfortably past a normal ingest (30-60s for a 20-page
// paper) and past the queue's own retry window.
const STALLED_AFTER_MS = 3 * 60 * 1000

function withStalled(paper) {
  if (TERMINAL.has(paper.processing_status) || !paper.created_at) return paper
  const waited = Date.now() - new Date(paper.created_at).getTime()
  return waited > STALLED_AFTER_MS ? { ...paper, stalled: true } : paper
}

export function usePapers() {
  const [papers, setPapers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  // The file currently being uploaded, or null. Held here rather than in the
  // rail so the name survives a re-render mid-upload.
  const [uploading, setUploading] = useState(null)
  const timer = useRef(null)

  const refresh = useCallback(async () => {
    try {
      setPapers((await api.listPapers()).map(withStalled))
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // Poll only while something is still being ingested, and stop as soon as
  // everything has settled — an idle rail should not be making requests.
  useEffect(() => {
    const pending = papers.some((p) => !TERMINAL.has(p.processing_status))
    clearTimeout(timer.current)
    if (!pending) return undefined

    timer.current = setTimeout(refresh, POLL_MS)
    return () => clearTimeout(timer.current)
  }, [papers, refresh])

  const upload = useCallback(
    async (file) => {
      // The request itself is not instant: a 10MB PDF is read, sniffed,
      // page-counted, hashed, put in Cloud Storage and inserted before the 202
      // comes back. Without this the Add button looked inert for several
      // seconds and people clicked it again.
      setUploading(file.name)
      try {
        const created = await api.uploadPaper(file)
        await refresh()
        return created
      } finally {
        setUploading(null)
      }
    },
    [refresh],
  )

  return { papers, loading, error, uploading, refresh, upload }
}
