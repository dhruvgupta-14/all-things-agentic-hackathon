import { useCallback, useEffect, useRef, useState } from 'react'

import { api } from '../api/client'

// Ingestion is a background job: POST returns 202 with status `queued`, and
// the paper becomes searchable some seconds later. These are the states from
// which it will not move on its own.
const TERMINAL = new Set(['ready', 'partially_ready', 'failed'])
const POLL_MS = 2500

export function usePapers() {
  const [papers, setPapers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const timer = useRef(null)

  const refresh = useCallback(async () => {
    try {
      setPapers(await api.listPapers())
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
      const created = await api.uploadPaper(file)
      await refresh()
      return created
    },
    [refresh],
  )

  return { papers, loading, error, refresh, upload }
}
