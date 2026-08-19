import { useCallback, useEffect, useState } from 'react'

import { api } from '../api/client'

/**
 * The learner model, as the reader is allowed to see it.
 *
 * Scores arrive already decayed from `last_reinforced_at`, and the weakness
 * thresholds come down with the payload rather than being duplicated here —
 * a UI that decided for itself what "weak" means would drift from the gate
 * that actually fires callbacks, and the two would disagree on screen.
 */
export function useMemory() {
  const [concepts, setConcepts] = useState([])
  const [thresholds, setThresholds] = useState({ weakBelow: 0.4, confidenceFloor: 0.3 })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const body = await api.listConcepts()
      setConcepts(body.concepts ?? [])
      setThresholds({
        weakBelow: body.weak_below ?? 0.4,
        confidenceFloor: body.confidence_floor ?? 0.3,
      })
      setError(null)
    } catch (caught) {
      setError(caught)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { concepts, thresholds, loading, error, refresh }
}

/** One concept in full: evidence, provenance, neighbours. */
export function useConcept(conceptId) {
  const [concept, setConcept] = useState(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    if (!conceptId) {
      setConcept(null)
      return
    }
    setLoading(true)
    try {
      setConcept(await api.getConcept(conceptId))
    } catch {
      setConcept(null)
    } finally {
      setLoading(false)
    }
  }, [conceptId])

  useEffect(() => {
    load()
  }, [load])

  const correct = useCallback(
    async (score, note) => {
      const updated = await api.correctConcept(conceptId, score, note)
      setConcept(updated)
      return updated
    },
    [conceptId],
  )

  return { concept, loading, correct, reload: load }
}

export function useGraph(enabled) {
  const [graph, setGraph] = useState({ nodes: [], edges: [] })
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    setLoading(true)
    api
      .getGraph()
      .then((body) => {
        if (!cancelled) setGraph(body)
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [enabled])

  return { graph, loading }
}
