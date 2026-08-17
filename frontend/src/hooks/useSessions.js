import { useCallback, useEffect, useState } from 'react'

import { api } from '../api/client'

export function useSessions() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      setSessions(await api.listSessions())
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const create = useCallback(
    async (paperId) => {
      const created = await api.createSession(paperId)
      await refresh()
      return created
    },
    [refresh],
  )

  return { sessions, loading, refresh, create }
}
