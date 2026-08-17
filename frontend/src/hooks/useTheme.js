import { useCallback, useEffect, useState } from 'react'

const KEY = 'theme'

function stored() {
  try {
    return localStorage.getItem(KEY)
  } catch {
    return null
  }
}

/**
 * Light/dark, persisted. The initial class is set by an inline script in
 * index.html so a dark reload never flashes the light palette first; this hook
 * only has to keep the two in step afterwards.
 */
export function useTheme() {
  const [theme, setTheme] = useState(() => stored() ?? 'light')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      /* private browsing: the toggle still works for this session */
    }
  }, [theme])

  const toggle = useCallback(
    () => setTheme((current) => (current === 'dark' ? 'light' : 'dark')),
    [],
  )

  return { theme, toggle }
}
