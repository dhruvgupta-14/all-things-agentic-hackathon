/** Inline SVG only — no icon package, no network fetch, no bundle weight. */

const base = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  viewBox: '0 0 24 24',
}

export function IconSun({ className = 'h-4 w-4' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
    </svg>
  )
}

export function IconMoon({ className = 'h-4 w-4' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
    </svg>
  )
}

export function IconClose({ className = 'h-4 w-4' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  )
}

export function IconPlus({ className = 'h-4 w-4' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M12 5v14M5 12h14" />
    </svg>
  )
}

export function IconSend({ className = 'h-4 w-4' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M5 12h13M12 5l7 7-7 7" />
    </svg>
  )
}

export function IconChevron({ className = 'h-3.5 w-3.5' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M9 6l6 6-6 6" />
    </svg>
  )
}

export function IconDoc({ className = 'h-4 w-4' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </svg>
  )
}

export function IconCheck({ className = 'h-3.5 w-3.5' }) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M4 12.5l5 5L20 6.5" />
    </svg>
  )
}
