import { useEffect, useState } from 'react'

import { api } from '../api/client'
import { IconClose } from './Icons'

/**
 * The slide-over a marker opens.
 *
 * The passage is fetched rather than taken from the SSE payload, because
 * `GET /api/citations/{turn_id}/{chunk_id}` is the endpoint that proves the
 * point: it resolves only through a `turn_retrievals` row with `was_cited`
 * true, on a turn the caller owns. If the text renders here, the citation is
 * a record of retrieval, not an assertion by the model.
 */
export function CitationPanel({ open, turnId, citation, onClose }) {
  const [source, setSource] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!open || !turnId || !citation) return undefined

    let cancelled = false
    setSource(null)
    setError(null)

    api
      .getCitation(turnId, citation.chunk_id)
      .then((data) => !cancelled && setSource(data))
      .catch((err) => !cancelled && setError(err.message))

    return () => {
      cancelled = true
    }
  }, [open, turnId, citation])

  useEffect(() => {
    if (!open) return undefined
    const onKey = (event) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-ink/20"
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        role="dialog"
        aria-label="Cited passage"
        className="fixed right-0 top-0 z-50 flex h-full w-[420px] flex-col border-l border-line bg-surface shadow-xl"
      >
        <header className="flex items-start gap-3 border-b border-line px-5 py-4">
          <span className="mt-[2px] rounded border border-cite-line bg-cite-soft px-1.5 py-[1px] font-mono text-[11px] font-semibold text-cite">
            {citation?.marker}
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[13px] font-semibold text-ink">
              {source?.section_heading || `Section ${citation?.section_path}`}
            </h2>
            <p className="mt-0.5 truncate text-[11px] text-muted">
              §{citation?.section_path} · page {citation?.page_start}
              {citation?.page_end !== citation?.page_start && `–${citation?.page_end}`}
              {citation?.similarity != null && (
                <span className="tabular-nums"> · similarity {citation.similarity.toFixed(3)}</span>
              )}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-muted transition-colors duration-100 hover:bg-raised hover:text-ink"
          >
            <IconClose />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-5">
          {error && <p className="text-[13px] text-danger">{error}</p>}
          {!error && !source && (
            <p className="text-[13px] text-faint">Loading the passage…</p>
          )}
          {source && (
            <>
              {source.paper_title && (
                <p className="mb-3 text-[11px] uppercase tracking-wider text-faint">
                  {source.paper_title}
                </p>
              )}
              {/* Serif, and quoted: this is the paper's own words, not ours. */}
              <blockquote className="border-l-2 border-cite-line pl-4 font-serif text-[15px] leading-[1.7] text-ink">
                {source.content}
              </blockquote>
            </>
          )}
        </div>

        <footer className="border-t border-line px-5 py-3 text-[10px] leading-relaxed text-faint">
          Retrieved during this turn and flagged as cited. Reachable only through
          the turn that used it.
        </footer>
      </aside>
    </>
  )
}
