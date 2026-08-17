import { useState } from 'react'

import { IconChevron } from './Icons'
import { GROUNDING, phaseLabel } from '../lib/phases'

const TONE = {
  ok: 'border-ok/35 bg-ok/10 text-ok',
  warn: 'border-warn/35 bg-warn/10 text-warn',
  danger: 'border-danger/35 bg-danger/10 text-danger',
}

export function GroundingBadge({ status }) {
  const meta = GROUNDING[status]
  if (!meta) return null
  return (
    <span
      title={meta.hint}
      className={`rounded border px-1.5 py-[1px] text-[10px] font-medium uppercase tracking-wide ${TONE[meta.tone]}`}
    >
      {meta.label}
    </span>
  )
}

/**
 * The one-line footer under a completed answer.
 *
 * Collapsed it is a summary; expanded it is the turn's actual provenance —
 * which passages were retrieved, how similar each was, and which of them the
 * answer cited. This is what a judge opens to check that the citation is a
 * record of retrieval rather than a claim the model made.
 */
export function DebugStrip({ message, onOpenCitation }) {
  const [open, setOpen] = useState(false)
  const { citations, tools, latencyMs, grounding, turnId, memory } = message

  const summary = [
    `${citations.length} source${citations.length === 1 ? '' : 's'}`,
    tools.length ? `${tools.length} tool call${tools.length === 1 ? '' : 's'}` : null,
    memory?.length ? `${memory.length} memory` : null,
    latencyMs != null ? `${(latencyMs / 1000).toFixed(1)}s` : null,
  ].filter(Boolean)

  return (
    <div className="mt-3 border-t border-line/70 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="group flex items-center gap-2 text-[11px] text-faint transition-colors duration-100 hover:text-muted"
      >
        <IconChevron
          className={`h-3 w-3 transition-transform duration-150 ${open ? 'rotate-90' : ''}`}
        />
        <GroundingBadge status={grounding} />
        <span className="tabular-nums">{summary.join(' · ')}</span>
      </button>

      {open && (
        <div className="mt-2.5 space-y-3 rounded-lg border border-line bg-raised/60 p-3">
          <Row label="turn">
            <code className="font-mono text-[11px] text-muted">{turnId ?? '—'}</code>
          </Row>

          <Row label="phases">
            <span className="text-[11px] text-muted">
              {message.phases.length
                ? message.phases.map((p) => phaseLabel(p.phase)).join(' → ')
                : 'not recorded for a reloaded turn'}
            </span>
          </Row>

          {tools.length > 0 && (
            <Row label="tools">
              <div className="flex flex-wrap gap-1">
                {tools.map((tool, index) => (
                  <span
                    key={`${tool}-${index}`}
                    className="rounded border border-line bg-surface px-1.5 py-0.5 font-mono text-[10px] text-muted"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </Row>
          )}

          <Row label="cited">
            {citations.length === 0 ? (
              <span className="text-[11px] text-muted">nothing was cited</span>
            ) : (
              <ul className="space-y-1">
                {citations.map((citation) => (
                  <li key={citation.chunk_id}>
                    <button
                      type="button"
                      disabled={!turnId}
                      onClick={turnId ? () => onOpenCitation(citation) : undefined}
                      className="flex w-full items-baseline gap-2 rounded px-1 py-0.5 text-left text-[11px] transition-colors duration-100 enabled:hover:bg-surface disabled:cursor-default"
                    >
                      <span className="font-mono font-semibold text-cite">
                        {citation.marker}
                      </span>
                      <span className="text-muted">
                        §{citation.section_path} · p.{citation.page_start}
                      </span>
                      <span className="ml-auto tabular-nums text-faint">
                        {citation.similarity.toFixed(3)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Row>
        </div>
      )}
    </div>
  )
}

function Row({ label, children }) {
  return (
    <div className="flex gap-3">
      <span className="w-14 shrink-0 pt-[1px] text-[10px] uppercase tracking-wider text-faint">
        {label}
      </span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}
