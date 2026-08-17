import { useRef, useState } from 'react'

import { IconDoc, IconMoon, IconPlus, IconSun } from './Icons'

const STATUS = {
  queued: { label: 'Queued', tone: 'text-faint', dot: 'bg-faint' },
  processing: { label: 'Processing', tone: 'text-warn', dot: 'bg-warn animate-breathe' },
  ready: null,
  partially_ready: { label: 'Partial', tone: 'text-warn', dot: 'bg-warn' },
  failed: { label: 'Failed', tone: 'text-danger', dot: 'bg-danger' },
}

function paperLabel(paper) {
  return paper.title || paper.nickname || 'Untitled paper'
}

function PaperRow({ paper, active, onSelect }) {
  const status = STATUS[paper.processing_status]
  const usable = paper.processing_status === 'ready' || paper.processing_status === 'partially_ready'

  return (
    <button
      type="button"
      disabled={!usable}
      onClick={() => onSelect(paper)}
      className={[
        'rail-item flex items-start gap-2',
        active ? 'rail-item-active' : 'rail-item-idle',
        usable ? '' : 'cursor-default opacity-70 hover:bg-transparent',
      ].join(' ')}
    >
      <IconDoc className="mt-[2px] h-3.5 w-3.5 shrink-0 opacity-60" />
      <span className="min-w-0 flex-1">
        <span className="line-clamp-2 break-words">{paperLabel(paper)}</span>
        {(status || paper.needs_reindex) && (
          <span className="mt-1 flex items-center gap-1.5">
            {status && (
              <>
                <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} />
                <span className={`text-[11px] ${status.tone}`}>{status.label}</span>
              </>
            )}
            {paper.needs_reindex && (
              <span
                className="text-[11px] text-warn"
                title="Embedded by a different model — not searchable until re-indexed."
              >
                needs re-index
              </span>
            )}
          </span>
        )}
      </span>
    </button>
  )
}

function SessionRow({ session, active, onSelect }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(session)}
      className={[
        'rail-item flex items-baseline gap-2',
        active ? 'rail-item-active' : 'rail-item-idle',
      ].join(' ')}
    >
      <span className="min-w-0 flex-1 truncate">
        {session.paper_title || 'No paper'}
      </span>
      <span className="shrink-0 text-[11px] tabular-nums text-faint">
        {session.turn_count}
      </span>
    </button>
  )
}

export function Rail({
  papers,
  sessions,
  activeSessionId,
  activePaperId,
  onSelectPaper,
  onSelectSession,
  onNewSession,
  onUpload,
  theme,
  onToggleTheme,
}) {
  const [dragging, setDragging] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const fileInput = useRef(null)

  async function accept(files) {
    const file = files?.[0]
    if (!file) return
    setUploadError(null)
    try {
      await onUpload(file)
    } catch (err) {
      setUploadError(err.message)
    }
  }

  return (
    <aside
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        accept(e.dataTransfer.files)
      }}
      className={[
        'relative flex w-[264px] shrink-0 flex-col border-r border-line bg-surface',
        dragging ? 'ring-2 ring-inset ring-accent' : '',
      ].join(' ')}
    >
      <div className="flex items-center gap-2 px-4 pb-3 pt-4">
        <h1 className="flex-1 font-serif text-[15px] font-semibold tracking-tight text-ink">
          Reading Companion
        </h1>
        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          className="rounded p-1.5 text-muted transition-colors duration-100 hover:bg-raised hover:text-ink"
        >
          {theme === 'dark' ? <IconSun /> : <IconMoon />}
        </button>
      </div>

      <div className="flex-1 space-y-6 overflow-y-auto px-2 pb-4">
        <section>
          <div className="flex items-baseline justify-between pr-1">
            <h2 className="rail-heading">Papers</h2>
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              className="rounded px-1.5 py-0.5 text-[11px] text-muted transition-colors duration-100 hover:bg-raised hover:text-ink"
            >
              Add
            </button>
          </div>

          <input
            ref={fileInput}
            type="file"
            accept="application/pdf,.pdf"
            className="hidden"
            onChange={(e) => {
              accept(e.target.files)
              e.target.value = ''
            }}
          />

          <div className="space-y-0.5">
            {papers.map((paper) => (
              <PaperRow
                key={paper.paper_id}
                paper={paper}
                active={paper.paper_id === activePaperId}
                onSelect={onSelectPaper}
              />
            ))}
            {!papers.length && (
              <p className="px-2.5 py-2 text-[12px] leading-relaxed text-faint">
                Drop a PDF anywhere in this panel to add it.
              </p>
            )}
          </div>

          {uploadError && (
            <p className="mt-2 px-2.5 text-[11px] text-danger">{uploadError}</p>
          )}
        </section>

        <section>
          <div className="flex items-baseline justify-between pr-1">
            <h2 className="rail-heading">Sessions</h2>
            <button
              type="button"
              onClick={onNewSession}
              aria-label="New session"
              className="rounded p-1 text-muted transition-colors duration-100 hover:bg-raised hover:text-ink"
            >
              <IconPlus className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="space-y-0.5">
            {sessions.map((session) => (
              <SessionRow
                key={session.session_id}
                session={session}
                active={session.session_id === activeSessionId}
                onSelect={onSelectSession}
              />
            ))}
            {!sessions.length && (
              <p className="px-2.5 py-2 text-[12px] text-faint">No sessions yet.</p>
            )}
          </div>
        </section>
      </div>

      {dragging && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-surface/90">
          <p className="text-[13px] font-medium text-accent">Drop the PDF to ingest</p>
        </div>
      )}
    </aside>
  )
}
