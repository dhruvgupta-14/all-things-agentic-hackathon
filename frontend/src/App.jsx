import { useCallback, useEffect, useMemo, useState } from 'react'

import { api } from './api/client'
import { CitationPanel } from './components/CitationPanel'
import { Composer } from './components/Composer'
import { Conversation } from './components/Conversation'
import { MemoryPanel } from './components/MemoryPanel'
import { Rail } from './components/Rail'
import { useConversation } from './hooks/useConversation'
import { usePapers } from './hooks/usePapers'
import { useSessions } from './hooks/useSessions'
import { useTheme } from './hooks/useTheme'

const SUGGESTIONS = [
  'What is the reparameterization trick?',
  'How is the objective derived?',
  'What do the experiments actually show?',
]

export default function App() {
  const { theme, toggle } = useTheme()
  const { papers, upload } = usePapers()
  const { sessions, refresh: refreshSessions, create } = useSessions()

  const [sessionId, setSessionId] = useState(null)
  const [open, setOpen] = useState(null)
  const [memoryOpen, setMemoryOpen] = useState(false)

  // Identity is established once; it also provisions the user row on first run.
  useEffect(() => {
    api.me().catch(() => {})
  }, [])

  const { messages, loading, streaming, send } = useConversation(sessionId, {
    onTurnComplete: refreshSessions,
  })

  const session = useMemo(
    () => sessions.find((s) => s.session_id === sessionId) ?? null,
    [sessions, sessionId],
  )
  const paper = useMemo(
    () => papers.find((p) => p.paper_id === session?.active_paper_id) ?? null,
    [papers, session],
  )

  const openSessionForPaper = useCallback(
    async (selected) => {
      // Reuse this paper's most recent session rather than accumulating empty
      // ones every time the rail is clicked.
      const existing = sessions.find((s) => s.active_paper_id === selected.paper_id)
      if (existing) {
        setSessionId(existing.session_id)
        return
      }
      const created = await create(selected.paper_id)
      setSessionId(created.session_id)
    },
    [sessions, create],
  )

  const startBlankSession = useCallback(async () => {
    const created = await create(null)
    setSessionId(created.session_id)
  }, [create])

  const turnIdFor = useCallback(
    (citation) =>
      messages.find((m) => m.citations?.some((c) => c.chunk_id === citation.chunk_id))
        ?.turnId ?? null,
    [messages],
  )

  const openCitation = useCallback(
    (citation) => setOpen({ citation, turnId: turnIdFor(citation) }),
    [turnIdFor],
  )

  return (
    <div className="flex h-screen overflow-hidden">
      <Rail
        papers={papers}
        sessions={sessions}
        activeSessionId={sessionId}
        activePaperId={session?.active_paper_id ?? null}
        onSelectPaper={openSessionForPaper}
        onSelectSession={(s) => setSessionId(s.session_id)}
        onNewSession={startBlankSession}
        onUpload={upload}
        theme={theme}
        onToggleTheme={toggle}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[52px] shrink-0 items-center gap-3 border-b border-line bg-surface px-8">
          {sessionId ? (
            <>
              <h2 className="truncate font-serif text-[14px] font-semibold text-ink">
                {paper?.title || session?.paper_title || 'No paper selected'}
              </h2>
              <span className="text-[11px] text-faint">
                {session?.turn_count ?? 0} turn
                {(session?.turn_count ?? 0) === 1 ? '' : 's'}
              </span>
              {/* The reader has to know their next message is graded, not asked. */}
              {session?.activity === 'QUIZ_PENDING' && (
                <span className="rounded border border-accent/40 bg-accent-soft px-1.5 py-0.5 text-[11px] text-accent-ink">
                  Awaiting your answer
                </span>
              )}
            </>
          ) : (
            <h2 className="text-[13px] text-faint">No session open</h2>
          )}

          <button
            type="button"
            onClick={() => setMemoryOpen((value) => !value)}
            aria-pressed={memoryOpen}
            className={`ml-auto shrink-0 rounded px-2 py-1 text-[12px] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
              memoryOpen ? 'bg-accent-soft text-accent-ink' : 'text-muted hover:text-ink'
            }`}
          >
            What I remember
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {sessionId ? (
            <Conversation
              messages={messages}
              loading={loading}
              onOpenCitation={openCitation}
              emptyState={
                <EmptyConversation
                  paperTitle={paper?.title ?? session?.paper_title}
                  onPick={send}
                />
              }
            />
          ) : (
            <NoSession hasPapers={papers.length > 0} />
          )}
        </div>

        {sessionId && (
          <Composer
            disabled={streaming}
            streaming={streaming}
            onSend={send}
            placeholder={
              session?.activity === 'QUIZ_PENDING'
                ? 'Answer the question above…'
                : paper
                  ? `Ask about ${paper.title ?? 'this paper'}…`
                  : 'Ask a question…'
            }
          />
        )}
      </main>

      <MemoryPanel open={memoryOpen} onClose={() => setMemoryOpen(false)} />

      <CitationPanel
        open={Boolean(open)}
        turnId={open?.turnId}
        citation={open?.citation}
        onClose={() => setOpen(null)}
      />
    </div>
  )
}

function EmptyConversation({ paperTitle, onPick }) {
  return (
    <div className="mx-auto flex h-full w-full max-w-prose flex-col justify-center px-8 pb-16">
      <h3 className="font-serif text-[22px] leading-snug text-ink">
        {paperTitle ? `Reading ${paperTitle}` : 'Ask about your papers'}
      </h3>
      <p className="mt-2 max-w-[46ch] text-[13px] leading-relaxed text-muted">
        Answers are composed only from passages retrieved out of the paper, and
        every citation is checked against what was actually retrieved before a
        single word is streamed.
      </p>
      <ul className="mt-6 space-y-1.5">
        {SUGGESTIONS.map((suggestion) => (
          <li key={suggestion}>
            <button
              type="button"
              onClick={() => onPick(suggestion)}
              className="rounded-lg border border-line px-3 py-2 text-left text-[13px] text-muted transition-colors duration-100 hover:border-accent/50 hover:text-ink"
            >
              {suggestion}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function NoSession({ hasPapers }) {
  return (
    <div className="flex h-full items-center justify-center px-8">
      <p className="max-w-[42ch] text-center text-[13px] leading-relaxed text-faint">
        {hasPapers
          ? 'Choose a paper from the left to start reading it.'
          : 'Drop a PDF into the panel on the left to get started.'}
      </p>
    </div>
  )
}
