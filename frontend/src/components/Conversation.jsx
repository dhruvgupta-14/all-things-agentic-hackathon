import { useEffect, useRef } from 'react'

import { Answer } from './Answer'
import { DebugStrip } from './DebugStrip'
import { TurnStepper } from './TurnStepper'

function UserMessage({ content }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-[14px] leading-relaxed text-ink">
        {content}
      </div>
    </div>
  )
}

function AssistantMessage({ message, onOpenCitation }) {
  const { content, streaming, error, phases, tools } = message

  if (error) {
    return (
      <div className="rounded-lg border border-danger/30 bg-danger/[0.06] px-4 py-3">
        <p className="text-[13px] text-danger">{error.message}</p>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-wider text-danger/60">
          {error.code}
        </p>
      </div>
    )
  }

  // Nothing verified yet, so nothing to read: show what the agent is doing.
  if (!content) {
    return streaming ? (
      <TurnStepper phases={phases} tools={tools} />
    ) : (
      <p className="text-[13px] text-muted">No answer was produced.</p>
    )
  }

  return (
    <div>
      <Answer
        content={content}
        citations={message.citations}
        turnId={message.turnId}
        onOpenCitation={onOpenCitation}
      />
      {streaming && (
        <span className="ml-0.5 inline-block h-[1.05em] w-[2px] animate-caret bg-accent align-text-bottom" />
      )}
      {!streaming && message.grounding && (
        <DebugStrip message={message} onOpenCitation={onOpenCitation} />
      )}
    </div>
  )
}

export function Conversation({ messages, loading, emptyState, onOpenCitation }) {
  const bottom = useRef(null)

  // Follow the stream. `auto` rather than `smooth`: with tokens arriving every
  // few milliseconds, smooth scrolling never catches up.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-[13px] text-faint">
        Loading conversation…
      </div>
    )
  }

  if (!messages.length) return emptyState

  return (
    <div className="mx-auto w-full max-w-prose space-y-7 px-8 py-8">
      {messages.map((message) =>
        message.role === 'user' ? (
          <UserMessage key={message.key} content={message.content} />
        ) : (
          <AssistantMessage
            key={message.key}
            message={message}
            onOpenCitation={onOpenCitation}
          />
        ),
      )}
      <div ref={bottom} />
    </div>
  )
}
