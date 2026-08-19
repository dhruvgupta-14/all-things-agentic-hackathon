import { useCallback, useEffect, useRef, useState } from 'react'

import { api } from '../api/client'
import { streamTurn } from '../api/stream'
import { rememberCitations, recallCitations } from '../lib/citationCache'
import { errorMessage } from '../lib/phases'

let localKey = 0
const nextKey = () => `local-${++localKey}`

function userMessage(content) {
  return {
    key: nextKey(),
    role: 'user',
    content,
    turnId: null,
    citations: [],
    memory: [],
    phases: [],
    tools: [],
    grounding: null,
    latencyMs: null,
    streaming: false,
    error: null,
  }
}

function assistantShell() {
  return {
    key: nextKey(),
    role: 'assistant',
    content: '',
    turnId: null,
    citations: [],
    memory: [],
    phases: [],
    tools: [],
    grounding: null,
    latencyMs: null,
    streaming: true,
    error: null,
  }
}

/**
 * One session's messages, and the machine that drives a turn.
 *
 * History is rebuilt from PostgreSQL rather than kept in memory, so a reload
 * restores the conversation exactly. Only the in-flight turn lives here.
 */
export function useConversation(sessionId, { onTurnComplete } = {}) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const abort = useRef(null)

  useEffect(() => {
    if (!sessionId) {
      setMessages([])
      return undefined
    }

    let cancelled = false
    setLoading(true)

    api
      .getMessages(sessionId)
      .then(async (transcript) => {
        if (cancelled) return
        const shown = transcript.filter(
          (m) => m.role === 'user' || m.role === 'assistant',
        )

        // Citations come from the server, which is the only place they are
        // authoritative — the local cache is a fallback for when that request
        // fails, not the source of truth it used to be. Without this a
        // transcript opened on another machine had inert markers.
        const restored = await Promise.all(
          shown.map(async (m) => {
            if (m.role !== 'assistant' || !m.turn_id) return []
            try {
              const body = await api.getTurnCitations(m.turn_id)
              return body.citations ?? []
            } catch {
              return recallCitations(m.turn_id)
            }
          }),
        )
        if (cancelled) return

        setMessages(
          shown.map((m, index) => ({
            key: m.message_id,
            role: m.role,
            content: m.content,
            turnId: m.turn_id,
            citations: restored[index],
            memory: [],
            phases: [],
            tools: [],
            grounding: null,
            latencyMs: null,
            streaming: false,
            error: null,
          })),
        )
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [sessionId])

  const patch = useCallback((key, update) => {
    setMessages((current) =>
      current.map((message) =>
        message.key === key ? { ...message, ...update(message) } : message,
      ),
    )
  }, [])

  const send = useCallback(
    async (text) => {
      const question = text.trim()
      if (!question || !sessionId || streaming) return

      const shell = assistantShell()
      setMessages((current) => [...current, userMessage(question), shell])
      setStreaming(true)

      // Held outside React state so `done` can cache the set without reading
      // back through an updater, which StrictMode would run twice.
      let citations = []

      const controller = new AbortController()
      abort.current = controller

      try {
        for await (const event of streamTurn({
          sessionId,
          message: question,
          signal: controller.signal,
        })) {
          switch (event.event) {
            case 'state':
              patch(shell.key, (m) => ({
                phases: [...m.phases, event],
                tools: event.tools_called?.length ? event.tools_called : m.tools,
              }))
              break

            case 'token':
              patch(shell.key, (m) => ({ content: m.content + event.text }))
              break

            case 'citations':
              // Rendered inert at this point: the click-through is addressed
              // by turn_id, which only arrives with `done`.
              citations = event.citations ?? []
              patch(shell.key, () => ({ citations }))
              break

            case 'memory_used':
              patch(shell.key, () => ({ memory: event.memory }))
              break

            case 'done':
              rememberCitations(event.turn_id, citations)
              patch(shell.key, () => ({
                turnId: event.turn_id,
                grounding: event.grounding_status,
                latencyMs: event.latency_ms,
                streaming: false,
              }))
              break

            case 'error':
              patch(shell.key, () => ({
                streaming: false,
                error: {
                  code: event.code,
                  message: errorMessage(event.code, event.message),
                },
              }))
              break

            default:
              break
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          patch(shell.key, () => ({
            streaming: false,
            error: {
              code: 'transport_error',
              message:
                err.status === 404
                  ? 'That session no longer exists.'
                  : 'Lost the connection to the server before the turn finished.',
            },
          }))
        }
      } finally {
        // A stream that ends without `done` must not leave a live spinner.
        patch(shell.key, () => ({ streaming: false }))
        setStreaming(false)
        abort.current = null
        onTurnComplete?.()
      }
    },
    [sessionId, streaming, patch, onTurnComplete],
  )

  const cancel = useCallback(() => abort.current?.abort(), [])

  useEffect(() => () => abort.current?.abort(), [])

  return { messages, loading, streaming, send, cancel }
}
