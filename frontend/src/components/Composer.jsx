import { useEffect, useRef, useState } from 'react'

import { IconSend } from './Icons'

const MAX_CHARS = 8000 // matches MAX_MESSAGE_CHARS in app/routers/sessions.py

export function Composer({ disabled, streaming, onSend, placeholder }) {
  const [value, setValue] = useState('')
  const textarea = useRef(null)

  // Grow with the question, up to a point.
  useEffect(() => {
    const el = textarea.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  function submit() {
    const question = value.trim()
    if (!question || disabled) return
    setValue('')
    onSend(question)
  }

  return (
    <div className="border-t border-line bg-surface px-8 py-4">
      <div className="mx-auto flex w-full max-w-prose items-end gap-2 rounded-xl border border-line bg-bg px-3 py-2 focus-within:border-accent/60">
        <textarea
          ref={textarea}
          rows={1}
          value={value}
          maxLength={MAX_CHARS}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          className="flex-1 resize-none bg-transparent py-1.5 text-[14px] leading-relaxed text-ink outline-none placeholder:text-faint disabled:cursor-not-allowed"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send"
          className="mb-0.5 rounded-lg bg-accent p-2 text-accent-ink transition-opacity duration-100 disabled:opacity-30"
        >
          <IconSend />
        </button>
      </div>

      <p className="mx-auto mt-2 max-w-prose text-[11px] text-faint">
        {streaming
          ? 'Answering — citations are verified before the first word appears.'
          : 'Enter to send · Shift+Enter for a new line'}
      </p>
    </div>
  )
}
