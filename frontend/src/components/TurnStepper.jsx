import { IconCheck } from './Icons'
import { phaseLabel } from '../lib/phases'

/**
 * What fills the answer slot while there is nothing to show yet.
 *
 * Citations are verified before the first token is streamed, so the gap
 * between asking and the first character is the whole agent loop — tens of
 * seconds. This is the one place in the UI where animation is doing real work:
 * it is the difference between "the agent is searching" and "nothing is
 * happening".
 *
 * The ladder is append-only. A row exists because its `state` event arrived,
 * never because it was expected to.
 */
export function TurnStepper({ phases, tools }) {
  if (!phases.length) return null

  const last = phases.length - 1

  return (
    <div className="space-y-2.5 py-1">
      {phases.map((phase, index) => {
        const active = index === last
        return (
          <div
            key={`${phase.phase}-${index}`}
            className="flex animate-stepIn items-start gap-2.5"
            style={{ animationDelay: `${Math.min(index, 4) * 30}ms` }}
          >
            <span
              className={[
                'mt-[3px] flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border',
                active
                  ? 'animate-breathe border-accent bg-accent-soft'
                  : 'border-ok/40 bg-ok/10 text-ok',
              ].join(' ')}
            >
              {!active && <IconCheck className="h-2.5 w-2.5" />}
            </span>

            <div className="min-w-0">
              <span
                className={[
                  'text-[13px] leading-tight',
                  active ? 'text-ink' : 'text-muted',
                ].join(' ')}
              >
                {phaseLabel(phase.phase)}
                {active && <span className="animate-caret text-accent">…</span>}
              </span>

              {phase.tools_called?.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {summarize(phase.tools_called).map((tool) => (
                    <span
                      key={tool}
                      className="rounded border border-line bg-raised px-1.5 py-0.5 font-mono text-[10px] text-muted"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )
      })}

      {tools?.length > 0 && !phases[last]?.tools_called?.length && (
        <div className="pl-6 text-[11px] text-faint">
          {tools.length} tool call{tools.length === 1 ? '' : 's'}
        </div>
      )}
    </div>
  )
}

/** `[a, a, b]` reads better as `a ×2`, `b`. */
function summarize(tools) {
  const counts = new Map()
  for (const tool of tools) counts.set(tool, (counts.get(tool) ?? 0) + 1)
  return [...counts].map(([tool, n]) => (n > 1 ? `${tool} ×${n}` : tool))
}
