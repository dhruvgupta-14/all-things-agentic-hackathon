import { useState } from 'react'

import { useConcept, useMemory } from '../hooks/useMemory'

/**
 * What the system believes about the reader, and why.
 *
 * The "why" is the point. A score with no evidence behind it is an assertion;
 * this panel shows the observations that produced it, each linked to the turn
 * it came from. It is also where a reader can say we got it wrong — a
 * correction outranks inference and is never silently overwritten.
 *
 * Amber is reserved for citations everywhere in this app, so nothing here
 * uses it: understanding is shown on the accent, and weakness in `warn`.
 *
 * The concept graph is deliberately not drawn here. Its edges are still built
 * at ingest and still drive the cross-paper callback, the memory prefetch and
 * quiz sequencing — see app/services/callbacks.py — they are simply not shown.
 */

function scoreLabel(concept, weakBelow) {
  if (concept.understanding_score === null) return 'Not assessed'
  if (concept.is_weak) return 'Needs another pass'
  if (concept.understanding_score >= 0.75) return 'Solid'
  return 'Getting there'
}

function ScoreBar({ score, weak }) {
  if (score === null) {
    return (
      <div className="h-1 w-full rounded-full bg-line" aria-hidden="true" />
    )
  }
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-line">
      <div
        className={`h-full rounded-full ${weak ? 'bg-warn' : 'bg-accent'}`}
        style={{ width: `${Math.max(3, Math.round(score * 100))}%` }}
      />
    </div>
  )
}

function Confidence({ value, floor }) {
  if (value === null || value === undefined) return null
  const sure = value >= floor
  return (
    <span
      className="text-[11px] text-faint"
      title={
        sure
          ? 'Enough evidence for the system to act on this'
          : 'Too little evidence to make a claim — this is a reason to ask, not to announce'
      }
    >
      {sure ? 'confident' : 'unsure'} · {value.toFixed(2)}
    </span>
  )
}

export function MemoryPanel({ open, onClose, onOpenTurn }) {
  const { concepts, thresholds, loading, refresh } = useMemory()
  const [selectedId, setSelectedId] = useState(null)

  if (!open) return null

  return (
    <aside className="flex w-[420px] shrink-0 flex-col border-l border-line bg-surface">
      <header className="flex h-[52px] shrink-0 items-center justify-between border-b border-line px-5">
        <div>
          <h2 className="font-serif text-[14px] font-semibold text-ink">
            What I remember
          </h2>
          <p className="text-[11px] text-faint">
            {concepts.length} concept{concepts.length === 1 ? '' : 's'} from your papers
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded px-2 py-1 text-[12px] text-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          Close
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {loading && (
          <p className="px-5 py-4 text-[12px] text-faint">Reading memory…</p>
        )}

        {!loading && concepts.length === 0 && (
          <p className="px-5 py-4 text-[12px] leading-relaxed text-faint">
            Nothing yet. Concepts appear here as papers are ingested, and gain
            scores as you work through them.
          </p>
        )}

        <ul className="divide-y divide-line">
          {concepts.map((concept) => (
            <li key={concept.concept_id}>
              <button
                type="button"
                onClick={() =>
                  setSelectedId(
                    selectedId === concept.concept_id ? null : concept.concept_id,
                  )
                }
                aria-expanded={selectedId === concept.concept_id}
                className="w-full px-5 py-3 text-left transition-colors hover:bg-raised focus-visible:outline focus-visible:-outline-offset-2 focus-visible:outline-accent"
              >
                <div className="flex items-baseline justify-between gap-3">
                  <span className="truncate font-serif text-[13px] text-ink">
                    {concept.canonical_name}
                  </span>
                  <span
                    className={`shrink-0 text-[11px] ${
                      concept.is_weak ? 'text-warn' : 'text-muted'
                    }`}
                  >
                    {scoreLabel(concept, thresholds.weakBelow)}
                  </span>
                </div>

                <div className="mt-2">
                  <ScoreBar score={concept.understanding_score} weak={concept.is_weak} />
                </div>

                <div className="mt-1.5 flex items-center gap-2">
                  <Confidence
                    value={concept.score_confidence}
                    floor={thresholds.confidenceFloor}
                  />
                  {concept.effective_style && (
                    <span className="text-[11px] text-faint">
                      · {concept.effective_style} explanations work
                    </span>
                  )}
                  <span className="ml-auto text-[11px] text-faint">
                    {concept.evidence_count} signal
                    {concept.evidence_count === 1 ? '' : 's'}
                  </span>
                </div>
              </button>

              {selectedId === concept.concept_id && (
                <ConceptDetail
                  conceptId={concept.concept_id}
                  onOpenTurn={onOpenTurn}
                  onCorrected={refresh}
                />
              )}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  )
}

function ConceptDetail({ conceptId, onOpenTurn, onCorrected }) {
  const { concept, loading, correct } = useConcept(conceptId)
  const [saving, setSaving] = useState(false)

  if (loading || !concept) {
    return <p className="px-5 pb-4 text-[11px] text-faint">Loading evidence…</p>
  }

  const applyCorrection = async (score) => {
    setSaving(true)
    try {
      await correct(score, 'Corrected from the memory panel.')
      onCorrected?.()
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="animate-stepIn border-t border-line bg-raised px-5 py-4">
      {concept.source_papers.length > 0 && (
        <p className="mb-3 text-[11px] text-faint">
          From {concept.source_papers.map((p) => p.title).join(' · ')}
        </p>
      )}

      {concept.related.length > 0 && (
        <div className="mb-4">
          <h4 className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-faint">
            Connected to
          </h4>
          <ul className="flex flex-wrap gap-1.5">
            {concept.related.map((related) => (
              <li
                key={related.concept_id}
                className="rounded border border-line px-1.5 py-0.5 text-[11px] text-muted"
                title={`${related.relationship_type} · confidence ${related.confidence.toFixed(2)}`}
              >
                {related.name}
                <span className="ml-1 text-faint">
                  {related.relationship_type.replace(/_/g, ' ')}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <h4 className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-faint">
        Why I think this
      </h4>
      {concept.evidence.length === 0 ? (
        <p className="text-[11px] leading-relaxed text-faint">
          Nothing observed yet — this concept came from the paper, not from
          anything you have said.
        </p>
      ) : (
        <ul className="space-y-2">
          {concept.evidence.map((item) => (
            <li key={item.observation_id} className="text-[11px] leading-relaxed">
              <div className="flex items-baseline gap-2">
                <span className="text-muted">
                  {item.signal_type.replace(/_/g, ' ')}
                </span>
                {item.weight === 0 && (
                  <span className="text-faint" title="Records that the concept came up; claims nothing about understanding">
                    no weight
                  </span>
                )}
                {item.resolved_a_struggle && (
                  <span className="text-ok">resolved a struggle</span>
                )}
                <span className="ml-auto shrink-0 text-faint">
                  {item.observed_at?.slice(0, 10)}
                </span>
              </div>
              {item.note && <p className="text-faint">{item.note}</p>}
              {item.turn_id && onOpenTurn && (
                <button
                  type="button"
                  onClick={() => onOpenTurn(item.turn_id)}
                  className="mt-0.5 text-accent underline decoration-dotted underline-offset-2 transition-colors hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                >
                  see the exchange
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      <div className="mt-4 border-t border-line pt-3">
        <h4 className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-faint">
          Got this wrong?
        </h4>
        <p className="mb-2 text-[11px] leading-relaxed text-faint">
          Your correction outranks anything inferred, and is never overwritten.
        </p>
        <div className="flex gap-1.5">
          <button
            type="button"
            disabled={saving}
            onClick={() => applyCorrection(0.9)}
            className="rounded border border-line px-2 py-1 text-[11px] text-muted transition-colors hover:border-accent/50 hover:text-ink disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            I know this well
          </button>
          <button
            type="button"
            disabled={saving}
            onClick={() => applyCorrection(0.15)}
            className="rounded border border-line px-2 py-1 text-[11px] text-muted transition-colors hover:border-accent/50 hover:text-ink disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            I still find this hard
          </button>
        </div>
        {concept.user_override_score !== null && (
          <p className="mt-2 text-[11px] text-accent">
            You set this to {concept.user_override_score.toFixed(2)}.
          </p>
        )}
      </div>
    </div>
  )
}
