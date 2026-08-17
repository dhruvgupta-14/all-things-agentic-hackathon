/**
 * Turn phases, as named by the SSE contract (app/schemas/sse.py).
 *
 * The stepper is append-only: a row appears when its `state` event actually
 * arrives, and the row above it flips to done. Nothing is shown speculatively.
 * That matters because the pipeline does not currently emit every phase —
 * `consulting_memory` never fires, since no tool reads learner memory yet, and
 * a checklist that ticked off "consulting memory" would be claiming work that
 * did not happen.
 *
 * Labels for the unemitted phases are defined anyway, so that when those tools
 * land their events render without a frontend change.
 */

export const PHASE_LABELS = {
  started: 'Opening turn',
  retrieving: 'Searching the paper',
  consulting_memory: 'Consulting what you know',
  composing: 'Composing the answer',
  verifying: 'Checking every citation',
  persisted: 'Saved',
}

export function phaseLabel(phase) {
  return PHASE_LABELS[phase] ?? phase
}

export const GROUNDING = {
  grounded: {
    label: 'grounded',
    hint: 'Every claim marker resolves to a retrieved passage.',
    tone: 'ok',
  },
  degraded: {
    label: 'uncited',
    hint: 'Passages were retrieved, but the answer cited none of them.',
    tone: 'warn',
  },
  no_evidence: {
    label: 'no evidence',
    hint: 'Nothing in the retrieved passages supports this answer.',
    tone: 'danger',
  },
}

export const ERROR_MESSAGES = {
  agent_unavailable:
    'The model is temporarily unavailable. Nothing was saved — try again.',
  scope_violation:
    'The turn was stopped: something tried to read outside this paper.',
  empty_response: 'The model returned nothing. Try rephrasing the question.',
  internal_error: 'The turn could not be completed.',
}

export function errorMessage(code, fallback) {
  return ERROR_MESSAGES[code] ?? fallback ?? 'The turn could not be completed.'
}
