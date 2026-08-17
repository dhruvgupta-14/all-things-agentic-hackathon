import { createContext, useContext, useMemo } from 'react'
import Markdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'

import { normalizeMath } from '../lib/normalizeMath'
import { remarkCitations } from '../lib/remarkCitations'

const CitationContext = createContext({ byMarker: new Map(), turnId: null, onOpen: null })

const REMARK = [remarkMath, remarkCitations]
const REHYPE = [[rehypeKatex, { throwOnError: false, strict: false }]]

/**
 * A `[1]` in the answer.
 *
 * Three states, and the difference between them is deliberate rather than
 * cosmetic:
 *
 *   resolved + turn known  — clickable; opens the passage it points at
 *   resolved, no turn yet  — styled but inert, for the moment between the
 *                            `citations` event and `done`
 *   unresolved             — plain text; the marker survived into a transcript
 *                            whose citation set this browser has not cached
 *
 * A pill never claims to be openable when it is not.
 */
function CitationPill(props) {
  const { byMarker, turnId, onOpen } = useContext(CitationContext)
  const marker = props['data-marker']
  const citation = marker ? byMarker.get(marker) : undefined

  if (!citation) {
    return <span className="text-muted">[{marker}]</span>
  }

  const clickable = Boolean(turnId && onOpen)
  const label = `Source ${marker}: section ${citation.section_path}, page ${citation.page_start}`

  return (
    <button
      type="button"
      disabled={!clickable}
      onClick={clickable ? () => onOpen(citation) : undefined}
      title={clickable ? label : 'Source available once the turn is saved'}
      aria-label={label}
      className={[
        'mx-[1px] inline-flex h-[1.15em] min-w-[1.5em] translate-y-[-0.15em] items-center',
        'justify-center rounded-[4px] border px-1 align-middle font-sans text-[0.68em]',
        'font-semibold leading-none tabular-nums',
        'border-cite-line bg-cite-soft text-cite',
        clickable
          ? 'cursor-pointer transition-colors duration-100 hover:bg-cite hover:text-bg'
          : 'cursor-default opacity-70',
      ].join(' ')}
    >
      {marker}
    </button>
  )
}

export function Answer({ content, citations, turnId, onOpenCitation }) {
  // Recomputed per token during streaming; it is a line-wise string pass over
  // an answer of a few kilobytes, which is not worth memoising around.
  const markdown = normalizeMath(content)

  const value = useMemo(
    () => ({
      // Markers are 1-based positions into the retrieval set; the payload
      // carries them as "[1]", so index by the bare digits.
      byMarker: new Map(
        (citations ?? []).map((c) => [c.marker.replace(/[[\]]/g, ''), c]),
      ),
      turnId,
      onOpen: onOpenCitation,
    }),
    [citations, turnId, onOpenCitation],
  )

  return (
    <CitationContext.Provider value={value}>
      <div className="answer">
        <Markdown
          remarkPlugins={REMARK}
          rehypePlugins={REHYPE}
          components={{ cite: CitationPill }}
        >
          {markdown}
        </Markdown>
      </div>
    </CitationContext.Provider>
  )
}
