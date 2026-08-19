import { useMemo, useState } from 'react'

/**
 * The concept graph: typed, directed, confidence-weighted edges.
 *
 * The layout is deterministic — nodes sit on a circle ordered by the data,
 * not by a force simulation. A physics layout looks livelier and settles
 * somewhere slightly different every run, which is exactly wrong for
 * something that has to be pointed at during a recorded demo.
 *
 * Edges are drawn once. Symmetric relationship types are stored once with a
 * canonical orientation, so drawing both directions would double the lines and
 * make the type distribution lie.
 *
 * Purely presentational: the panel owns the fetch and hands the data down, so
 * the layout can be rendered from a fixture and asserted on.
 */

const RELATIONSHIP_STYLE = {
  prerequisite_of: { dash: null, label: 'prerequisite of' },
  component_of: { dash: null, label: 'component of' },
  specialisation_of: { dash: '4 2', label: 'specialisation of' },
  contrasts_with: { dash: '2 3', label: 'contrasts with' },
  equivalent_notation: { dash: '1 3', label: 'equivalent notation' },
  co_occurs_with: { dash: '1 4', label: 'co-occurs with' },
}

const SIZE = 380
const RADIUS = 148

export function ConceptGraph({ graph = { nodes: [], edges: [] }, loading = false }) {
  const [hovered, setHovered] = useState(null)

  const placed = useMemo(() => {
    const nodes = graph.nodes ?? []
    const centre = SIZE / 2
    return nodes.map((node, index) => {
      // Start at the top and go clockwise, so the same data always draws the
      // same picture.
      const angle = (index / Math.max(1, nodes.length)) * Math.PI * 2 - Math.PI / 2
      return {
        ...node,
        x: centre + Math.cos(angle) * RADIUS,
        y: centre + Math.sin(angle) * RADIUS,
        angle,
      }
    })
  }, [graph.nodes])

  const byId = useMemo(
    () => Object.fromEntries(placed.map((node) => [node.concept_id, node])),
    [placed],
  )

  const crossPaper = useMemo(() => {
    const set = new Set()
    for (const edge of graph.edges ?? []) {
      const a = byId[edge.source]
      const b = byId[edge.target]
      if (!a || !b) continue
      const papersA = new Set(a.papers ?? [])
      const shared = (b.papers ?? []).some((paper) => papersA.has(paper))
      if (!shared && (a.papers?.length ?? 0) && (b.papers?.length ?? 0)) {
        set.add(`${edge.source}:${edge.target}:${edge.type}`)
      }
    }
    return set
  }, [graph.edges, byId])

  if (loading) {
    return <p className="px-5 py-4 text-[12px] text-faint">Drawing the graph…</p>
  }

  if (placed.length === 0) {
    return (
      <p className="px-5 py-4 text-[12px] leading-relaxed text-faint">
        No concepts yet. The graph is built when a paper is ingested, before
        you ask anything.
      </p>
    )
  }

  const active = hovered ? byId[hovered] : null
  const activeEdges = (graph.edges ?? []).filter(
    (edge) => !hovered || edge.source === hovered || edge.target === hovered,
  )

  return (
    <div className="px-3 py-4">
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="mx-auto block h-auto w-full max-w-[380px]"
          role="img"
          aria-label={`Concept graph: ${placed.length} concepts, ${(graph.edges ?? []).length} relationships`}
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 7 4 L 0 7 z" fill="var(--faint)" />
            </marker>
          </defs>

          {activeEdges.map((edge, index) => {
            const a = byId[edge.source]
            const b = byId[edge.target]
            if (!a || !b) return null
            const style = RELATIONSHIP_STYLE[edge.type] ?? { dash: null }
            const bridges = crossPaper.has(`${edge.source}:${edge.target}:${edge.type}`)
            const lit = hovered && (edge.source === hovered || edge.target === hovered)
            return (
              <line
                key={`${edge.source}-${edge.target}-${edge.type}-${index}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={bridges ? 'var(--accent)' : 'var(--line)'}
                strokeWidth={lit ? 1.6 : bridges ? 1.2 : 0.8}
                strokeDasharray={style.dash ?? undefined}
                opacity={hovered && !lit ? 0.15 : bridges ? 0.85 : 0.5}
                markerEnd="url(#arrow)"
              />
            )
          })}

          {placed.map((node) => {
            const dimmed =
              hovered &&
              hovered !== node.concept_id &&
              !activeEdges.some(
                (edge) =>
                  edge.source === node.concept_id || edge.target === node.concept_id,
              )
            const onRight = Math.cos(node.angle) > -0.01
            return (
              <g
                key={node.concept_id}
                opacity={dimmed ? 0.25 : 1}
                onMouseEnter={() => setHovered(node.concept_id)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: 'default' }}
              >
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.evidence_count > 0 ? 5 : 3.5}
                  fill={
                    node.is_weak
                      ? 'var(--warn)'
                      : node.evidence_count > 0
                        ? 'var(--accent)'
                        : 'var(--faint)'
                  }
                />
                <text
                  x={node.x + (onRight ? 9 : -9)}
                  y={node.y + 3}
                  textAnchor={onRight ? 'start' : 'end'}
                  className="fill-muted font-sans"
                  style={{ fontSize: '7.5px' }}
                >
                  {node.name.length > 24 ? `${node.name.slice(0, 23)}…` : node.name}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      <div className="mt-3 space-y-1.5 px-2">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-faint">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-accent" /> assessed
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-warn" /> needs work
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-faint" /> not yet assessed
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-px w-4 bg-accent" /> spans two papers
          </span>
        </div>

        {active ? (
          <p className="text-[11px] leading-relaxed text-muted">
            <span className="font-serif text-ink">{active.name}</span>
            {active.papers?.length > 0 && (
              <span className="text-faint"> · {active.papers.join(' · ')}</span>
            )}
          </p>
        ) : (
          <p className="text-[11px] leading-relaxed text-faint">
            {placed.length} concepts, {(graph.edges ?? []).length} relationships. Hover a
            concept to isolate its connections.
          </p>
        )}
      </div>
    </div>
  )
}
