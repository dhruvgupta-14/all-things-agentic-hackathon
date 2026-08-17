/**
 * Turns `[1]` in the answer text into a `<cite data-marker="1">` element.
 *
 * Done as a remark plugin rather than a regex over the rendered HTML, so the
 * substitution only ever touches markdown *text* nodes. Markers inside inline
 * code, fenced blocks, or `$...$` math are different node types and are left
 * exactly as the model wrote them — an index into a matrix, `x[1]`, must not
 * turn into a citation.
 *
 * The emitted node is an `emphasis` carrying `data.hName`, because every
 * built-in mdast handler applies `data` on the way to hast. `cite` is chosen
 * as the tag name because KaTeX's output is spans and divs, so overriding it
 * in react-markdown cannot collide with rendered mathematics.
 */

import { visit } from 'unist-util-visit'

const MARKER = /\[(\d{1,2})\]/g

export function remarkCitations() {
  return (tree) => {
    visit(tree, 'text', (node, index, parent) => {
      if (!parent || index === null || index === undefined) return
      if (!node.value.includes('[')) return

      const pieces = []
      let cursor = 0
      MARKER.lastIndex = 0

      let match
      while ((match = MARKER.exec(node.value)) !== null) {
        if (match.index > cursor) {
          pieces.push({ type: 'text', value: node.value.slice(cursor, match.index) })
        }
        pieces.push({
          type: 'emphasis',
          data: {
            hName: 'cite',
            hProperties: { 'data-marker': match[1] },
          },
          children: [{ type: 'text', value: match[0] }],
        })
        cursor = match.index + match[0].length
      }

      if (!pieces.length) return
      if (cursor < node.value.length) {
        pieces.push({ type: 'text', value: node.value.slice(cursor) })
      }

      parent.children.splice(index, 1, ...pieces)
      return index + pieces.length
    })
  }
}
