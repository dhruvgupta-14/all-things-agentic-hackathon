/**
 * Promote a single-line `$$…$$` to a display equation.
 *
 * remark-math only treats `$$` as *block* math when the delimiters sit on
 * their own lines. A line that opens and closes on itself —
 *
 *     $$\mathbb{E}_{q}[f(z)] = \mathbb{E}_{p}[f(g)]$$
 *
 * — parses as inline math instead, and KaTeX then renders it cramped into the
 * paragraph flow rather than centred on its own line. The model writes
 * equations in exactly that form, so without this the one thing a
 * paper-reading tool must get right looks wrong.
 *
 * This rewrites only the delimiters, never the mathematics, and only when a
 * line consists of nothing else. Fenced code is skipped, so a shell snippet
 * containing `$$` is left alone.
 */

const WHOLE_LINE = /^([ \t]*)\$\$(.+?)\$\$[ \t]*$/
const FENCE = /^\s*(```|~~~)/

export function normalizeMath(markdown) {
  if (!markdown || !markdown.includes('$$')) return markdown

  let inFence = false

  return markdown
    .split('\n')
    .map((line) => {
      if (FENCE.test(line)) {
        inFence = !inFence
        return line
      }
      if (inFence) return line

      const match = line.match(WHOLE_LINE)
      if (!match) return line

      const [, indent, body] = match
      // Guard against `$$a$$ and $$b$$` on one line: that is two inline spans,
      // not one display block, and must keep its original meaning.
      if (body.includes('$$')) return line

      return `${indent}$$\n${indent}${body.trim()}\n${indent}$$`
    })
    .join('\n')
}
