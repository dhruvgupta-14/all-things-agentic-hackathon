/**
 * Temporary harness: does `[1]` survive mdast -> hast as <cite data-marker>,
 * and is it correctly NOT applied inside code and mathematics?
 */
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'

import { remarkCitations } from '../src/lib/remarkCitations.js'

const pipeline = unified()
  .use(remarkParse)
  .use(remarkMath)
  .use(remarkCitations)
  .use(remarkRehype)
  .use(rehypeKatex, { throwOnError: false, strict: false })

async function hast(md) {
  return pipeline.run(pipeline.parse(md))
}

function cites(tree) {
  const found = []
  visit(tree, 'element', (node) => {
    if (node.tagName === 'cite') found.push(node.properties)
  })
  return found
}

function text(tree) {
  let out = ''
  visit(tree, 'text', (node) => {
    out += node.value
  })
  return out
}

let failures = 0
function check(name, condition, detail) {
  if (condition) {
    console.log(`  PASS  ${name}`)
  } else {
    failures += 1
    console.log(`  FAIL  ${name}${detail ? ` -> ${detail}` : ''}`)
  }
}

// 1. The real answer from the live turn.
const real =
  '### What It Is\n\nThe reparameterization trick expresses a continuous random ' +
  'variable $z \\sim q_\\phi(z|x)$ as a deterministic function [1]. It is used ' +
  'again in section 4 [2], and once more here [1].\n'
const a = await hast(real)
check('markers become <cite>', cites(a).length === 3, JSON.stringify(cites(a)))
check(
  'data-marker carries the number',
  cites(a).map((p) => p['data-marker']).join(',') === '1,2,1',
  JSON.stringify(cites(a)),
)
check('inline math still rendered by katex', JSON.stringify(a).includes('katex'))

// 2. Markers must not be created inside code or mathematics.
const b = await hast('Index `x[1]` and $a_{[2]}$ and\n\n```py\ny = v[3]\n```\n')
check('no citation inside inline code, math or a fence', cites(b).length === 0,
  JSON.stringify(cites(b)))

// 3. Two-digit markers, and non-markers left alone.
const c = await hast('See [12] but not [123] nor [] nor [a].')
check('two-digit marker resolves', cites(c).some((p) => p['data-marker'] === '12'))
check('three-digit is not a marker', !cites(c).some((p) => p['data-marker'] === '123'))
check('[123] survives as literal text', text(c).includes('[123]'), text(c))
check('[a] survives as literal text', text(c).includes('[a]'), text(c))

// 4. Markers spanning emphasis / list items.
const d = await hast('- first [1]\n- second **bold** [2]\n')
check('markers inside list items resolve', cites(d).length === 2)

// 5. Adjacent markers.
const e = await hast('Both [1][2] agree.')
check('adjacent markers both resolve', cites(e).length === 2, JSON.stringify(cites(e)))

// 6. Display-math normalisation, including on partially streamed text.
const { normalizeMath } = await import('../src/lib/normalizeMath.js')

check(
  'a whole-line $$…$$ becomes a display block',
  normalizeMath('a\n\n$$x = y$$\n\nb') === 'a\n\n$$\nx = y\n$$\n\nb',
  JSON.stringify(normalizeMath('a\n\n$$x = y$$\n\nb')),
)
check(
  'inline $…$ is untouched',
  normalizeMath('the value $z$ here') === 'the value $z$ here',
)
check(
  'mid-sentence $$…$$ is left inline',
  normalizeMath('see $$x$$ inline') === 'see $$x$$ inline',
)
check(
  'two spans on one line keep their meaning',
  normalizeMath('$$a$$ and $$b$$') === '$$a$$ and $$b$$',
)
check(
  '$$ inside a fence is left alone',
  normalizeMath('```sh\n$$x$$\n```') === '```sh\n$$x$$\n```',
)
check(
  'a half-streamed equation is not mangled',
  normalizeMath('text\n\n$$x = ') === 'text\n\n$$x = ',
)
check('normalising twice changes nothing',
  normalizeMath(normalizeMath('$$x$$')) === normalizeMath('$$x$$'))

const normalized = await hast(normalizeMath('$$\\mathbb{E}[f(z)] = g$$'))
check('the normalised block reaches katex as display',
  JSON.stringify(normalized).includes('katex-display'))
check('citations still resolve alongside display math',
  (await hast(normalizeMath('Given [1]:\n\n$$x = y$$\n'))).children.length > 0 &&
    cites(await hast(normalizeMath('Given [1]:\n\n$$x = y$$\n'))).length === 1)

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`)
process.exit(failures === 0 ? 0 : 1)
