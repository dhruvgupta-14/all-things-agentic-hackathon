/**
 * Temporary harness: does the SSE reader recover the exact event sequence when
 * the transport splits frames at hostile byte boundaries?
 *
 * The wire text below is a real turn's framing, produced by app/schemas/sse.py.
 */
globalThis.localStorage = { getItem: () => null, setItem: () => {} }

const FRAMES = [
  'event: state\ndata: {"phase":"started","activity":"FREE","tools_called":[]}\n\n',
  'event: state\ndata: {"phase":"retrieving","activity":"FREE","tools_called":[]}\n\n',
  'event: state\ndata: {"phase":"verifying","activity":"FREE","tools_called":["retrieve_paper_context","retrieve_paper_context"]}\n\n',
  'event: token\ndata: {"text":"The reparameterization "}\n\n',
  'event: token\ndata: {"text":"trick expresses z ~ q [1]."}\n\n',
  'event: citations\ndata: {"citations":[{"marker":"[1]","chunk_id":"c1","paper_id":"p1","section_path":"2.4","page_start":4,"page_end":4,"similarity":0.797}]}\n\n',
  'event: memory_used\ndata: {"memory":[]}\n\n',
  'event: state\ndata: {"phase":"persisted","activity":"FREE","tools_called":["retrieve_paper_context"]}\n\n',
  'event: done\ndata: {"turn_id":"t1","grounding_status":"grounded","latency_ms":70375}\n\n',
]
const WIRE = FRAMES.join('')
const EXPECTED = [
  'state', 'state', 'state', 'token', 'token',
  'citations', 'memory_used', 'state', 'done',
]

function bodyChunkedEvery(n) {
  const bytes = new TextEncoder().encode(WIRE)
  let offset = 0
  return {
    getReader() {
      return {
        read: async () => {
          if (offset >= bytes.length) return { done: true, value: undefined }
          const value = bytes.slice(offset, offset + n)
          offset += n
          return { done: false, value }
        },
        cancel: async () => {},
      }
    },
  }
}

let failures = 0
function check(name, condition, detail) {
  if (condition) console.log(`  PASS  ${name}`)
  else {
    failures += 1
    console.log(`  FAIL  ${name}${detail ? ` -> ${detail}` : ''}`)
  }
}

const { streamTurn } = await import('../src/api/stream.js')

async function collect(body) {
  globalThis.fetch = async () => ({ ok: true, status: 200, body })
  const events = []
  for await (const event of streamTurn({ sessionId: 's', message: 'q' })) {
    events.push(event)
  }
  return events
}

// Every chunk size from "one byte at a time" to "the whole stream at once".
for (const size of [1, 3, 17, 64, 200, WIRE.length]) {
  const events = await collect(bodyChunkedEvery(size))
  check(
    `chunked every ${size} byte(s): sequence intact`,
    events.map((e) => e.event).join(',') === EXPECTED.join(','),
    events.map((e) => e.event).join(','),
  )
}

// Payloads survive reassembly, not just event names.
const events = await collect(bodyChunkedEvery(7))
const text = events.filter((e) => e.event === 'token').map((e) => e.text).join('')
check('token text reassembles exactly',
  text === 'The reparameterization trick expresses z ~ q [1].', text)

const done = events.at(-1)
check('done carries turn_id', done.turn_id === 't1')
check('done carries latency', done.latency_ms === 70375)

const citations = events.find((e) => e.event === 'citations').citations
check('citation payload intact',
  citations[0].marker === '[1]' && citations[0].similarity === 0.797,
  JSON.stringify(citations))

const verifying = events.filter((e) => e.event === 'state')[2]
check('tools_called array intact', verifying.tools_called.length === 2)

// A stream that stops mid-turn must not invent a `done`.
const truncated = new TextEncoder().encode(FRAMES.slice(0, 4).join(''))
globalThis.fetch = async () => ({
  ok: true, status: 200,
  body: { getReader: () => { let sent = false; return {
    read: async () => (sent ? { done: true } : ((sent = true), { done: false, value: truncated })),
    cancel: async () => {},
  }}},
})
const partial = []
for await (const e of streamTurn({ sessionId: 's', message: 'q' })) partial.push(e)
check('a truncated stream yields no done event',
  !partial.some((e) => e.event === 'done'), partial.map((e) => e.event).join(','))

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`)
process.exit(failures === 0 ? 0 : 1)
