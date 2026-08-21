/**
 * Drive one real turn through the frontend's own client and stream modules,
 * against the deployed service, signed in as a real Firebase user.
 *
 * This is the check that the contract holds end to end — event order, payload
 * shapes, the citation click-through, and the durable transcript a reload
 * rebuilds from. It costs real model quota, so it runs one turn.
 *
 * It used to point at the Vite dev server and rely on the backend's
 * development auth bypass, which meant it exercised a request path and an
 * identity that no longer exist anywhere. It now signs in the way a judge
 * does and talks to the origin a judge would.
 *
 *   PAPER_COMPANION_URL   the service to drive (defaults to the deployed one)
 *   DEMO_EMAIL            a real Firebase account in this project
 *   DEMO_PASSWORD
 *
 *   node verify/live.mjs
 */
globalThis.localStorage = { getItem: () => null, setItem: () => {} }

const ORIGIN =
  process.env.PAPER_COMPANION_URL ??
  'https://paper-companion-929850602194.us-central1.run.app'

// The web API key is a public project identifier, not a credential — the same
// one that ships in the browser bundle. Every request it can make is still
// checked by the backend against a verified ID token.
const FIREBASE_API_KEY = 'AIzaSyDBoycPkygePPPkrzT3CoKSGLy7C5nLQ20'

const email = process.env.DEMO_EMAIL
const password = process.env.DEMO_PASSWORD
if (!email || !password) {
  console.error(
    'DEMO_EMAIL and DEMO_PASSWORD must be set: there is no auth bypass to fall
' +
      'back on, in this harness or anywhere else.',
  )
  process.exit(2)
}

const realFetch = globalThis.fetch

// Identity Toolkit directly rather than the Firebase SDK: the SDK expects a
// browser, and this is the one place the harness needs a token rather than a
// session.
const signIn = await realFetch(
  `https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=${FIREBASE_API_KEY}`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, returnSecureToken: true }),
  },
)
if (!signIn.ok) {
  const detail = await signIn.text()
  console.error(`sign-in failed (${signIn.status}): ${detail}`)
  process.exit(1)
}
const { idToken } = await signIn.json()
console.log(`signed in as ${email} against ${ORIGIN}`)

globalThis.fetch = (url, init) =>
  realFetch(typeof url === 'string' && url.startsWith('/') ? ORIGIN + url : url, init)

const { api, setTokenProvider } = await import('../src/api/client.js')
const { streamTurn } = await import('../src/api/stream.js')

// The same seam `main.jsx` uses in the browser, so every request below carries
// a bearer token exactly as a real one would.
setTokenProvider(async () => idToken)

let failures = 0
const check = (name, ok, detail) => {
  if (ok) console.log(`  PASS  ${name}`)
  else {
    failures += 1
    console.log(`  FAIL  ${name}${detail ? ` -> ${detail}` : ''}`)
  }
}

console.log('\n== papers ==')
const papers = await api.listPapers()
console.log(papers.map((p) => `  ${p.title} [${p.processing_status}]`).join('\n'))
const paper = papers.find((p) => p.processing_status === 'ready')
check('a ready paper exists', Boolean(paper))

console.log('\n== session ==')
const session = await api.createSession(paper.paper_id)
console.log(`  created ${session.session_id}`)
const sessions = await api.listSessions()
check(
  'the new session appears in GET /api/sessions',
  sessions.some((s) => s.session_id === session.session_id),
)
check(
  'it is listed first (most recently active)',
  sessions[0].session_id === session.session_id,
  sessions[0].session_id,
)
check('it carries the paper title', sessions[0].paper_title === paper.title)

console.log('\n== turn ==')
const started = Date.now()
const at = () => `${String((Date.now() - started) / 1000).padStart(6)}s`
const order = []
let text = ''
let citations = []
let done = null

for await (const event of streamTurn({
  sessionId: session.session_id,
  message: 'What is the reparameterization trick?',
})) {
  order.push(event.event)
  if (event.event === 'state') {
    console.log(`  ${at()}  state: ${event.phase}` +
      (event.tools_called?.length ? `  tools=${event.tools_called.join(',')}` : ''))
  } else if (event.event === 'token') {
    text += event.text
  } else if (event.event === 'citations') {
    citations = event.citations
    console.log(`  ${at()}  citations: ${citations.length}`)
  } else if (event.event === 'memory_used') {
    console.log(`  ${at()}  memory_used: ${event.memory.length}`)
  } else if (event.event === 'done') {
    done = event
    console.log(`  ${at()}  done: ${event.grounding_status} in ${event.latency_ms}ms`)
  } else if (event.event === 'error') {
    console.log(`  ${at()}  ERROR ${event.code}: ${event.message}`)
  }
}

if (!done) {
  console.log('\n  turn did not complete; stopping here')
  process.exit(1)
}

console.log('\n== answer ==')
console.log(text.split('\n').map((l) => `  ${l}`).join('\n').slice(0, 900))

console.log('\n== contract ==')
const tokenCount = order.filter((e) => e === 'token').length
check('tokens streamed in slices', tokenCount > 1, `${tokenCount} token events`)
check('last event is done', order.at(-1) === 'done', order.at(-1))
check(
  'citations precede memory_used precede done',
  order.indexOf('citations') < order.indexOf('memory_used') &&
    order.indexOf('memory_used') < order.lastIndexOf('done'),
  order.join(','),
)
check(
  'every token arrives before citations',
  order.lastIndexOf('token') < order.indexOf('citations'),
)
check('state events precede the first token',
  order.indexOf('state') < order.indexOf('token'))
check('grounding status is a known value',
  ['grounded', 'degraded', 'no_evidence'].includes(done.grounding_status),
  done.grounding_status)

console.log('\n== citations ==')
for (const c of citations) {
  console.log(`  ${c.marker} -> §${c.section_path} p.${c.page_start} sim=${c.similarity.toFixed(3)}`)
}

const markersInText = [...text.matchAll(/\[(\d{1,2})\]/g)].map((m) => m[0])
const markersCited = citations.map((c) => c.marker)
check(
  'every marker in the answer has a citation payload',
  markersInText.every((m) => markersCited.includes(m)),
  `text=${[...new Set(markersInText)].join(',')} payload=${markersCited.join(',')}`,
)

console.log('\n== click-through ==')
if (citations.length) {
  const source = await api.getCitation(done.turn_id, citations[0].chunk_id)
  check('the cited passage resolves', Boolean(source.content))
  check('it reports the same section', source.section_path === citations[0].section_path)
  console.log(`  §${source.section_path} "${source.section_heading}" p.${source.page_start}`)
  console.log(`  "${source.content.slice(0, 220).replace(/\s+/g, ' ')}…"`)

  // A chunk id alone must open nothing.
  try {
    await api.getCitation('00000000-0000-0000-0000-000000000000', citations[0].chunk_id)
    check('a foreign turn_id is rejected', false, 'it returned 200')
  } catch (err) {
    check('a foreign turn_id is rejected', err.status === 404, String(err.status))
  }
}

console.log('\n== reload ==')
const transcript = await api.getMessages(session.session_id)
check('the transcript has both messages', transcript.length === 2, `${transcript.length}`)
check('roles are user then assistant',
  transcript.map((m) => m.role).join(',') === 'user,assistant')
check('the assistant message matches what was streamed',
  transcript[1].content === text)
check('the assistant message carries the turn_id',
  transcript[1].turn_id === done.turn_id)

// A reload has to restore clickable pills from the server, not from this
// browser's localStorage — otherwise the transcript is inert on any other
// machine, which is what HANDOFF 6.5 recorded as a defect.
const rehydrated = await api.getTurnCitations(done.turn_id)
check('a reloaded turn recovers its citations from the server',
  rehydrated.citations.length === citations.length,
  `${rehydrated.citations.length} vs ${citations.length} streamed`)
check('the recovered markers match what was streamed',
  rehydrated.citations.map((c) => c.marker).sort().join(',') ===
    citations.map((c) => c.marker).sort().join(','))
check('every recovered citation carries a chunk_id to click through to',
  rehydrated.citations.every((c) => Boolean(c.chunk_id) && Boolean(c.section_path)))

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`)
process.exit(failures === 0 ? 0 : 1)
