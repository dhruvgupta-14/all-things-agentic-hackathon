/**
 * Temporary harness: render the real components through Vite's SSR pipeline.
 *
 * A successful `vite build` only proves the modules resolve. This proves they
 * actually render — a bad hook order, a component that is undefined at use
 * site, or a plugin that throws on the answer text would all pass the build
 * and fail here.
 */
globalThis.localStorage = { getItem: () => null, setItem: () => {} }

const { createServer } = await import('vite')
const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' })

let failures = 0
const check = (name, ok, detail) => {
  if (ok) console.log(`  PASS  ${name}`)
  else {
    failures += 1
    console.log(`  FAIL  ${name}${detail ? ` -> ${detail}` : ''}`)
  }
}

try {
  // React itself is loaded natively; only our own JSX needs Vite's transform,
  // and both must resolve to the same React instance.
  const { renderToStaticMarkup } = await import('react-dom/server')
  const React = (await import('react')).default
  const App = (await server.ssrLoadModule('/src/App.jsx')).default
  const { Answer } = await server.ssrLoadModule('/src/components/Answer.jsx')
  const { TurnStepper } = await server.ssrLoadModule('/src/components/TurnStepper.jsx')
  const { Conversation } = await server.ssrLoadModule('/src/components/Conversation.jsx')

  console.log('\n== app shell ==')
  const { Workspace } = await server.ssrLoadModule('/src/App.jsx')
  const { SignIn } = await server.ssrLoadModule('/src/components/SignIn.jsx')

  // `App` gates on auth before anything else, and effects do not run under
  // SSR, so it renders the resolving state — which is the correct behaviour
  // and is what stops a returning reader seeing the login form flash.
  const gate = renderToStaticMarkup(React.createElement(App))
  check('App renders', gate.length > 0)
  check('it resolves the session before showing anything', gate.includes('Signing you in'))

  const login = renderToStaticMarkup(
    React.createElement(SignIn, { onSignIn: () => {}, error: null }),
  )
  check('the sign-in screen renders', login.includes('Sign in'))
  check('it takes an email and a password', login.includes('type="password"'))
  check(
    'an error is announced to assistive tech',
    renderToStaticMarkup(
      React.createElement(SignIn, { onSignIn: () => {}, error: 'Nope.' }),
    ).includes('role="alert"'),
  )

  const shell = renderToStaticMarkup(React.createElement(Workspace, {}))
  check('the rail is present', shell.includes('Reading Companion'))
  check('the empty state is shown with no session', shell.includes('Drop a PDF'))
  check(
    'sign-out is hidden when there is no session to end',
    !shell.includes('Sign out'),
  )
  check(
    'sign-out appears once signed in',
    renderToStaticMarkup(
      React.createElement(Workspace, { onSignOut: () => {}, user: { email: 'a@b.c' } }),
    ).includes('Sign out'),
  )

  console.log('\n== answer, with the real turn text ==')
  const text =
    'The reparameterization trick expresses $z \\sim q_\\phi(z|x)$ as a ' +
    'deterministic function [1].\n\n$$\\mathbb{E}_{q}[f(z)] = \\mathbb{E}_{p}[f(g)]$$\n\n' +
    'It also appears in section 4 [2], and the variance argument is in [5]. ' +
    'An array index `x[1]` must stay literal.\n'
  const citations = [
    { marker: '[1]', chunk_id: 'c1', paper_id: 'p1', section_path: '2.4', page_start: 4, page_end: 4, similarity: 0.779 },
    { marker: '[5]', chunk_id: 'c5', paper_id: 'p1', section_path: '2.2', page_start: 3, page_end: 3, similarity: 0.678 },
  ]

  const answered = renderToStaticMarkup(
    React.createElement(Answer, { content: text, citations, turnId: 't1', onOpenCitation: () => {} }),
  )
  check('math is rendered by katex', answered.includes('katex'))
  check('display math is rendered', answered.includes('katex-display'))
  check('a resolved marker becomes a button', /<button[^>]*>1<\/button>/.test(answered))
  check('marker [5] resolves even though [2],[3],[4] are absent',
    /<button[^>]*>5<\/button>/.test(answered))
  check('an unresolved marker renders as plain text, not a button',
    !/<button[^>]*>2<\/button>/.test(answered) && answered.includes('[2]'))
  check('a code-span index is left alone', answered.includes('<code>x[1]</code>'))

  console.log('\n== inert before done ==')
  const inert = renderToStaticMarkup(
    React.createElement(Answer, { content: 'Grounded [1].', citations, turnId: null, onOpenCitation: () => {} }),
  )
  check('the pill is disabled until turn_id arrives', inert.includes('disabled='))

  console.log('\n== stepper ==')
  const stepper = renderToStaticMarkup(
    React.createElement(TurnStepper, {
      phases: [
        { phase: 'started', activity: 'FREE', tools_called: [] },
        { phase: 'retrieving', activity: 'FREE', tools_called: [] },
        { phase: 'verifying', activity: 'FREE', tools_called: ['retrieve_paper_context', 'retrieve_paper_context'] },
      ],
      tools: ['retrieve_paper_context'],
    }),
  )
  check('phases are labelled for a reader', stepper.includes('Searching the paper'))
  check('repeated tool calls are summarised', stepper.includes('×2'))
  check('an unemitted phase is never shown', !stepper.includes('Consulting what you know'))

  console.log('\n== error turn ==')
  const errored = renderToStaticMarkup(
    React.createElement(Conversation, {
      messages: [
        { key: 'u', role: 'user', content: 'What is it?' },
        { key: 'a', role: 'assistant', content: '', citations: [], phases: [], tools: [],
          streaming: false, error: { code: 'agent_unavailable', message: 'The model is temporarily unavailable.' } },
      ],
      loading: false,
      emptyState: null,
      onOpenCitation: () => {},
    }),
  )
  check('a failed turn shows its typed code', errored.includes('agent_unavailable'))
  check('the question is still on screen', errored.includes('What is it?'))

  console.log('\n== learner memory ==')
  const { MemoryPanel } = await server.ssrLoadModule('/src/components/MemoryPanel.jsx')
  const { ConceptGraph } = await server.ssrLoadModule('/src/components/ConceptGraph.jsx')

  // Closed is the common case and must cost nothing.
  check(
    'the panel renders nothing while closed',
    renderToStaticMarkup(React.createElement(MemoryPanel, { open: false })) === '',
  )

  const panel = renderToStaticMarkup(React.createElement(MemoryPanel, { open: true }))
  check('the panel renders when open', panel.includes('What I remember'))
  check('both views are reachable', panel.includes('Graph') && panel.includes('Concepts'))

  // The graph is deterministic by design, so the same nodes must always
  // produce the same coordinates — that is what makes it safe to point at
  // during a recording.
  const nodes = [
    { concept_id: 'a', name: 'Reparameterization trick', understanding_score: 0.35,
      score_confidence: 0.7, evidence_count: 2, is_weak: true, papers: ['VAE'] },
    { concept_id: 'b', name: 'Simplified Training Objective', understanding_score: null,
      score_confidence: null, evidence_count: 0, is_weak: false, papers: ['DDPM'] },
  ]
  const edges = [
    { source: 'b', target: 'a', type: 'prerequisite_of', confidence: 0.85,
      discovery_method: 'model' },
  ]
  const graphProps = { graph: { nodes, edges } }
  const first = renderToStaticMarkup(React.createElement(ConceptGraph, graphProps))
  const second = renderToStaticMarkup(React.createElement(ConceptGraph, graphProps))
  check('the graph renders its nodes', first.includes('Reparameterization trick'))
  check('the same data draws the same picture', first === second)
  check(
    'a cross-paper edge is drawn on the accent',
    first.includes('stroke="var(--accent)"'),
  )
  check(
    'an empty graph says so instead of drawing nothing',
    renderToStaticMarkup(
      React.createElement(ConceptGraph, { graph: { nodes: [], edges: [] } }),
    ).includes('No concepts yet'),
  )
} finally {
  await server.close()
}

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`)
process.exit(failures === 0 ? 0 : 1)
