import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { setTokenProvider } from './api/client'
import App from './App'
import './index.css'
import { currentIdToken } from './lib/firebase'

// Identity is wired into the transport here rather than imported by it, so
// `api/client.js` stays loadable in Node for the offline verification
// harnesses. This is the only place the two meet.
setTokenProvider(currentIdToken)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
