import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Build configuration only. There is deliberately no dev server proxy here.
//
// The SPA is served by FastAPI from the same origin as the API, in the
// container and on localhost alike — see app/spa.py. A proxy would recreate
// the one thing this arrangement exists to avoid: a development setup whose
// request path differs from the deployed one, where a same-origin assumption
// holds locally and quietly fails once it is real.
//
// The development loop is `npm run build`, then run the backend; it serves
// whatever is in `dist`.
export default defineConfig({
  plugins: [react()],
})
