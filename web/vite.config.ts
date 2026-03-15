import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/upload': 'http://localhost:8420',
      '/status': 'http://localhost:8420',
      '/download': 'http://localhost:8420',
      '/cancel': 'http://localhost:8420',
      '/api': 'http://localhost:8420',
      '/ws': {
        target: 'ws://localhost:8420',
        ws: true,
      },
    },
  },
})
