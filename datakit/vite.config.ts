import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: true,
    port: 5174,
    strictPort: true,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/label': 'http://127.0.0.1:8000',
    },
  },
  preview: {
    host: true,
    port: 4174,
    strictPort: true,
  },
  build: {
    target: 'es2022',
  },
})

