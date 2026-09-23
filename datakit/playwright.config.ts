import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  testMatch: /.*\.spec\.ts/,
  webServer: {
    command: 'vite --host 127.0.0.1 --port 5174',
    url: 'http://127.0.0.1:5174',
    reuseExistingServer: true,
    timeout: 30000,
  },
  use: {
    baseURL: 'http://127.0.0.1:5174',
    viewport: { width: 375, height: 812 },
  },
  projects: [
    {
      name: 'android-chrome-375',
      use: { ...devices['Pixel 5'], viewport: { width: 375, height: 812 } },
    },
    {
      name: 'ios-safari-375',
      use: { ...devices['iPhone 12'], viewport: { width: 375, height: 812 } },
    },
  ],
})
