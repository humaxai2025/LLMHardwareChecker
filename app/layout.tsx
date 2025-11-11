import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ThemeProvider } from '../lib/ThemeContext'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'LLM Hardware Compatibility Checker v1.0.0',
  description: 'Discover which Large Language Models your system can run locally. Get personalized recommendations with detailed installation instructions.',
  keywords: 'LLM, Large Language Models, Hardware Compatibility, Ollama, Local AI, Machine Learning',
  authors: [{ name: 'Sriram Srinivasan' }],
  robots: 'index, follow',
  openGraph: {
    title: 'LLM Hardware Compatibility Checker v1.0.0',
    description: 'Check if your system can run Large Language Models locally. Get personalized recommendations and installation guides.',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'LLM Hardware Compatibility Checker v1.0.0',
    description: 'Check if your system can run Large Language Models locally.',
  },
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <meta name="theme-color" content="#3b82f6" />
      </head>
      <body className={inter.className} suppressHydrationWarning>
        <ThemeProvider>
          <div id="root">
            {children}
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}