import { createPagesMiddleware } from '@aiedx/fetchlens-cloudflare'

type PagesContext = {
  request: Request
  next: () => Promise<Response>
  waitUntil: (promise: Promise<unknown>) => void
  env: Record<string, unknown>
}

const fetchlens = createPagesMiddleware({
  siteId: 'cohort.bubblnet.com',
  apiEndpoint: 'https://fetchlens.ai',
  siteTag: 'fl_pub_6f78bbfb4264f7b4b76c3b86272e1b49',
})

export const onRequest = async (context: PagesContext) => {
  const ua = context.request.headers.get('user-agent') || ''
  if (/GPTBot/i.test(ua)) {
    return new Response('Blocked', { status: 403 })
  }
  return fetchlens(context)
}
