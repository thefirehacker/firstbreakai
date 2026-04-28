import { createPagesMiddleware } from '@aiedx/fetchlens-cloudflare'

const CANONICAL_HOST = 'cohort.bubblnet.com'

async function canonicalHostRedirect(context: {
  request: Request
  next: () => Promise<Response>
}): Promise<Response> {
  const url = new URL(context.request.url)
  if (url.hostname.endsWith('.pages.dev')) {
    const target = `https://${CANONICAL_HOST}${url.pathname}${url.search}`
    return Response.redirect(target, 301)
  }
  return context.next()
}

const fetchlens = createPagesMiddleware({
  siteId: CANONICAL_HOST,
  apiEndpoint: 'https://fetchlens.ai',
  siteTag: 'fl_pub_6f78bbfb4264f7b4b76c3b86272e1b49',
  observeOnly: true,
  blockVulnScans: false,
})

export const onRequest = [canonicalHostRedirect, fetchlens]