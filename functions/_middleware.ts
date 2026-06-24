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

export const onRequest = [canonicalHostRedirect]
