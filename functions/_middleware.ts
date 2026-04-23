import { createPagesMiddleware } from '@aiedx/fetchlens-cloudflare'

export const onRequest = createPagesMiddleware({
  siteId: 'cohort.bubblnet.com',
  apiEndpoint: 'https://fetchlens.ai',
  siteTag: 'fl_pub_6f78bbfb4264f7b4b76c3b86272e1b49',
  observeOnly: true,
  blockVulnScans: false,
})