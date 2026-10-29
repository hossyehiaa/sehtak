// Screenshot helper: node shots.js <baseUrl> <outDir> [label]
// Uses the locally installed Google Chrome. All /api/v1/triage calls are mocked,
// so no trial messages, real data, or real accounts are touched.
const { chromium } = require('playwright-core');
const path = require('path');
const fs = require('fs');

const base = process.argv[2] || 'http://127.0.0.1:8765';
const out = process.argv[3] || './shots';
const label = process.argv[4] || 'run';
fs.mkdirSync(out, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors = [];
  for (const lang of ['ar', 'en']) {
    for (const vp of [{ name: 'mobile', width: 390, height: 844 }, { name: 'desktop', width: 1440, height: 900 }]) {
      const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
      await ctx.addInitScript((l) => { localStorage.setItem('sehtak_lang', l); }, lang);
      const page = await ctx.newPage();
      page.on('pageerror', (e) => errors.push(`[${lang}/${vp.name}] ${e.message}`));
      await page.route('**/api/v1/triage/**', (r) => r.fulfill({ status: 503, body: '{"detail":"mock"}', contentType: 'application/json' }));
      await page.goto(base + '/', { waitUntil: 'networkidle' });
      await page.waitForTimeout(400);
      await page.screenshot({ path: path.join(out, `${label}-${lang}-${vp.name}.png`), fullPage: false });
      await ctx.close();
    }
  }
  await browser.close();
  console.log(errors.length ? 'PAGE ERRORS:\n' + errors.join('\n') : 'no page errors');
})();
