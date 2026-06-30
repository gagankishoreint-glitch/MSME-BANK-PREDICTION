import asyncio, json
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        errors = []
        page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
        page.on('pageerror', lambda e: errors.append(e.message))
        await page.goto('http://127.0.0.1:8000/dashboard/index.html')
        await page.wait_for_timeout(2000)
        await page.evaluate('loadDossier("BR-0002")')
        await page.wait_for_timeout(1000)
        print('ERRORS:', errors)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
