from playwright.async_api import async_playwright
import asyncio
import random

async def scrape_page(url):
    """Scrapes webpage title, text, and links asynchronously."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Run in non-headless mode to bypass bot detection
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            print(f"Scraping: {url}")
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")

            # Introduce random delay to mimic human interaction
            await asyncio.sleep(random.uniform(2, 5))

            title = await page.title()
            content = await page.inner_text("body")
            content = content[:5000]  # Truncate content

            # Extract all hyperlinks
            links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a'))
                .map(a => a.href)
                .filter(link => link.startsWith('http'))
            """)

            # Extract metadata
            metadata = await page.evaluate("""
                () => { 
                    return { 
                        description: document.querySelector('meta[name="description"]')?.content || '', 
                        keywords: document.querySelector('meta[name="keywords"]')?.content || '' 
                    }; 
                }
            """)

            return {
                "title": title,
                "content": content,
                "links": links[:20],  # Limit to 20 links
                "metadata": metadata
            }

        except Exception as e:
            return {"error": f"Failed to scrape: {str(e)}"}

        finally:
            await browser.close()
# Run the async function properly
if __name__ == "__main__":
    result = asyncio.run(scrape_page("https://fitgirl-repacks.site/"))
    print(result)
