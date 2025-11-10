# servers/fastapi/services/web_scraper_connector.py

import re
import aiohttp
import os
from fastapi import HTTPException

SCRAPER_SERVICE_URL = os.getenv("SCRAPER_SERVICE_URL", "https://scraper-service-1-w52m.onrender.com")


def is_url(text: str) -> bool:
    """Detect if input text is a URL."""
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return re.match(url_pattern, text.strip()) is not None


async def fetch_scraped_content(url: str) -> str:
    """Send URL to external scraper service and return extracted text."""
    if not SCRAPER_SERVICE_URL:
        raise HTTPException(status_code=500, detail="Scraper service URL not configured")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SCRAPER_SERVICE_URL}/scrape",
                json={"url": url},
                timeout=60,
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Scraper service error: {await response.text()}"
                    )
                data = await response.json()
                return data.get("content", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content from scraper: {str(e)}")


async def get_content_or_scrape(input_text: str) -> str:
    """
    Determines whether input is a URL or plain text.
    If URL → fetch scraped content.
    Otherwise → return text as-is.
    """
    if is_url(input_text):
        print(f"🔗 Detected URL input: {input_text}")
        scraped_text = await fetch_scraped_content(input_text)
        return scraped_text or f"Could not extract text from {input_text}"
    else:
        print(f"🧠 Using direct text input.")
        return input_text

