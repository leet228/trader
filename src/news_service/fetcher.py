from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import aiohttp

NEWSAPI_URL = "https://newsapi.org/v2/everything"


async def fetch_rss(url: str) -> list[dict]:
    items: list[dict] = []
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            text = await resp.text()
    if url.startswith("https://api.gdeltproject.org"):
        # simplistic parse of JSON summary
        try:
            data = await _fetch_json(url)
            headlines = data.get("articles", []) if isinstance(data, dict) else []
            for art in headlines:
                title = art.get("title", "")
                link = art.get("url", "")
                if not title:
                    continue
                ts = datetime.now(timezone.utc)
                items.append(
                    {
                        "id": hashlib.sha1(f"{title}{link}".encode()).hexdigest(),
                        "headline": title,
                        "url": link,
                        "ts": ts,
                        "raw_pub": art,
                    }
                )
            return items
        except Exception:
            return []
    root = ET.fromstring(text)
    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        pub = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        if not title:
            continue
        ts = datetime.now(timezone.utc)
        items.append(
            {
                "id": hashlib.sha1(f"{title}{link}".encode()).hexdigest(),
                "headline": title,
                "url": link,
                "ts": ts,
                "raw_pub": pub,
            }
        )
    return items


async def _fetch_json(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            return await resp.json()


async def fetch_newsapi(api_key: str, query: str, page_size: int = 50) -> list[dict]:
    """Fetch news from NewsAPI Everything endpoint with a broad market/crypto query."""
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    headers = {"X-Api-Key": api_key}
    async with aiohttp.ClientSession() as session:
        async with session.get(NEWSAPI_URL, params=params, headers=headers, timeout=15) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    articles = data.get("articles", []) if isinstance(data, dict) else []
    items: list[dict] = []
    for art in articles:
        title = art.get("title") or ""
        url = art.get("url") or ""
        if not title or not url:
            continue
        ts = datetime.now(timezone.utc)
        items.append(
            {
                "id": hashlib.sha1(f"{title}{url}".encode()).hexdigest(),
                "headline": title,
                "url": url,
                "ts": ts,
                "raw_pub": art,
            }
        )
    return items

