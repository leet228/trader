from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import aiohttp


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

