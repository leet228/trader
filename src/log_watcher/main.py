from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from typing import Iterable

import docker
import httpx

# Simple log watcher: follows docker logs for configured services and sends Telegram
# alerts on lines containing any of the patterns (case-insensitive).


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_ADMIN_ID")

# Comma-separated service/container names to watch; defaults to all main services.
SERVICES = os.getenv(
    "LOG_WATCH_SERVICES",
    "market_service,pattern_service,trader_service,trainer_service,telegram_bot_service,news_service,ai_nlp_service",
).split(",")

# Patterns to trigger alerts (case-insensitive).
PATTERNS = [p.strip() for p in os.getenv("LOG_WATCH_PATTERNS", "ERROR,Exception,Traceback,failed").split(",") if p.strip()]
RATE_LIMIT_SECONDS = int(os.getenv("LOG_WATCH_RATE_LIMIT_SEC", "60"))  # per service


async def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": text}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json=payload)
    except Exception:
        # swallow errors to avoid log loop
        return


def _matches(line: str, patterns: Iterable[str]) -> bool:
    lower = line.lower()
    return any(p.lower() in lower for p in patterns)


async def watch_container(container_name: str) -> None:
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
    except Exception:
        return
    last_sent = 0.0
    # use current time to avoid replaying old logs
    since = int(time.time())
    try:
        for log in container.logs(stream=True, follow=True, since=since):
            try:
                line = log.decode(errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            if not _matches(line, PATTERNS):
                continue
            now = time.time()
            if now - last_sent < RATE_LIMIT_SECONDS:
                continue
            last_sent = now
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            msg = f"[log alert] {container_name} {ts} UTC\n{line[:3000]}"
            await send_telegram(msg)
    except Exception:
        return


async def main() -> None:
    tasks = [asyncio.create_task(watch_container(name.strip())) for name in SERVICES if name.strip()]
    if not tasks:
        return
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
