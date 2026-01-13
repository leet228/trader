from __future__ import annotations

import httpx

from .config import get_settings
from .logger import logger


async def send_telegram_message(text: str, chat_id: int | None = None) -> None:
    settings = get_settings()
    chat = chat_id or settings.telegram_admin_id
    if not settings.telegram_bot_token or not chat:
        return
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {"chat_id": chat, "text": text, "parse_mode": "Markdown"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("telegram send failed", error=str(exc))

