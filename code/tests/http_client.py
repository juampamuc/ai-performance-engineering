from __future__ import annotations

import asyncio
from typing import Any

import httpx


def asgi_request(app: Any, method: str, url: str, **kwargs: Any) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=kwargs.pop("timeout", 20.0),
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(_run())


def asgi_stream_text(app: Any, method: str, url: str, **kwargs: Any) -> tuple[int, str]:
    async def _run() -> tuple[int, str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=kwargs.pop("timeout", 20.0),
        ) as client:
            async with client.stream(method, url, **kwargs) as response:
                chunks: list[str] = []
                async for text in response.aiter_text():
                    chunks.append(text)
                return response.status_code, "".join(chunks)

    return asyncio.run(_run())
