"""Asynchronous SDK example with AsyncOpenAI."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from examples.common import ROOT_DIR, load_settings, print_header


async def fetch_reply(client: AsyncOpenAI, model: str, prompt: str) -> str:
    """Send one async request and return the generated text."""
    response = await client.responses.create(model=model, input=prompt)
    return response.output_text


async def main_async() -> None:
    """Run multiple requests concurrently and print each result."""
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    settings = load_settings()
    client = AsyncOpenAI()
    prompts = [
        "RAG の前処理で重要な観点を 1 つ説明してください。",
        "評価指標の設計で重要な観点を 1 つ説明してください。",
        "運用監視で重要な観点を 1 つ説明してください。",
    ]
    print_header("async usage")
    replies = await asyncio.gather(*(fetch_reply(client, settings["model"], prompt) for prompt in prompts))
    for idx, reply in enumerate(replies, start=1):
        print(f"[{idx}] {reply}")


if __name__ == "__main__":
    asyncio.run(main_async())
