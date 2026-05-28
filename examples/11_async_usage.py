"""Asynchronous SDK example with AsyncOpenAI.

This script demonstrates concurrent execution of multiple OpenAI requests
using the AsyncOpenAI client and asyncio.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from examples.common import ROOT_DIR, load_settings, print_header


async def fetch_reply(client: AsyncOpenAI, model: str, prompt: str) -> str:
    """Send one async request and return the generated text.

    Args:
        client: AsyncOpenAI client instance.
        model: Model name (e.g., 'gpt-4.1-mini').
        prompt: User prompt text.

    Returns:
        Generated text from the model.
    """
    # 1. 非同期で OpenAI API にリクエストを送信
    #    await を使うことで、他の非同期タスクと並行して処理できる
    response = await client.responses.create(model=model, input=prompt)

    # 2. 生成されたテキストを返す
    return response.output_text


async def main_async() -> None:
    """Run multiple requests concurrently and print each result.

    This function:
    1. Loads environment variables from .env
    2. Creates AsyncOpenAI client
    3. Sends 3 prompts concurrently using asyncio.gather()
    4. Prints each reply in order
    """
    # 1. プロジェクトルートの .env から環境変数を読み込む
    load_dotenv(ROOT_DIR / ".env")

    # 2. OPENAI_API_KEY が設定されていない場合はエラーを発生
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # 3. モデル設定（.env の OPENAI_MODEL など）を読み込む
    settings = load_settings()

    # 4. 非同期クライアントを作成
    client = AsyncOpenAI()

    # 5. 並列で実行するプロンプト（質問）リスト
    prompts = [
        "RAG の前処理で重要な観点を 1 つ説明してください。",
        "評価指標の設計で重要な観点を 1 つ説明してください。",
        "運用監視で重要な観点を 1 つ説明してください。",
    ]

    # 6. セクション見出しを表示
    print_header("async usage")

    # 7. 複数プロンプトを並列で同時実行
    #    asyncio.gather() で 3 つの fetch_reply を同時に起動
    #    各リクエストは非同期なので、1 つ目が終わるのを待たずに次も実行
    replies = await asyncio.gather(
        *(fetch_reply(client, settings["model"], prompt) for prompt in prompts)
    )

    # 8. 各リクエストの返答を順番に出力
    for idx, reply in enumerate(replies, start=1):
        print(f"[{idx}] {reply}")


if __name__ == "__main__":
    # 9. main_async 非同期関数を実行エントリーポイントとして実行
    asyncio.run(main_async())
