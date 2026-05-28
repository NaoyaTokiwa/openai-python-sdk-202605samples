"""Structured JSON output example.

This script demonstrates how to request structured JSON output from the OpenAI Responses API
and parse it into a Python dictionary.
"""

import json

from examples.common import create_client, load_settings, print_header


def main() -> None:
    """Request JSON output and pretty-print the parsed result."""
    
    # 1. 環境変数からモデル設定を読み込む（.env の OPENAI_MODEL など）
    settings = load_settings()
    
    # 2. OpenAI API キーを読み取ってクライアントを生成
    client = create_client()
    
    # 3. セクション見出しを表示
    print_header("structured output")
    
    # 4. Responses API を呼び出して JSON 形式の出力を要求
    #    - model: 使用するモデル（.env の OPENAI_MODEL、デフォルト：gpt-4.1-mini など）
    #    - input: AI への指示（Python SDK の学習計画を JSON で 3 つ返すように要求）
    #    - text.format.type: "json_object" を指定して JSON 形式を強制
    response = client.responses.create(
        model=settings["model"],
        input=(
            "Python SDK の学習計画を JSON で返してください。"
            "fields: topic, priority, estimated_hours, hands_on_task. 3 items."
        ),
        text={"format": {"type": "json_object"}},
    )
    
    # 5. API の返答（テキスト）を JSON としてパース
    #    response.output_text は文字列なので、json.loads() で Python dict に変換
    payload = json.loads(response.output_text)
    
    # 6. パースした JSON を見やすく整形して出力
    #    - ensure_ascii=False: 日本語などをそのまま表示
    #    - indent=2: 2 文字インデントで整形
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
