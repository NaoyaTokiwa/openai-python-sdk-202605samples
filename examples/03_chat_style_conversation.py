"""Multi-turn conversation example using the Responses API."""

from examples.common import create_client, load_settings, print_header


def main() -> None:
    """Send a system instruction and a short two-turn conversation context."""
    settings = load_settings()
    client = create_client()
    print_header("conversation")
    response = client.responses.create(
        model=settings["model"],
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "あなたは簡潔で実務的な AI アシスタントです。"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "OpenAI Python SDK で何ができますか？"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "テキスト生成、画像生成、埋め込み、音声処理などに使えます。"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "その中で RAG 開発と相性が良い機能を 2 つ挙げてください。"}]},
        ],
        temperature=0.2,
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
