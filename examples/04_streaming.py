"""Streaming example for incremental text output."""
"""
Python SDK のストリーミング応答の利点は、処理結果をリアルタイムで逐次受け取れるため、応答待ち時間を短縮し、ユーザー体験を向上させることができる点です。これにより、大量データの処理や対話型アプリケーションでの応答速度が改善されます。
"""

from examples.common import create_client, load_settings, print_header


def main() -> None:
    """Receive model output as a stream and print text chunks incrementally."""
    settings = load_settings()
    client = create_client()
    print_header("streaming")
    with client.responses.stream(
        model=settings["model"],
        input="Python SDK のストリーミング応答の利点を日本語で簡潔に説明してください。",
        temperature=0.3,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
        final_response = stream.get_final_response()
    print("\n\n[final output_text]\n")
    print(final_response.output_text)


if __name__ == "__main__":
    main()
