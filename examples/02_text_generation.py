"""Text generation example with tunable sampling parameters."""

from examples.common import create_client, load_settings, print_header


def main() -> None:
    """Generate a short explanation in Japanese with configurable parameters."""
    settings = load_settings()
    client = create_client()
    print_header("text generation")
    response = client.responses.create(
        model=settings["model"],
        input=(
            "機械学習エンジニア向けに、OpenAI Python SDK を使う利点を"
            "日本語で 3 点、箇条書きで説明してください。"
        ),
        max_output_tokens=220,  # 出力の最大トークン数を指定。1トークンは約4文字に相当
        temperature=0.4,  # 出力のランダム性を制御。0に近いほど常に決まった回答を返し、1.0に近いほどランダムで創造的な回答を返す
        )
    print(response.output_text)



if __name__ == "__main__":
    main()
