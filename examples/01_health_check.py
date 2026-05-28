"""Minimal connectivity check using the current OpenAI Python SDK."""

from examples.common import create_client, load_settings, print_header


def main() -> None:
    """Run a minimal Responses API request and print the result."""
    settings = load_settings()
    client = create_client()
    print_header("health check")
    response = client.responses.create(
        model=settings["model"],
        input="OpenAI Python SDK の疎通確認として、短い日本語の挨拶を 1 文だけ返してください。",
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
