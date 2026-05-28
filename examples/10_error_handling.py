"""Error-handling patterns for OpenAI SDK requests."""

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from examples.common import load_settings, print_header


def main() -> None:
    """Execute a request and demonstrate explicit exception handling branches."""
    settings = load_settings()
    print_header("error handling")

    # True にすると、存在しない接続先を使って APIConnectionError を試せる
    force_connection_error = False

    # True にすると、存在しないモデル名を使って APIStatusError を試せる
    force_status_error = True

    if force_connection_error:
        # 存在しないローカルエンドポイントに向けて接続エラーを発生させる
        client = OpenAI(
            api_key="sk-dummy-for-connection-test",
            base_url="http://127.0.0.1:9999/v1",
        )
        model_name = settings["model"]
    else:
        # 通常時、または status error テスト時の通常クライアント
        client = OpenAI()
        model_name = "gpt-does-not-exist-for-demo" if force_status_error else settings["model"]

    try:
        response = client.responses.create(
            model=model_name,
            input="例外処理のサンプルとして、OpenAI SDK の運用上の注意を 1 文で返してください。",
        )
        print(response.output_text)

    except RateLimitError as exc:
        print(f"rate limit error: {exc}")

    except APIConnectionError as exc:
        print(f"connection error: {exc}")

    except APIStatusError as exc:
        print(f"status error: {exc.status_code} / {exc.response}")
        if getattr(exc, "body", None):
            print(f"error body: {exc.body}")


if __name__ == "__main__":
    main()
