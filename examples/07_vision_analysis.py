"""Vision input example using an image plus a text prompt."""

from examples.common import create_client, load_settings, print_header


SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/500px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def main() -> None:
    """Analyze a remote image and ask the model for a concise description."""
    settings = load_settings()
    client = create_client()
    print_header("vision analysis")
    response = client.responses.create(
        model=settings["model"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "この画像の内容を日本語で 2 文で説明してください。"},
                    {"type": "input_image", "image_url": SAMPLE_IMAGE_URL},
                ],
            }
        ],
    )
    print(response.output_text)


if __name__ == "__main__":
    main()  

# output例：この画像は広がる緑の草原の中に木製の通路が真っ直ぐ伸びている風景を写しています。青空と所々に浮かぶ雲が穏やかな自然の雰囲気を醸し出しています。
