"""Audio transcription example using a local sample file."""

from pathlib import Path

from examples.common import create_client, ensure_artifact_dir, load_settings, print_header


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "sample.mp3"  # ここには、短い音声ファイル（例: sample.mp3）を配置してください。ファイル形式は WAV や MP3 など、OpenAI API がサポートする形式であれば問題ありません。https://ondoku3.com/ja/ で音声合成したファイルを使用することもできます。


def main() -> None:
    """Transcribe a local audio file and save the transcript as text."""
    settings = load_settings()
    client = create_client()
    out_dir = ensure_artifact_dir("08_audio_transcription")
    output_path = out_dir / "transcript.txt"
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing audio file: {DATA_FILE}. Add a short MP3 file before running this example.")
    print_header("audio transcription")
    with DATA_FILE.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=settings["transcription_model"],
            file=audio_file,
        )
    output_path.write_text(transcript.text, encoding="utf-8")
    print(transcript.text)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
