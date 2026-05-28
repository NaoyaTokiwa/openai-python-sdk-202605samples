# OpenAI Python SDK Samples

一創の記事「OpenAI Python SDK とは？AI モデル活用を容易にする開発者向けツールの概要」で扱われている主なトピックを、**現在の OpenAI Python SDK** でそのまま試せるように整理したサンプル集です。

記事内には旧 API スタイルの例も含まれていますが、OpenAI は新規実装では Responses API と現行 SDK の利用を案内しているため、このリポジトリでは **現行の書き方に寄せて** 再構成しています。

## 収録内容

- `examples/01_health_check.py`: 最小疎通確認
- `examples/02_text_generation.py`: テキスト生成
- `examples/03_chat_style_conversation.py`: 会話履歴付き応答
- `examples/04_streaming.py`: ストリーミング出力
- `examples/05_embeddings.py`: 埋め込みとコサイン類似度
- `examples/06_image_generation.py`: 画像生成
- `examples/07_vision_analysis.py`: 画像入力の理解
- `examples/08_audio_transcription.py`: 音声文字起こし
- `examples/09_structured_output.py`: JSON 形式の構造化出力
- `examples/10_error_handling.py`: 例外処理
- `examples/11_async_usage.py`: 非同期実行

## セットアップ
* python3.9.6の仮想環境下にて動作確認済みです

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

`.env` に `OPENAI_API_KEY` を設定してください。

## 実行方法

```bash
python -m examples.01_health_check
python -m examples.02_text_generation
python -m examples.03_chat_style_conversation
python -m examples.04_streaming
python -m examples.05_embeddings
python -m examples.06_image_generation
python -m examples.07_vision_analysis
python -m examples.09_structured_output
python -m examples.10_error_handling
python -m examples.11_async_usage
```

音声認識サンプルのみ `data/sample.wav` が必要です。https://ondoku3.com/ja/ で音声合成したファイルを使用することもできます。

###　参考
* https://developers.openai.com/api/docs/libraries?language=python
* https://github.com/openai/openai-python
