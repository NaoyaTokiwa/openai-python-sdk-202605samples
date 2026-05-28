"""Image generation example using the current image API.

This script demonstrates how to generate an image using OpenAI's image generation
API and save it locally as a PNG file.
"""

from pathlib import Path

from examples.common import (
    create_client,
    ensure_artifact_dir,
    load_settings,
    print_header,
    save_base64_file,
)


def main() -> None:
    """Generate an image and save it under the artifacts directory."""
    
    # 1. 環境変数からモデル設定を読み込む（.env の OPENAI_MODEL など）
    settings = load_settings()
    
    # 2. OpenAI API キーを読み取ってクライアントを生成
    client = create_client()
    
    # 3. 出力先ディレクトリ（artifacts/06_image_generation）を作成・取得
    out_dir = ensure_artifact_dir("06_image_generation")
    
    # 4. 保存する画像ファイルのパスを決定（sample.png）
    output_path = Path(out_dir) / "sample.png"
    
    # 5. セクション見出しを表示
    print_header("image generation")
    
    # 6. OpenAI 画像生成 API を呼び出して画像を生成
    #    - model: 画像生成モデル（.env の OPENAI_IMAGE_MODEL、デフォルト:gpt-image-1）
    #    - prompt: 生成したい画像の内容を英語で記述
    #    - size: 出力画像サイズ（1024x1024）
    result = client.images.generate(
        model=settings["image_model"],
        prompt="A clean flat illustration of a machine learning engineer working with graphs, vectors, and Python on a desk, modern Japanese tech style",  # デスク上でグラフ・ベクトル・Python を使って作業している機械学習エンジニアを、現代的な日本のテックスタイルで描いた、クリーンなフラットイラストレーション
        size="1024x1024",
    )
    
    # 7. 生成された画像を Base64 形式で取得（response.data[0].b64_json）
    image_base64 = result.data[0].b64_json
    
    # 8. Base64 をデコードして PNG ファイルとして保存
    save_base64_file(image_base64, output_path)
    
    # 9. 保存先パスを出力
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
