"""Embedding generation and cosine-similarity example."""

from __future__ import annotations

import numpy as np

from examples.common import create_client, load_settings, print_header, summarize_vector


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    """Generate embeddings and compare semantic similarity among sample texts."""
    settings = load_settings()
    client = create_client()
    texts = [
        "機械学習エンジニアとして RAG を実装したい。",
        "ベクトル検索を使って関連文書を取り出したい。",
        "週末は公園で子どもと遊びたい。",
    ]
    print_header("embeddings")
    response = client.embeddings.create(model=settings["embedding_model"], input=texts)
    vectors = [item.embedding for item in response.data]
    print("response.data[0].embedding:",response.data[0].embedding)
    for idx, vector in enumerate(vectors, start=1):
        print(f"text{idx} preview: {summarize_vector(vector)}")
    print(f"similarity(text1, text2): {cosine_similarity(vectors[0], vectors[1]):.4f}")
    print(f"similarity(text1, text3): {cosine_similarity(vectors[0], vectors[2]):.4f}")


if __name__ == "__main__":
    main()


    """
    ==================== embeddings ====================
    text1 preview: 0.0076, 0.0148, -0.0508, -0.0136, 0.0120, -0.0125, -0.0000, 0.0226
    text2 preview: -0.0154, 0.0264, 0.0146, -0.0035, 0.0102, 0.0273, -0.0497, 0.0383
    text3 preview: 0.0463, -0.0017, -0.0670, -0.0350, 0.0082, -0.0335, -0.0050, 0.0605
    similarity(text1, text2): 0.1976  # text1 と text2 はどちらも技術的な内容で、RAG やベクトル検索に関連しているため、ある程度の類似性が見られます。
    similarity(text1, text3): 0.1899  # text1 と text3 は内容が異なり、text3 は週末の過ごし方に関するものであるため、類似性は低くなっていますが、完全に無関係というわけではないため、0.18程度の値になっています。
    """
