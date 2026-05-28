"""Shared utilities for OpenAI Python SDK runnable examples."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, str]:
    """Load environment variables and return commonly used model settings."""
    load_dotenv(ROOT_DIR / ".env")
    return {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "image_model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "transcription_model": os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
    }


def create_client() -> OpenAI:
    """Create an authenticated OpenAI client."""
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and configure your API key.")
    return OpenAI()


def ensure_artifact_dir(name: str) -> Path:
    """Create and return a per-example artifact directory."""
    path = ARTIFACTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_header(title: str) -> None:
    """Print a human-friendly console header."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def save_base64_file(encoded: str, output_path: Path) -> Path:
    """Decode a base64 string and persist it as a binary file."""
    output_path.write_bytes(base64.b64decode(encoded))
    return output_path


def first_text_output(response: Any) -> str:
    """Best-effort extraction of text content from a Responses API result."""
    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                chunks.append(getattr(content, "text", ""))
    return "\n".join(part for part in chunks if part)


def summarize_vector(values: Iterable[float], n: int = 8) -> str:
    """Create a compact preview string for an embedding vector."""
    head = list(values)[:n]
    return ", ".join(f"{v:.4f}" for v in head)
