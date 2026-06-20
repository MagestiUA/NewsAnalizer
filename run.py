"""NewsAnalizer — аналіз новинних відео з YouTube (батч-режим).

Запуск:
    python run.py                          # спитає посилання
    python run.py "url1, url2, url3"        # одразу з аргументу
    python run.py url1 url2 url3            # або кількома аргументами

Посилання розділяються комами / пробілами / новими рядками.
Помилка на одному відео не зупиняє решту.
"""

from __future__ import annotations

import re
import sys

from src import analyze, metadata, transcript
from src.llm import OllamaClient


def parse_urls(raw: str) -> list[str]:
    """Вхідний рядок → список URL (розділювачі: кома, пробіл, новий рядок)."""
    parts = re.split(r"[,\s]+", raw.strip())
    seen, urls = set(), []
    for p in parts:
        if p and p not in seen:        # дедуплікація зі збереженням порядку
            seen.add(p)
            urls.append(p)
    return urls


def process_one(url: str, client: OllamaClient) -> str:
    """Обробити одне відео. Повертає шлях до звіту (рядок)."""
    meta = metadata.fetch_metadata(url)
    print(f"  {meta.channel} · {meta.upload_date} · {meta.duration_min} хв · "
          f"розділів: {len(meta.chapters)}")
    print(f"  {meta.title}")
    text = transcript.get_transcript(meta.video_id, meta.chapters)
    print(f"  транскрипт: {len(text)} символів")
    out_path = analyze.analyze(meta, text, client=client)
    return str(out_path)


def main(raw: str) -> None:
    urls = parse_urls(raw)
    if not urls:
        print("Не знайдено жодного посилання.")
        return

    print(f"Батч: {len(urls)} відео.\n")
    client = OllamaClient()                # один старт Ollama на весь батч
    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    try:
        for i, url in enumerate(urls, 1):
            print(f"━━━ [{i}/{len(urls)}] {url}")
            try:
                ok.append(process_one(url, client))
            except Exception as exc:
                print(f"  [!] ПОМИЛКА: {exc}")
                failed.append((url, str(exc)))
            print()
    finally:
        client.shutdown()

    print("═" * 60)
    print(f"Готово: {len(ok)} успішно, {len(failed)} з помилкою.")
    for p in ok:
        print(f"  ✓ {p}")
    for url, err in failed:
        print(f"  ✗ {url} — {err}")


if __name__ == "__main__":
    arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Посилання (через кому): ")
    main(arg)
