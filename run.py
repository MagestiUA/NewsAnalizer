"""NewsAnalizer — аналіз новинних відео з YouTube (батч-режим).

Запуск:
    python run.py                          # спитає посилання й джерело тексту
    python run.py "url1, url2, url3"        # одразу з аргументу (спитає джерело)
    python run.py --asr  "url1, url2"       # примусове розпізнавання аудіо
    python run.py --subs "url1, url2"       # тільки субтитри (дефолт)

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


def process_one(url: str, client: OllamaClient, force_asr: bool) -> str:
    """Обробити одне відео. Повертає шлях до звіту (рядок)."""
    meta = metadata.fetch_metadata(url)
    print(f"  {meta.channel} · {meta.upload_date} · {meta.duration_min} хв · "
          f"розділів: {len(meta.chapters)}")
    print(f"  {meta.title}")
    text = transcript.get_transcript(meta.video_id, meta.chapters,
                                     force_asr=force_asr)
    print(f"  транскрипт: {len(text)} символів")
    out_path = analyze.analyze(meta, text, client=client)
    return str(out_path)


def main(raw: str, force_asr: bool) -> None:
    urls = parse_urls(raw)
    if not urls:
        print("Не знайдено жодного посилання.")
        return

    src_label = "розпізнавання аудіо (WhisperX)" if force_asr else "субтитри"
    print(f"Батч: {len(urls)} відео. Джерело тексту: {src_label}.\n")
    client = OllamaClient()                # один старт Ollama на весь батч
    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    try:
        for i, url in enumerate(urls, 1):
            print(f"━━━ [{i}/{len(urls)}] {url}")
            try:
                ok.append(process_one(url, client, force_asr))
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


def _read_args() -> tuple[str, bool]:
    """Розбирає argv: прапорці --asr/--subs + посилання. Доповнює інтерактивно."""
    force_asr: bool | None = None
    rest: list[str] = []
    for a in sys.argv[1:]:
        if a in ("--asr", "-a"):
            force_asr = True
        elif a in ("--subs", "-s"):
            force_asr = False
        else:
            rest.append(a)

    interactive = sys.stdin.isatty()
    raw = " ".join(rest) if rest else (input("Посилання (через кому): ")
                                       if interactive else "")
    if force_asr is None:
        # Без терміналу (піп/скрипт) не питаємо — дефолт субтитри.
        if interactive:
            ans = input("Джерело тексту — субтитри (0) чи розпізнавання аудіо (1)? [0]: ")
            force_asr = ans.strip() == "1"
        else:
            force_asr = False
    return raw, force_asr


if __name__ == "__main__":
    raw, force_asr = _read_args()
    main(raw, force_asr)
