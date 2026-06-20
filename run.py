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


def _fetch(url: str, force_asr: bool):
    """Метадані + транскрипт одного відео. Повертає (meta, text)."""
    meta = metadata.fetch_metadata(url)
    print(f"  {meta.channel} · {meta.upload_date} · {meta.duration_min} хв · "
          f"розділів: {len(meta.chapters)}")
    print(f"  {meta.title}")
    text = transcript.get_transcript(meta.video_id, meta.chapters,
                                     force_asr=force_asr)
    print(f"  транскрипт: {len(text)} символів")
    return meta, text


def main(raw: str, force_asr: bool) -> None:
    urls = parse_urls(raw)
    if not urls:
        print("Не знайдено жодного посилання.")
        return

    src_label = "розпізнавання аудіо (WhisperX)" if force_asr else "субтитри"
    print(f"Батч: {len(urls)} відео. Джерело тексту: {src_label}.")
    ok: list[str] = []
    failed: list[tuple[str, str]] = []

    # ── Фаза 1: транскрипти ──────────────────────────────────────────────────
    # WhisperX крутиться ізольовано; Ollama НЕ чіпаємо, щоб gemma не висіла у
    # VRAM під час розпізнавання (інакше whisper+gemma конкурують за пам'ять).
    print(f"\n[Фаза 1/2] Транскрипти ({len(urls)})...")
    jobs: list[tuple] = []
    for i, url in enumerate(urls, 1):
        print(f"━━━ [{i}/{len(urls)}] {url}")
        try:
            jobs.append(_fetch(url, force_asr))
        except Exception as exc:
            print(f"  [!] ПОМИЛКА транскрипту: {exc}")
            failed.append((url, str(exc)))
        print()

    # ── Фаза 2: аналіз ───────────────────────────────────────────────────────
    # WhisperX уже завершив роботу й звільнив VRAM → gemma бере всю пам'ять сама.
    if jobs:
        print(f"[Фаза 2/2] Аналіз LLM ({len(jobs)})...")
        client = OllamaClient()
        try:
            for i, (meta, text) in enumerate(jobs, 1):
                print(f"━━━ [{i}/{len(jobs)}] {meta.title}")
                try:
                    ok.append(str(analyze.analyze(meta, text, client=client)))
                except Exception as exc:
                    print(f"  [!] ПОМИЛКА аналізу: {exc}")
                    failed.append((meta.video_id, str(exc)))
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

    # Пробуємо інтерактив; EOFError = реально неінтерактивний запуск (піп) → дефолти.
    # (PyCharm-консоль не tty, але input() підтримує — тому не покладаємось на isatty.)
    if rest:
        raw = " ".join(rest)
    else:
        try:
            raw = input("Посилання (через кому): ")
        except EOFError:
            raw = ""
    if force_asr is None:
        try:
            ans = input("Джерело тексту — субтитри (0) чи розпізнавання аудіо (1)? [0]: ")
            force_asr = ans.strip() == "1"
        except EOFError:
            force_asr = False
    return raw, force_asr


if __name__ == "__main__":
    raw, force_asr = _read_args()
    main(raw, force_asr)
