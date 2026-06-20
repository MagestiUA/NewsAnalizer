"""Збірка запиту, виклик моделі, парсинг імені файлу й запис звіту."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from . import config
from .llm import OllamaClient
from .metadata import VideoMeta

# Заборонені у Windows-іменах символи.
_BAD_CHARS = re.compile(r'[<>:"/\\|?*]')


def _today() -> str:
    return dt.date.today().isoformat()


def _daypart() -> str:
    h = dt.datetime.now().hour
    if h < 12:
        return "ранок"
    if h < 18:
        return "день"
    return "вечір"


def _meta_header(meta: VideoMeta) -> str:
    return (
        "МЕТАДАНІ ВІДЕО:\n"
        f"- Канал/доповідач: {meta.channel or 'невідомо'}\n"
        f"- Дата завантаження відео: {meta.upload_date or 'невідомо'}\n"
        f"- Назва відео: {meta.title}\n"
        f"- Сьогоднішня дата (аналіз): {_today()}\n"
        f"- Поточний час доби (аналіз): {_daypart()}\n"
    )


def _build_user(meta: VideoMeta, transcript: str) -> str:
    return (
        _meta_header(meta)
        + "\nТРАНСКРИПТ (рядки «=== Назва ===» — авторські розділи відео, "
        "використовуй їх як підказку про структуру тем):\n"
        + transcript
    )


def _build_sort_user(meta: VideoMeta, items: str) -> str:
    return (
        _meta_header(meta)
        + "\nСПИСОК ТЕЗ — розсортуй за темами, НЕ змінюючи самі тези:\n"
        + items
    )


def _classify(client: OllamaClient, meta: VideoMeta, transcript: str) -> str:
    """ROUNDUP (швидке зведення) чи ANALYSIS (розбір аналітика). Дешевий виклик."""
    head = transcript[:4000]
    user = f"Назва відео: {meta.title}\n\nПочаток транскрипту:\n{head}"
    resp = client.chat(config.load(config.PROMPT_CLASSIFY), user).upper()
    return "ANALYSIS" if "ANALYSIS" in resp else "ROUNDUP"


def _split_filename(raw: str, meta: VideoMeta) -> tuple[str, str]:
    """Відділяє рядок FILENAME: ... від тіла звіту.

    Повертає (заголовок_для_H1, тіло_звіту). Якщо модель не дала FILENAME —
    будуємо запасний заголовок із метаданих.
    """
    lines = raw.splitlines()
    title = ""
    body_start = 0
    for i, line in enumerate(lines):
        m = re.match(r"\s*FILENAME:\s*(.+)", line, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            body_start = i + 1
            break
        if line.strip():           # перший непорожній рядок — не FILENAME
            break
    if not title:
        date = meta.upload_date or _today()
        title = f'Ефір "{meta.channel or "невідомо"}", Загальний брифінг, {date}, {_daypart()}'
    body = "\n".join(lines[body_start:]).strip()
    return title, body


def _safe_filename(title: str) -> str:
    """«Ефір "Канал", 2026-06-15, вечір» → «Ефір Канал - 2026-06-15 - вечір»."""
    name = title.replace('"', "").replace("«", "").replace("»", "")
    name = name.replace(",", " -")
    name = _BAD_CHARS.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" .-")
    return name[:150] or "Ефір без назви"


def _normalize_body(body: str) -> str:
    """Маркер напрямку — завжди ">". Прибрати LaTeX (\\rightarrow) і стрілку →."""
    body = re.sub(r"\$?\s*\\(?:rightarrow|to|Rightarrow)\s*\$?", " > ", body)
    body = body.replace("→", ">")
    body = re.sub(r" {2,}", " ", body)
    return body


def _resolve_path(title: str, video_id: str) -> Path:
    """Шлях до .md без перезапису ЧУЖОГО відео.

    Те саме відео (id вже у файлі) → ідемпотентний перезапис.
    Інше відео з тим самим іменем → дописуємо video_id у дужках.
    """
    base = _safe_filename(title)
    path = config.OUTPUT_DIR / f"{base}.md"
    if path.exists() and video_id not in path.read_text(encoding="utf-8"):
        path = config.OUTPUT_DIR / f"{base} ({video_id}).md"
    return path


def analyze(meta: VideoMeta, transcript: str,
            client: OllamaClient | None = None) -> Path:
    """Транскрипт → звіт. Повертає шлях до збереженого .md.

    client — якщо переданий (батч-режим), використовується спільний клієнт і
    Ollama НЕ глушиться тут. Якщо None — піднімаємо й глушимо самі (одне відео).
    """
    print(f"  [llm] аналіз {len(transcript)} символів (~{len(transcript)//3} ток)...")
    own_client = client is None
    if own_client:
        client = OllamaClient()
    try:
        ctype = _classify(client, meta, transcript)
        print(f"  [тип] {ctype}")
        if ctype == "ANALYSIS":
            # 2 проходи: дістати плоский список → розсортувати за темами
            items = client.chat(config.load(config.PROMPT_ANALYSIS_EXTRACT),
                                _build_user(meta, transcript))
            raw = client.chat(config.load(config.PROMPT_ANALYSIS_SORT),
                              _build_sort_user(meta, items))
        else:
            raw = client.chat(config.load(config.PROMPT_ROUNDUP),
                              _build_user(meta, transcript))
    finally:
        if own_client:
            client.shutdown()

    title, body = _split_filename(raw, meta)
    body = _normalize_body(body)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    out_path = _resolve_path(title, meta.video_id)
    content = (
        f"# {title}\n\n"
        f"> Джерело: {meta.title}\n"
        f"> Канал: {meta.channel} · Завантажено: {meta.upload_date or '—'} · "
        f"Тривалість: {meta.duration_min} хв · Аналіз: {_today()}\n"
        f"> https://www.youtube.com/watch?v={meta.video_id}\n\n"
        f"{body}\n"
    )
    out_path.write_text(content, encoding="utf-8")
    return out_path
