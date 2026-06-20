"""Дістати текстову версію відео — з розміткою розділів.

Каскад джерел:
  1. youtube-transcript-api — авто/ручні субтитри (укр. у пріоритеті);
  2. yt-dlp --write-auto-sub — резерв, якщо (1) недоступний (геоблок/вимкнено).

Фрагменти субтитрів склеюються в суцільний текст (модель НЕ має зшивати
порізані рядки в речення), а на стиках розділів (chapters із метаданих)
вставляються заголовки «=== Назва розділу ===». Це відтворює «текстову версію»
з YouTube, але без поденних таймкодів — щоб не роздувати токени.

Розпізнавання голосу (Whisper) НЕ використовуємо: ~99% новинних відео мають
текстову версію (ручні або авто-субтитри YouTube).
"""

from __future__ import annotations

import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from . import config

# Мови в порядку пріоритету.
PREFERRED_LANGS = ["uk", "uk-orig", "en"]

# Звукові ремарки авто-субтитрів — викидаємо.
_SOUND_CUE = re.compile(r"\[[^\]]{0,30}\]")

# Тип: фрагмент субтитрів = (час_початку_сек, текст).
Snippet = tuple[float, str]


def _clean(text: str) -> str:
    text = _SOUND_CUE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Джерело 1: youtube-transcript-api ─────────────────────────────────────────
def _snippets_via_api(video_id: str) -> list[Snippet] | None:
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=PREFERRED_LANGS)
    except Exception:
        try:                                  # будь-яка доступна мова
            fetched = next(iter(api.list(video_id))).fetch()
        except Exception:
            return None
    snippets = getattr(fetched, "snippets", fetched)
    out = [(float(getattr(s, "start", 0.0)), s.text)
           for s in snippets if getattr(s, "text", "").strip()]
    return out or None


# ─── Джерело 2: yt-dlp VTT (резерв) ────────────────────────────────────────────
def _snippets_via_ytdlp(video_id: str) -> list[Snippet] | None:
    import tempfile

    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": PREFERRED_LANGS + ["uk.*"],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmp) / "%(id)s"),
            "quiet": True,
            "no_warnings": True,
        }
        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        vtts = list(Path(tmp).glob("*.vtt"))
        if not vtts:
            return None
        vtts.sort(key=lambda p: (0 if "uk" in p.name.lower() else 1, p.name))
        return _parse_vtt(vtts[0].read_text(encoding="utf-8"))


_VTT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->")


def _parse_vtt(vtt: str) -> list[Snippet]:
    out: list[Snippet] = []
    cur_start: float | None = None
    last_text = ""
    for line in vtt.splitlines():
        line = line.strip()
        m = _VTT_TIME.match(line)
        if m:
            h, mi, s, ms = map(int, m.groups())
            cur_start = h * 3600 + mi * 60 + s + ms / 1000
            continue
        if (not line or line.startswith("WEBVTT") or line.isdigit()
                or line.startswith(("Kind:", "Language:"))):
            continue
        line = re.sub(r"<[^>]+>", "", line)        # inline-теги
        if not line or line == last_text:           # rolling-дублі
            continue
        out.append((cur_start or 0.0, line))
        last_text = line
    return out


# ─── Злиття субтитрів із розділами ─────────────────────────────────────────────
def _build_text(snippets: list[Snippet], chapters: list[dict] | None) -> str:
    """Суцільний текст; на стиках розділів — заголовок «=== Назва ===»."""
    if not chapters:
        return _clean(" ".join(t for _, t in snippets))

    chs = sorted(chapters, key=lambda c: c["start_time"])
    blocks: list[tuple[str | None, list[str]]] = [(None, [])]
    ci = 0
    for start, text in snippets:
        while ci < len(chs) and start >= chs[ci]["start_time"]:
            blocks.append((chs[ci]["title"], []))
            ci += 1
        blocks[-1][1].append(text)

    parts: list[str] = []
    for title, texts in blocks:
        body = _clean(" ".join(texts))
        if not body:
            continue
        parts.append(f"=== {title} ===\n{body}" if title else body)
    return "\n\n".join(parts)


# ─── Публічний API ─────────────────────────────────────────────────────────────
def get_transcript(video_id: str, chapters: list[dict] | None = None,
                   use_cache: bool = True, force_asr: bool = False) -> str:
    """Текстова версія. Кеш роздільний: cache/<id>.txt (субтитри) і
    cache/<id>.asr.txt (розпізнавання аудіо).

    force_asr=True — ігнорувати субтитри й одразу розпізнавати аудіо (WhisperX
    + діаризація). Корисно для діалогів: мітки [Спікер N] інформативніші за
    «голі» субтитри без розмітки спікерів.
    """
    config.CACHE_DIR.mkdir(exist_ok=True)
    cache_file = config.CACHE_DIR / (f"{video_id}.asr.txt" if force_asr
                                     else f"{video_id}.txt")
    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    if force_asr:
        print("      [i] Примусове розпізнавання аудіо (WhisperX)...")
        from . import asr
        text = asr.transcribe(video_id)
    else:
        snippets = _snippets_via_api(video_id) or _snippets_via_ytdlp(video_id)
        if snippets:
            text = _build_text(snippets, chapters)
        else:
            # Субтитрів немає взагалі (напр. запис прямого ефіру) → ASR-резерв.
            print("      [i] Субтитрів немає — резерв WhisperX (розпізнавання мовлення)...")
            from . import asr
            text = asr.transcribe(video_id)

    cache_file.write_text(text, encoding="utf-8")
    return text
