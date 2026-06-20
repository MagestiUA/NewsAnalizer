"""Метадані відео через yt-dlp: канал, дата, назва, id, тривалість."""

from __future__ import annotations

from dataclasses import dataclass

import yt_dlp


@dataclass
class VideoMeta:
    video_id: str
    title: str
    channel: str
    upload_date: str   # YYYY-MM-DD (порожнє, якщо невідомо)
    duration: int      # секунди
    chapters: list[dict]  # [{"start_time": float, "title": str}, ...]

    @property
    def duration_min(self) -> float:
        return round(self.duration / 60, 1)


def _fmt_date(raw: str | None) -> str:
    """yt-dlp дає upload_date як YYYYMMDD → YYYY-MM-DD."""
    if raw and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw or ""


def fetch_metadata(url: str) -> VideoMeta:
    """Витягує метадані без завантаження відео."""
    opts = {"skip_download": True, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    chapters = [
        {"start_time": float(c.get("start_time", 0)), "title": (c.get("title") or "").strip()}
        for c in (info.get("chapters") or [])
    ]
    return VideoMeta(
        video_id=info.get("id", ""),
        title=info.get("title", "") or "",
        channel=(info.get("channel") or info.get("uploader") or "").strip(),
        upload_date=_fmt_date(info.get("upload_date")),
        duration=int(info.get("duration") or 0),
        chapters=chapters,
    )
