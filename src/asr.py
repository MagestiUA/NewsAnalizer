"""ASR-резерв: WhisperX → текст. Лише коли субтитрів немає взагалі.

WhisperX large-v3 віддає вже зв'язний текст із пунктуацією (повні речення), тож
LLM-очистка не потрібна. Опційно — діаризація (розрізнення спікерів) через
pyannote: вмикається, якщо є HF_TOKEN. Тоді текст розмічається мітками
[Спікер 1] / [Спікер 2] ... (таймкоди не зберігаємо — вони нам не потрібні).

Важкі залежності (torch, whisperx) імпортуються ЛІНИВО, щоб субтитровий шлях
працював і без ML-стеку.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from . import config


def _download_audio(video_id: str, dst_dir: Path) -> Path:
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dst_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = list(dst_dir.glob(f"{video_id}.*"))
    if not files:
        raise RuntimeError("yt-dlp не зміг завантажити аудіо")
    return files[0]


def _spk_to_int(label) -> int | None:
    if not label:
        return None
    m = re.search(r"(\d+)", str(label))
    return int(m.group(1)) if m else None


def _join_plain(segments) -> str:
    parts = [(s.get("text") or "").strip() for s in segments]
    return " ".join(p for p in parts if p)


def _join_diarized(segments) -> str:
    """Групує підряд репліки одного спікера під міткою [Спікер N]."""
    blocks: list[tuple[int | None, list[str]]] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = _spk_to_int(s.get("speaker"))
        if blocks and blocks[-1][0] == spk:
            blocks[-1][1].append(text)
        else:
            blocks.append((spk, [text]))
    out = []
    for spk, texts in blocks:
        label = f"[Спікер {spk + 1}]" if spk is not None else "[Спікер ?]"
        out.append(f"{label} " + " ".join(texts))
    return "\n\n".join(out)


def transcribe(video_id: str, diarize: bool | None = None) -> str:
    """Розпізнає мовлення відео в текст. Якщо diarize (та є HF_TOKEN) — з мітками
    спікерів. diarize=None → авто: вмикаємо, коли токен присутній."""
    import whisperx

    hf_token = os.environ.get("HF_TOKEN")
    if diarize is None:
        diarize = bool(hf_token)
    if diarize and not hf_token:
        print("      [asr] діаризація вимкнена: немає HF_TOKEN")
        diarize = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("      [asr] завантаження аудіо...")
        audio_file = _download_audio(video_id, tmp_dir)

        print(f"      [asr] WhisperX {config.WHISPER_MODEL} "
              f"({config.WHISPER_DEVICE}/{config.WHISPER_COMPUTE})"
              f"{' + діаризація' if diarize else ''}...")
        model = whisperx.load_model(
            config.WHISPER_MODEL, config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )
        audio = whisperx.load_audio(str(audio_file))
        result = model.transcribe(audio, batch_size=16,
                                  language=config.WHISPER_LANG)
        lang = result.get("language", "?")

        segments = result.get("segments", [])
        if diarize:
            segments = _diarize(whisperx, audio, result, lang, hf_token)
            text = _join_diarized(segments)
        else:
            text = _join_plain(segments)

    n_spk = len({_spk_to_int(s.get("speaker")) for s in segments
                 if s.get("speaker")}) if diarize else 0
    print(f"      [asr] мова={lang}, символів={len(text)}"
          + (f", спікерів={n_spk}" if diarize else ""))
    if not text:
        raise RuntimeError("WhisperX повернув порожній текст")
    return text


def _diarize(whisperx, audio, result, lang: str, hf_token: str):
    """Вирівнювання слів → діаризація pyannote → призначення спікерів.
    Повертає список сегментів із полем 'speaker'. М'яко деградує при помилці."""
    device = config.WHISPER_DEVICE
    try:
        model_a, metadata = whisperx.load_align_model(language_code=lang,
                                                      device=device)
        result = whisperx.align(result["segments"], model_a, metadata,
                                audio, device, return_char_alignments=False)
        del model_a
    except Exception as exc:
        print(f"      [asr] вирівнювання пропущено для '{lang}': {exc}")

    from whisperx.diarize import DiarizationPipeline
    try:
        pipe = DiarizationPipeline(token=hf_token, device=device)
    except TypeError:
        pipe = DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = pipe(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result["segments"]
