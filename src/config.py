"""Константи й шляхи NewsAnalizer.

LLM-стек узято з проєкту AI_Video_Translatrer (Ollama + gemma4), але параметри
тут інші: завдання — ТОЧНА ЕКСТРАКЦІЯ фактів, а не художній переклад, тож
температура низька.
"""

from __future__ import annotations

from pathlib import Path

# ─── Шляхи ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompt.txt"
OUTPUT_DIR  = ROOT / "output"
CACHE_DIR   = ROOT / "cache"

# ─── Ollama / модель ─────────────────────────────────────────────────────────
HOST          = "http://localhost:11434"
MODEL         = "gemma4:26b-a4b-it-qat"

# "Лінивий" фіксований контекст: одне новинне відео (~30 хв ≈ 9-13k токенів)
# влазить із запасом. Не морити модель динамічним підбором.
NUM_CTX       = 65536

# Аналіз ≠ переклад: мінімум варіативності → максимум точності цифр/цитат,
# стабільний формат секцій. НЕ 0.0 (gemma інколи зациклюється на жадібному декоді).
ANALYZE_TEMP  = 0.2
TOP_P         = 0.9
TOP_K         = 40

# Запас часу на відповідь моделі (велике відео + роздуми gemma)
REQUEST_TIMEOUT = 1800

# ─── ASR-резерв (WhisperX) — коли субтитрів немає взагалі ─────────────────────
# Вмикається лише як фолбек у transcript.py. Таймкоди/діаризація не потрібні —
# беремо лише текст. Версії під RTX (cu128): torch 2.8.0, whisperx 3.8.6.
WHISPER_MODEL   = "large-v3"
WHISPER_DEVICE  = "cuda"      # "cpu" якщо немає GPU (буде дуже повільно)
WHISPER_COMPUTE = "float16"   # "int8" для CPU / економії VRAM
WHISPER_LANG    = None        # None = автовизначення мови (uk/ru)


def load_prompt() -> str:
    """Системний промпт аналітика з prompt.txt (редагується без коду)."""
    return PROMPT_FILE.read_text(encoding="utf-8").strip()
