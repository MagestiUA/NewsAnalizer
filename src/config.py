"""Константи й шляхи NewsAnalizer.

LLM-стек узято з проєкту AI_Video_Translatrer (Ollama + gemma4), але параметри
тут інші: завдання — ТОЧНА ЕКСТРАКЦІЯ фактів, а не художній переклад, тож
температура низька.
"""

from __future__ import annotations

from pathlib import Path

# ─── Шляхи ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = ROOT / "output"
CACHE_DIR   = ROOT / "cache"

# Промпти: класифікатор обирає формат, далі відповідний промпт.
PROMPT_CLASSIFY         = ROOT / "prompt_classify.txt"
PROMPT_ROUNDUP          = ROOT / "prompt_roundup.txt"          # рубрики, без атрибуції
PROMPT_ANALYSIS_EXTRACT = ROOT / "prompt_analysis_extract.txt"  # плоский список
PROMPT_ANALYSIS_SORT    = ROOT / "prompt_analysis_sort.txt"     # сортування списку

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


def load(path) -> str:
    """Текст промпт-файлу (редагується без коду)."""
    return path.read_text(encoding="utf-8").strip()
