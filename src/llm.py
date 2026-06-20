"""Тонка обгортка Ollama. Патерни керування сервером — з AI_Video_Translatrer."""

from __future__ import annotations

import subprocess
import sys
import time

import requests

from . import config


class OllamaClient:
    def __init__(self, model: str = config.MODEL, host: str = config.HOST):
        self.model = model
        self.host = host.rstrip("/")
        self._ensure_server()

    # ── Керування сервером ──────────────────────────────────────────────────
    def _ensure_server(self, retries: int = 20, delay: float = 1.5) -> None:
        try:
            requests.get(f"{self.host}/api/tags", timeout=3)
            return
        except Exception:
            pass
        print("  [i] Ollama не запущена — стартую сервер...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(retries):
            time.sleep(delay)
            try:
                requests.get(f"{self.host}/api/tags", timeout=3)
                print("  [i] Ollama готова.")
                return
            except Exception:
                pass
        raise RuntimeError("Ollama не відповіла після запуску — перевір ollama.exe")

    def shutdown(self) -> None:
        """Вивантажити модель і звільнити VRAM."""
        try:
            requests.post(f"{self.host}/api/generate",
                          json={"model": self.model, "keep_alive": 0}, timeout=10)
        except Exception:
            pass
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                               capture_output=True, check=False)
                subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"],
                               capture_output=True, check=False)
            else:
                subprocess.run(["pkill", "-f", "ollama"],
                               capture_output=True, check=False)
            print("  [i] Ollama зупинено, VRAM звільнено.")
        except Exception as exc:
            print(f"  [!] Не вдалось зупинити Ollama: {exc}")

    # ── Запит ───────────────────────────────────────────────────────────────
    def chat(self, system: str, user: str) -> str:
        r = requests.post(f"{self.host}/api/chat", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": config.ANALYZE_TEMP,
                "num_ctx": config.NUM_CTX,
                "num_predict": -1,
                "top_p": config.TOP_P,
                "top_k": config.TOP_K,
            },
        }, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"].get("content", "").strip()
