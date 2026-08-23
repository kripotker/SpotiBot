"""
Shared settings storage for SpotiBot. Both bot.py and gui_launcher.py read
and write this same file, so changes made in the GUI's Settings window can
take effect in the already-running bot without a restart (see
bot.reload_settings()).
"""
import json
import os

SETTINGS_PATH = "spotibot_settings.json"

# Covers every mishearing seen in real testing ("hey bot" -> "hey bought",
# "hey but", "hey bawt", etc.) without the two-part regex-alternation
# approach, which was hard to expose in a UI. Users can add/remove phrases
# freely from Settings > Voice.
DEFAULT_WAKE_PHRASES = [
    "hey bot", "hey bought", "hey but", "hey bawt", "hey paybot",
    "hay bot", "hay bought",
    "ok bot", "ok bought",
    "okay bot", "okay bought",
    "k bot", "k bought",
]

DEFAULTS = {
    "wake_phrases": DEFAULT_WAKE_PHRASES,
    "debug_voice": False,  # off by default - verbose per-packet/watchdog logging
    "idle_timeout": 0.8,  # seconds of silence before an utterance is processed
    "spotify_match_threshold": 0.45,  # 0-1, how confident a Spotify match must be to use it
    "whisper_model": "small.en",  # faster-whisper model name for speech-to-text
    "whisper_device": "cpu",  # "cpu" or "cuda" - GPU accel is CUDA-only (ctranslate2 has no ROCm/Vulkan backend)
    "show_vram_guide": True,  # auto-popup the model-size/VRAM guide when opening the HF browser
    "llama_model_path": r"models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "llama_ngl": "99",
    "llama_server_port": "8080",
}


def load_settings() -> dict:
    """Always returns every key (falling back to defaults for anything
    missing/corrupt), so callers never need to guard against absent keys."""
    settings = dict(DEFAULTS)
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                settings.update(saved)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable file - fall back to defaults silently
    return settings


def save_settings(settings: dict):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)