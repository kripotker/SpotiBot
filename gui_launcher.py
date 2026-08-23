"""
GUI launcher for SpotiBot - runs bot.py in a background thread and launches
llama-server as a managed subprocess, streaming both into a proper app
window (separate tabs) instead of two terminals.

Run this instead of bot.py directly:
    python gui_launcher.py

Wake words, debug logging, and AI model/backend preferences are all
editable from the app's Settings menu (top-left of the window) once it's
running - no need to touch environment variables for day-to-day use.
Everything is persisted in spotibot_settings.json (see settings_store.py).

On first launch (and from Settings > AI Model > "Re-run GPU/backend
setup..."), the GPU picker tries to detect your hardware (gpu_detect.py)
and pre-selects a matching backend as a clearly-labeled suggestion - it
never auto-downloads anything without you clicking confirm. Same idea in
the Hugging Face model browser: it pre-fills a search and highlights a
recommended quantization sized to your VRAM, but you still pick and
confirm the actual file.

Environment variables remain available as power-user overrides for the
llama-server launch (Settings > AI Model covers the same ground for
everyone else):
    LLAMA_SERVER_PATH   (default: llama-server.exe)
    LLAMA_MODEL_PATH    (default: models\\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)
    LLAMA_SERVER_PORT   (default: 8080)
    LLAMA_NGL           (default: 99)

To build a standalone .exe (see README.md for the full prerequisite list -
this does NOT bundle llama-server.exe itself, your GGUF model, or ffmpeg):
    pip install pyinstaller
    pyinstaller --onefile --noconsole --name SpotiBot ^
        --add-data "ping.mp3;." ^
        --collect-all faster_whisper ^
        --collect-all ctranslate2 ^
        --collect-all discord ^
        gui_launcher.py
"""
import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
import zipfile
from tkinter import filedialog, messagebox, scrolledtext, ttk

import gpu_detect
import settings_store
import theme as th

# NOTE: bot_module is intentionally NOT imported here. bot.py does real
# work at import time (loading the Whisper model, Spotify client setup,
# etc.), which can take anywhere from a few seconds to tens of seconds.
# Importing it before the window exists means the whole process sits
# there with nothing visible on screen during that time - looks exactly
# like "nothing launches," and with a --noconsole build, any import-time
# crash would be completely silent too. Import happens inside the
# background thread instead, AFTER the window is already showing and
# stdout is already redirected, so loading progress and any startup error
# both show up in the log.
bot_module = None

_settings = settings_store.load_settings()

LLAMA_SERVER_PATH = os.getenv("LLAMA_SERVER_PATH", "llama-server.exe")
DEFAULT_MODEL_PATH = settings_store.DEFAULTS["llama_model_path"]
# env vars are a power-user override; settings file (editable from Settings
# > AI Model in the GUI) is the normal path.
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH") or _settings["llama_model_path"]
LLAMA_SERVER_PORT = os.getenv("LLAMA_SERVER_PORT") or _settings["llama_server_port"]
LLAMA_NGL = os.getenv("LLAMA_NGL") or _settings["llama_ngl"]

# Only known for the default model - if LLAMA_MODEL_PATH was customized to
# point at something else, we have no way to know what to fetch.
MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

# --- llama.cpp backend auto-download ---
# Pinned to a specific release tag for reproducibility - llama.cpp cuts
# releases very frequently, and an "always latest" link could silently
# start shipping a build with a breaking change. Bump this manually if you
# want a newer build; check https://github.com/ggml-org/llama.cpp/releases
LLAMA_RELEASE_TAG = "b10549"
LLAMA_RELEASE_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE_TAG}"
LLAMA_CPP_INSTALL_DIR = "llama_cpp"
LLAMA_CPP_CONFIG_PATH = os.path.join(LLAMA_CPP_INSTALL_DIR, "install_config.json")

# Each entry: key, short label, speed badge, plain-language explanation,
# main zip filename template, optional extra zip (e.g. CUDA runtime DLLs).
LLAMA_BACKENDS = [
    (
        "cpu-x64", "CPU only (x64)", "🐢 Slowest, but always works",
        "Runs the AI purely on your processor, no graphics card involved at all. "
        "This is the slowest option - responses will take noticeably longer - but "
        "it works on literally any Windows PC, no matter what hardware you have. "
        "Good starting point if you don't have a dedicated graphics card, or just "
        "want to get things running without fuss.",
        "llama-{tag}-bin-win-cpu-x64.zip", None,
    ),
    (
        "cpu-arm64", "CPU only (ARM64)", "🐢 Slowest, but always works",
        "Same as CPU only (x64) above, but built for ARM-based Windows laptops "
        "(Snapdragon-powered devices like Surface Pro X or Snapdragon X Elite "
        "machines). Use this instead of the x64 CPU option if you're on one of "
        "these ARM laptops.",
        "llama-{tag}-bin-win-cpu-arm64.zip", None,
    ),
    (
        "opencl-adreno-arm64", "Qualcomm Adreno GPU (ARM64)", "⚡ Medium",
        "For Snapdragon X Elite/Plus Windows laptops that have an Adreno graphics "
        "chip built into the processor. Uses that built-in GPU for a real speed "
        "boost over the plain CPU option on these specific laptops.",
        "llama-{tag}-bin-win-opencl-adreno-arm64.zip", None,
    ),
    (
        "cuda12-x64", "NVIDIA GPU - CUDA 12", "🚀 Fast",
        "For NVIDIA graphics cards (anything called GeForce GTX or RTX). Uses "
        "your graphics card to run the AI much faster than the CPU alone. This "
        "version works with older NVIDIA cards too - GTX 900-series, GTX "
        "10-series, GTX 16-series, and all RTX 20/30/40-series cards - so it's "
        "the safer NVIDIA pick if you're not sure how new your card is.",
        "llama-{tag}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ),
    (
        "cuda13-x64", "NVIDIA GPU - CUDA 13", "🚀 Fast",
        "For newer NVIDIA graphics cards only: RTX 20-series or newer. This is "
        "required (not optional) for the newest RTX 50-series cards. Just as "
        "fast as CUDA 12, but it will NOT work on older GTX 900/10/16-series "
        "cards - pick 'CUDA 12' instead if you have one of those, or if this "
        "option fails to start.",
        "llama-{tag}-bin-win-cuda-13.3-x64.zip", "cudart-llama-bin-win-cuda-13.3-x64.zip",
    ),
    (
        "cuda13-arm64", "NVIDIA GPU - CUDA 13 (ARM64, preview)", "🚀 Fast (niche)",
        "A specialized option for ARM-based devices with NVIDIA's newest "
        "Blackwell chips (e.g. NVIDIA Jetson Thor). Almost nobody needs this - "
        "if you're on a normal Windows PC or laptop, this isn't for you.",
        "llama-{tag}-bin-win-cuda-13.4-arm64.zip", "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    ),
    (
        "vulkan-x64", "Vulkan", "⚡ Medium-Fast, very reliable",
        "A general-purpose graphics option that works on most AMD, Intel, and "
        "NVIDIA graphics cards. Usually not quite as fast as a vendor-specific "
        "option (CUDA/ROCm) on the same card, but it's the most reliable "
        "'it just works' choice. Good pick if you're unsure what else to "
        "choose, or if a more specific option below fails to start.",
        "llama-{tag}-bin-win-vulkan-x64.zip", None,
    ),
    (
        "openvino-x64", "Intel GPU - OpenVINO", "⚡ Medium",
        "For PCs with Intel graphics - either an Intel Arc graphics card, or "
        "the graphics built into most Intel processors. Optimized specifically "
        "for Intel hardware by Intel's own toolkit.",
        "llama-{tag}-bin-win-openvino-2026.3-x64.zip", None,
    ),
    (
        "sycl-x64", "Intel GPU - SYCL", "⚡ Medium",
        "Another option for the same Intel graphics hardware as OpenVINO above "
        "(Arc cards or built-in Intel graphics), using a different underlying "
        "toolkit. If OpenVINO doesn't work well for you, try this instead - "
        "results can vary by exact GPU model.",
        "llama-{tag}-bin-win-sycl-x64.zip", None,
    ),
    (
        "rocm-x64", "AMD GPU - ROCm 7.14", "🚀 Fast",
        "For AMD Radeon graphics cards. Uses your GPU for a big speed boost, "
        "similar to what CUDA does for NVIDIA cards. Confirmed working well on "
        "the RX 7900 XT (RX 7000-series), and AMD's newer RX 9000-series is "
        "also supported. If this fails to start on your card, try 'Vulkan' "
        "instead, which also works on AMD hardware.",
        "llama-{tag}-bin-win-rocm-7.14-x64.zip", None,
    ),
]

# Each entry: key (the exact faster-whisper model name), label, speed
# badge, plain-language description covering size/accuracy/speed/language
# support. Sizes are the faster-whisper (CTranslate2, int8) download
# sizes, not the original OpenAI checkpoint sizes - notably smaller.
WHISPER_MODELS = [
    (
        "tiny.en", "Tiny (English)", "🚀 Fastest",
        "~75MB download. The smallest, fastest model - lowest accuracy of "
        "the set, and will mishear unusual words/names more often. English "
        "only. Good if responsiveness matters more than accuracy, or on "
        "very limited hardware.",
    ),
    (
        "tiny", "Tiny (Multilingual)", "🚀 Fastest",
        "~75MB download. Same speed/size as Tiny (English), but understands "
        "~99 languages instead of English only - quality varies a lot by "
        "language, best on high-resource ones (Spanish, French, German, "
        "Chinese, Japanese, etc).",
    ),
    (
        "base.en", "Base (English)", "🚀 Very fast",
        "~145MB download, roughly 2x slower than Tiny. Noticeably more "
        "accurate than Tiny, especially on names and less common words. "
        "English only. A reasonable default if Tiny mishears too often.",
    ),
    (
        "base", "Base (Multilingual)", "🚀 Very fast",
        "~145MB download, same speed as Base (English). Multilingual "
        "version, ~99 languages.",
    ),
    (
        "small.en", "Small (English)", "⚡ Fast",
        "~484MB download, roughly 4-6x slower than Tiny. Solid accuracy, "
        "handles proper nouns and accents well. English only. This is "
        "SpotiBot's default - a good balance for short voice commands.",
    ),
    (
        "small", "Small (Multilingual)", "⚡ Fast",
        "~484MB download, same speed as Small (English). Multilingual, "
        "~99 languages.",
    ),
    (
        "medium.en", "Medium (English)", "⚡ Moderate",
        "~1.5GB download, roughly 10x+ slower than Tiny. Best accuracy of "
        "the '.en' models, likely overkill for short commands like 'hey "
        "bot, play X' - the extra time per response is more noticeable "
        "than the accuracy gain for this use case. English only.",
    ),
    (
        "medium", "Medium (Multilingual)", "⚡ Moderate",
        "~1.5GB download, same speed as Medium (English). Multilingual, "
        "~99 languages.",
    ),
    (
        "large-v3", "Large v3 (Multilingual)", "🐢 Slow",
        "~2.9GB download, the most accurate model here but also the "
        "slowest - many seconds per transcription on CPU. Multilingual "
        "only (there's no English-only 'large' variant). Only really "
        "worth it if accuracy matters far more than response time.",
    ),
    (
        "large-v3-turbo", "Large v3 Turbo (Multilingual)", "⚡ Fast for its accuracy",
        "~1.5GB download. A pruned/distilled version of Large v3 - keeps "
        "most of its accuracy while running close to Medium speed, "
        "noticeably faster than the full Large v3. Multilingual. Probably "
        "the best pick if you want Large-level accuracy without the full "
        "Large-level wait.",
    ),
]


class QueueWriter:
    """File-like object standing in for stdout/stderr - pushes writes into
    a thread-safe queue instead of a terminal, since background threads
    can't touch Tkinter widgets directly."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(text)

    def flush(self):
        pass


# Log lines that flip the status indicator - matched as simple substrings
# against everything printed, so no changes to print() calls elsewhere are
# needed to keep this in sync.
BOT_STATUS_TRIGGERS = [
    ("Bot successfully logged in", "● Online", th.SUCCESS),
    ("👂 Listening!", "● Listening", th.SUCCESS),
    ("🔇", "● Processing speech...", th.WARNING),
    ("[Playback] Finished", "● Online", th.SUCCESS),
    ("🎶 Now playing", "● Playing music", th.SUCCESS),
    ("FATAL", "● Crashed - see log", th.DANGER),
]

# llama-server prints this line once it's actually ready to take requests.
AI_STATUS_TRIGGERS = [
    ("listening on http://", "● Ready", th.SUCCESS),
    ("error", "● Error - see log", th.WARNING),
    ("[FATAL]", "● Crashed - see log", th.DANGER),
]


class LogTab:
    """One tab: a scrolled log view plus its own status pill, fed by its own queue."""

    def __init__(self, notebook: ttk.Notebook, title: str, triggers: list):
        self.queue: queue.Queue = queue.Queue()
        self.triggers = triggers

        frame = th.frame(notebook)
        notebook.add(frame, text=title)

        status_bar = th.frame(frame)
        status_bar.pack(fill="x", padx=16, pady=(14, 8))
        self.status_var = tk.StringVar(value="● Starting...")
        self.status_label = th.pill(status_bar, self.status_var, fg=th.WARNING)
        self.status_label.pack(side="left")

        card_outer, card_inner = th.card(frame)
        card_outer.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.text_area = scrolledtext.ScrolledText(
            card_inner, bg=th.SURFACE, fg=th.TEXT_SECONDARY, insertbackground=th.TEXT_PRIMARY,
            font=(th.FONT_MONO, 10), wrap="word", state="disabled", borderwidth=0, highlightthickness=0,
            padx=10, pady=8,
        )
        self.text_area.pack(fill="both", expand=True)

    def append(self, msg: str):
        self.text_area.configure(state="normal")
        self.text_area.insert("end", msg)
        self.text_area.see("end")
        self.text_area.configure(state="disabled")

        for trigger, label_text, color in self.triggers:
            if trigger.lower() in msg.lower():
                self.status_var.set(label_text)
                self.status_label.configure(fg=color)
                break

    def drain(self):
        try:
            while True:
                self.append(self.queue.get_nowait())
        except queue.Empty:
            pass


class GpuPickerDialog(tk.Toplevel):
    """
    First-launch modal: pick a llama.cpp backend matching your hardware,
    then download + extract it. Runs once - the resolved llama-server.exe
    path gets saved to LLAMA_CPP_CONFIG_PATH so this never shows again
    unless that file/folder is deleted.

    Tries to auto-detect the GPU (gpu_detect.py) and pre-selects a
    matching backend, shown as a clearly-labeled suggestion banner at the
    top - detection only ever changes what's pre-selected, never what
    happens without a click. Nothing downloads until "Download & Continue".
    """

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.result_path = None  # set on success, left None if cancelled/failed

        self.title("SpotiBot - First-time setup")
        self.geometry("660x600")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        th.apply_dark_titlebar(self)
        self.grab_set()  # modal

        gpus = gpu_detect.detect_gpus()
        self._gpus = gpus
        suggested_key, self._suggestion_reason = gpu_detect.suggest_backend(gpus)
        self.choice_var = tk.StringVar(value=suggested_key)
        self._suggested_key = suggested_key
        self._detail_frames = {}  # key -> Frame, for expand/collapse toggling
        self._build_choice_ui()

    def _build_choice_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        th.label(self, "Which GPU/backend does this PC have?", kind="title", bg=th.BG).pack(
            pady=(20, 4), padx=20, anchor="w"
        )
        th.label(
            self, "Click 'more info' on any option for a plain-language explanation.\n"
                  "Only asked once - change it later from Settings > AI Model.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(pady=(0, 12), padx=20, anchor="w")

        # Suggestion banner - accent-colored card, clearly distinct from the
        # neutral list below, but changes nothing until the person confirms.
        suggested_label = next((b[1] for b in LLAMA_BACKENDS if b[0] == self._suggested_key), self._suggested_key)
        banner_outer = tk.Frame(self, bg=th.ACCENT)
        banner_outer.pack(fill="x", padx=20, pady=(0, 14))
        banner_inner = tk.Frame(banner_outer, bg=th.SURFACE)
        banner_inner.pack(fill="x", padx=1, pady=1)
        th.label(
            banner_inner, f"💡 Suggested for your PC: {suggested_label}",
            kind="body", bg=th.SURFACE, fg=th.ACCENT, font=(th.FONT, 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))
        th.label(
            banner_inner, self._suggestion_reason, kind="secondary", bg=th.SURFACE, justify="left", wraplength=580,
        ).pack(anchor="w", padx=14, pady=(0, 10))

        # Scrollable list, since expanded descriptions push this well past
        # what fits in a fixed-height dialog.
        outer = th.frame(self)
        outer.pack(fill="both", expand=True, padx=20)
        canvas = tk.Canvas(outer, bg=th.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        list_frame = th.frame(canvas)

        list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=list_frame, anchor="nw", width=600)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        for key, label_text, speed, description, *_ in LLAMA_BACKENDS:
            row_outer, row_inner = th.card(list_frame)
            row_outer.pack(fill="x", pady=5)

            header = th.frame(row_inner, bg=th.SURFACE)
            header.pack(fill="x", padx=10, pady=8)

            th.radiobutton(
                header, label_text, self.choice_var, key,
                bg=th.SURFACE, font=(th.FONT, 10, "bold"),
            ).pack(side="left")
            th.label(header, speed, kind="secondary", bg=th.SURFACE, fg=th.ACCENT).pack(side="left", padx=(8, 0))
            if key == self._suggested_key:
                th.pill(header, tk.StringVar(value="Suggested"), fg=th.ACCENT_TEXT, bg=th.ACCENT).pack(side="left", padx=(8, 0))

            toggle_btn = tk.Label(
                header, text="▸ more info", bg=th.SURFACE, fg=th.INFO,
                font=(th.FONT, 9, "underline"), cursor="hand2",
            )
            toggle_btn.pack(side="right", padx=(0, 4))

            detail = th.frame(row_inner, bg=th.SURFACE_ALT)
            th.label(
                detail, description, kind="secondary", bg=th.SURFACE_ALT,
                justify="left", wraplength=550, anchor="w",
            ).pack(fill="x", padx=14, pady=10)
            self._detail_frames[key] = detail

            def make_toggle(btn=toggle_btn, det=detail):
                def toggle(_event=None):
                    if det.winfo_ismapped():
                        det.pack_forget()
                        btn.configure(text="▸ more info")
                    else:
                        det.pack(fill="x", padx=10, pady=(0, 8))
                        btn.configure(text="▾ hide info")
                return toggle

            toggle_btn.bind("<Button-1>", make_toggle())

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=20, pady=16)
        th.button(btn_frame, "Download & Continue", self._on_confirm, kind="primary").pack(side="right")
        th.button(btn_frame, "Skip (I'll set up llama-server myself)", self._on_cancel, kind="ghost").pack(side="left")

    def _build_progress_ui(self):
        for widget in self.winfo_children():
            widget.destroy()

        th.label(self, "Setting up llama.cpp...", kind="title", bg=th.BG).pack(pady=(20, 10), padx=20, anchor="w")

        card_outer, card_inner = th.card(self)
        card_outer.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.progress_text = scrolledtext.ScrolledText(
            card_inner, bg=th.SURFACE, fg=th.TEXT_SECONDARY, font=(th.FONT_MONO, 9),
            wrap="word", state="disabled", borderwidth=0, highlightthickness=0, padx=10, pady=8,
        )
        self.progress_text.pack(fill="both", expand=True)

    def _log(self, msg: str):
        self.progress_text.configure(state="normal")
        self.progress_text.insert("end", msg)
        self.progress_text.see("end")
        self.progress_text.configure(state="disabled")
        self.update()  # this dialog blocks its own event loop during download, so pump it manually

    def _on_cancel(self):
        self.unbind_all("<MouseWheel>")  # bind_all is app-wide, don't leak it past this dialog
        self.result_path = None
        self.grab_release()
        self.destroy()

    def _on_confirm(self):
        self.unbind_all("<MouseWheel>")
        key = self.choice_var.get()
        backend = next(b for b in LLAMA_BACKENDS if b[0] == key)
        self._build_progress_ui()
        try:
            self.result_path = _download_and_install_backend(backend, self._log)
        except Exception as e:
            self._log(f"\n[FATAL] Setup failed: {e}\n")
            self.result_path = None

        if self.result_path:
            self._maybe_prompt_whisper_gpu()

        self.grab_release()
        self.destroy()

    def _maybe_prompt_whisper_gpu(self):
        """
        Independent of which llama.cpp backend was actually chosen -
        Whisper's GPU path is CUDA-specific and works as long as an
        NVIDIA GPU with enough VRAM is present, regardless of whether the
        LLM itself is running on CUDA, Vulkan, or anything else. Only
        offered at >=8GB VRAM, and skipped entirely if Whisper is already
        set to GPU (no need to ask again).
        """
        current_device = settings_store.load_settings().get("whisper_device", "cpu")
        if current_device == "cuda":
            return

        nvidia_gpus = [g for g in self._gpus if g.get("vendor") == "nvidia" and (g.get("vram_gb") or 0) >= 8]
        if not nvidia_gpus:
            return

        best = max(nvidia_gpus, key=lambda g: g["vram_gb"])
        prompt = WhisperGpuPromptDialog(self, best["name"], best["vram_gb"])
        self.wait_window(prompt)

        if prompt.result:
            s = settings_store.load_settings()
            s["whisper_device"] = "cuda"
            settings_store.save_settings(s)
            if bot_module:
                bot_module.reload_whisper_model(s["whisper_model"], "cuda")


class WhisperModelDialog(tk.Toplevel):
    """
    Pick a faster-whisper speech-to-text model, with the same expandable
    radio-list pattern as GpuPickerDialog. No download-progress step here -
    unlike the llama.cpp backend, faster-whisper downloads a new model
    transparently the first time it's used, and that progress already
    surfaces naturally in the Bot tab's log (bot.py's prints, including
    huggingface_hub's own download progress, already flow through there).
    Sets self.result_key on confirm; None if cancelled.
    """

    def __init__(self, parent: tk.Widget, current_model: str, current_device: str = "cpu"):
        super().__init__(parent)
        self.result_key = None
        self.result_device = None

        self.title("Choose Speech-to-Text Model")
        self.geometry("640x600")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        th.apply_dark_titlebar(self)
        self.transient(parent)
        self.grab_set()

        self.choice_var = tk.StringVar(value=current_model)
        self.device_var = tk.StringVar(value=current_device)

        th.label(self, "Which speech-to-text model?", kind="title", bg=th.BG).pack(
            pady=(20, 4), padx=20, anchor="w"
        )
        th.label(
            self, "This is what transcribes your voice commands (separate from the AI\n"
                  "model that understands them). Smaller = faster but less accurate.\n"
                  "Click 'more info' on any option for details.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(pady=(0, 10), padx=20, anchor="w")

        device_outer, device_inner = th.card(self)
        device_outer.pack(fill="x", padx=20, pady=(0, 12))
        th.label(device_inner, "Run on:", kind="body", bg=th.SURFACE, font=(th.FONT, 9, "bold")).pack(
            anchor="w", padx=12, pady=(10, 4)
        )
        device_row = th.frame(device_inner, bg=th.SURFACE)
        device_row.pack(anchor="w", padx=12, pady=(0, 4))
        th.radiobutton(device_row, "CPU", self.device_var, "cpu", bg=th.SURFACE).pack(side="left")
        th.radiobutton(device_row, "GPU", self.device_var, "cuda", bg=th.SURFACE).pack(side="left", padx=(16, 0))
        th.label(
            device_inner, "GPU acceleration only works with NVIDIA cards (via CUDA) - AMD and Intel\n"
                          "GPUs aren't supported for speech-to-text regardless of your llama.cpp backend.\n"
                          "Requires a working CUDA/cuDNN setup; if it fails to load, switch back to CPU.",
            kind="muted", bg=th.SURFACE, justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 10))

        outer = th.frame(self)
        outer.pack(fill="both", expand=True, padx=20)
        canvas = tk.Canvas(outer, bg=th.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        list_frame = th.frame(canvas)

        list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=list_frame, anchor="nw", width=580)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._canvas = canvas

        for key, label_text, speed, description in WHISPER_MODELS:
            row_outer, row_inner = th.card(list_frame)
            row_outer.pack(fill="x", pady=5)

            header = th.frame(row_inner, bg=th.SURFACE)
            header.pack(fill="x", padx=10, pady=8)

            th.radiobutton(
                header, label_text, self.choice_var, key, bg=th.SURFACE, font=(th.FONT, 10, "bold"),
            ).pack(side="left")
            th.label(header, speed, kind="secondary", bg=th.SURFACE, fg=th.ACCENT).pack(side="left", padx=(8, 0))
            if key == current_model:
                th.pill(header, tk.StringVar(value="Current"), fg=th.ACCENT_TEXT, bg=th.ACCENT).pack(side="left", padx=(8, 0))

            toggle_btn = tk.Label(
                header, text="▸ more info", bg=th.SURFACE, fg=th.INFO,
                font=(th.FONT, 9, "underline"), cursor="hand2",
            )
            toggle_btn.pack(side="right", padx=(0, 4))

            detail = th.frame(row_inner, bg=th.SURFACE_ALT)
            th.label(
                detail, description, kind="secondary", bg=th.SURFACE_ALT,
                justify="left", wraplength=530, anchor="w",
            ).pack(fill="x", padx=14, pady=10)

            def make_toggle(btn=toggle_btn, det=detail):
                def toggle(_event=None):
                    if det.winfo_ismapped():
                        det.pack_forget()
                        btn.configure(text="▸ more info")
                    else:
                        det.pack(fill="x", padx=10, pady=(0, 8))
                        btn.configure(text="▾ hide info")
                return toggle

            toggle_btn.bind("<Button-1>", make_toggle())

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=20, pady=16)
        th.button(btn_frame, "Use This Model", self._on_confirm, kind="primary").pack(side="right")
        th.button(btn_frame, "Cancel", self._on_cancel, kind="ghost").pack(side="left")

    def _on_cancel(self):
        self._canvas.unbind_all("<MouseWheel>")
        self.result_key = None
        self.result_device = None
        self.grab_release()
        self.destroy()

    def _on_confirm(self):
        self._canvas.unbind_all("<MouseWheel>")
        device = self.device_var.get()

        if device == "cuda" and not _cuda_runtime_ready():
            dl_dialog = CudaRuntimeDownloadDialog(self)
            self.wait_window(dl_dialog)
            if not dl_dialog.success:
                messagebox.showerror(
                    "SpotiBot", "Couldn't set up GPU support - pick CPU instead, or try again.", parent=self,
                )
                return  # keep the picker open so they can retry or switch to CPU
            _add_bundled_binaries_to_path()

        self.result_key = self.choice_var.get()
        self.result_device = device
        self.grab_release()
        self.destroy()


class WhisperGpuPromptDialog(tk.Toplevel):
    """
    Shown once after a successful GPU/backend setup, only when an NVIDIA
    GPU with >=8GB VRAM was actually detected - asks whether to also run
    Whisper (speech-to-text) on GPU. The warning text scales with how much
    headroom that VRAM figure actually gives: right at the 8GB minimum,
    running the LLM and Whisper on GPU together can leave little room for
    anything else sharing that VRAM (games, other GPU apps, and SpotiBot's
    own responsiveness if it runs short mid-use).
    """

    def __init__(self, parent: tk.Widget, gpu_name: str, vram_gb: float):
        super().__init__(parent)
        self.result = False

        self.title("Speed up voice transcription?")
        self.geometry("480x420")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_no)
        th.apply_dark_titlebar(self)
        self.transient(parent)
        self.grab_set()

        th.label(self, "Also run Whisper on your GPU?", kind="title", bg=th.BG).pack(
            pady=(20, 4), padx=20, anchor="w"
        )
        th.label(
            self, f"Detected {gpu_name} (~{vram_gb:.0f}GB VRAM). Whisper is what transcribes\n"
                  "your voice commands - running it on GPU instead of CPU can make that\n"
                  "noticeably faster.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(padx=20, pady=(0, 14), anchor="w")

        if vram_gb < 12:
            warn_color = th.WARNING
            warn_text = (
                f"⚠️ With only ~{vram_gb:.0f}GB VRAM, this is close to the minimum for running "
                f"both the AI model and Whisper on GPU at once. That shared VRAM budget can get "
                f"tight fast - expect it to heavily impact performance in games or other "
                f"GPU-heavy apps running alongside SpotiBot, and potentially SpotiBot's own "
                f"responsiveness too if VRAM runs low mid-use. If you often game or run other "
                f"GPU work while SpotiBot is open, CPU is the safer choice here."
            )
        elif vram_gb < 16:
            warn_color = th.TEXT_SECONDARY
            warn_text = (
                f"With ~{vram_gb:.0f}GB VRAM you likely have reasonable headroom for both, but "
                f"keep an eye on performance if you're gaming or running other GPU-heavy work "
                f"at the same time."
            )
        else:
            warn_color = th.TEXT_SECONDARY
            warn_text = (
                f"With ~{vram_gb:.0f}GB VRAM you should have plenty of headroom for both "
                f"comfortably."
            )

        card_outer, card_inner = th.card(self)
        card_outer.pack(fill="x", padx=20, pady=(0, 16))
        th.label(
            card_inner, warn_text, kind="secondary", bg=th.SURFACE, fg=warn_color,
            justify="left", wraplength=420,
        ).pack(padx=14, pady=12)

        th.label(
            self, "You can change this anytime in Settings > Voice > Change Whisper Model.",
            kind="muted", bg=th.BG, justify="left",
        ).pack(padx=20, anchor="w")

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=20, pady=16, side="bottom")
        th.button(btn_frame, "Yes, use GPU for Whisper too", self._on_yes, kind="primary").pack(
            side="right"
        )
        th.button(btn_frame, "No, keep Whisper on CPU", self._on_no, kind="secondary").pack(side="left")

    def _on_yes(self):
        if not _cuda_runtime_ready():
            dl_dialog = CudaRuntimeDownloadDialog(self)
            self.wait_window(dl_dialog)
            if not dl_dialog.success:
                messagebox.showerror(
                    "SpotiBot", "Couldn't set up GPU support - staying on CPU for Whisper.", parent=self.master,
                )
                self.result = False
                self.grab_release()
                self.destroy()
                return
            _add_bundled_binaries_to_path()  # make the just-downloaded DLLs discoverable this run, not just next launch

        self.result = True
        self.grab_release()
        self.destroy()

    def _on_no(self):
        self.result = False
        self.grab_release()
        self.destroy()


def _hf_headers() -> dict:
    """
    An HF_TOKEN in .env gets used automatically for Hugging Face API/
    download requests - higher rate limits, and required for any gated
    models. Entirely optional; unauthenticated requests still work, just
    slower and more rate-limit-prone. Checked at call time (not import
    time) since gui_launcher.py doesn't call load_dotenv() itself.
    """
    token = os.environ.get("HF_TOKEN") or _read_env_file().get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _download_file(url: str, dest_path: str, log, headers: dict = None):
    import requests

    with requests.get(url, stream=True, allow_redirects=True, timeout=30, headers=headers or {}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 10 == 0:
                        log(f"  {pct}% ({downloaded / 1e6:.0f}MB / {total / 1e6:.0f}MB)\n")
                        last_pct = pct


def _download_zip(url: str, dest_dir: str, log):
    filename = url.rsplit("/", 1)[-1]
    log(f"Downloading {filename}...\n")
    zip_path = os.path.join(dest_dir, filename)
    _download_file(url, zip_path, log)

    log(f"Extracting {filename}...\n")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)


def _hf_search_models(query: str) -> list:
    import requests

    params = {"search": query, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": 30}
    r = requests.get("https://huggingface.co/api/models", params=params, headers=_hf_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def _hf_list_gguf_files(repo_id: str) -> list:
    import re
    import requests

    r = requests.get(
        f"https://huggingface.co/api/models/{repo_id}/tree/main", headers=_hf_headers(), timeout=15,
    )
    r.raise_for_status()
    files = r.json()

    # Exclude split-file parts (e.g. "...-q4_k_m-00001-of-00002.gguf") -
    # downloading just one part isn't a usable model on its own, it needs
    # merging with llama-gguf-split first. The complete single-file
    # version (e.g. "...-q4_k_m.gguf") is normally listed in the same repo
    # too, so this doesn't lose real options, just broken partial ones.
    split_part_pattern = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)

    return [
        f for f in files
        if isinstance(f, dict)
        and f.get("path", "").lower().endswith(".gguf")
        and not split_part_pattern.search(f.get("path", ""))
    ]


def _download_and_install_backend(backend, log) -> str:
    key, label_text, speed, description, main_zip_tmpl, extra_zip_name = backend
    install_dir = os.path.join(LLAMA_CPP_INSTALL_DIR, key)
    os.makedirs(install_dir, exist_ok=True)

    main_zip_name = main_zip_tmpl.format(tag=LLAMA_RELEASE_TAG)
    _download_zip(f"{LLAMA_RELEASE_BASE}/{main_zip_name}", install_dir, log)

    if extra_zip_name:
        # CUDA builds need the matching cudart DLLs alongside the binary -
        # extract into the same folder so llama-server.exe finds them.
        _download_zip(f"{LLAMA_RELEASE_BASE}/{extra_zip_name}", install_dir, log)

    log("Locating llama-server.exe...\n")
    matches = []
    for dirpath, _, filenames in os.walk(install_dir):
        for fn in filenames:
            if fn.lower() == "llama-server.exe":
                matches.append(os.path.join(dirpath, fn))
    if not matches:
        raise RuntimeError(f"llama-server.exe not found anywhere in the extracted '{key}' build")
    server_path = matches[0]

    with open(LLAMA_CPP_CONFIG_PATH, "w") as f:
        json.dump({"backend_key": key, "server_path": server_path}, f, indent=2)

    log(f"\n✅ Done. Using: {server_path}\n")
    return server_path


def resolve_llama_server_path(root: tk.Tk) -> str:
    """
    Priority order:
    1. Previously-installed backend from LLAMA_CPP_CONFIG_PATH, if the exe
       still exists (normal case after first run).
    2. An explicit LLAMA_SERVER_PATH env override, if it resolves to a
       real file or something on PATH - respects manual setups, skips
       the picker entirely.
    3. The bare default ("llama-server.exe") if it's already on PATH.
    4. Otherwise, show the GPU picker.
    """
    if os.path.exists(LLAMA_CPP_CONFIG_PATH):
        try:
            with open(LLAMA_CPP_CONFIG_PATH) as f:
                cfg = json.load(f)
            if os.path.exists(cfg.get("server_path", "")):
                return cfg["server_path"]
        except (json.JSONDecodeError, OSError):
            pass  # fall through to picker on a corrupt config

    env_override = os.environ.get("LLAMA_SERVER_PATH")
    if env_override:
        if os.path.exists(env_override) or shutil.which(env_override):
            return env_override
        # explicitly set but not found - still fall through to the picker
        # rather than silently ignoring the user's setting

    if shutil.which(LLAMA_SERVER_PATH):
        return LLAMA_SERVER_PATH

    dialog = GpuPickerDialog(root)
    root.wait_window(dialog)
    return dialog.result_path or LLAMA_SERVER_PATH  # fall back to old behavior if skipped


# --- Settings window helpers ---

def _read_backend_info() -> str:
    if not os.path.exists(LLAMA_CPP_CONFIG_PATH):
        return "No backend installed yet (using PATH or a manually-set LLAMA_SERVER_PATH)."
    try:
        with open(LLAMA_CPP_CONFIG_PATH) as f:
            cfg = json.load(f)
        key = cfg.get("backend_key", "?")
        label_text = next((b[1] for b in LLAMA_BACKENDS if b[0] == key), key)
        return f"Currently installed: {label_text}\n{cfg.get('server_path', '?')}"
    except (OSError, json.JSONDecodeError):
        return "Backend config file exists but is unreadable."


def _read_env_file() -> dict:
    env = {}
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _write_env_file(updates: dict):
    """Rewrites .env, replacing matching keys in place and preserving
    everything else (comments, unrelated keys, ordering)."""
    lines = []
    seen = set()
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k = stripped.split("=", 1)[0].strip()
                    if k in updates:
                        lines.append(f"{k}={updates[k]}\n")
                        seen.add(k)
                        continue
                lines.append(line if line.endswith("\n") else line + "\n")
    for k, v in updates.items():
        if k not in seen:
            lines.append(f"{k}={v}\n")
    with open(".env", "w") as f:
        f.writelines(lines)


def _needs_initial_setup() -> bool:
    """Whether the required credentials are missing from .env - controls
    whether InitialSetupDialog shows on this launch. Returning users with
    a filled-in .env never see this again."""
    env = _read_env_file()
    required = ["DISCORD_BOT_TOKEN", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"]
    return not all(env.get(k, "").strip() for k in required)


class TokenGuideDialog(tk.Toplevel):
    """
    Standalone reference window: what each credential is for, and exactly
    where/how to get it. Deliberately NOT modal and NOT grab_set() - it's
    meant to stay open side-by-side with InitialSetupDialog (and with the
    person's actual browser) while they copy tokens across, not block
    interaction with either.
    """

    SECTIONS = [
        (
            "Discord Bot Token", "https://discord.com/developers/applications",
            "Required - this is how the bot logs into Discord.",
            [
                "Click the link above, log in, and click \"New Application\".",
                "Name it anything (e.g. \"SpotiBot\"), then open the \"Bot\" tab on the left.",
                "Click \"Reset Token\" (or \"Copy\" if one's already shown) to get the token.",
                "On this same Bot page, scroll to \"Privileged Gateway Intents\" and enable "
                "both \"Message Content Intent\" and \"Server Members Intent\" - the bot won't "
                "start without these.",
                "To actually invite the bot to your server: go to \"OAuth2\" > \"URL Generator\", "
                "check the \"bot\" scope, check \"Connect\", \"Speak\", \"Send Messages\", and "
                "\"View Channels\" under Bot Permissions, then open the generated URL and pick "
                "your server.",
            ],
        ),
        (
            "Spotify Client ID & Secret", "https://developer.spotify.com/dashboard",
            "Required - used to look up song/artist info when you ask the bot to play something.",
            [
                "Click the link above and log in with any Spotify account (free accounts work fine).",
                "Click \"Create app\". Name/description can be anything.",
                "For \"Redirect URI\", anything valid works (e.g. http://localhost:8888/callback) - "
                "this bot doesn't use that flow, but Spotify requires one to be set.",
                "Once created, the Client ID is shown directly on the app's page.",
                "Click \"View client secret\" on the same page to reveal the Client Secret.",
            ],
        ),
        (
            "Hugging Face Token", "https://huggingface.co/settings/tokens",
            "Optional - without one, downloading the AI model and Whisper speech-to-text "
            "models still works, just slower and more likely to hit rate limits on repeated "
            "downloads/searches.",
            [
                "Click the link above and log in (free account works fine).",
                "Click \"New token\".",
                "A \"Read\" access token is enough - no need for write/fine-grained access.",
                "Copy the generated token.",
            ],
        ),
    ]

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("Where do I get these?")
        self.geometry("560x680")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        th.apply_dark_titlebar(self)

        th.label(self, "Getting your credentials", kind="title", bg=th.BG).pack(
            pady=(20, 4), padx=20, anchor="w"
        )
        th.label(
            self, "This window can stay open while you fill in the other one - copy each\n"
                  "token over as you get it.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(pady=(0, 14), padx=20, anchor="w")

        outer = th.frame(self)
        outer.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        canvas = tk.Canvas(outer, bg=th.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        list_frame = th.frame(canvas)
        list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=list_frame, anchor="nw", width=520)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>") if e.widget is self else None)

        for title, url, subtitle, steps in self.SECTIONS:
            section_outer, section_inner = th.card(list_frame)
            section_outer.pack(fill="x", pady=(0, 14))

            th.label(section_inner, title, kind="heading", bg=th.SURFACE).pack(
                anchor="w", padx=14, pady=(12, 2)
            )
            th.label(
                section_inner, subtitle, kind="secondary", bg=th.SURFACE, justify="left", wraplength=480,
            ).pack(anchor="w", padx=14, pady=(0, 8))

            link = tk.Label(
                section_inner, text=url, bg=th.SURFACE, fg=th.INFO,
                font=(th.FONT, 9, "underline"), cursor="hand2",
            )
            link.pack(anchor="w", padx=14, pady=(0, 10))
            link.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

            for i, step in enumerate(steps, 1):
                th.label(
                    section_inner, f"{i}. {step}", kind="secondary", bg=th.SURFACE,
                    justify="left", wraplength=480,
                ).pack(anchor="w", padx=14, pady=(0, 6))

            th.label(section_inner, "", bg=th.SURFACE).pack(pady=(0, 2))  # bottom breathing room


class InitialSetupDialog(tk.Toplevel):
    """
    First-launch credential entry, shown before anything else (even the
    GPU/backend picker) if .env is missing the required keys. Deliberately
    NOT modal/grab_set() - meant to coexist with TokenGuideDialog (and the
    person's browser) rather than block interaction with them.

    self.completed is only set True on a successful "Continue" - main()
    checks this after the dialog closes and exits cleanly rather than
    proceeding into an app that has no way to actually log into Discord.
    """

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.completed = False

        self.title("SpotiBot - First-Time Setup")
        self.geometry("560x520")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        th.apply_dark_titlebar(self)

        th.label(self, "Welcome to SpotiBot", kind="title", bg=th.BG).pack(pady=(20, 4), padx=20, anchor="w")
        th.label(
            self, "A few credentials are needed before SpotiBot can start. These are yours\n"
                  "alone - stored locally in a .env file next to the app, never sent anywhere\n"
                  "except the services they belong to (Discord/Spotify/Hugging Face).",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(pady=(0, 6), padx=20, anchor="w")

        guide_link = tk.Label(
            self, text="❓ Where do I get these? (opens a guide)", bg=th.BG, fg=th.INFO,
            font=(th.FONT, 9, "underline"), cursor="hand2",
        )
        guide_link.pack(padx=20, anchor="w", pady=(0, 16))
        guide_link.bind("<Button-1>", lambda e: self._raise_guide())
        self._guide_dialog = None  # set by main() after both windows are created

        self.entries = {}
        fields = [
            ("DISCORD_BOT_TOKEN", "Discord Bot Token", False),
            ("SPOTIFY_CLIENT_ID", "Spotify Client ID", False),
            ("SPOTIFY_CLIENT_SECRET", "Spotify Client Secret", False),
            ("HF_TOKEN", "Hugging Face Token (optional)", True),
        ]
        for key, label_text, optional in fields:
            row = th.frame(self)
            row.pack(fill="x", padx=20, pady=6)
            th.label(row, label_text, kind="body", bg=th.BG, width=24, anchor="w").pack(side="left")

            entry = th.entry(row, show="*", font=(th.FONT_MONO, 9))
            entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=3)
            self.entries[key] = entry

            def make_toggle(e=entry):
                def toggle():
                    e.configure(show="" if e.cget("show") == "*" else "*")
                return toggle
            th.button(row, "👁", make_toggle(), kind="secondary", width=3, padx=4).pack(side="left")

        th.label(self, "", bg=th.BG).pack(pady=4)  # spacer

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=20, pady=16, side="bottom")
        th.button(btn_frame, "Continue", self._on_continue, kind="primary").pack(side="right")
        th.button(btn_frame, "Exit", self._on_close, kind="ghost").pack(side="left")

    def _raise_guide(self):
        if self._guide_dialog is not None and self._guide_dialog.winfo_exists():
            self._guide_dialog.deiconify()
            self._guide_dialog.lift()
            self._guide_dialog.focus_force()

    def _on_continue(self):
        values = {k: e.get().strip() for k, e in self.entries.items()}

        missing = [
            label for key, label in [
                ("DISCORD_BOT_TOKEN", "Discord Bot Token"),
                ("SPOTIFY_CLIENT_ID", "Spotify Client ID"),
                ("SPOTIFY_CLIENT_SECRET", "Spotify Client Secret"),
            ] if not values[key]
        ]
        if missing:
            messagebox.showwarning(
                "SpotiBot", f"These are required before continuing:\n\n" + "\n".join(f"• {m}" for m in missing),
                parent=self,
            )
            return

        if not values["HF_TOKEN"]:
            proceed = messagebox.askyesno(
                "SpotiBot",
                "You didn't enter a Hugging Face token. This is optional, but without one, "
                "downloading the AI model and Whisper speech-to-text models may be slower "
                "and more likely to hit rate limits.\n\n"
                "You can always add one later from Settings > Credentials.\n\n"
                "Continue without it?",
                parent=self,
            )
            if not proceed:
                return

        _write_env_file(values)

        self.completed = True
        if self._guide_dialog is not None and self._guide_dialog.winfo_exists():
            self._guide_dialog.destroy()
        self.destroy()

    def _on_close(self):
        if messagebox.askyesno(
            "SpotiBot", "Exit setup? SpotiBot needs these credentials to run.", parent=self,
        ):
            self.completed = False
            if self._guide_dialog is not None and self._guide_dialog.winfo_exists():
                self._guide_dialog.destroy()
            self.destroy()


class VramGuideDialog(tk.Toplevel):
    """
    Standalone popup: reference table for what model size is realistic at
    different VRAM levels, with the bracket matching this PC's detected
    VRAM highlighted. Pops up automatically when opening the Hugging Face
    browser unless disabled here (or from Settings > Advanced) - always
    reachable on demand too, via the "Model size guide" link in the
    browser's search screen.
    """

    def __init__(self, parent: tk.Widget, gpus: list):
        super().__init__(parent)
        self.title("Model Size Guide")
        self.geometry("500x560")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        th.apply_dark_titlebar(self)
        self.transient(parent)
        self.grab_set()

        th.label(self, "Choosing a model size", kind="title", bg=th.BG).pack(pady=(20, 4), padx=20, anchor="w")
        th.label(
            self, "Bigger models (more B = billions of parameters) generally give smarter,\n"
                  "more nuanced answers, but need more VRAM and respond more slowly.\n"
                  "Quantization (Q4/Q5/Q6/Q8) trades a little accuracy for a smaller, faster\n"
                  "file at the same parameter count - Q4_K_M is usually the sweet spot.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(padx=20, pady=(0, 14), anchor="w")

        table_outer, table_inner = th.card(self)
        table_outer.pack(fill="x", padx=20)

        current_tier = gpu_detect.guidance_tier_index(gpus)
        for i, (_upper, vram_label, size_label, speed_label) in enumerate(gpu_detect.VRAM_MODEL_GUIDANCE):
            is_current = i == current_tier
            row_bg = "#173323" if is_current else th.SURFACE
            row = tk.Frame(table_inner, bg=row_bg)
            row.pack(fill="x")
            marker = "→ " if is_current else "   "
            th.label(
                row, f"{marker}{vram_label}", kind="secondary", bg=row_bg,
                fg=(th.ACCENT if is_current else th.TEXT_SECONDARY), width=11, anchor="w", font=(th.FONT_MONO, 9),
            ).pack(side="left", padx=(10, 6), pady=7)
            th.label(
                row, size_label, kind="secondary", bg=row_bg, width=30, anchor="w", font=(th.FONT, 9),
            ).pack(side="left")
            th.label(row, speed_label, kind="secondary", bg=row_bg, anchor="w", font=(th.FONT, 9)).pack(
                side="left", padx=(0, 10)
            )

        self.dont_show_var = tk.BooleanVar(value=False)
        th.checkbutton(
            self, "Don't show this automatically when browsing Hugging Face",
            self.dont_show_var, bg=th.BG,
        ).pack(anchor="w", padx=20, pady=(16, 0))

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=20, pady=16)
        th.button(btn_frame, "Got it", self._on_close, kind="primary").pack(side="right")

    def _on_close(self):
        if self.dont_show_var.get():
            s = settings_store.load_settings()
            s["show_vram_guide"] = False
            settings_store.save_settings(s)
        self.grab_release()
        self.destroy()


# File-size filter buckets for the HF browser's file list (upper bound in
# GB; the last entry has no upper bound, shown as "24GB+"). Distinct from
# VRAM_MODEL_GUIDANCE's brackets - this filters raw file sizes you're
# looking at, not a VRAM-based suggestion.
FILE_SIZE_FILTERS = [4, 8, 12, 16, 20, 24, float("inf")]


def _file_size_filter_label(upper_bound: float) -> str:
    return f"<{upper_bound:.0f}GB" if upper_bound != float("inf") else "24GB+"


class HuggingFaceBrowserDialog(tk.Toplevel):
    """
    Search Hugging Face for GGUF models and download one directly, instead
    of hand-typing a repo name and filename. Two-step flow: search results
    (repos) -> file list within a chosen repo (quantizations, filterable
    by size). Sets self.result_path on a successful download; caller still
    needs to Apply & Restart to actually use it.

    Pre-fills the search query and highlights a recommended quantization
    based on detected GPU VRAM (gpu_detect.py) - a suggestion only, never
    picked automatically.
    """

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.result_path = None

        gpus = gpu_detect.detect_gpus()
        self._suggested_query, self._suggested_quant, self._suggestion_reason = gpu_detect.suggest_model(gpus)

        self.title("Browse Hugging Face Models")
        self.geometry("700x740")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        th.apply_dark_titlebar(self)
        self.transient(parent)
        self.grab_set()

        self._gpus = gpus
        self._repo_ids = []
        self._files = []
        self._filtered_results = []  # list of {repo_id, filename, size, downloads} when a search-screen size filter is active
        self._search_size_filter = None  # None = show repo names only (fast); a bound = fetch+flatten matching files
        self._files_size_filter = None  # separate filter state for the per-repo file-list screen
        self._build_search_ui()

        if settings_store.load_settings().get("show_vram_guide", True):
            self.after(150, lambda: VramGuideDialog(self, gpus))

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _build_filter_row(self, parent, active_filter, on_change):
        """Shared 'All / <4GB / <8GB / .../ 24GB+' toggle-button row used
        on both the search screen and the per-repo file list. Returns the
        {bound_or_None: Label} dict so the caller can restyle on change."""
        buttons = {}

        def make_handler(bound):
            return lambda e=None: on_change(bound)

        all_btn = tk.Label(
            parent, text="All", bg=(th.ACCENT if active_filter is None else th.SURFACE_ALT),
            fg=(th.ACCENT_TEXT if active_filter is None else th.TEXT_SECONDARY),
            font=(th.FONT, 8, "bold"), padx=8, pady=3, cursor="hand2",
        )
        all_btn.pack(side="left", padx=2)
        all_btn.bind("<Button-1>", make_handler(None))
        buttons[None] = all_btn

        for bound in FILE_SIZE_FILTERS:
            is_active = bound == active_filter
            btn = tk.Label(
                parent, text=_file_size_filter_label(bound),
                bg=(th.ACCENT if is_active else th.SURFACE_ALT), fg=(th.ACCENT_TEXT if is_active else th.TEXT_SECONDARY),
                font=(th.FONT, 8, "bold"), padx=8, pady=3, cursor="hand2",
            )
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", make_handler(bound))
            buttons[bound] = btn

        return buttons

    def _refresh_filter_buttons(self, buttons, active_filter):
        for bound, btn in buttons.items():
            active = bound == active_filter
            btn.configure(
                bg=th.ACCENT if active else th.SURFACE_ALT,
                fg=th.ACCENT_TEXT if active else th.TEXT_SECONDARY,
            )

    def _build_search_ui(self):
        self._clear()

        th.label(self, "Search Hugging Face for a GGUF model", kind="title", bg=th.BG).pack(
            pady=(20, 4), padx=20, anchor="w"
        )
        th.label(
            self, "Look for repo names ending in '-GGUF'. No account needed, but adding an\n"
                  "HF_TOKEN to your .env avoids rate limits on repeated searches.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(pady=(0, 6), padx=20, anchor="w")

        guide_link = tk.Label(
            self, text="ℹ️ Model size guide", bg=th.BG, fg=th.INFO, font=(th.FONT, 9, "underline"), cursor="hand2",
        )
        guide_link.pack(padx=20, anchor="w", pady=(0, 10))
        guide_link.bind("<Button-1>", lambda e: VramGuideDialog(self, self._gpus))

        # GPU-based suggestion banner, same visual language as the GPU picker's.
        banner_outer = tk.Frame(self, bg=th.ACCENT)
        banner_outer.pack(fill="x", padx=20, pady=(0, 12))
        banner_inner = tk.Frame(banner_outer, bg=th.SURFACE)
        banner_inner.pack(fill="x", padx=1, pady=1)
        th.label(
            banner_inner, f"💡 Suggested for your PC: {self._suggested_quant} quantization",
            kind="body", bg=th.SURFACE, fg=th.ACCENT, font=(th.FONT, 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))
        th.label(
            banner_inner, self._suggestion_reason, kind="secondary", bg=th.SURFACE, justify="left", wraplength=590,
        ).pack(anchor="w", padx=14, pady=(0, 10))

        search_frame = th.frame(self)
        search_frame.pack(fill="x", padx=20)
        self.search_entry = th.entry(search_frame)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=4)
        self.search_entry.insert(0, self._suggested_query)
        self.search_entry.bind("<Return>", lambda e: self._do_search())
        th.button(search_frame, "Search", self._do_search, kind="secondary").pack(side="left")

        filter_row = th.frame(self)
        filter_row.pack(fill="x", padx=20, pady=(10, 0))
        th.label(filter_row, "Filter by size:", kind="secondary", bg=th.BG).pack(side="left", padx=(0, 8))

        def on_filter_change(bound):
            self._search_size_filter = bound
            self._refresh_filter_buttons(self._search_filter_buttons, bound)
            self._do_search()

        self._search_filter_buttons = self._build_filter_row(filter_row, self._search_size_filter, on_filter_change)
        th.label(
            self, "A size filter checks actual quant files across the top matching repos and\n"
                  "shows those directly (slower than plain search - fetches each repo's file list).",
            kind="muted", bg=th.BG, justify="left",
        ).pack(padx=20, pady=(4, 0), anchor="w")

        list_container = th.frame(self)
        list_container.pack(fill="both", expand=True, padx=20, pady=10)
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        self.results_listbox = th.listbox(list_container, yscrollcommand=scrollbar.set)
        self.results_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.results_listbox.yview)
        self.results_listbox.bind("<Double-Button-1>", lambda e: self._select_result())

        self.status_label = th.label(self, "", kind="secondary", bg=th.BG)
        self.status_label.pack(padx=20, anchor="w")

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=20, pady=16)
        self.select_btn = th.button(btn_frame, "View Files ›", self._select_result, kind="primary")
        self.select_btn.pack(side="right")
        th.button(btn_frame, "Cancel", self._on_cancel, kind="ghost").pack(side="left")

        self._do_search()  # auto-search with the suggested/default query on open

    def _do_search(self):
        query = self.search_entry.get().strip()
        self.results_listbox.delete(0, "end")
        self._repo_ids = []
        self._filtered_results = []

        try:
            results = _hf_search_models(query)
        except Exception as e:
            self.status_label.configure(text=f"Search failed: {e}")
            return

        if not results:
            self.status_label.configure(text="No results - try different search terms.")
            return

        if self._search_size_filter is None:
            self.select_btn.configure(text="View Files ›")
            for r in results:
                repo_id = r.get("id", "?")
                downloads = r.get("downloads", 0)
                self._repo_ids.append(repo_id)
                self.results_listbox.insert("end", f"{repo_id}   ({downloads:,} downloads)")
            self.status_label.configure(
                text=f"{len(results)} result(s) - double-click one, or select and click 'View Files'"
            )
            return

        # Size filter active: fetch each top repo's file list and flatten
        # matching files into the results directly, sorted by (repo)
        # downloads - real per-file API calls, so bounded to a sane count.
        self.select_btn.configure(text="Use This File ›")
        bound = self._search_size_filter
        idx = FILE_SIZE_FILTERS.index(bound)
        lower_bound = FILE_SIZE_FILTERS[idx - 1] if idx > 0 else 0

        TOP_N = 12
        top_results = results[:TOP_N]
        quant_needle = self._suggested_quant.lower()
        matches = []
        for i, r in enumerate(top_results, 1):
            repo_id = r.get("id", "?")
            downloads = r.get("downloads", 0)
            self.status_label.configure(text=f"Checking files ({i}/{len(top_results)}): {repo_id}...")
            self.update()
            try:
                files = _hf_list_gguf_files(repo_id)
            except Exception:
                continue
            for f in files:
                size = f.get("size")
                if size and lower_bound <= size / 1e9 < bound:
                    matches.append({
                        "repo_id": repo_id, "filename": f["path"], "size": size, "downloads": downloads,
                    })

        matches.sort(key=lambda m: m["downloads"], reverse=True)
        self._filtered_results = matches

        if not matches:
            self.status_label.configure(
                text=f"No files in {_file_size_filter_label(bound)} range among top {len(top_results)} repos checked."
            )
            return

        for i, m in enumerate(matches):
            size_str = f"{m['size'] / 1e9:.2f} GB"
            is_recommended = quant_needle in m["filename"].lower()
            prefix = "⭐ " if is_recommended else "   "
            self.results_listbox.insert(
                "end", f"{prefix}{m['repo_id']} / {m['filename']}   ({size_str}, {m['downloads']:,} dl)"
            )
            if is_recommended:
                self.results_listbox.itemconfig(i, bg="#173323", fg=th.ACCENT)

        self.status_label.configure(
            text=f"{len(matches)} file(s) in {_file_size_filter_label(bound)} range, sorted by downloads "
                 f"(top {len(top_results)} repos checked)"
        )

    def _select_result(self):
        sel = self.results_listbox.curselection()
        if not sel:
            return
        if self._search_size_filter is not None:
            if not self._filtered_results:
                return
            match = self._filtered_results[sel[0]]
            self._start_download(match["repo_id"], match["filename"])
        else:
            self._build_files_ui(self._repo_ids[sel[0]])

    def _build_files_ui(self, repo_id: str):
        self._clear()
        self._files_size_filter = None
        self._current_repo_id = repo_id

        th.label(self, repo_id, kind="title", bg=th.BG, wraplength=610).pack(pady=(20, 4), padx=20, anchor="w")
        th.label(
            self, f"⭐ marks files matching the suggested {self._suggested_quant} quantization for your PC.\n"
                  "Smaller (Q2/Q3) is faster and less accurate, larger (Q6/Q8) is the opposite.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(pady=(0, 10), padx=20, anchor="w")

        filter_row = th.frame(self)
        filter_row.pack(fill="x", padx=20, pady=(0, 8))
        th.label(filter_row, "Filter by size:", kind="secondary", bg=th.BG).pack(side="left", padx=(0, 8))

        def on_filter_change(bound):
            self._files_size_filter = bound
            self._refresh_filter_buttons(self._files_filter_buttons, bound)
            self._render_files_list()

        self._files_filter_buttons = self._build_filter_row(filter_row, self._files_size_filter, on_filter_change)

        list_container = th.frame(self)
        list_container.pack(fill="both", expand=True, padx=20)
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        self.files_listbox = th.listbox(list_container, yscrollcommand=scrollbar.set)
        self.files_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.files_listbox.yview)

        self.files_status = th.label(self, "Loading files...", kind="secondary", bg=th.BG)
        self.files_status.pack(padx=20, anchor="w", pady=(6, 0))
        self.update()

        try:
            self._files = _hf_list_gguf_files(repo_id)
        except Exception as e:
            self.files_status.configure(text=f"Failed to list files: {e}")
            self._files = []

        self._render_files_list()

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=20, pady=16)
        th.button(
            btn_frame, "Download & Use This Model",
            lambda: self._download_selected_from_files_list(repo_id), kind="primary",
        ).pack(side="right")
        th.button(btn_frame, "‹ Back to Search", self._build_search_ui, kind="secondary").pack(side="left")
        th.button(btn_frame, "Cancel", self._on_cancel, kind="ghost").pack(side="left", padx=(8, 0))

    def _render_files_list(self):
        """Repopulates the file listbox from self._files, applying the
        active size filter (if any) and the recommended-quant highlight."""
        self.files_listbox.delete(0, "end")

        visible = self._files
        if self._files_size_filter is not None:
            idx = FILE_SIZE_FILTERS.index(self._files_size_filter)
            lower_bound = FILE_SIZE_FILTERS[idx - 1] if idx > 0 else 0
            upper_bound = self._files_size_filter
            visible = [
                f for f in self._files
                if f.get("size") and lower_bound <= f["size"] / 1e9 < upper_bound
            ]

        if not self._files:
            self.files_status.configure(text="No .gguf files found in this repo.")
        elif not visible:
            self.files_status.configure(text=f"0 files in this size range (of {len(self._files)} total).")
        else:
            self.files_status.configure(text=f"{len(visible)} of {len(self._files)} file(s) shown")

        quant_needle = self._suggested_quant.lower()
        self._visible_files = visible
        for idx, f in enumerate(visible):
            size = f.get("size")
            size_str = f"{size / 1e9:.2f} GB" if size else "size unknown"
            is_recommended = quant_needle in f["path"].lower()
            prefix = "⭐ " if is_recommended else "   "
            self.files_listbox.insert("end", f"{prefix}{f['path']}   ({size_str})")
            if is_recommended:
                self.files_listbox.itemconfig(idx, bg="#173323", fg=th.ACCENT)

    def _download_selected_from_files_list(self, repo_id: str):
        sel = self.files_listbox.curselection()
        if not sel:
            messagebox.showwarning("SpotiBot", "Select a file first.", parent=self)
            return
        filename = self._visible_files[sel[0]]["path"]
        self._start_download(repo_id, filename)

    def _start_download(self, repo_id: str, filename: str):
        self._clear()
        th.label(self, f"Downloading {filename}...", kind="title", bg=th.BG, wraplength=610).pack(
            pady=(20, 10), padx=20, anchor="w"
        )
        card_outer, card_inner = th.card(self)
        card_outer.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        progress_text = scrolledtext.ScrolledText(
            card_inner, bg=th.SURFACE, fg=th.TEXT_SECONDARY, font=(th.FONT_MONO, 9),
            wrap="word", state="disabled", borderwidth=0, highlightthickness=0, padx=10, pady=8,
        )
        progress_text.pack(fill="both", expand=True)

        def log(msg):
            progress_text.configure(state="normal")
            progress_text.insert("end", msg)
            progress_text.see("end")
            progress_text.configure(state="disabled")
            self.update()

        try:
            os.makedirs("models", exist_ok=True)
            dest_path = os.path.join("models", os.path.basename(filename))
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            _download_file(url, dest_path, log, headers=_hf_headers())
            log(f"\n✅ Done: {dest_path}\n")
            self.result_path = dest_path
        except Exception as e:
            log(f"\n[FATAL] Download failed: {e}\n")
            self.result_path = None
            return  # keep the dialog open so the error stays visible

        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        self.result_path = None
        self.grab_release()
        self.destroy()


class SettingsDialog(tk.Toplevel):
    """
    Settings window with categorized tabs, matching the shape of any other
    app's settings menu: Voice (wake words), AI Model (model/GPU backend),
    Credentials (.env values), Advanced (debug logging + tuning knobs).

    Cheap/non-disruptive changes (wake words, debug toggle, tuning values)
    all live under one "Save" button and apply live via
    bot_module.reload_settings() - no restart needed. Disruptive actions
    (restarting the AI model, re-running GPU setup, uninstalling a
    backend) are separate explicit buttons in the AI Model tab, since
    those shouldn't fire just because you tweaked a wake word.
    """

    def __init__(self, parent: tk.Tk, app: "SpotiBotGUI"):
        super().__init__(parent)
        self.app = app
        self.settings = settings_store.load_settings()

        self.title("SpotiBot Settings")
        self.geometry("640x580")
        self.configure(bg=th.BG)
        self.transient(parent)
        th.apply_dark_titlebar(self)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=(12, 0))

        self._build_voice_tab(notebook)
        self._build_ai_model_tab(notebook)
        self._build_credentials_tab(notebook)
        self._build_advanced_tab(notebook)

        btn_frame = th.frame(self)
        btn_frame.pack(fill="x", padx=12, pady=12)
        th.button(btn_frame, "Save", self._save_all, kind="primary").pack(side="right")
        th.button(btn_frame, "Close", self.destroy, kind="secondary").pack(side="right", padx=(0, 8))

    def _section_label(self, parent, text):
        th.label(parent, text, kind="heading", bg=th.BG).pack(anchor="w", pady=(16, 4), padx=12)

    # --- Voice tab ---

    def _build_voice_tab(self, notebook):
        frame = th.frame(notebook)
        notebook.add(frame, text="🎤 Voice")

        th.label(frame, "Wake words", kind="heading", bg=th.BG).pack(anchor="w", padx=12, pady=(14, 2))
        th.label(
            frame, "Phrases that trigger a voice command (e.g. 'hey bot'). Add variants if\n"
                   "the AI keeps mishearing your wake word - check the console log after a\n"
                   "missed command to see what it actually transcribed.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 10))

        list_container = th.frame(frame)
        list_container.pack(fill="both", expand=True, padx=12)
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        self.wake_listbox = th.listbox(list_container, height=6, yscrollcommand=scrollbar.set, font=(th.FONT_MONO, 10))
        self.wake_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.wake_listbox.yview)
        for phrase in self.settings["wake_phrases"]:
            self.wake_listbox.insert("end", phrase)

        add_frame = th.frame(frame)
        add_frame.pack(fill="x", padx=12, pady=10)
        self.wake_entry = th.entry(add_frame)
        self.wake_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=4)
        self.wake_entry.bind("<Return>", lambda e: self._add_wake_phrase())
        th.button(add_frame, "Add", self._add_wake_phrase, kind="secondary").pack(side="left")

        btn_row = th.frame(frame)
        btn_row.pack(fill="x", padx=12, pady=(0, 4))
        th.button(btn_row, "Remove Selected", self._remove_wake_phrase, kind="secondary").pack(side="left")
        th.button(btn_row, "Restore Defaults", self._restore_wake_defaults, kind="secondary").pack(side="left", padx=(8, 0))

        self._section_label(frame, "Speech-to-text model")
        whisper_row = th.frame(frame)
        whisper_row.pack(fill="x", padx=12, pady=(0, 14))
        self.whisper_model_var = tk.StringVar(value=self.settings["whisper_model"])
        self.whisper_device_var = tk.StringVar(value=self.settings.get("whisper_device", "cpu"))
        current_label = next(
            (m[1] for m in WHISPER_MODELS if m[0] == self.whisper_model_var.get()), self.whisper_model_var.get(),
        )
        device_label = "GPU" if self.whisper_device_var.get() == "cuda" else "CPU"
        self.whisper_label_var = tk.StringVar(value=f"Current: {current_label}  ({device_label})")
        th.label(whisper_row, textvariable=self.whisper_label_var, kind="body", bg=th.BG).pack(side="left")
        th.button(whisper_row, "Change...", self._change_whisper_model, kind="secondary").pack(side="right")

    def _change_whisper_model(self):
        dialog = WhisperModelDialog(self, self.whisper_model_var.get(), self.whisper_device_var.get())
        self.wait_window(dialog)
        if not dialog.result_key:
            return
        if dialog.result_key == self.whisper_model_var.get() and dialog.result_device == self.whisper_device_var.get():
            return  # nothing actually changed

        self.whisper_model_var.set(dialog.result_key)
        self.whisper_device_var.set(dialog.result_device)
        new_label = next((m[1] for m in WHISPER_MODELS if m[0] == dialog.result_key), dialog.result_key)
        device_label = "GPU" if dialog.result_device == "cuda" else "CPU"
        self.whisper_label_var.set(f"Current: {new_label}  ({device_label})")

        s = settings_store.load_settings()
        s["whisper_model"] = dialog.result_key
        s["whisper_device"] = dialog.result_device
        settings_store.save_settings(s)

        if bot_module:
            bot_module.reload_whisper_model(dialog.result_key, dialog.result_device)
            messagebox.showinfo(
                "SpotiBot",
                f"Loading '{new_label}' on {device_label} now - watch the Bot tab for progress "
                f"(larger models download the first time they're used). If GPU loading fails, "
                f"check the Bot tab log and switch back to CPU here if needed.",
                parent=self,
            )
        else:
            messagebox.showinfo(
                "SpotiBot", f"Saved - will use '{new_label}' ({device_label}) once the bot finishes starting.",
                parent=self,
            )

    def _add_wake_phrase(self):
        phrase = self.wake_entry.get().strip().lower()
        if not phrase:
            return
        existing = [self.wake_listbox.get(i) for i in range(self.wake_listbox.size())]
        if phrase not in existing:
            self.wake_listbox.insert("end", phrase)
        self.wake_entry.delete(0, "end")

    def _remove_wake_phrase(self):
        for i in reversed(self.wake_listbox.curselection()):
            self.wake_listbox.delete(i)

    def _restore_wake_defaults(self):
        self.wake_listbox.delete(0, "end")
        for phrase in settings_store.DEFAULT_WAKE_PHRASES:
            self.wake_listbox.insert("end", phrase)

    # --- AI Model tab ---

    def _build_ai_model_tab(self, notebook):
        frame = th.frame(notebook)
        notebook.add(frame, text="🧠 AI Model")

        self._section_label(frame, "Model")
        self.model_path_var = tk.StringVar(value=LLAMA_MODEL_PATH)
        card_outer, card_inner = th.card(frame)
        card_outer.pack(fill="x", padx=12)
        th.label(
            card_inner, textvariable=self.model_path_var, kind="secondary", bg=th.SURFACE,
            fg=th.TEXT_SECONDARY, font=(th.FONT_MONO, 9), anchor="w", wraplength=560, justify="left",
            padx=8, pady=8,
        ).pack(fill="x")

        btn_row = th.frame(frame)
        btn_row.pack(fill="x", padx=12, pady=(8, 0))
        th.button(btn_row, "Browse local file...", self._browse_model, kind="secondary").pack(side="left")
        th.button(btn_row, "Browse Hugging Face...", self._browse_huggingface, kind="secondary").pack(side="left", padx=(8, 0))

        self._section_label(frame, "Performance")
        perf_frame = th.frame(frame)
        perf_frame.pack(fill="x", padx=12)

        th.label(perf_frame, "GPU layers (-ngl):", kind="body", bg=th.BG).grid(row=0, column=0, sticky="w", pady=3)
        self.ngl_var = tk.StringVar(value=LLAMA_NGL)
        th.entry(perf_frame, textvariable=self.ngl_var, width=8).grid(row=0, column=1, sticky="w", padx=8, ipady=3)
        th.label(
            perf_frame, "(99 = offload everything to GPU, 0 = CPU only)", kind="muted", bg=th.BG,
        ).grid(row=0, column=2, sticky="w")

        th.label(perf_frame, "Server port:", kind="body", bg=th.BG).grid(row=1, column=0, sticky="w", pady=3)
        self.port_var = tk.StringVar(value=LLAMA_SERVER_PORT)
        th.entry(perf_frame, textvariable=self.port_var, width=8).grid(row=1, column=1, sticky="w", padx=8, ipady=3)

        th.button(frame, "Apply & Restart AI Model", self._apply_model_settings, kind="primary").pack(
            anchor="w", padx=12, pady=(12, 0)
        )

        self._section_label(frame, "GPU backend (llama.cpp build)")
        self.backend_label_var = tk.StringVar(value=_read_backend_info())
        th.label(
            frame, textvariable=self.backend_label_var, kind="secondary", bg=th.BG,
            justify="left", anchor="w", wraplength=560,
        ).pack(fill="x", padx=12)

        btn_row2 = th.frame(frame)
        btn_row2.pack(fill="x", padx=12, pady=(10, 0))
        th.button(btn_row2, "Re-run GPU/backend setup...", self._rerun_gpu_setup, kind="secondary").pack(side="left")
        th.button(btn_row2, "Uninstall current backend", self._uninstall_backend, kind="danger").pack(side="left", padx=(8, 0))

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select a GGUF model file",
            filetypes=[("GGUF model files", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

    def _browse_huggingface(self):
        dialog = HuggingFaceBrowserDialog(self)
        self.wait_window(dialog)
        if dialog.result_path:
            self.model_path_var.set(dialog.result_path)

    def _apply_model_settings(self):
        global LLAMA_MODEL_PATH, LLAMA_NGL, LLAMA_SERVER_PORT
        LLAMA_MODEL_PATH = self.model_path_var.get().strip() or DEFAULT_MODEL_PATH
        LLAMA_NGL = self.ngl_var.get().strip() or "99"
        LLAMA_SERVER_PORT = self.port_var.get().strip() or "8080"

        s = settings_store.load_settings()
        s["llama_model_path"] = LLAMA_MODEL_PATH
        s["llama_ngl"] = LLAMA_NGL
        s["llama_server_port"] = LLAMA_SERVER_PORT
        settings_store.save_settings(s)

        self.app.restart_llama_server()
        messagebox.showinfo("SpotiBot", "AI model settings applied - restarting the server now.", parent=self)

    def _rerun_gpu_setup(self):
        global LLAMA_SERVER_PATH
        if os.path.exists(LLAMA_CPP_CONFIG_PATH):
            try:
                os.remove(LLAMA_CPP_CONFIG_PATH)
            except OSError:
                pass

        dialog = GpuPickerDialog(self)
        self.wait_window(dialog)
        if dialog.result_path:
            LLAMA_SERVER_PATH = dialog.result_path
            self.backend_label_var.set(_read_backend_info())
            self.app.restart_llama_server()
            messagebox.showinfo("SpotiBot", f"Now using: {LLAMA_SERVER_PATH}", parent=self)

    def _uninstall_backend(self):
        if not os.path.exists(LLAMA_CPP_CONFIG_PATH):
            messagebox.showinfo("SpotiBot", "No installed backend found to uninstall.", parent=self)
            return
        if not messagebox.askyesno(
            "SpotiBot",
            "This stops the AI model and deletes the installed llama.cpp backend from disk. Continue?",
            parent=self,
        ):
            return

        key = None
        try:
            with open(LLAMA_CPP_CONFIG_PATH) as f:
                key = json.load(f).get("backend_key")
        except (OSError, json.JSONDecodeError):
            pass

        # Stop the server first - Windows won't let us delete the exe
        # while a running process still has it open.
        if self.app.llama_process and self.app.llama_process.poll() is None:
            try:
                self.app.llama_process.terminate()
                self.app.llama_process.wait(timeout=5)
            except Exception:
                pass

        if key:
            shutil.rmtree(os.path.join(LLAMA_CPP_INSTALL_DIR, key), ignore_errors=True)
        try:
            os.remove(LLAMA_CPP_CONFIG_PATH)
        except OSError:
            pass

        self.backend_label_var.set(_read_backend_info())
        messagebox.showinfo(
            "SpotiBot",
            "Backend uninstalled. The AI model is stopped until you set one up again "
            "('Re-run GPU/backend setup...').",
            parent=self,
        )

    # --- Credentials tab ---

    def _build_credentials_tab(self, notebook):
        frame = th.frame(notebook)
        notebook.add(frame, text="🔑 Credentials")

        th.label(
            frame, "Stored in your .env file. Restart SpotiBot after changing Discord/Spotify\n"
                   "values. HF_TOKEN takes effect immediately (used for Hugging Face searches\n"
                   "in Settings > AI Model) - no restart needed for that one.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(anchor="w", padx=12, pady=(14, 12))

        env = _read_env_file()
        self.cred_vars = {}
        for key, display in [
            ("DISCORD_BOT_TOKEN", "Discord Bot Token"),
            ("SPOTIFY_CLIENT_ID", "Spotify Client ID"),
            ("SPOTIFY_CLIENT_SECRET", "Spotify Client Secret"),
            ("HF_TOKEN", "Hugging Face Token (optional)"),
        ]:
            row = th.frame(frame)
            row.pack(fill="x", padx=12, pady=5)
            th.label(row, display, kind="body", bg=th.BG, width=20, anchor="w").pack(side="left")

            var = tk.StringVar(value=env.get(key, ""))
            self.cred_vars[key] = var
            entry = th.entry(row, textvariable=var, show="*", font=(th.FONT_MONO, 9))
            entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=3)

            def make_toggle(e=entry):
                def toggle():
                    e.configure(show="" if e.cget("show") == "*" else "*")
                return toggle

            th.button(row, "👁", make_toggle(), kind="secondary", width=3, padx=4).pack(side="left")

        th.button(frame, "Save Credentials", self._save_credentials, kind="primary").pack(
            anchor="w", padx=12, pady=14
        )

    def _save_credentials(self):
        updates = {k: v.get().strip() for k, v in self.cred_vars.items()}
        _write_env_file(updates)
        messagebox.showinfo("SpotiBot", "Credentials saved. Restart SpotiBot for the changes to take effect.", parent=self)

    # --- Advanced tab ---

    def _build_advanced_tab(self, notebook):
        frame = th.frame(notebook)
        notebook.add(frame, text="⚙ Advanced")

        self.debug_var = tk.BooleanVar(value=self.settings["debug_voice"])
        th.checkbutton(
            frame, "Enable verbose debug logging (per-packet voice info, watchdog heartbeat)",
            self.debug_var, bg=th.BG,
        ).pack(anchor="w", padx=12, pady=(16, 2))
        th.label(
            frame, "Off by default - only turn this on if you're debugging a voice issue, it's noisy.",
            kind="muted", bg=th.BG, justify="left",
        ).pack(anchor="w", padx=12)

        self.show_vram_guide_var = tk.BooleanVar(value=self.settings.get("show_vram_guide", True))
        th.checkbutton(
            frame, "Show model size guide when browsing Hugging Face",
            self.show_vram_guide_var, bg=th.BG,
        ).pack(anchor="w", padx=12, pady=(10, 0))

        tuning_frame = th.frame(frame)
        tuning_frame.pack(fill="x", padx=12, pady=(16, 0))

        th.label(tuning_frame, "Silence timeout (seconds):", kind="body", bg=th.BG).grid(row=0, column=0, sticky="w", pady=4)
        self.idle_timeout_var = tk.StringVar(value=str(self.settings["idle_timeout"]))
        th.entry(tuning_frame, textvariable=self.idle_timeout_var, width=8).grid(row=0, column=1, sticky="w", padx=8, ipady=3)
        th.label(
            tuning_frame, "How long to wait after you stop talking before processing", kind="muted", bg=th.BG,
        ).grid(row=0, column=2, sticky="w")

        th.label(tuning_frame, "Spotify match confidence (0-1):", kind="body", bg=th.BG).grid(row=1, column=0, sticky="w", pady=4)
        self.spotify_threshold_var = tk.StringVar(value=str(self.settings["spotify_match_threshold"]))
        th.entry(tuning_frame, textvariable=self.spotify_threshold_var, width=8).grid(row=1, column=1, sticky="w", padx=8, ipady=3)
        th.label(
            tuning_frame, "Higher = stricter before trusting a Spotify search match", kind="muted", bg=th.BG,
        ).grid(row=1, column=2, sticky="w")

        self._section_label(frame, "Maintenance")
        btn_col = th.frame(frame)
        btn_col.pack(fill="x", padx=12)
        th.button(btn_col, "Clear debug files (captured audio, ffmpeg log)", self._clear_debug_files, kind="secondary").pack(anchor="w")
        th.button(btn_col, "Reset all settings to defaults", self._reset_all_settings, kind="danger").pack(anchor="w", pady=(8, 0))

    def _clear_debug_files(self):
        import glob
        removed = 0
        for pattern in ["debug_last_capture.wav", "ffmpeg_debug.log", "raw_discord_audio_*.wav"]:
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        messagebox.showinfo("SpotiBot", f"Removed {removed} debug file(s).", parent=self)

    def _reset_all_settings(self):
        if not messagebox.askyesno(
            "SpotiBot", "Reset ALL settings (wake words, debug, tuning) to defaults?", parent=self,
        ):
            return
        settings_store.save_settings(dict(settings_store.DEFAULTS))
        if bot_module:
            bot_module.reload_settings()
        parent = self.master
        app = self.app
        self.destroy()
        SettingsDialog(parent, app)  # reopen fresh with defaults loaded

    # --- Save all (Voice/Advanced tabs - cheap, live-reloadable settings) ---

    def _save_all(self):
        wake_phrases = [self.wake_listbox.get(i) for i in range(self.wake_listbox.size())]
        if not wake_phrases:
            messagebox.showwarning("SpotiBot", "You need at least one wake phrase.", parent=self)
            return

        try:
            idle_timeout = float(self.idle_timeout_var.get())
            spotify_threshold = float(self.spotify_threshold_var.get())
        except ValueError:
            messagebox.showwarning(
                "SpotiBot", "Silence timeout and Spotify match confidence must be numbers.", parent=self,
            )
            return

        s = settings_store.load_settings()
        s["wake_phrases"] = wake_phrases
        s["debug_voice"] = bool(self.debug_var.get())
        s["show_vram_guide"] = bool(self.show_vram_guide_var.get())
        s["idle_timeout"] = idle_timeout
        s["spotify_match_threshold"] = spotify_threshold
        settings_store.save_settings(s)

        if bot_module:
            bot_module.reload_settings()

        messagebox.showinfo("SpotiBot", "Settings saved and applied.", parent=self)


class GradientCanvas(tk.Canvas):
    """
    Subtle dark blue/teal/green gradient, evoking the app's accent palette
    (matches the reference aurora-style image, at a much darker/more
    subtle level appropriate for a background rather than the focal
    content). Tkinter widgets are fully opaque - there's no way to make
    something show "through" a Frame - so this is only visible in the
    margins/gaps around the solid UI panels placed on top of it, the same
    visual language most gradient-background app designs actually use.
    Redraws on resize.
    """

    STOPS = ["#0a0e14", "#0d2b28", "#123a2e", "#0a0e14"]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, highlightthickness=0, bd=0, **kwargs)
        self.bind("<Configure>", self._redraw)

    def _redraw(self, event=None):
        self.delete("gradient")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return

        steps = 120
        stops = self.STOPS
        n = len(stops) - 1
        for i in range(steps):
            t = i / steps
            seg = min(int(t * n), n - 1)
            local_t = (t * n) - seg
            c1 = self.winfo_rgb(stops[seg])
            c2 = self.winfo_rgb(stops[seg + 1])
            r = int(c1[0] + (c2[0] - c1[0]) * local_t) >> 8
            g = int(c1[1] + (c2[1] - c1[1]) * local_t) >> 8
            b = int(c1[2] + (c2[2] - c1[2]) * local_t) >> 8
            color = f"#{r:02x}{g:02x}{b:02x}"
            y0 = int(h * i / steps)
            y1 = int(h * (i + 1) / steps) + 1
            self.create_rectangle(0, y0, w, y1, fill=color, outline=color, tags="gradient")
        self.tag_lower("gradient")


class QueueSidebar:
    """
    Left panel showing the live song queue. Reads directly from bot_module's
    module-level state (bot.now_playing / bot.song_queue) - both processes
    are actually the same Python process (bot.py runs in a background
    thread, not a subprocess), so this is just reading shared memory, no
    IPC needed. Call refresh() periodically from the main thread.
    """

    def __init__(self, parent):
        outer, inner = th.card(parent)
        outer.pack(fill="both", expand=True)
        self.frame = inner

        th.label(inner, "QUEUE", kind="secondary", bg=th.SURFACE, font=(th.FONT, 9, "bold")).pack(
            anchor="w", padx=14, pady=(14, 8)
        )

        now_playing_outer = tk.Frame(inner, bg=th.ACCENT)
        now_playing_outer.pack(fill="x", padx=14, pady=(0, 14))
        now_playing_inner = tk.Frame(now_playing_outer, bg=th.SURFACE_ALT)
        now_playing_inner.pack(fill="x", padx=1, pady=1)
        th.label(
            now_playing_inner, "NOW PLAYING", kind="muted", bg=th.SURFACE_ALT, fg=th.ACCENT, font=(th.FONT, 8, "bold"),
        ).pack(anchor="w", padx=10, pady=(8, 0))
        self.now_playing_var = tk.StringVar(value="Nothing playing")
        th.label(
            now_playing_inner, textvariable=self.now_playing_var, kind="body", bg=th.SURFACE_ALT,
            wraplength=190, justify="left",
        ).pack(anchor="w", padx=10, pady=(2, 10))

        th.label(inner, "UP NEXT", kind="secondary", bg=th.SURFACE, font=(th.FONT, 9, "bold")).pack(
            anchor="w", padx=14, pady=(0, 6)
        )

        list_container = th.frame(inner, bg=th.SURFACE)
        list_container.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        self.queue_listbox = th.listbox(
            list_container, font=(th.FONT, 9), yscrollcommand=scrollbar.set, activestyle="none",
        )
        self.queue_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.queue_listbox.yview)

        self._last_rendered = None  # avoid needless listbox rebuilds every poll tick

    def refresh(self, bot_module):
        if bot_module is None:
            self.now_playing_var.set("Bot still starting...")
            return

        now_playing = getattr(bot_module, "now_playing", None)
        self.now_playing_var.set(now_playing or "Nothing playing")

        current_queue = list(getattr(bot_module, "song_queue", []))
        if current_queue == self._last_rendered:
            return  # nothing changed - don't touch the widget (avoids scroll-position jumps)
        self._last_rendered = current_queue

        self.queue_listbox.delete(0, "end")
        if not current_queue:
            self.queue_listbox.insert("end", "Queue is empty")
        else:
            for i, title in enumerate(current_queue, 1):
                self.queue_listbox.insert("end", f"{i}. {title}")


class VoiceMembersSidebar:
    """
    Right panel: everyone currently in the bot's voice channel, with a
    checkbox controlling whether each person's voice is authorized to
    trigger the bot. Reads/writes bot_module's active sink state directly
    (same process, no IPC) - changes apply on the very next audio packet,
    live mid-session, not just at !listen time.
    """

    def __init__(self, parent):
        outer, inner = th.card(parent)
        outer.pack(fill="both", expand=True)
        self.frame = inner

        th.label(inner, "VOICE CHANNEL", kind="secondary", bg=th.SURFACE, font=(th.FONT, 9, "bold")).pack(
            anchor="w", padx=14, pady=(14, 6)
        )
        th.label(
            inner, "Who the bot listens to for wake words.\nChanges apply instantly.",
            kind="muted", bg=th.SURFACE, justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        preset_row = th.frame(inner, bg=th.SURFACE)
        preset_row.pack(fill="x", padx=14, pady=(0, 10))
        th.button(preset_row, "All", self._select_all, kind="secondary", padx=10, pady=3, size=8).pack(side="left")
        th.button(preset_row, "None", self._select_none, kind="secondary", padx=10, pady=3, size=8).pack(
            side="left", padx=(6, 0)
        )

        list_container = th.frame(inner, bg=th.SURFACE)
        list_container.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        self.list_canvas = tk.Canvas(
            list_container, bg=th.SURFACE, highlightthickness=0, yscrollcommand=scrollbar.set,
        )
        self.list_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.list_canvas.yview)

        self.checkbox_frame = th.frame(self.list_canvas, bg=th.SURFACE)
        self._canvas_window = self.list_canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")
        self.checkbox_frame.bind(
            "<Configure>", lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all"))
        )
        self.list_canvas.bind(
            "<Configure>", lambda e: self.list_canvas.itemconfig(self._canvas_window, width=e.width)
        )

        self.empty_label = th.label(
            self.checkbox_frame, "Not connected to a voice channel.", kind="muted", bg=th.SURFACE,
        )
        self.empty_label.pack(anchor="w", pady=8)

        self._member_vars = {}      # user_id -> tk.BooleanVar
        self._member_rows = {}      # user_id -> Checkbutton widget
        self._listen_all = True     # matches bot.py's sink default (authorized_user_ids=None)
        self._suppress_callback = False
        self._last_member_ids = None  # avoid needless rebuilds when membership hasn't actually changed
        self._bot_module_ref = None   # stashed so the All/None button handlers (no args) can reach it

    def _current_selected_ids(self):
        return {uid for uid, var in self._member_vars.items() if var.get()}

    def _send_state(self):
        if self._bot_module_ref is None:
            return
        if self._listen_all:
            self._bot_module_ref.set_authorized_listeners(None)
        else:
            self._bot_module_ref.set_authorized_listeners(self._current_selected_ids())

    def _on_checkbox_toggle(self):
        if self._suppress_callback:
            return
        self._listen_all = False  # a manual per-person toggle always exits "everyone" mode
        self._send_state()

    def _select_all(self):
        self._suppress_callback = True
        for var in self._member_vars.values():
            var.set(True)
        self._suppress_callback = False
        self._listen_all = True
        self._send_state()

    def _select_none(self):
        self._suppress_callback = True
        for var in self._member_vars.values():
            var.set(False)
        self._suppress_callback = False
        self._listen_all = False
        self._send_state()

    def refresh(self, bot_module):
        self._bot_module_ref = bot_module
        if bot_module is None:
            return

        members = bot_module.get_active_voice_channel_members()
        if members is None:
            # Fetch failed/timed out this tick (routinely happens while
            # Whisper is mid-transcription, briefly starving the event
            # loop) - keep showing the last-known-good state instead of
            # flickering to "empty" every time, then repopulating 150ms
            # later once the next poll succeeds.
            return

        member_ids = tuple(sorted(uid for uid, _ in members))

        if member_ids == self._last_member_ids:
            return  # membership unchanged - don't touch widgets (avoids scroll-position jumps)
        self._last_member_ids = member_ids

        current_ids = {uid for uid, _ in members}
        for uid in list(self._member_vars.keys()):
            if uid not in current_ids:
                self._member_rows.pop(uid).destroy()
                self._member_vars.pop(uid)

        if not members:
            self.empty_label.pack(anchor="w", pady=8)
        else:
            self.empty_label.pack_forget()

        for uid, name in members:
            if uid in self._member_vars:
                continue  # already have a row for them
            # New members default to checked only in "listen to everyone"
            # mode - if someone had narrowed the list down deliberately,
            # a person who just joined shouldn't silently become
            # authorized without an explicit check.
            var = tk.BooleanVar(value=self._listen_all)
            self._member_vars[uid] = var
            cb = th.checkbutton(self.checkbox_frame, name, var, bg=th.SURFACE)
            cb.configure(command=self._on_checkbox_toggle)
            cb.pack(anchor="w", pady=2)
            self._member_rows[uid] = cb


class SpotiBotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.llama_process = None

        root.title("SpotiBot")
        root.geometry("1320x680")
        root.configure(bg=th.BG)

        # Gradient canvas fills the whole window; the actual UI lives in a
        # "shell" frame embedded on top of it, inset by a margin - Tkinter
        # widgets are fully opaque, so the gradient can only ever show
        # through in that margin/gap, not underneath the shell itself.
        bg_canvas = GradientCanvas(root, bg=th.BG)
        bg_canvas.pack(fill="both", expand=True)

        shell = th.frame(bg_canvas)
        shell_window = bg_canvas.create_window(0, 0, window=shell, anchor="nw")

        def _resize_shell(event):
            margin = 18
            bg_canvas.coords(shell_window, margin, margin)
            bg_canvas.itemconfig(
                shell_window,
                width=max(event.width - margin * 2, 100),
                height=max(event.height - margin * 2, 100),
            )
        bg_canvas.bind("<Configure>", _resize_shell)

        # Header: accent badge + wordmark, with a thin accent underline
        # separating it from the tab content - the one-time visual signal
        # that this is a designed app, not default tkinter gray-on-gray.
        header = th.frame(shell)
        header.pack(fill="x", padx=6, pady=(2, 0))
        badge = tk.Label(
            header, text="🎧", bg=th.ACCENT, fg=th.ACCENT_TEXT, font=(th.FONT, 20),
            width=2, height=1,
        )
        badge.pack(side="left")
        th.label(header, "SpotiBot", kind="title", bg=th.BG, font=(th.FONT, 22, "bold")).pack(
            side="left", padx=(12, 0)
        )

        underline = tk.Frame(shell, bg=th.ACCENT, height=2)
        underline.pack(fill="x", padx=6, pady=(12, 16))

        # Body: fixed-width queue sidebar on the left, fixed-width voice-
        # members panel on the right, log tabs filling the middle. Order
        # matters for pack() here - both fixed-width sides need to be
        # packed before the expanding middle notebook, or the notebook
        # claims all the space before the right panel gets a chance to.
        body = th.frame(shell)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 12))

        sidebar_container = th.frame(body, width=240)
        sidebar_container.pack(side="left", fill="y", padx=(0, 16))
        sidebar_container.pack_propagate(False)
        self.sidebar = QueueSidebar(sidebar_container)

        members_container = th.frame(body, width=240)
        members_container.pack(side="right", fill="y", padx=(16, 0))
        members_container.pack_propagate(False)
        self.voice_members = VoiceMembersSidebar(members_container)

        notebook = ttk.Notebook(body)
        notebook.pack(side="left", fill="both", expand=True)

        self.bot_tab = LogTab(notebook, "🤖 Bot", BOT_STATUS_TRIGGERS)
        self.ai_tab = LogTab(notebook, "🧠 AI Model", AI_STATUS_TRIGGERS)

        # Bottom status bar: a real, computed "at a glance" summary (not
        # decorative) - reflects the same status text already shown in
        # each tab's own pill, plus which model is currently loaded.
        status_bar_outer, status_bar_inner = th.card(shell)
        status_bar_outer.pack(fill="x", padx=6, pady=(0, 2))
        self.status_bar_var = tk.StringVar(value="Starting up...")
        self.status_bar_label = th.label(
            status_bar_inner, textvariable=self.status_bar_var, kind="secondary", bg=th.SURFACE,
        )
        self.status_bar_label.pack(side="left", anchor="w", padx=14, pady=8)

        # Settings now lives here instead of a top menu bar - a gear icon
        # docked at the bottom-right, consistent with where a settings
        # control usually sits in a polished app.
        gear_btn = tk.Label(
            status_bar_inner, text="⚙", bg=th.SURFACE, fg=th.TEXT_SECONDARY,
            font=(th.FONT, 15), cursor="hand2", padx=12, pady=6,
        )
        gear_btn.pack(side="right", padx=(0, 6))
        gear_btn.bind("<Button-1>", lambda e: self.open_settings())
        gear_btn.bind("<Enter>", lambda e: gear_btn.configure(fg=th.ACCENT))
        gear_btn.bind("<Leave>", lambda e: gear_btn.configure(fg=th.TEXT_SECONDARY))

        # bot.py's prints go through sys.stdout/stderr -> this queue.
        sys.stdout = QueueWriter(self.bot_tab.queue)
        sys.stderr = QueueWriter(self.bot_tab.queue)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._start_llama_server()
        self._start_bot_thread()
        self._poll_queues()

    def _update_status_bar(self):
        bot_status = self.bot_tab.status_var.get()
        ai_status = self.ai_tab.status_var.get()
        bot_ok = any(s in bot_status for s in ("Online", "Listening", "Playing"))
        ai_ok = "Ready" in ai_status

        model_name = os.path.basename(LLAMA_MODEL_PATH)
        if bot_ok and ai_ok:
            self.status_bar_var.set(f"🧠 {model_name}   •   ✅ All systems ready")
            self.status_bar_label.configure(fg=th.SUCCESS)
        else:
            waiting_on = []
            if not bot_ok:
                waiting_on.append("bot")
            if not ai_ok:
                waiting_on.append("AI model")
            self.status_bar_var.set(f"🧠 {model_name}   •   ⏳ Waiting on: {', '.join(waiting_on)}")
            self.status_bar_label.configure(fg=th.WARNING)

    # --- Settings ---

    def open_settings(self):
        SettingsDialog(self.root, self)

    def restart_llama_server(self):
        def set_restarting_status():
            self.ai_tab.status_var.set("● Restarting...")
            self.ai_tab.status_label.configure(fg=th.WARNING)

        def do_restart():
            self.ai_tab.queue.put("\n[GUI] Restarting AI model server...\n")
            self.root.after(0, set_restarting_status)  # Tkinter widgets aren't thread-safe - hop to the main thread
            if self.llama_process and self.llama_process.poll() is None:
                try:
                    self.llama_process.terminate()
                    self.llama_process.wait(timeout=5)
                except Exception:
                    try:
                        self.llama_process.kill()
                    except Exception:
                        pass
            self.llama_process = None
            self._start_llama_server()

        threading.Thread(target=do_restart, daemon=True).start()

    # --- llama-server subprocess ---

    def _ensure_model_downloaded(self) -> bool:
        if os.path.exists(LLAMA_MODEL_PATH):
            return True

        if LLAMA_MODEL_PATH != DEFAULT_MODEL_PATH:
            self.ai_tab.queue.put(
                f"[FATAL] Model not found at '{LLAMA_MODEL_PATH}'. Auto-download only knows how to "
                f"fetch the default Qwen2.5-1.5B-Instruct model - since LLAMA_MODEL_PATH was "
                f"customized, place that model file there yourself.\n"
            )
            return False

        target_dir = os.path.dirname(LLAMA_MODEL_PATH) or "."
        os.makedirs(target_dir, exist_ok=True)
        tmp_path = LLAMA_MODEL_PATH + ".part"

        self.ai_tab.queue.put(
            f"[GUI] Model not found - downloading Qwen2.5-1.5B-Instruct (Q4_K_M, ~1.1GB) "
            f"from Hugging Face. This only happens once.\n"
        )

        try:
            def on_progress(msg):
                self.ai_tab.queue.put(f"[GUI] Downloading model...{msg}")
            _download_file(MODEL_DOWNLOAD_URL, tmp_path, on_progress)
            os.replace(tmp_path, LLAMA_MODEL_PATH)
            self.ai_tab.queue.put(f"[GUI] Model download complete: {LLAMA_MODEL_PATH}\n\n")
            return True
        except Exception as e:
            self.ai_tab.queue.put(f"[FATAL] Model download failed: {e}\n")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False

    def _start_llama_server(self):
        cmd = [LLAMA_SERVER_PATH, "-m", LLAMA_MODEL_PATH, "--port", LLAMA_SERVER_PORT, "-ngl", LLAMA_NGL]

        def run():
            if not self._ensure_model_downloaded():
                return

            self.ai_tab.queue.put(f"[GUI] Launching: {' '.join(cmd)}\n\n")
            try:
                # CREATE_NO_WINDOW: this is a GUI app - don't let
                # llama-server pop up its own separate console window.
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.llama_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=creationflags,
                )
            except FileNotFoundError:
                self.ai_tab.queue.put(
                    f"[FATAL] Couldn't find '{LLAMA_SERVER_PATH}'. Set LLAMA_SERVER_PATH in your "
                    f".env to its full path, or place llama-server.exe next to this app.\n"
                )
                return
            except Exception as e:
                self.ai_tab.queue.put(f"[FATAL] Failed to launch llama-server: {e}\n")
                return

            for line in self.llama_process.stdout:
                self.ai_tab.queue.put(line)

            code = self.llama_process.wait()
            self.ai_tab.queue.put(f"\n[llama-server exited with code {code}]\n")

        self.llama_thread = threading.Thread(target=run, daemon=True)
        self.llama_thread.start()

    # --- bot.py ---

    def _start_bot_thread(self):
        def run():
            global bot_module
            try:
                print("[GUI] Loading bot.py (this can take a while - Whisper model, etc.)...")
                import bot as _bot_module
                bot_module = _bot_module
            except Exception as e:
                import traceback
                self.bot_tab.queue.put(f"\n[FATAL] bot.py failed to load: {e}\n")
                self.bot_tab.queue.put(traceback.format_exc())
                return

            try:
                bot_module.main()
            except Exception as e:
                self.bot_tab.queue.put(f"\n[FATAL] Bot crashed: {e}\n")

        self.bot_thread = threading.Thread(target=run, daemon=True)
        self.bot_thread.start()

    # --- shared polling / shutdown ---

    def _poll_queues(self):
        self.bot_tab.drain()
        self.ai_tab.drain()
        self.sidebar.refresh(bot_module)
        self.voice_members.refresh(bot_module)
        self._update_status_bar()
        self.root.after(150, self._poll_queues)

    def _on_close(self):
        self.bot_tab.status_var.set("● Shutting down...")
        self.bot_tab.status_label.configure(fg=th.WARNING)
        self.ai_tab.status_var.set("● Shutting down...")
        self.ai_tab.status_label.configure(fg=th.WARNING)
        self.root.update()  # force a repaint before we block below

        loop = getattr(getattr(bot_module, "bot", None), "loop", None) if bot_module else None
        if loop and loop.is_running():
            async def shutdown():
                # Disconnect voice explicitly first (stops recording +
                # leaves the call) - bot.close() alone should handle this
                # too, but being explicit avoids relying on that.
                for vc in list(bot_module.bot.voice_clients):
                    vc._auto_listen_enabled = False  # don't let auto-restart fire mid-shutdown
                    try:
                        vc.stop_recording()
                    except Exception:
                        pass
                    try:
                        await vc.disconnect(force=True)
                    except Exception:
                        pass
                await bot_module.bot.close()

            fut = asyncio.run_coroutine_threadsafe(shutdown(), loop)
            try:
                fut.result(timeout=5)
            except Exception as e:
                self.bot_tab.queue.put(f"[GUI] Bot shutdown didn't finish cleanly ({e}), exiting anyway.\n")

        self.bot_thread.join(timeout=3)

        # llama-server is a plain subprocess, not a Python thread - it
        # doesn't get cleaned up automatically. Without this it keeps
        # running invisibly (still holding GPU memory) after the window
        # closes.
        if self.llama_process and self.llama_process.poll() is None:
            try:
                self.llama_process.terminate()
                self.llama_process.wait(timeout=5)
            except Exception:
                self.llama_process.kill()

        self.root.destroy()
        os._exit(0)


CUDA_RUNTIME_DIR = "cuda_runtime"
CUDA_RUNTIME_PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]


def _cuda_runtime_ready() -> bool:
    """Whether the on-demand-downloaded cuBLAS/cuDNN DLLs are already
    present from a previous session - avoids re-downloading every time
    someone switches Whisper to GPU."""
    if not os.path.isdir(CUDA_RUNTIME_DIR):
        return False
    return any(f.lower().startswith("cublas64") for f in os.listdir(CUDA_RUNTIME_DIR))


def _pypi_wheel_url(package_name: str):
    """Finds the latest win_amd64 wheel download URL for a package via
    PyPI's JSON API - the same file `pip install <package_name>` would
    fetch, just grabbed directly since a frozen exe has no pip available
    at runtime to do this the normal way."""
    import requests

    r = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=15)
    r.raise_for_status()
    data = r.json()
    version = data["info"]["version"]
    for f in data["releases"].get(version, []):
        if f["filename"].endswith(".whl") and "win_amd64" in f["filename"]:
            return f["url"], f["filename"]
    raise RuntimeError(f"No win_amd64 wheel found for {package_name} {version}")


def _download_and_extract_cuda_dlls(log):
    """
    Downloads the cuBLAS/cuDNN wheels and extracts just the .dll files
    (wheels are plain zip files) into CUDA_RUNTIME_DIR, flattened - the
    wheel's internal folder structure doesn't matter once these are just
    sitting in one folder added to PATH.
    """
    os.makedirs(CUDA_RUNTIME_DIR, exist_ok=True)
    for package_name in CUDA_RUNTIME_PACKAGES:
        log(f"Finding latest {package_name} wheel...\n")
        url, filename = _pypi_wheel_url(package_name)
        whl_path = os.path.join(CUDA_RUNTIME_DIR, filename)

        log(f"Downloading {filename}...\n")
        _download_file(url, whl_path, log)

        log(f"Extracting DLLs from {filename}...\n")
        extracted_any = False
        with zipfile.ZipFile(whl_path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".dll"):
                    target = os.path.join(CUDA_RUNTIME_DIR, os.path.basename(name))
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted_any = True
        os.remove(whl_path)

        if not extracted_any:
            log(f"⚠️ No DLLs found inside {filename} - this package's wheel layout may have changed.\n")

    log("✅ GPU support files ready.\n")


def _add_bundled_binaries_to_path():
    """
    If this is running as a PyInstaller --onefile exe with ffmpeg.exe/
    ffprobe.exe/deno.exe bundled in via --add-binary, they land in
    sys._MEIPASS (PyInstaller's per-run extraction folder) at runtime.
    Prepending that folder to PATH makes them discoverable exactly the
    way a normal system install would be - every existing ffmpeg/deno
    call in bot.py and yt-dlp already just looks them up by name on PATH,
    so nothing else needs to change for this to work.

    No-op when running from source (python gui_launcher.py) - sys.frozen
    only exists in a PyInstaller build, so this quietly does nothing and
    falls back to whatever's already on the system PATH.

    Also adds CUDA_RUNTIME_DIR if it exists (from a previous on-demand
    GPU-Whisper download) - safe/idempotent to call this more than once
    per run, e.g. right after a fresh download completes.
    """
    if getattr(sys, "frozen", False):
        bundled_dir = sys._MEIPASS
        os.environ["PATH"] = bundled_dir + os.pathsep + os.environ.get("PATH", "")

    cuda_dir = os.path.abspath(CUDA_RUNTIME_DIR)
    if os.path.isdir(cuda_dir):
        os.environ["PATH"] = cuda_dir + os.pathsep + os.environ.get("PATH", "")


class CudaRuntimeDownloadDialog(tk.Toplevel):
    """
    Downloads and extracts the NVIDIA cuBLAS/cuDNN redistributable DLLs
    faster-whisper's GPU path needs, on demand - only triggered when
    someone actually opts into GPU Whisper, instead of bundling ~1GB of
    CUDA runtime into every build regardless of whether anyone uses it.
    """

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.success = False

        self.title("Setting up GPU support")
        self.geometry("560x420")
        self.configure(bg=th.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # don't allow closing mid-download
        th.apply_dark_titlebar(self)
        self.transient(parent)
        self.grab_set()

        th.label(self, "Downloading GPU support files...", kind="title", bg=th.BG).pack(
            pady=(20, 8), padx=20, anchor="w"
        )
        th.label(
            self, "One-time download (~1GB) of the NVIDIA CUDA runtime libraries Whisper's\n"
                  "GPU mode needs. Only happens once - future switches to GPU reuse these.",
            kind="secondary", bg=th.BG, justify="left",
        ).pack(padx=20, pady=(0, 12), anchor="w")

        card_outer, card_inner = th.card(self)
        card_outer.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.progress_text = scrolledtext.ScrolledText(
            card_inner, bg=th.SURFACE, fg=th.TEXT_SECONDARY, font=(th.FONT_MONO, 9),
            wrap="word", state="disabled", borderwidth=0, highlightthickness=0, padx=10, pady=8,
        )
        self.progress_text.pack(fill="both", expand=True)

        self.close_btn = th.button(self, "Please wait...", lambda: None, kind="secondary")
        self.close_btn.pack(pady=(0, 16))
        self.close_btn.configure(state="disabled")

        self.after(150, self._run)

    def _log(self, msg: str):
        self.progress_text.configure(state="normal")
        self.progress_text.insert("end", msg)
        self.progress_text.see("end")
        self.progress_text.configure(state="disabled")
        self.update()

    def _run(self):
        try:
            _download_and_extract_cuda_dlls(self._log)
            self.success = True
        except Exception as e:
            self._log(f"\n[FATAL] Failed to set up GPU support: {e}\n")
            self.success = False

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.close_btn.configure(
            text="Continue" if self.success else "Close", command=self.destroy, state="normal",
            bg=th.ACCENT if self.success else th.SURFACE_ALT,
        )


def main():
    global LLAMA_SERVER_PATH
    _add_bundled_binaries_to_path()

    root = tk.Tk()
    root.withdraw()  # hide the (still empty) main window during first-launch setup
    th.configure_ttk_style()  # must run AFTER a real Tk root exists, or ttk.Style() implicitly
                               # creates its own hidden root - which then shows up as a stray
                               # blank "tk"-titled window since nothing ever configures it.

    root.title("SpotiBot")
    root.configure(bg=th.BG)
    th.apply_dark_titlebar(root)

    if _needs_initial_setup():
        setup_dialog = InitialSetupDialog(root)
        guide_dialog = TokenGuideDialog(root)
        setup_dialog._guide_dialog = guide_dialog

        # Side by side - setup form on the left, guide on the right, so
        # both are usable at once without either covering the other.
        root.update_idletasks()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        gap = 16
        total_w = 560 + gap + 560
        left_x = max((screen_w - total_w) // 2, 0)
        top_y = max((screen_h - 680) // 2, 0)
        setup_dialog.geometry(f"+{left_x}+{top_y}")
        guide_dialog.geometry(f"+{left_x + 560 + gap}+{top_y}")

        root.wait_window(setup_dialog)  # blocks here until Continue/Exit

        if not setup_dialog.completed:
            return  # user chose to exit setup rather than complete it - close cleanly, don't proceed

    LLAMA_SERVER_PATH = resolve_llama_server_path(root)

    root.deiconify()
    SpotiBotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()