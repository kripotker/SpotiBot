"""
IMPORTANT: This bot runs on py-cord's still-unmerged PR #3159, which is
the only Python Discord library with working DAVE voice-receive decryption
as of writing. Install it in this exact venv (do NOT mix with discord.py or
discord-ext-voice-recv - they conflict on the `discord` import name):

    pip uninstall discord.py discord-ext-voice-recv -y
    pip install "git+https://github.com/Pycord-Development/pycord@refs/pull/3159/head"

Track https://github.com/Pycord-Development/pycord/pull/3159 for merge status.

Wake words, debug logging, and a couple of other knobs live in
spotibot_settings.json (see settings_store.py) - edit them via the GUI's
Settings window, or by hand. reload_settings() lets a running bot pick up
changes without a restart.
"""
import asyncio
import json
import os
import re
import sys
import threading
import time
import traceback
import wave
from difflib import SequenceMatcher

import discord
import discord.sinks
import numpy as np
import openai
import spotipy
import yt_dlp
from discord.ext import commands
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from spotipy.oauth2 import SpotifyClientCredentials

import settings_store

load_dotenv()

_settings = settings_store.load_settings()

WAKE_PHRASES = list(_settings["wake_phrases"])
# env var kept as a power-user override for running bot.py standalone
# (outside the GUI) without touching the settings file.
DEBUG_VOICE = bool(_settings["debug_voice"]) or os.getenv("DEBUG_VOICE", "0") == "1"
SPOTIFY_MATCH_THRESHOLD = float(_settings["spotify_match_threshold"])

# discord.py auto-loads its bundled Opus DLL at import time, but that
# happens before discord.py's own logging is configured, so a failed
# auto-load fails *silently* - is_loaded() is the only way to catch it.
if not discord.opus.is_loaded():
    discord.opus._load_default()
print(f"Opus Library Loaded: {discord.opus.is_loaded()}")
if not discord.opus.is_loaded():
    raise RuntimeError(
        "Opus failed to load even after an explicit _load_default() call. "
        "Check that this py-cord install includes the bundled opus DLL."
    )


# --- CONFIGURATION ---
# WAKE_PHRASES, DEBUG_VOICE, SPOTIFY_MATCH_THRESHOLD loaded from settings
# above. The "Listening!" message shown to users uses WAKE_PHRASES[0] as
# an example phrase.

WHISPER_MODEL_NAME = _settings["whisper_model"]
WHISPER_DEVICE = _settings.get("whisper_device", "cpu")
_whisper_compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"

# Same "models" folder gui_launcher.py downloads the Qwen GGUF into,
# instead of faster-whisper's default (the global Hugging Face cache
# under your user profile). Note this still lands as an HF-style cache
# subfolder (models/models--Systran--faster-whisper-<name>/...) rather
# than a single flat file, since faster-whisper passes this straight
# through to huggingface_hub's own snapshot download - just keeping it
# under one project folder instead of scattered in the user cache.
WHISPER_DOWNLOAD_ROOT = "models"
os.makedirs(WHISPER_DOWNLOAD_ROOT, exist_ok=True)

stt_model = None
try:
    stt_model = WhisperModel(
        WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=_whisper_compute_type,
        download_root=WHISPER_DOWNLOAD_ROOT,
    )
except Exception as e:
    print(f"[Startup] Failed to load Whisper model '{WHISPER_MODEL_NAME}' on {WHISPER_DEVICE}: {e}")
    if WHISPER_DEVICE != "cuda":
        raise  # not a GPU-related failure - nothing sensible to fall back to, let it crash loudly

    error_str = str(e)
    if any(k in error_str.lower() for k in ("cublas", "cudnn")):
        print(
            "[Startup] This means the NVIDIA CUDA runtime libraries (cuBLAS/cuDNN) aren't "
            "installed on this PC - having a CUDA-capable GPU isn't enough by itself. Get "
            "the NVIDIA CUDA Toolkit (12.x) from https://developer.nvidia.com/cuda-downloads "
            "if you want GPU Whisper working, or leave this on CPU."
        )
    print("[Startup] Falling back to CPU for now, and saving that so future launches don't "
          "keep retrying a broken GPU setup automatically.")
    WHISPER_DEVICE = "cpu"
    _whisper_compute_type = "int8"
    stt_model = WhisperModel(
        WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=_whisper_compute_type,
        download_root=WHISPER_DOWNLOAD_ROOT,
    )
    _fallback_settings = settings_store.load_settings()
    _fallback_settings["whisper_device"] = "cpu"
    settings_store.save_settings(_fallback_settings)


def reload_whisper_model(model_name: str, device: str = "cpu"):
    """
    Swaps the loaded Whisper model (and/or compute device) at runtime -
    called from the GUI's Settings > Voice tab so a change doesn't require
    restarting the whole bot (which would drop the Discord voice
    connection). Loads in a background thread since it can take real time,
    and only swaps the global once the new model is actually ready, so an
    in-flight transcription on the old model isn't interrupted mid-load.

    device="cuda" requires an NVIDIA GPU - faster-whisper's GPU support is
    CUDA-only via ctranslate2, which has no ROCm/Vulkan/OpenVINO backend.
    AMD and Intel GPUs can't accelerate this regardless of what llama.cpp
    backend is installed for the separate LLM.
    """
    def _load():
        global stt_model, WHISPER_MODEL_NAME, WHISPER_DEVICE
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[Settings] Loading Whisper model '{model_name}' on {device}...")
        try:
            new_model = WhisperModel(
                model_name, device=device, compute_type=compute_type, download_root=WHISPER_DOWNLOAD_ROOT,
            )
            stt_model = new_model
            WHISPER_MODEL_NAME = model_name
            WHISPER_DEVICE = device
            print(f"[Settings] Whisper model switched to '{model_name}' ({device}).")
        except Exception as e:
            error_str = str(e)
            print(f"[Settings] Failed to load Whisper model '{model_name}' on {device}: {e}")
            if device == "cuda" and any(k in error_str.lower() for k in ("cublas", "cudnn")):
                print(
                    "[Settings] This means the NVIDIA CUDA runtime libraries (cuBLAS/cuDNN) "
                    "aren't installed on this PC - having a CUDA-capable GPU isn't enough by "
                    "itself, these are a separate install. Get the NVIDIA CUDA Toolkit "
                    "(12.x) from https://developer.nvidia.com/cuda-downloads, or switch back "
                    "to CPU in Settings > Voice > Change Whisper Model. Staying on the "
                    "previous model for now, nothing else was affected."
                )
            elif device == "cuda":
                print("[Settings] GPU (CUDA) Whisper needs an NVIDIA GPU with a working CUDA/cuDNN "
                      "setup - if you don't have that, switch back to CPU in Settings > Voice.")

    threading.Thread(target=_load, daemon=True).start()

llm_client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

SYSTEM_PROMPT = """You are a voice command parser for a music bot. Analyze the user command and return ONLY a valid JSON object.
JSON Schema:
{
  "action": "play" | "skip" | "next" | "pause" | "resume" | "unknown",
  "query": "song name or search query string if action is play, otherwise null"
}"""

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
))

intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # needed for channel.members to reliably reflect who's in voice -
                         # ALSO requires enabling "Server Members Intent" in the Discord
                         # Developer Portal (Bot page) for this application, or this stays
                         # empty regardless of the code-side setting here.
bot = commands.Bot(command_prefix="!", intents=intents)

song_queue = []
now_playing = None  # title of the currently-playing track, or None - read by gui_launcher.py's queue sidebar
active_sink = None  # the currently active VoiceCommandSink, or None - read/controlled by gui_launcher.py's voice-members panel

YTDL_OPTIONS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "default_search": "ytsearch",
    # Deno alone (the JS runtime) isn't enough - yt-dlp also needs the
    # actual challenge-solver script downloaded to solve YouTube's
    # signature/n-challenges.
    "remote_components": ["ejs:github"],
    # YouTube's client requirements shift week-to-week; several clients now
    # gate formats behind a PO Token that headers alone can't satisfy.
    # Widen the fallback list rather than pin to one or two clients.
    "extractor_args": {"youtube": {"player_client": ["web", "tv", "android", "web_safari"]}},
}
FFMPEG_OPTIONS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}


def _resource_path(filename: str) -> str:
    """
    Resolves a bundled resource file (like ping.mp3) to an absolute path,
    correctly whether running from source or as a PyInstaller onefile/
    onedir build - all of these can have a working directory that's
    different from wherever the file actually lives, which is exactly why
    a bare "ping.mp3" breaks the moment someone launches the exe from a
    shortcut/pinned icon instead of running it from inside its own folder.
    sys._MEIPASS is set correctly by PyInstaller for both onefile and
    onedir builds; falling back to this script's own folder covers
    running from source.
    """
    base_dir = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)


def play_cue(vc):
    if vc and vc.is_connected() and not vc.is_playing():
        cue_path = _resource_path("ping.mp3")
        if not os.path.exists(cue_path):
            print(f"[play_cue] ping.mp3 not found at expected path: {cue_path}")
            return
        # Capture ffmpeg's stderr the same way play_next() already does -
        # without this, a broken cue plays silently with zero diagnostic
        # trail, which is exactly what made this bug invisible until now.
        cue_log = open("ffmpeg_cue_debug.log", "wb")
        vc.play(discord.FFmpegPCMAudio(cue_path, stderr=cue_log))


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation to spaces (not delete), collapse whitespace."""
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def match_wake_phrase(normalized: str):
    """
    Returns the remaining command text if `normalized` starts with any
    configured wake phrase, else None. Checks longest phrases first so a
    short phrase's prefix can't swallow part of a longer one (e.g. "hey
    bot" matching inside what should be "hey bought to" first). Strips a
    trailing "to " too, since Whisper sometimes inserts it
    ("hey bought to play..." -> "play...").
    """
    for phrase in sorted(WAKE_PHRASES, key=len, reverse=True):
        phrase_norm = normalize_text(phrase)
        if not phrase_norm:
            continue
        if normalized == phrase_norm:
            return ""
        if normalized.startswith(phrase_norm + " "):
            rest = normalized[len(phrase_norm):].strip()
            if rest.startswith("to "):
                rest = rest[3:].strip()
            return rest
    return None


def reload_settings():
    """
    Re-reads spotibot_settings.json and applies changes live. Called by the
    GUI's Settings window after saving, so wake-word/debug/threshold
    changes take effect immediately without restarting the bot. (Model/GPU
    backend changes still need an AI-model restart - those aren't handled
    here, see gui_launcher.py.)
    """
    global WAKE_PHRASES, DEBUG_VOICE, SPOTIFY_MATCH_THRESHOLD
    s = settings_store.load_settings()
    WAKE_PHRASES = list(s["wake_phrases"])
    DEBUG_VOICE = bool(s["debug_voice"]) or os.getenv("DEBUG_VOICE", "0") == "1"
    SPOTIFY_MATCH_THRESHOLD = float(s["spotify_match_threshold"])
    VoiceCommandSink.IDLE_TIMEOUT = float(s["idle_timeout"])
    print(f"[Settings] Reloaded - {len(WAKE_PHRASES)} wake phrase(s), "
          f"debug={'on' if DEBUG_VOICE else 'off'}, "
          f"idle_timeout={VoiceCommandSink.IDLE_TIMEOUT}s, "
          f"spotify_threshold={SPOTIFY_MATCH_THRESHOLD}")


# --- VOICE PROCESSING SINK ---

class VoiceCommandSink(discord.sinks.Sink):
    """
    Buffers raw PCM per-speaker and hands each person's audio to Whisper
    independently once they've paused - supports multiple people talking
    in the same session, not just whoever ran !listen. Runs on py-cord's
    start_recording/Sink API (PR #3159):
      - write() takes (data, user), reversed from discord-ext-voice-recv
      - is_opus() (not wants_opus()) controls whether we get raw PCM
      - data.pcm still holds raw 48kHz stereo 16-bit PCM

    authorized_user_ids controls who the bot actually listens to (audio
    from anyone else is discarded in write() before it's ever buffered):
      - None    = everyone currently in the channel, including anyone who
                  joins later this session (the default)
      - set()   = no one (bot connected but not listening to anyone)
      - {id,..} = only these specific Discord user IDs
    Editable live from the GUI's voice-members panel via
    set_authorized_listeners() below - takes effect on the very next
    packet, no restart needed.

    Discord only sends voice packets while someone is actively talking, so
    write() stops firing entirely once an utterance ends - there's no
    trailing "silence" packet to detect. A watchdog task (independent of
    write() being called) tracks each speaker's wall-clock idle time
    instead, so multiple people going idle at different times are each
    processed independently rather than sharing one buffer/timer (which
    would otherwise interleave two people's speech into one garbled clip).
    """

    IDLE_TIMEOUT = float(_settings["idle_timeout"])  # seconds of no new packets = end of utterance
    MIN_BUFFER_BYTES = 48000 * 2    # ~0.5s of 48kHz stereo 16-bit PCM

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.buffers = {}          # user_id -> bytearray
        self.last_audio_time = {}  # user_id -> float | None
        self.watchdog_task = None
        self.authorized_user_ids = None  # None = everyone (see class docstring)
        self._write_calls = 0
        self._mismatched_calls = 0

    def is_authorized(self, user_id) -> bool:
        if self.authorized_user_ids is None:
            return True
        return user_id in self.authorized_user_ids

    def is_opus(self):
        return False

    def write(self, data, user):
        self._write_calls += 1

        user_id = getattr(user, "id", user)
        if not self.is_authorized(user_id):
            self._mismatched_calls += 1
            if DEBUG_VOICE and (self._mismatched_calls in (1, 50, 200) or self._mismatched_calls % 500 == 0):
                print(f"[write] packet from unauthorized user_id={user_id} "
                      f"(ignored #{self._mismatched_calls}, total write() calls={self._write_calls})")
            return

        payload = getattr(data, "pcm", data)  # VoiceData wraps raw PCM in .pcm
        if not payload:
            return

        if user_id not in self.buffers:
            self.buffers[user_id] = bytearray()
        self.buffers[user_id].extend(payload)
        self.last_audio_time[user_id] = time.monotonic()

    async def _watchdog(self):
        if DEBUG_VOICE:
            print("[watchdog] started")
        last_heartbeat = time.monotonic()
        try:
            while True:
                await asyncio.sleep(0.2)

                if DEBUG_VOICE:
                    now = time.monotonic()
                    if now - last_heartbeat > 15:
                        print(f"[watchdog] alive, write() calls so far: {self._write_calls}")
                        last_heartbeat = now

                now = time.monotonic()
                for user_id in list(self.buffers.keys()):
                    buf = self.buffers.get(user_id)
                    last_time = self.last_audio_time.get(user_id)
                    if (
                        last_time is not None
                        and buf and len(buf) > self.MIN_BUFFER_BYTES
                        and now - last_time > self.IDLE_TIMEOUT
                    ):
                        audio_data = bytearray(buf)
                        self.buffers[user_id] = bytearray()
                        self.last_audio_time[user_id] = None

                        print(f"🔊 Silence detected (user {user_id}), processing {len(audio_data)} bytes...")
                        await self.process_audio(audio_data, user_id)
        except asyncio.CancelledError:
            if DEBUG_VOICE:
                print("[watchdog] cancelled (expected on !leave)")
            raise
        except Exception:
            print("[watchdog] CRASHED with an unhandled exception - this silently killed listening:")
            traceback.print_exc()

    def cleanup(self):
        self.buffers.clear()
        self.last_audio_time.clear()
        if self.watchdog_task:
            self.watchdog_task.cancel()
        super().cleanup()

    async def process_audio(self, pcm_bytes, user_id=None):
        try:
            # 48kHz stereo s16le -> 16kHz mono float32, which is what
            # faster-whisper expects. [::2]-style channel dropping is NOT
            # resampling; average L/R to mono, then average every 3 samples
            # for a cheap anti-aliased 48k -> 16k decimation.
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            usable_len = len(audio_array) - (len(audio_array) % 2)
            stereo = audio_array[:usable_len].reshape(-1, 2)
            mono_48k = stereo.mean(axis=1)

            trim = len(mono_48k) - (len(mono_48k) % 3)
            audio_array = mono_48k[:trim].reshape(-1, 3).mean(axis=1)

            peak = float(np.abs(audio_array).max()) if audio_array.size else 0.0
            print(f"🔈 Captured {len(audio_array) / 16000:.2f}s of audio, peak amplitude {peak:.3f}")

            segments, _ = await asyncio.to_thread(
                stt_model.transcribe,
                audio_array,
                beam_size=1,
                no_speech_threshold=0.3,
            )
            raw_text = " ".join(seg.text for seg in segments).strip().lower()
            print(f"🎙️ Transcribed: '{raw_text}'")

            # Always dump the capture (not just failures) so any false
            # trigger or empty transcription can actually be listened to
            # instead of guessed at from the log alone. Per-user filename
            # since multiple people can be buffered/processed independently
            # now - a shared filename would let two speakers clobber the
            # same debug file mid-session.
            debug_path = f"debug_last_capture_{user_id}.wav" if user_id is not None else "debug_last_capture.wav"
            with wave.open(debug_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                pcm16 = np.clip(audio_array * 32768.0, -32768, 32767).astype(np.int16)
                wf.writeframes(pcm16.tobytes())

            if not raw_text:
                print(f"⚠️ Empty transcription - wrote captured audio to {debug_path} for review")
                return

            normalized = normalize_text(raw_text)
            command_text = match_wake_phrase(normalized)
            if command_text is None:
                print(f"[Debug] No wake word - captured audio saved to {debug_path} if you want to check it")
                return

            play_cue(self.ctx.voice_client)

            if not command_text:
                await self.ctx.send("Listening! What would you like to play?")
                return

            print(f"⚡ Wake-word matched. Command: '{command_text}'")

            def parse_intent():
                response = llm_client.chat.completions.create(
                    model="qwen2.5-1.5b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": command_text},
                    ],
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)

            intent = await asyncio.to_thread(parse_intent)
            action = intent.get("action")
            query = intent.get("query")
            print(f"🤖 Action: {action} | Query: {query}")

            # Dispatch as independent tasks, not inline awaits. write()/
            # process_audio() runs INSIDE this sink's own watchdog task, so
            # any command that tears down/replaces the recording session
            # (e.g. skip -> voice_client.stop() -> sink.cleanup() ->
            # watchdog_task.cancel()) would cancel the very task running
            # this code, cutting it off mid-dispatch if awaited directly.
            if action == "play" and query:
                bot.loop.create_task(self.ctx.invoke(bot.get_command("play"), query=query))
            elif action in ("skip", "next"):
                bot.loop.create_task(self.ctx.invoke(bot.get_command("next")))
            elif action == "pause" and self.ctx.voice_client and self.ctx.voice_client.is_playing():
                self.ctx.voice_client.pause()
                bot.loop.create_task(self.ctx.send("Paused!"))
            elif action == "resume" and self.ctx.voice_client and self.ctx.voice_client.is_paused():
                self.ctx.voice_client.resume()
                bot.loop.create_task(self.ctx.send("Resumed!"))

        except Exception as e:
            print(f"[Audio Processing Error]: {e}")


# --- LISTENING SESSION MANAGEMENT ---

async def start_listening(ctx, *, announce=True):
    """
    Starts (or restarts) recording on ctx.voice_client, self-healing if the
    session ever ends unexpectedly - this branch's DAVE/MLS key-rotation
    handling, or voice_client.stop() as a side effect, can tear the
    recording session down without a clean signal. Previously this
    required manually !leave + !listen; now it retries automatically, with
    a circuit breaker so a persistently broken session doesn't loop forever.
    """
    global active_sink
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        # The underlying voice connection itself is gone, not just the
        # recording layer on top of it - this used to just silently give
        # up here, leaving the bot invisibly disconnected until a human
        # noticed and manually retyped !listen. Try a real rejoin instead.
        if not ctx.author.voice:
            print("[Voice] Can't (re)connect - no longer in a voice channel.")
            return
        try:
            print("[Voice] No active voice connection - (re)connecting...")
            vc = await ctx.author.voice.channel.connect()
        except Exception as e:
            print(f"[Voice] (Re)connect failed: {e}")
            return

    # Persist restart tracking on the voice_client itself (not a local var)
    # so it survives across the recursive auto-restart calls below. Reset
    # only on an explicit !listen (announce=True), so genuine user-initiated
    # restarts don't inherit a stale failure count.
    if announce or not hasattr(vc, "_restart_state"):
        vc._restart_state = {"count": 0, "last": 0.0}
    restart_state = vc._restart_state

    # Preserve the authorized-listener selection across auto-restarts - a
    # self-healing session recreates the sink from scratch, which would
    # otherwise silently reset back to "everyone" and lose whatever the
    # GUI's voice-members panel had configured. A genuine fresh !listen
    # (announce=True) intentionally starts back at the "everyone" default.
    previous_authorized = active_sink.authorized_user_ids if active_sink else None

    sink = VoiceCommandSink(ctx)
    if not announce:
        sink.authorized_user_ids = previous_authorized
    active_sink = sink

    async def _on_recording_finished(*args):
        print(f"[Voice] Recording session ended. callback received: {args!r}")
        for arg in args:
            if isinstance(arg, BaseException):
                traceback.print_exception(type(arg), arg, arg.__traceback__)

        if not getattr(vc, "_auto_listen_enabled", False):
            return  # stopped intentionally (!leave), don't self-heal

        now = time.monotonic()
        if now - restart_state["last"] < 10:
            restart_state["count"] += 1
        else:
            restart_state["count"] = 1
        restart_state["last"] = now

        if restart_state["count"] > 5:
            print("[Voice] Session died 5+ times in quick succession - giving up auto-restart.")
            await ctx.send("⚠️ Voice listening keeps dropping - stopped auto-restarting. Try `!leave` then `!listen`.")
            return

        print("[Voice] Auto-restarting listening session...")
        await start_listening(ctx, announce=False)

    vc.start_recording(sink, _on_recording_finished)
    sink.watchdog_task = bot.loop.create_task(sink._watchdog())
    vc._auto_listen_enabled = True

    if announce:
        play_cue(vc)
        example_phrase = WAKE_PHRASES[0] if WAKE_PHRASES else "hey bot"
        await ctx.send(f"👂 Listening! Say **'{example_phrase} play [song]'** or **'{example_phrase} next'** to control music.")


async def _get_active_voice_channel_members_coro():
    """
    The actual discord.py state read - MUST run on the bot's own event
    loop thread, never called directly. discord.py's member/channel
    caches (channel.members, bot.get_all_members()) are mutated
    continuously by the event loop on every voice-state/member gateway
    event - reading them from any other thread unsynchronized is a real
    race condition, not just a theoretical one, and it's specifically the
    kind that can misbehave intermittently and unpredictably depending on
    exactly when it lands relative to a mutation.
    """
    if not bot.voice_clients:
        return []

    channel = getattr(bot.voice_clients[0], "channel", None)
    seen = {}

    if channel:
        for m in getattr(channel, "members", []):
            if not getattr(m, "bot", False):
                seen[m.id] = getattr(m, "display_name", None) or getattr(m, "name", str(m.id))

    if active_sink is not None:
        # These are our own dicts (not discord.py's internals), mutated by
        # write() on py-cord's voice-receive thread - still cross-thread
        # from here, but lower risk than the caches above. Defensive
        # try/except rather than assuming it's perfectly safe either.
        try:
            heard_ids = set(active_sink.buffers.keys()) | set(active_sink.last_audio_time.keys())
        except RuntimeError:
            heard_ids = set()
        for uid in heard_ids:
            if uid in seen:
                continue
            member = discord.utils.get(bot.get_all_members(), id=uid)
            seen[uid] = member.display_name if member else f"User {uid}"

    if DEBUG_VOICE and channel and not seen:
        print(f"[VoiceMembers] Connected to {channel!r} but found 0 members via cache or sink activity.")

    return list(seen.items())


def get_active_voice_channel_members():
    """
    [(user_id, display_name), ...] for everyone currently in the bot's
    connected voice channel (bots excluded); None if the fetch itself
    failed or timed out. Used by the GUI's live voice-members panel.

    Marshals the actual read onto the bot's own event loop thread via
    run_coroutine_threadsafe (same pattern already used for
    after_playing -> play_next elsewhere in this file) instead of
    touching discord.py's internal caches directly from the GUI thread -
    see _get_active_voice_channel_members_coro's docstring for why that
    matters here specifically.

    Returns None (not []) on failure/timeout, distinct from a genuinely
    empty channel - Whisper's transcription is CPU-bound and briefly
    starves the event loop of a chance to run this scheduled coroutine
    while it's working, which routinely pushes this past a short timeout.
    That's expected, not a real error - the caller should just keep
    showing its last-known-good state rather than flickering to "no one
    here" every single time a transcription happens to be in progress.
    """
    if bot.loop is None or not bot.loop.is_running():
        return None
    try:
        fut = asyncio.run_coroutine_threadsafe(_get_active_voice_channel_members_coro(), bot.loop)
        return fut.result(timeout=2.0)
    except Exception as e:
        if DEBUG_VOICE:
            print(f"[VoiceMembers] Failed to fetch channel members: {e!r}")
        return None


def get_authorized_listener_ids():
    """None = everyone, set() = no one, set of ids = explicit list, None
    if there's no active listening session at all (not connected/not
    listening) - the GUI panel treats that last case as "nothing to show"."""
    return active_sink.authorized_user_ids if active_sink else None


def set_authorized_listeners(user_ids):
    """
    Updates who the bot listens to, live, mid-session - called from the
    GUI's voice-members panel. user_ids: None = everyone (including
    future joiners), or any iterable of Discord user IDs for an explicit
    list (pass an empty list/set for "no one"). Takes effect on the very
    next audio packet; no restart of the recording session needed since
    write() just checks this on every call already.
    """
    if active_sink is None:
        return
    active_sink.authorized_user_ids = None if user_ids is None else set(user_ids)
    if user_ids is None:
        print("[Voice] Authorized listeners updated: everyone")
    elif not user_ids:
        print("[Voice] Authorized listeners updated: no one")
    else:
        print(f"[Voice] Authorized listeners updated: {sorted(user_ids)}")


# --- MUSIC PLAYBACK LOGIC ---

async def play_next(ctx, retries=2):
    global now_playing
    if not song_queue or not ctx.voice_client:
        now_playing = None
        return

    next_query = song_queue.pop(0)

    try:
        with yt_dlp.YoutubeDL(YTDL_OPTIONS) as ydl:
            info = ydl.extract_info(f"ytsearch:{next_query}", download=False)["entries"][0]
            stream_url = info["url"]
            title = info["title"]

        # Some client extractions require the exact request headers yt-dlp
        # used, or the CDN 403s - forward them to ffmpeg instead of guessing.
        headers = info.get("http_headers") or {}
        before_options = FFMPEG_OPTIONS["before_options"]
        if headers:
            header_lines = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            before_options += f' -headers "{header_lines}"'
        else:
            print("[Playback] WARNING: yt-dlp returned no http_headers for this format - "
                  "playing without them, may hit a silent 403.")

        def after_playing(error):
            if error:
                print(f"[Playback Error]: {error}")
            else:
                print(f"[Playback] Finished: {title}")
            fut = asyncio.run_coroutine_threadsafe(play_next(ctx), bot.loop)
            try:
                fut.result()
            except Exception:
                pass

        # Capture ffmpeg's actual stderr - a 403/empty response can
        # otherwise look identical to a clean finish in after_playing.
        ffmpeg_log = open("ffmpeg_debug.log", "wb")
        source = discord.FFmpegPCMAudio(
            stream_url,
            before_options=before_options,
            options=FFMPEG_OPTIONS["options"],
            stderr=ffmpeg_log,
        )
        ctx.voice_client.play(source, after=after_playing)
        now_playing = title
        await ctx.send(f"🎶 Now playing: **{title}**")

    except Exception as e:
        print(f"[YTDL Error]: {e}")
        if retries > 0:
            # YouTube's client requirements shift fast enough that the same
            # query can fail once and succeed on the next attempt.
            print(f"[YTDL] Retrying '{next_query}' ({retries} attempt(s) left)...")
            song_queue.insert(0, next_query)
            await play_next(ctx, retries=retries - 1)
        else:
            await ctx.send(f"Failed to play track: **{next_query}**")
            await play_next(ctx)


# --- COMMANDS ---

@bot.command()
async def play(ctx, *, query: str):
    """Play a track via query string or Spotify link."""
    if not ctx.voice_client:
        await ctx.invoke(bot.get_command("listen"))

    if "open.spotify.com/track/" in query:
        try:
            track = await asyncio.to_thread(sp.track, query)
            query = f"{track['name']} {track['artists'][0]['name']}"
        except Exception as e:
            print(f"[Spotify Error] Failed to resolve track link: {e}")
    else:
        try:
            results = await asyncio.to_thread(sp.search, q=query, type="track", limit=5)
            items = results.get("tracks", {}).get("items", [])

            # Whisper mishears proper nouns; a blind top-1 Spotify match can
            # land on something with no relation to the query. Score
            # candidates against the query and only accept a confident
            # match - otherwise keep the raw text for YouTube search
            # (more typo-tolerant on free text).
            query_norm = normalize_text(query)
            best_score, best_resolved = 0.0, None
            for track in items:
                candidate = f"{track['name']} {track['artists'][0]['name']}"
                score = SequenceMatcher(None, query_norm, normalize_text(candidate)).ratio()
                if score > best_score:
                    best_score, best_resolved = score, candidate

            if best_resolved and best_score >= SPOTIFY_MATCH_THRESHOLD:
                print(f"🎧 Spotify resolved '{query}' -> '{best_resolved}' (score {best_score:.2f})")
                query = best_resolved
            elif best_resolved:
                print(f"🎧 Spotify match too weak ('{best_resolved}', score {best_score:.2f}) - "
                      f"keeping raw query for YouTube search")
        except Exception as e:
            print(f"[Spotify Error] Search failed, falling back to raw query: {e}")

    song_queue.append(query)

    if not ctx.voice_client.is_playing() and not ctx.voice_client.is_paused():
        await play_next(ctx)
    else:
        await ctx.send(f"Queued: **{query}**")


@bot.command(name="next", aliases=["skip"])
async def next_song(ctx):
    """Skip to the next song in the queue."""
    if ctx.voice_client and (ctx.voice_client.is_playing() or ctx.voice_client.is_paused()):
        ctx.voice_client.stop()
        await ctx.send("⏭️ Skipped to the next song!")
    elif song_queue:
        await play_next(ctx)
    else:
        await ctx.send("Queue is empty!")

    # voice_client.stop() tears down the recording sink's cleanup() as a
    # side effect on this branch, without going through the normal
    # "recording ended" callback - restart proactively rather than rely on
    # that callback firing. Scheduled as its own task (not awaited inline)
    # since this may be running from inside the very watchdog task that
    # just got cancelled by the stop() call above.
    if getattr(ctx.voice_client, "_auto_listen_enabled", False):
        print("[Voice] Queue advanced - ensuring voice listening remains active...")
        bot.loop.create_task(start_listening(ctx, announce=False))


@bot.command()
async def listen(ctx):
    """Join channel and start listening for voice commands."""
    if not ctx.author.voice:
        return await ctx.send("Join a voice channel first!")

    vc = ctx.voice_client

    if vc and vc.is_connected() and getattr(vc, "_auto_listen_enabled", False):
        # Already actively listening - calling start_recording() again on
        # a session that's already recording raises ClientException
        # ("Already recording audio") and crashes the command instead of
        # doing anything useful. Just say so instead.
        return await ctx.send("👂 Already listening!")

    if not vc:
        # This branch's voice handshake has proven flaky before (see the
        # whole DAVE saga elsewhere in this file) - a transient
        # "connection reset" mid-handshake is a known-possible blip, not
        # necessarily a real problem. Retry a couple times with a short
        # backoff before giving up, instead of making the person manually
        # retype !listen themselves for what's often a one-off hiccup.
        last_error = None
        for attempt in range(3):
            try:
                vc = await ctx.author.voice.channel.connect()
                break
            except Exception as e:
                last_error = e
                print(f"[Voice] Connect attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.5)
        else:
            await ctx.send(
                f"❌ Couldn't connect to voice after 3 tries ({last_error}). "
                f"Try `!listen` again in a moment - this is usually transient."
            )
            return

    await start_listening(ctx)


@bot.command()
async def leave(ctx):
    """Disconnect from voice channel."""
    global song_queue, now_playing, active_sink
    song_queue = []
    now_playing = None
    active_sink = None
    if ctx.voice_client:
        ctx.voice_client._auto_listen_enabled = False  # intentional stop, don't self-heal
        try:
            ctx.voice_client.stop_recording()
        except discord.sinks.errors.RecordingException:
            pass  # wasn't recording, that's fine
        await ctx.voice_client.disconnect()


@bot.event
async def on_ready():
    print(f"Bot successfully logged in as {bot.user.name}")


# --- DEBUG SINK ---

class DebugWaveSink(discord.sinks.Sink):
    """Records raw audio from every speaker for !debugmic, with no wake-word/STT processing."""

    def __init__(self):
        super().__init__()
        self.buffers = {}

    def is_opus(self):
        return False

    def write(self, data, user):
        payload = getattr(data, "pcm", data)
        if not payload:
            return
        self.buffers.setdefault(user, bytearray()).extend(payload)


async def _debugmic_finished(sink, ctx):
    if not sink.buffers:
        await ctx.send("⚠️ No audio captured - check console for connection errors.")
        return
    for user, frames in sink.buffers.items():
        user_id = getattr(user, "id", user)
        out_path = f"raw_discord_audio_{user_id}.wav"
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(bytes(frames))
        print(f"Wrote {out_path} ({len(frames)} bytes)")
    await ctx.send("✅ Done! Check `raw_discord_audio_*.wav` in your SPOTIBOT folder.")


@bot.command()
async def debugmic(ctx):
    """Records exactly what the bot hears directly from Discord for 5 seconds, no processing."""
    if not ctx.author.voice:
        return await ctx.send("Join a voice channel!")

    vc = ctx.voice_client
    if not vc:
        vc = await ctx.author.voice.channel.connect()

    await ctx.send("🔴 Recording raw audio for 5 seconds... speak into your mic!")

    vc.start_recording(DebugWaveSink(), _debugmic_finished, ctx)
    await asyncio.sleep(5)

    try:
        vc.stop_recording()
    except discord.sinks.errors.RecordingException:
        pass  # already stopped/finished on its own


def main():
    bot.run(os.getenv("DISCORD_BOT_TOKEN"))


if __name__ == "__main__":
    main()