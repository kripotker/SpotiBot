# SpotiBot

Created by [Kripotker](https://github.com/Kripotker).

A Discord voice bot: say a wake word, it transcribes your speech locally
(faster-whisper), figures out what you meant using a local AI model
(llama.cpp), and plays music via YouTube/Spotify search - all running on
your own PC, no cloud AI services involved.

## ⚠️ Before you start - please actually read this part

This bot depends on **an unmerged, experimental pull request**
([Pycord-Development/pycord#3159](https://github.com/Pycord-Development/pycord/pull/3159))
to receive voice audio at all. Discord's mandatory end-to-end voice
encryption (DAVE), fully enforced since March 2026, broke voice-receiving
in every stable release of every Python Discord library. That PR is
currently the only known working fix, anywhere, in Python.

What that means in practice:

- This is genuinely experimental software glued on top of someone else's
  in-progress, unreviewed code. It mostly works well, but you may
  occasionally hit rough edges that have nothing to do with SpotiBot's
  own code - voice sessions dropping and needing a `!leave` + `!listen`,
  odd errors during the initial voice handshake, etc. Built-in
  self-healing covers most of these automatically; see Troubleshooting
  below for the rest.
- **This release has been reviewed and confirmed clean by Microsoft's
  malware analysis team (WDSI), but you may still see a Windows
  SmartScreen warning on download/first run.** That's a separate,
  reputation-based system, not a sign anything's actually wrong - see the
  Download section and the dedicated Windows Defender section below for
  the full explanation.
- If/when that PR gets merged into a stable release, this project should
  become meaningfully more reliable. Until then, expect "experimental" to
  mean what it says.
- **Most of this codebase was written with substantial AI assistance**
  (Claude, by Anthropic) - see Credits below for what that actually
  means in practice.

---

# Part 1: Using SpotiBot (the app)

This part is for anyone who just wants to run SpotiBot - no coding, no
Python, nothing installed beyond what's listed below.

## What you'll need before downloading

Nothing you can bundle into an app can hand you these - they're
inherently tied to *your* accounts:

- **A Discord account** and a server where you can add bots.
- **A Spotify account** (free tier is fine) - just to look up song/artist
  info, not to play anything through Spotify itself.
- Optionally, a **Hugging Face account** - only matters if you plan to
  download extra AI models later; the default one works without this.

You do **not** need Python, Visual Studio, or any developer tools for
this part - those only matter in Part 2.

## Download

**This release has been reviewed by Microsoft's malware analysis team
(WDSI) and confirmed clean - no malware or Defender signature issue
found.** You may still see a **Windows SmartScreen** warning ("Windows
protected your PC" / unrecognized app) the first time you download or run
it, especially early on. That's expected and unrelated to the WDSI
result: SmartScreen and Defender are two separate systems. SmartScreen's
warning is based purely on download reputation (how new/uncommon a file
is), not on any actual malware determination - Microsoft's own support
team confirmed there's no way to manually clear that warning short of
paid code-signing and the reputation that builds up over time from
regular downloads. It'll fade as more people use this safely; it isn't a
sign anything is wrong with this specific file, which has already been
independently checked.

Grab the latest `SpotiBot-windows.zip` from
this repo's [Releases page](../../releases), extract it *anywhere you have write
permission* (your Desktop or Documents folder are fine - avoid
`Program Files`, since SpotiBot creates a few folders next to itself on
first run and needs permission to do that), and run `SpotiBot.exe` inside
the extracted folder.

**Important:** don't move `SpotiBot.exe` out of its folder by itself - it
needs the other files sitting alongside it (an `_internal` folder full of
supporting files) to actually run. Move or shortcut the whole extracted
folder, not just the exe.

## If Windows Defender flags it - here's why it might do that, and what to do

This file has already been submitted to and cleared by Microsoft's
malware analysis team (WDSI) - no malware or Defender signature issue
found, confirmed directly by Microsoft support. Individual machines can
still occasionally disagree with that in the moment (cloud protection
data doesn't always sync instantly everywhere), so you may still see
Windows quarantine the exe the first time you run it, or a SmartScreen
warning on download. Here's the honest reason why this can still happen,
not just "ignore it": SpotiBot downloads and runs other programs at
startup (the AI model backend, and possibly GPU support files) - which is
also a pattern real malware uses. It's a false alarm
here, but it's not an unreasonable thing for antivirus software to flag.

**To fix it:**
1. Windows Security -> Virus & threat protection -> Protection history ->
   find SpotiBot -> **Restore**.
2. Add an exclusion so it doesn't happen again: Virus & threat protection
   settings -> Manage settings -> Exclusions -> Add an exclusion -> Folder
   -> select your SpotiBot folder.

This is a reasonable ask if you trust where you got this from (this
repo), but it's also exactly why you should never do this for software
from a source you don't trust.

## First launch, step by step

The very first time you run `SpotiBot.exe`, two setup screens appear
before the main app does - both only ever show up once.

### 1. Credentials setup

A window asking for your Discord Bot Token and Spotify Client ID/Secret
(required), plus an optional Hugging Face token. A second window opens
alongside it with direct links and click-by-click instructions for
getting each one - keep both open side by side while you go get them.

**Quick reference for what each one is and where to get it:**

| Credential | Where to get it | Required? |
|---|---|---|
| Discord Bot Token | [discord.com/developers/applications](https://discord.com/developers/applications) -> New Application -> Bot tab | Yes |
| Spotify Client ID + Secret | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) -> Create app | Yes |
| Hugging Face Token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) -> New token | No - just makes downloads faster/less rate-limited |

**One thing easy to miss on the Discord Bot page:** scroll down to
"Privileged Gateway Intents" and turn on both **Message Content Intent**
and **Server Members Intent**, then save. The bot will refuse to start
without these - it's a Discord-side setting, not something this app can
turn on for you.

### 2. GPU/AI backend setup

A window tries to detect your graphics card and pre-selects a matching
option (CPU-only, NVIDIA/CUDA, AMD/ROCm, Vulkan, etc.) with a
plain-language explanation of each choice. This downloads the actual AI
engine (llama.cpp) that runs the assistant - nothing is installed without
you clicking "Download & Continue" yourself.

If you have an NVIDIA GPU with at least 8GB VRAM, you may also be asked
whether to run the speech-to-text model on your GPU too, for faster
responses - read the on-screen warning about VRAM sharing before saying
yes, especially if you also game while running this.

After this, the main app window opens, and the actual AI model download
starts automatically in the background (~1GB, one-time).

## Using the bot

Invite the bot to your server (the Discord Bot Token guide above covers
this), join a voice channel, then:

| Command | Voice equivalent | What it does |
|---|---|---|
| `!listen` | - | Joins your voice channel and starts listening |
| `!play <song>` | "hey bot, play [song]" | Queues and plays a track |
| `!next` / `!skip` | "hey bot, skip" | Skips to the next track |
| - | "hey bot, pause" / "hey bot, resume" | Pauses/resumes playback |
| `!leave` | - | Disconnects and stops listening |
| `!debugmic` | - | Records 5 seconds of raw audio to a file, for troubleshooting |

**Wake word mishearings are normal and fixable without touching code** -
if "hey bot" keeps coming out as "hey bought" or similar in the console
log, just add that exact wording as another wake phrase in Settings, see
below.

## The app window, explained

- **Left sidebar (Queue):** what's currently playing and what's queued
  next - reflects the real state of the bot live.
- **Right sidebar (Voice Channel):** everyone currently in the bot's
  voice channel, each with a checkbox controlling whether *their* voice
  can trigger the bot. Use "All"/"None" for the whole group, or check
  individual boxes for specific people. Changes apply instantly, mid-call
  - no need to reconnect.
- **Middle tabs:** live logs for the Discord bot itself and the AI model
  server - useful for seeing what's actually happening, and essential for
  troubleshooting if something's not working.
- **Bottom-right gear icon:** opens Settings.

## Settings

- **Voice** - add/remove wake-word phrases; change the speech-to-text
  model (bigger = more accurate but slower) and whether it runs on CPU or
  GPU; swap the notification sound (played when the bot starts listening
  or hears a wake word) for any `.mp3`/`.wav`/`.ogg`/`.m4a` file of your
  own.
- **AI Model** - browse and download different AI models (search Hugging
  Face directly, with size/speed guidance based on your GPU), switch GPU
  backends, or uninstall the current one to free disk space.
- **Credentials** - edit your Discord/Spotify/Hugging Face tokens without
  digging through files (restart the app after changing Discord/Spotify
  values; Hugging Face applies immediately).
- **Advanced** - verbose debug logging (off by default), fine-tuning
  knobs, and a reset-to-defaults option.

## Troubleshooting

- **Bot doesn't respond to anyone:** check the right sidebar - if it's
  narrowed to specific people, someone talking who isn't checked will be
  silently ignored. Click "All" to reset it.
- **Empty/garbled transcriptions:** check your mic isn't clipping (very
  loud, distorted audio transcribes badly) - Discord's own input volume
  meter should stay under the red zone while you talk.
- **A voice session seems stuck/unresponsive for a while:** this usually
  self-heals within a few minutes without you doing anything, thanks to
  built-in retry logic. If it doesn't recover, `!leave` then `!listen`
  forces a clean reconnect.
- **"Already listening!" instead of doing something:** you ran `!listen`
  while a session was already active - not an error, just informational.
- Turn on Settings -> Advanced -> verbose debug logging for much more
  detailed console output if you need to dig into any of the above
  further.

---

# Part 2: Building from source / development

This part is for anyone who wants to modify SpotiBot, understand how it
works internally, or build their own `.exe`.

## Prerequisites

- **Python 3.10-3.13** (not 3.14 - the py-cord PR branch's metadata
  currently excludes it).
- **Git.**
- **ffmpeg** on `PATH` (or bundled at build time - see below).
- **[Deno](https://deno.land/)**, for yt-dlp's JS challenge solving:
  `winget install DenoLand.Deno`

## Setup

```powershell
git clone <this-repo-url>
cd SpotiBot
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip uninstall discord.py discord-ext-voice-recv -y
python -m pip install "git+https://github.com/Pycord-Development/pycord@refs/pull/3159/head" -r requirements.txt pyinstaller
```

If `Activate.ps1` refuses to run ("running scripts is disabled on this
system"), run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
in the same terminal once, then retry.

Create a `.env` file next to `bot.py` (or just run `gui_launcher.py` once
- the same setup wizard from Part 1 handles this for you too):

```
DISCORD_BOT_TOKEN=your_bot_token
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
# HF_TOKEN=your_huggingface_token   # optional

# Optional overrides - only needed if llama-server.exe/model live
# somewhere non-default:
# LLAMA_SERVER_PATH=C:\path\to\llama-server.exe
# LLAMA_MODEL_PATH=models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
# LLAMA_SERVER_PORT=8080
# LLAMA_NGL=99
```

Place a short `ping.mp3` cue file next to `bot.py` (played when the bot
starts listening or hears a wake word).

## Running from source

```powershell
python gui_launcher.py
```

`bot.py` can also be run directly (console output instead of the GUI),
but you'd need `llama-server.exe` running separately yourself in that
case - the GUI normally manages that for you.

## Project structure

| File | What it does |
|---|---|
| `bot.py` | The actual Discord bot - voice handling, wake-word detection, Whisper transcription, LLM intent parsing, playback |
| `gui_launcher.py` | The desktop app - runs `bot.py` in a background thread, manages the AI model subprocess, all the Settings/setup UI |
| `theme.py` | Shared design system (colors, fonts, styled-widget factories) used across every window |
| `gpu_detect.py` | Best-effort local GPU detection, used only to *suggest* backends/models - never auto-applies anything |
| `settings_store.py` | Shared JSON-backed settings (`spotibot_settings.json`), read/written by both `bot.py` and `gui_launcher.py` live |
| `requirements.txt` | Python dependencies (py-cord installed separately - see Setup) |

## Building your own `.exe`

```powershell
pyinstaller --onedir --noupx --noconsole --name SpotiBot --add-data "ping.mp3;." --collect-all faster_whisper --collect-all ctranslate2 --collect-all discord --collect-all yt_dlp --collect-all nacl gui_launcher.py
```

`--onedir` (not `--onefile`) avoids a slow self-extraction step on every
launch. `--noupx` and using a recent PyInstaller version both help reduce
antivirus false positives - see the Troubleshooting section in Part 1 for
why those happen at all.

### Optional: bundling ffmpeg and Deno into the exe too

By default the built exe still expects `ffmpeg`/`ffprobe` and `deno` to
already be installed and on `PATH`. You can bundle them directly instead,
for a more complete out-of-the-box experience:

1. Download a static Windows ffmpeg build (e.g. the "essentials" build
   from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)) and
   copy `ffmpeg.exe`/`ffprobe.exe` into this folder.
2. Download `deno.exe` from the Windows x86_64 zip on
   [github.com/denoland/deno/releases](https://github.com/denoland/deno/releases)
   (the standalone portable build, not the winget install) and copy it
   here too.
3. Add these flags to the build command above:
   ```
   --add-binary "ffmpeg.exe;." --add-binary "ffprobe.exe;." --add-binary "deno.exe;."
   ```

`gui_launcher.py` automatically detects these at startup (only when
running as the built exe) and adds them to `PATH` for that session -
nothing else needs to change for this to work.

**License note:** the gyan.dev ffmpeg build is GPLv3-licensed. Bundling
it this way is legally fine (FFmpeg runs as a separate subprocess, not
linked into this app), but see `THIRD_PARTY_LICENSES.md` for what that
does and doesn't require, and for a lighter-weight LGPL alternative if
you'd rather avoid GPL entirely.

### Optional: bundling GPU-Whisper support too

The NVIDIA cuBLAS/cuDNN libraries needed for GPU-accelerated Whisper are
*not* bundled by default - they're downloaded on-demand only if someone
actually opts into GPU Whisper from the app itself, keeping the default
build much smaller. This is intentional; there's no supported way to
force-bundle these into the exe short of undoing that on-demand logic in
`gui_launcher.py` yourself.

## Publishing a release

Build with `--onedir`, zip the whole `dist\SpotiBot\` folder (not just
the exe), and attach it to a GitHub Release rather than committing it to
the repo - the built output is large and isn't source code (see
`.gitignore`).

Include `README.md`, `THIRD_PARTY_LICENSES.md`, and `LICENSE` in the zip
too, not just on the repo page - anyone holding just the exe (forwarded
by a friend, no idea it came from GitHub) should still be able to see
what's bundled, under what terms, and how to actually use it.

```powershell
Compress-Archive -Path "dist\SpotiBot\*", "README.md", "THIRD_PARTY_LICENSES.md", "LICENSE" -DestinationPath "SpotiBot-windows.zip"
```

Then on GitHub: Releases -> Create a new release -> tag it -> attach the
zip -> publish.

## License

SpotiBot's own code is licensed under the [MIT License](LICENSE) - free to
use, modify, and redistribute, including commercially, as long as the
copyright notice stays attached.

That covers only the code in this repository. See `THIRD_PARTY_LICENSES.md`
for everything bundled into the built exe (FFmpeg, Deno, and the various
Python libraries) and their own, separate licensing terms.

## Credits

SpotiBot was created by [Kripotker](https://github.com/Kripotker).

**AI disclosure:** the large majority of the code in this repository was
written by Claude (Anthropic), working iteratively with Kripotker across
an extended back-and-forth. Direction, testing on real hardware, bug
reports from real usage, and design/feature decisions all came from
Kripotker, who also wrote some parts of the code directly and debugged
some issues independently - most debugging beyond that happened jointly,
working through real logs and errors together. Claude also drafted this
documentation. If you're evaluating this project's code quality,
architecture, or reliability, that development process is worth keeping
in mind.