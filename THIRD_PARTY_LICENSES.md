# Third-Party Licenses

SpotiBot's own code (`bot.py`, `gui_launcher.py`, `theme.py`, `gpu_detect.py`,
`settings_store.py`) is original and covered by the [MIT License](LICENSE)
in this repository. The built `.exe` also bundles several third-party
tools and libraries, each under their own license. This document
summarizes what's included and what that means for you.

**This is a good-faith summary, not legal advice, and the author of this
project is not a lawyer.** If you're building something where license
compliance actually matters for you (commercial use, redistribution at
scale, etc.), verify the details yourself against each project's own
license file linked below.

---

## Bundled binaries (external tools, not linked into the app)

These are separate programs SpotiBot calls as subprocesses - not code
compiled or linked into the app itself.

### FFmpeg / ffprobe

- **License: GPLv3** (using the standard gyan.dev "essentials" build -
  see [www.gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/))
- **Source:** [ffmpeg.org](https://ffmpeg.org) / [github.com/GyanD/codexffmpeg](https://github.com/GyanD/codexffmpeg)
- **Full license text:** [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt)

FFmpeg is redistributed here as a separate, unmodified executable that
SpotiBot invokes as a subprocess (not linked into the app) - this is
"mere aggregation" under the GPL, which does not require SpotiBot's own
code to be GPL-licensed. It does mean this notice, the license text, and
a pointer to FFmpeg's own source (both linked above) need to travel
alongside the binary, which is why this file exists.

**If you build your own release and only need audio (no video)**, an
LGPL-only build is available from
[BtbN/FFmpeg-Builds on GitHub](https://github.com/BtbN/FFmpeg-Builds/releases)
(look for filenames containing `lgpl`) - it drops GPL-only components like
libx264 that this project never uses anyway, and carries lighter
redistribution terms.

### Deno

- **License: MIT**
- **Source:** [github.com/denoland/deno](https://github.com/denoland/deno)
- **Full license text:** [MIT License](https://github.com/denoland/deno/blob/main/LICENSE.md)

Used by yt-dlp to solve YouTube's JavaScript challenges. MIT is
permissive - the only real obligation is preserving the copyright/license
notice, which this file does.

---

## PyInstaller (the packaging tool itself)

PyInstaller is GPL-licensed but ships a specific **Bootloader Exception**:
the executables it produces can be distributed under any license you
choose, and you don't need to include PyInstaller's own license file with
your app. This is why bundling SpotiBot into a `.exe` at all doesn't
force this project into GPL. See
[pyinstaller.org/en/stable/license.html](https://pyinstaller.org/en/stable/license.html)
for the exact terms.

---

## Bundled Python libraries

PyInstaller bundles the actual Python packages SpotiBot depends on
directly into the `.exe`. The major ones and their licenses:

| Package | License | Notes |
|---|---|---|
| py-cord (PR #3159 branch) | MIT | Fork of discord.py |
| faster-whisper | MIT | |
| CTranslate2 | MIT | faster-whisper's inference backend |
| numpy | BSD-3-Clause | |
| PyNaCl | Apache-2.0 / BSD (dual) | Discord voice encryption |
| yt-dlp | Unlicense (public domain) | |
| spotipy | MIT | |
| python-dotenv | BSD-3-Clause | |
| openai (Python client) | Apache-2.0 | Used only to talk to your local llama-server, not OpenAI's actual API |
| requests | Apache-2.0 | |

All of the above are permissive licenses (attribution-only, no
requirement to disclose SpotiBot's own source because of them).

**This table covers the direct, major dependencies - it is not
necessarily exhaustive** (each of these pulls in its own smaller
dependencies too). For a complete, current list generated directly from
your actual build environment, run this in your venv before building:

```powershell
pip install pip-licenses
pip-licenses --format=markdown --with-urls
```

That prints every installed package and its license in one table - worth
doing periodically as dependencies get updated, and better ground truth
than any manually-maintained list (including this one) can promise to
stay in sync with.

---

## GPU-related components (downloaded on demand, not bundled in the exe)

These are *not* included in the `.exe` itself - they're downloaded
separately at runtime only if you actually opt into them (see the README):

- **llama.cpp** (GPU/CPU backend for the local AI model) - MIT License.
  [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **NVIDIA cuBLAS / cuDNN** (GPU Whisper support) - proprietary, under
  [NVIDIA's own license terms](https://docs.nvidia.com/cuda/eula/index.html) -
  downloaded directly from NVIDIA's official PyPI packages, not
  redistributed by this project.