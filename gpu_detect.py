"""
Best-effort local GPU detection, used to *suggest* a llama.cpp backend and
a Hugging Face model/quant that should run well on this machine.

Detection failing, being wrong, or finding nothing is never a hard error -
callers always fall back to showing a normal, fully-manual picker either
way. Nothing here downloads or installs anything; it only produces a
suggestion string plus a reason, which the caller displays and the person
must still confirm themselves.
"""
import re
import subprocess

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


def _run(cmd, timeout=8) -> str:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _registry_vram_gb() -> dict:
    """
    Win32_VideoController's AdapterRAM is a 32-bit field that overflows on
    cards above 4GB - a long-documented Windows quirk, and exactly what
    caused a 20GB card to read as ~4GB. GPU drivers separately write the
    real figure into the registry under each display adapter's class key
    as a proper 64-bit value, which doesn't have that overflow problem.
    Returns {driver_description: vram_gb}, matched against Win32_VideoController's
    Name by the caller (best-effort - both come from the same driver
    metadata, so they should usually agree, but this isn't guaranteed).
    """
    ps_script = (
        "Get-ItemProperty -Path "
        "'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
        "-Name 'HardwareInformation.qwMemorySize','DriverDesc' -ErrorAction SilentlyContinue | "
        "ForEach-Object { \"$($_.DriverDesc)|$($_.'HardwareInformation.qwMemorySize')\" }"
    )
    out = _run(["powershell", "-NoProfile", "-Command", ps_script])
    results = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        name, _, size_str = line.rpartition("|")
        name = name.strip()
        try:
            size_bytes = int(size_str.strip())
        except ValueError:
            continue
        if name and size_bytes > 0:
            results[name] = round(size_bytes / (1024 ** 3), 1)
    return results


def detect_gpus() -> list:
    """Returns [{"name": str, "vram_gb": float|None, "vendor": str}, ...]."""
    gpus = []

    # nvidia-smi, when present, gives an exact name + VRAM figure directly -
    # far more reliable than anything else here.
    nv_out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if nv_out:
        for line in nv_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                vram_gb = round(float(parts[1]) / 1024, 1)
            except ValueError:
                vram_gb = None
            gpus.append({"name": parts[0], "vram_gb": vram_gb, "vendor": "nvidia"})
        if gpus:
            return gpus

    # Generic fallback via WMI/PowerShell - works for any vendor, and also
    # covers the case where an NVIDIA card is present but nvidia-smi isn't
    # on PATH.
    registry_vram = _registry_vram_gb()  # accurate, not capped - preferred when a name match is found

    ps_out = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }",
    ])
    for line in ps_out.splitlines():
        if "|" not in line:
            continue
        name, _, ram_str = line.rpartition("|")
        name = name.strip()
        if not name:
            continue

        lname = name.lower()
        if any(k in lname for k in ("nvidia", "geforce", "rtx", "gtx")):
            vendor = "nvidia"
        elif any(k in lname for k in ("amd", "radeon")):
            vendor = "amd"
        elif "intel" in lname:
            vendor = "intel"
        elif any(k in lname for k in ("adreno", "qualcomm")):
            vendor = "qualcomm"
        else:
            vendor = "unknown"

        # Prefer the registry figure - only fall back to the flaky
        # AdapterRAM field if no matching registry entry was found.
        vram_gb = registry_vram.get(name)
        if vram_gb is None:
            try:
                ram_bytes = int(ram_str.strip())
                if ram_bytes > 0:
                    vram_gb = round(ram_bytes / (1024 ** 3), 1)
            except ValueError:
                pass

        gpus.append({"name": name, "vram_gb": vram_gb, "vendor": vendor})

    return gpus


def _nvidia_generation(name: str):
    """Rough architecture classification from the model number, used when
    we only have a name string (no compute capability from nvidia-smi)."""
    m = re.search(r"\b(RTX|GTX)\s*(\d{3,4})", name.upper())
    if not m:
        return None
    series, number = m.group(1), int(m.group(2))
    if series == "GTX":
        return "turing" if number >= 1600 else "pre-turing"  # GTX 16xx is Turing
    gen = number // 1000  # 20xx->2, 30xx->3, 40xx->4, 50xx->5
    return {2: "turing", 3: "ampere", 4: "ada"}.get(gen, "blackwell" if gen >= 5 else None)


def suggest_backend(gpus: list):
    """Returns (backend_key, reason_str). Always returns something usable,
    even with no GPU detected at all (falls back to CPU)."""
    if not gpus:
        return "cpu-x64", "No dedicated GPU detected - CPU is the safe default."

    gpu = max(gpus, key=lambda g: (g.get("vram_gb") or 0))
    name, vendor = gpu["name"], gpu["vendor"]

    if vendor == "nvidia":
        gen = _nvidia_generation(name)
        if gen == "pre-turing":
            return "cuda12-x64", f"Detected {name} - CUDA 13 dropped support for this generation, using CUDA 12."
        if gen in ("turing", "ampere", "ada", "blackwell"):
            return "cuda13-x64", f"Detected {name} - CUDA 13 is recommended for this generation."
        return "cuda12-x64", f"Detected {name} - using CUDA 12 for broad compatibility."

    if vendor == "amd":
        return "rocm-x64", f"Detected {name} - ROCm should give the best performance."
    if vendor == "intel":
        return "openvino-x64", f"Detected {name} - OpenVINO is Intel's own optimized backend."
    if vendor == "qualcomm":
        return "opencl-adreno-arm64", f"Detected {name} - using the Adreno GPU backend."

    return "vulkan-x64", f"Detected {name} - using Vulkan as a broadly-compatible option."


def suggest_model(gpus: list):
    """Returns (search_query, recommended_quant, reason_str) for the HF
    browser to pre-fill and highlight, sized to available VRAM."""
    if not gpus:
        return (
            "qwen2.5 1.5b instruct gguf", "Q4_K_M",
            "No GPU detected at all - recommending a small model that runs reasonably on CPU.",
        )

    known_vram = [g["vram_gb"] for g in gpus if g.get("vram_gb")]
    if not known_vram:
        names = ", ".join(g["name"] for g in gpus)
        return (
            "qwen2.5 1.5b instruct gguf", "Q4_K_M",
            f"Detected {names}, but couldn't determine its VRAM - recommending a small model "
            f"to be safe. If you know it has more headroom, search for a larger model manually.",
        )

    vram_gb = max(known_vram)
    if vram_gb < 6:
        return "qwen2.5 1.5b instruct gguf", "Q4_K_M", f"~{vram_gb:.0f}GB VRAM detected - a small 1.5B model fits comfortably."
    if vram_gb < 10:
        return "qwen2.5 7b instruct gguf", "Q4_K_M", f"~{vram_gb:.0f}GB VRAM detected - a 7B model fits well at this quant."
    if vram_gb < 16:
        return "qwen2.5 7b instruct gguf", "Q5_K_M", f"~{vram_gb:.0f}GB VRAM detected - room for a higher-quality quant of a 7B model."
    return "qwen2.5 14b instruct gguf", "Q4_K_M", f"~{vram_gb:.0f}GB VRAM detected - a larger 14B model fits and answers noticeably better."


# General reference guidance for the Hugging Face browser's help panel -
# each entry: (upper VRAM bound in GB, exclusive; label for that bracket,
# what model sizes fit, and roughly how it feels to use). Static/
# well-established rules of thumb, not measured on this specific machine.
VRAM_MODEL_GUIDANCE = [
    (4, "< 4GB", "1-2B models (e.g. Qwen2.5-1.5B)", "🚀 Very fast - near-instant responses"),
    (8, "4-8GB", "3-8B models (e.g. Qwen2.5-7B, Llama-3-8B)", "⚡ Fast - roughly 1-3 seconds per response"),
    (12, "8-12GB", "8-13B models", "⚡ Moderate - a few seconds per response"),
    (16, "12-16GB", "13-20B models (e.g. Qwen2.5-14B)", "🐢 Slower - several seconds, noticeably smarter answers"),
    (24, "16-24GB", "20-34B models", "🐢 Slow - many seconds per response, quality over speed"),
    (float("inf"), "24GB+", "34B+ models", "🐌 Very slow on one consumer GPU - approaching datacenter territory"),
]


def guidance_tier_index(gpus: list) -> int:
    """Which VRAM_MODEL_GUIDANCE row matches detected hardware, for the
    caller to highlight. Returns 0 (the smallest/safest bracket) if VRAM
    is unknown or no GPU was found - matches suggest_model()'s same
    safe-default behavior."""
    known_vram = [g["vram_gb"] for g in gpus if g.get("vram_gb")]
    vram_gb = max(known_vram) if known_vram else 0
    for i, (upper_bound, *_rest) in enumerate(VRAM_MODEL_GUIDANCE):
        if vram_gb < upper_bound:
            return i
    return len(VRAM_MODEL_GUIDANCE) - 1
