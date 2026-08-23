"""
Design system for the SpotiBot GUI - one place for colors, type, and
styled-widget factories, so every window/dialog looks like part of the
same product instead of ad-hoc gray tkinter defaults.
"""
import os
import tkinter as tk
from tkinter import ttk

# --- Palette ---
BG = "#0f1115"           # app background
SURFACE = "#171a21"       # cards/panels, one step up from BG
SURFACE_ALT = "#1e222b"   # hover states, alternate rows, inputs
BORDER = "#2a2f3a"

TEXT_PRIMARY = "#eef1f6"
TEXT_SECONDARY = "#a8b0bd"
TEXT_MUTED = "#6b7280"

ACCENT = "#1DB954"        # brand accent (Spotify green - thematically on-brand)
ACCENT_HOVER = "#1ed760"
ACCENT_TEXT = "#08130c"   # dark text for use on the bright accent color

DANGER = "#e5484d"
DANGER_HOVER = "#ff5a5f"
WARNING = "#e0a030"
INFO = "#4a9eff"
SUCCESS = ACCENT

FONT = "Segoe UI"
FONT_MONO = "Consolas"


def apply_dark_titlebar(window):
    """
    Makes a window's native Windows title bar dark instead of the default
    white/light one that clashes with the app's dark theme (the white bar
    visible at the top of every screenshot). Windows 10 (1809+) and 11
    only, via the documented but easy-to-miss DwmSetWindowAttribute /
    DWMWA_USE_IMMERSIVE_DARK_MODE call. Silently does nothing anywhere
    else, or if anything about this fails - it's cosmetic, never worth
    breaking the app over.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        window.update_idletasks()  # ensure the native window handle actually exists yet
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1)
        # Attribute ID differs across Windows 10/11 builds - 20 is current
        # (post-20H1), 19 is the original early-Win10 value. Try both;
        # whichever doesn't apply on a given build is simply a no-op.
        for attribute in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            )
    except Exception:
        pass


def configure_ttk_style():
    """Call once, before creating any ttk widgets. Restyles Notebook and
    Scrollbar to match the palette - their OS-default look is the single
    biggest giveaway of an unstyled tkinter app."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure(
        "TNotebook.Tab", background=SURFACE, foreground=TEXT_SECONDARY,
        borderwidth=0, font=(FONT, 10),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", SURFACE_ALT), ("!selected", SURFACE)],
        foreground=[("selected", TEXT_PRIMARY), ("!selected", TEXT_SECONDARY)],
        # "expand" (previously used here) turned out not to be respected
        # by clam's tab layout the way other ttk themes handle it - it had
        # no effect, or made the selected tab shrink instead. Padding is a
        # much more universally-honored per-state option: giving the
        # selected tab more vertical padding directly makes it taller/more
        # prominent than the unselected ones, with no dependence on how a
        # given theme implements tab "expansion."
        padding=[("selected", (16, 12)), ("!selected", (16, 6))],
    )

    style.configure(
        "Vertical.TScrollbar", background=SURFACE_ALT, troughcolor=BG,
        bordercolor=BG, arrowcolor=TEXT_SECONDARY, relief="flat", borderwidth=0,
    )
    style.map("Vertical.TScrollbar", background=[("active", BORDER)])

    style.configure(
        "TProgressbar", background=ACCENT, troughcolor=SURFACE_ALT,
        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT,
    )


def button(parent, text, command, kind="secondary", **kwargs):
    """kind: 'primary' (accent-filled), 'secondary' (default), 'danger', or 'ghost'."""
    palette = {
        "primary": (ACCENT, ACCENT_HOVER, ACCENT_TEXT),
        "secondary": (SURFACE_ALT, BORDER, TEXT_PRIMARY),
        "danger": (DANGER, DANGER_HOVER, "#ffffff"),
        "ghost": (BG, SURFACE_ALT, TEXT_SECONDARY),
    }
    bg, hover_bg, fg = palette.get(kind, palette["secondary"])
    weight = "bold" if kind in ("primary", "danger") else "normal"

    btn = tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=hover_bg, activeforeground=fg,
        font=(FONT, kwargs.pop("size", 10), kwargs.pop("weight", weight)),
        relief="flat", bd=0, cursor="hand2",
        padx=kwargs.pop("padx", 14), pady=kwargs.pop("pady", 8),
        **kwargs,
    )
    btn.bind("<Enter>", lambda e: btn.configure(bg=hover_bg))
    btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
    return btn


def label(parent, text="", kind="body", **kwargs):
    """kind: 'title', 'heading', 'body', 'secondary', or 'muted'."""
    styles = {
        "title": (TEXT_PRIMARY, (FONT, 18, "bold")),
        "heading": (TEXT_PRIMARY, (FONT, 12, "bold")),
        "body": (TEXT_PRIMARY, (FONT, 10)),
        "secondary": (TEXT_SECONDARY, (FONT, 9)),
        "muted": (TEXT_MUTED, (FONT, 8)),
    }
    fg, font = styles.get(kind, styles["body"])
    return tk.Label(
        parent, text=text, bg=kwargs.pop("bg", BG), fg=kwargs.pop("fg", fg),
        font=kwargs.pop("font", font), **kwargs,
    )


def entry(parent, textvariable=None, **kwargs):
    e = tk.Entry(
        parent, textvariable=textvariable, bg=SURFACE_ALT, fg=TEXT_PRIMARY,
        insertbackground=TEXT_PRIMARY, relief="flat", bd=0,
        font=kwargs.pop("font", (FONT, 10)),
        highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
        **kwargs,
    )
    return e


def listbox(parent, **kwargs):
    return tk.Listbox(
        parent, bg=SURFACE_ALT, fg=TEXT_PRIMARY, selectbackground=ACCENT,
        selectforeground=ACCENT_TEXT, relief="flat", bd=0, highlightthickness=0,
        font=kwargs.pop("font", (FONT_MONO, 9)), **kwargs,
    )


def frame(parent, **kwargs):
    return tk.Frame(parent, bg=kwargs.pop("bg", BG), **kwargs)


def card(parent, **kwargs):
    """A subtly-bordered panel: an outer frame in BORDER color with a 1px
    gap showing through to an inner SURFACE frame - the standard trick for
    a bordered card in plain tkinter (no rounded corners, but reads as a
    distinct, elevated panel rather than flat background)."""
    outer = tk.Frame(parent, bg=BORDER, **kwargs)
    inner = tk.Frame(outer, bg=SURFACE)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    return outer, inner


def pill(parent, textvariable, fg=TEXT_SECONDARY, bg=SURFACE_ALT):
    """Small rounded-looking status chip (padding-as-pill, same technique
    as `card` - no true rounded corners in plain tkinter, but the padded
    colored background reads as a chip rather than plain text)."""
    return tk.Label(
        parent, textvariable=textvariable, bg=bg, fg=fg,
        font=(FONT, 9, "bold"), padx=10, pady=4,
    )


def checkbutton(parent, text, variable, **kwargs):
    bg = kwargs.pop("bg", BG)
    return tk.Checkbutton(
        parent, text=text, variable=variable, bg=bg, fg=TEXT_PRIMARY,
        selectcolor=SURFACE_ALT, activebackground=bg, activeforeground=TEXT_PRIMARY,
        font=(FONT, 9), anchor="w", relief="flat", bd=0, highlightthickness=0, **kwargs,
    )


def radiobutton(parent, text, variable, value, **kwargs):
    return tk.Radiobutton(
        parent, text=text, variable=variable, value=value,
        bg=kwargs.pop("bg", BG), fg=TEXT_PRIMARY, selectcolor=SURFACE_ALT,
        activebackground=kwargs.pop("bg", BG), activeforeground=TEXT_PRIMARY,
        relief="flat", bd=0, highlightthickness=0,
        font=kwargs.pop("font", (FONT, 10)), **kwargs,
    )