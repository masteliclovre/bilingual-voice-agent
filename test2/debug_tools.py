"""Debugging helpers shared by the local and RunPod agents.

These utilities install verbose exception hooks so we can capture the full
traceback for crashes like ``Error in sys.excepthook`` and gather useful GPU
environment diagnostics when the process boots.  The log destination can be
overridden with the ``VOICE_AGENT_DEBUG_LOG`` environment variable.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict


def _log_path() -> Path:
    env_path = os.getenv("VOICE_AGENT_DEBUG_LOG", "agent_debug.log")
    path = Path(env_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_log(block: str) -> None:
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    message = f"[{timestamp}]\n{block.rstrip()}\n\n"
    path = _log_path()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message)


def install_exception_logging(component: str) -> None:
    """Capture uncaught exceptions for debugging the agent.

    ``sys.excepthook`` errors hide the underlying traceback.  By capturing both
    the original error and any hook failures into a persistent log file we can
    inspect the real failure after the fact.
    """

    log_header = f"Unhandled exception in {component}"

    def _hook(exc_type, exc_value, exc_tb):
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _write_log(f"{log_header}\n{tb_str}")
        # Still print to stderr so the process surfaces the failure.
        traceback.print_exception(exc_type, exc_value, exc_tb)

    def _thread_hook(args) -> None:
        tb_str = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        thread_header = f"Unhandled thread exception in {component} (thread: {args.thread.name})"
        _write_log(f"{thread_header}\n{tb_str}")
        if hasattr(threading, "__excepthook__"):
            threading.__excepthook__(args)  # type: ignore[attr-defined]
        else:
            sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook  # type: ignore[assignment]


def _check_module(name: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - optional modules
        return {"available": False, "error": type(exc).__name__, "detail": str(exc)}

    info: Dict[str, Any] = {"available": True}
    for attr, getter in attrs.items():
        try:
            if callable(getter):
                info[attr] = getter(module)
            else:
                info[attr] = getattr(module, getter)
        except Exception as exc:  # pragma: no cover - runtime specific
            info[attr] = {"error": type(exc).__name__, "detail": str(exc)}
    return info


def log_startup_diagnostics(component: str) -> None:
    """Dump useful runtime diagnostics to help debug GPU / dependency issues."""

    env_snapshot = {
        key: os.getenv(key)
        for key in (
            "WHISPER_DEVICE",
            "WHISPER_COMPUTE",
            "CUDA_VISIBLE_DEVICES",
            "RUNPOD_POD_ID",
            "HF_HUB_ENABLE_HF_TRANSFER",
        )
    }

    payload: Dict[str, Any] = {
        "component": component,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "env": env_snapshot,
    }

    payload["modules"] = {
        "torch": _check_module(
            "torch",
            {
                "version": lambda m: getattr(m, "__version__", "?"),
                "cuda_available": lambda m: bool(
                    getattr(getattr(m, "cuda", None), "is_available", lambda: False)()
                ),
                "cuda_device_count": lambda m: int(
                    getattr(getattr(m, "cuda", None), "device_count", lambda: 0)()
                ),
            },
        ),
        "ctranslate2": _check_module(
            "ctranslate2",
            {
                "version": lambda m: getattr(m, "__version__", "?"),
                "cuda_available": lambda m: bool(
                    getattr(m, "is_cuda_available", lambda: False)()
                ),
            },
        ),
        "sounddevice": _check_module("sounddevice", {"version": lambda m: getattr(m, "__version__", "?")}),
        "numpy": _check_module("numpy", {"version": lambda m: getattr(m, "__version__", "?")}),
    }

    _write_log(json.dumps(payload, indent=2, sort_keys=True))
