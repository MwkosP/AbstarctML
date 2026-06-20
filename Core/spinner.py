from __future__ import annotations

import itertools
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any


@contextmanager
def runSpinner(message: str, enabled: bool = True):
    """Show a terminal spinner while a block runs."""
    if not enabled or not sys.stderr.isatty():
        yield
        return

    stop_event = threading.Event()
    thread = threading.Thread(target=_spin, args=(message, stop_event), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        sys.stderr.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stderr.flush()


def withSpinner(message: str | None = None, enabled: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function with a terminal spinner."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            spinner_message = message or f"Running {fn.__name__}"
            with runSpinner(spinner_message, enabled=enabled):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def _spin(message: str, stop_event: threading.Event) -> None:
    for frame in itertools.cycle("|/-\\"):
        if stop_event.is_set():
            break

        sys.stderr.write(f"\r{frame} {message}")
        sys.stderr.flush()
        time.sleep(0.1)
