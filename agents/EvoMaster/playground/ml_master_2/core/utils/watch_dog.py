"""Watchdog module for timeout control and forced interruption."""
import ctypes
import os
import threading

RUN_TIMEOUT_SECONDS = int(os.getenv("MLMASTER2_RUN_TIMEOUT_SECONDS", "86400"))


# Must inherit from BaseException to prevent being caught by lower-level except Exception:
class GlobalTimeoutInterrupt(BaseException):
    """Global timeout exception for watchdog forced interruption."""
    pass


def _async_raise(target_tid, exception_type):
    """Forcefully raise an exception in the specified thread via C-API.

    Args:
        target_tid: Target thread ID.
        exception_type: Exception type to raise.

    Raises:
        ValueError: If thread ID is invalid.
        SystemError: If PyThreadState_SetAsyncExc call fails.
    """
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(target_tid),
        ctypes.py_object(exception_type)
    )
    if ret == 0:
        raise ValueError("Invalid thread ID")
    elif ret > 1:
        # If return value > 1, indicates abnormal state, need to undo operation
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), None)
        raise SystemError("PyThreadState_SetAsyncExc call failed")


class TimeoutWatchdog:
    """Timeout watchdog for enforcing hard time limits on execution.

    Uses a daemon thread to monitor execution time and forcefully interrupts
    the main thread by injecting a GlobalTimeoutInterrupt exception if the
    timeout is exceeded.
    """

    def __init__(self, timeout_seconds: int):
        """Initialize the watchdog.

        Args:
            timeout_seconds: Maximum execution time in seconds.
        """
        self.timeout_seconds = timeout_seconds
        self.cancel_event = threading.Event()
        self.main_thread_id = threading.get_ident()  # Record main thread ID that started watchdog
        self._thread = None

    def start(self):
        """Start the watchdog daemon thread."""
        self._thread = threading.Thread(target=self._watch, daemon=True, name="TimeoutWatchdog")
        self._thread.start()

    def _watch(self):
        """Internal watchdog loop."""
        # Wait for specified timeout, or until stop() is called to trigger event
        is_cancelled = self.cancel_event.wait(self.timeout_seconds)
        if not is_cancelled:
            # Time's up and not cancelled normally -> trigger main thread interruption!
            _async_raise(self.main_thread_id, GlobalTimeoutInterrupt)

    def stop(self):
        """Cancel the watchdog when main logic ends normally."""
        self.cancel_event.set()
