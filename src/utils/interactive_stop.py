import select
import sys
import threading

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


class GracefulStopController:
    """Listen for a quit key (default: q) and signal graceful early stop."""

    def __init__(self, enabled=True, logger=None):
        self.enabled = bool(enabled)
        self.stop_key = "q"
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread = None
        self._stdin_fd = None
        self._stdin_settings = None

    @property
    def stop_requested(self):
        return self._stop_event.is_set()

    def request_stop(self, reason=None):
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        message = reason or "Early stop requested."
        if self.logger:
            self.logger.log(message)
        else:
            print(message, flush=True)

    def _confirm_stop_request(self):
        """Confirm stop action after a quit-key press to avoid accidental termination."""
        print(
            f"Pressed '{self.stop_key}'. Stop new work and finalize partial results? [y/n]: ",
            end="",
            flush=True,
        )
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except Exception:
                print("n")
                return False
            if not ready:
                continue
            try:
                response = sys.stdin.read(1)
            except Exception:
                print("n")
                return False
            if not response:
                continue
            answer = response.strip().lower()
            if answer in ("y", "n"):
                print(answer, flush=True)
                return answer == "y"

    def _listen_for_keypress(self):
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except Exception:
                break

            if not ready:
                continue

            try:
                char = sys.stdin.read(1)
            except Exception:
                break

            if char and char.lower() == self.stop_key:
                if self._confirm_stop_request():
                    self._stop_event.set()
                    print(
                        f"Confirmed stop after '{self.stop_key}'. Stopping new submissions and finalizing available outputs.",
                        flush=True,
                    )
                    break
                print("Stop canceled. Continuing execution.", flush=True)

    def start(self):
        if not self.enabled:
            return

        if termios is None or tty is None:
            if self.logger:
                self.logger.log("Quit-key listener unavailable on this platform; continuing without key listener.")
            self.enabled = False
            return

        if not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            if self.logger:
                self.logger.log("Quit-key listener disabled (stdin is not a TTY).")
            self.enabled = False
            return

        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_settings = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
        except Exception as exc:
            if self.logger:
                self.logger.log(f"Quit-key listener disabled: failed to configure terminal input ({exc}).")
            self.enabled = False
            return

        self._thread = threading.Thread(target=self._listen_for_keypress, daemon=True)
        self._thread.start()
        if self.logger:
            self.logger.log(
                f"Interactive stop enabled: press '{self.stop_key}' to stop new work and finalize partial results."
            )

    def close(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        if self._stdin_fd is not None and self._stdin_settings is not None and termios is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_settings)
            except Exception:
                pass
