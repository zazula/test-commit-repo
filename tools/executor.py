"""Tool integration stubs."""

import contextlib
import io


def run_code(code: str) -> tuple[str, str]:
    """Execute code and return stdout and stderr as strings."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            exec(code, {})
        except Exception as exc:  # pragma: no cover - stub
            print(exc, file=stderr_buffer)
    return stdout_buffer.getvalue(), stderr_buffer.getvalue()
