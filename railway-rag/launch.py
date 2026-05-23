#!/usr/bin/env python3
"""
Railway RAG Launcher

Opens both local_builder.py (port 8501) and app.py (port 8502)
via Streamlit using the project's venv or system Python 3.11.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BUILDER_SCRIPT = ROOT_DIR / "local_builder.py"
APP_SCRIPT = ROOT_DIR / "app.py"
BUILDER_PORT = "8501"
APP_PORT = "8502"


def find_python() -> Path:
    """Locate Python from venv or system PATH. Prefers project venv."""
    candidates = [
        ROOT_DIR / "venv311",
        ROOT_DIR.parent / "venv311",
        ROOT_DIR.parent / "venv",
    ]
    for venv_dir in candidates:
        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"
        if venv_python.exists():
            return venv_python

    import shutil
    system_python = shutil.which("python") or shutil.which("python3")
    if system_python:
        print("WARNING: No project venv found. Using system Python — CUDA/GPU will NOT be available.", file=sys.stderr)
        print(f"  Checked: {[str(d) for d in candidates]}", file=sys.stderr)
        return Path(system_python)

    raise FileNotFoundError(
        "No Python found. Checked:\n"
        f"  {[str(d) for d in candidates]}\n"
        "  python / python3 on PATH"
    )


def launch_streamlit(python_path: Path, script: Path, port: str, name: str) -> subprocess.Popen:
    """Start a Streamlit app subprocess and return it."""
    if not script.exists():
        raise FileNotFoundError(f"{name} script not found: {script}")

    cmd = [
        str(python_path),
        "-m", "streamlit", "run",
        str(script),
        "--server.port", port,
        "--server.address", "localhost",
        "--server.headless", "true",
    ]
    print(f"  [{name}] Starting on http://localhost:{port}")
    return subprocess.Popen(cmd)


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            except Exception:
                pass


def main() -> None:
    print("=== Railway RAG Launcher ===")
    print()

    try:
        python_path = find_python()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Python:  {python_path}")
    print(f"CWD:     {ROOT_DIR}")
    print()

    missing = []
    if not BUILDER_SCRIPT.exists():
        missing.append(str(BUILDER_SCRIPT))
    if not APP_SCRIPT.exists():
        missing.append(str(APP_SCRIPT))
    if missing:
        print("Error: scripts not found:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    procs: list[subprocess.Popen] = []
    try:
        procs.append(launch_streamlit(python_path, BUILDER_SCRIPT, BUILDER_PORT, "Builder"))
        procs.append(launch_streamlit(python_path, APP_SCRIPT, APP_PORT, "Retrieval"))

        print()
        print(f"  Builder   \u2192 http://localhost:{BUILDER_PORT}")
        print(f"  Retrieval \u2192 http://localhost:{APP_PORT}")
        print()
        print("Press Ctrl+C to stop both applications.")
        print()

        time.sleep(2)
        webbrowser.open(f"http://localhost:{BUILDER_PORT}")
        webbrowser.open(f"http://localhost:{APP_PORT}")

        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
    finally:
        _stop(procs)
        print("Both applications stopped.")


if __name__ == "__main__":
    main()
