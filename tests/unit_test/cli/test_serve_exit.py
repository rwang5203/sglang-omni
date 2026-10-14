# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# note (Richard Wang): run_launcher stands in for a server that stopped. The
# object in the script's globals reports interpreter teardown.
SERVE_THAT_STOPS = """
import atexit, sys, threading, time
from sglang_omni.cli import serve
from sglang_omni.serve import launcher

marker = sys.argv[1]

def note(text):
    with open(marker, "a") as fh:
        fh.write(text)

class Teardown:
    # Module globals are cleared before this runs at teardown.
    def __del__(self, open=open, marker=marker):
        with open(marker, "a") as fh:
            fh.write("teardown,")

def fail():
    raise RuntimeError("thread exit callback failed")

def run_launcher(*args, **kwargs):
    threading.Thread(target=lambda: (time.sleep(0.2), note("thread,"))).start()
    atexit.register(note, "atexit,")
    if sys.argv[2:] == ["failing"]:
        threading._register_atexit(fail)
    sys.stdout.write("unflushed")

teardown = Teardown()
launcher.launch_server = run_launcher
serve.launch_server(None)
"""


@pytest.mark.parametrize("callbacks", ["passing", "failing"])
def test_serve_exits_without_interpreter_teardown(
    tmp_path: Path, callbacks: str
) -> None:
    marker = tmp_path / "marker.txt"
    command = [sys.executable, "-c", SERVE_THAT_STOPS, str(marker), callbacks]

    done = subprocess.run(command, capture_output=True, text=True, timeout=300)

    # A failing thread exit callback is reported and keeps the exit code, as
    # at a normal exit.
    assert done.returncode == 0, done.stderr
    assert "unflushed" in done.stdout
    if callbacks == "failing":
        assert "thread exit callback failed" in done.stderr
    else:
        # Threads finish before the atexit callbacks, as at a normal exit.
        assert marker.read_text() == "thread,atexit,"
