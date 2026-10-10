# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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

def run_launcher(*args, **kwargs):
    threading.Thread(target=lambda: (time.sleep(0.2), note("thread,"))).start()
    atexit.register(note, "atexit,")
    sys.stdout.write("unflushed")

teardown = Teardown()
launcher.launch_server = run_launcher
serve.launch_server(None)
"""


def test_serve_exits_without_interpreter_teardown(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    command = [sys.executable, "-c", SERVE_THAT_STOPS, str(marker)]

    done = subprocess.run(command, capture_output=True, text=True, timeout=300)

    assert done.returncode == 0, done.stderr
    assert "unflushed" in done.stdout
    # Threads finish before the atexit callbacks, as at a normal exit.
    assert marker.read_text() == "thread,atexit,"
