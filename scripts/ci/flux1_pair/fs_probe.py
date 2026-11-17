"""Bounded, listing-free filesystem probes for Hub-cache paths.

On nax-macos-2, listing a directory under the Hub cache blocked forever (runs 42463079723 and
42474515680) while `stat` returned in milliseconds. A probe stats (or reads a few bytes of) one
exact path in a child process with a timeout, so a blocked call stays in the child and is
reported as `Unreachable`, never as absent.
"""
import json
import subprocess
import sys

PROBE_TIMEOUT_SECONDS = 10.0

# The child stats or reads one path and prints a JSON line; a blocked filesystem call stays in it.
PROBE_CHILD = r"""
import json, os, sys
op, path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
row = {"exists": False}
try:
    info = os.stat(path)
    row = {"exists": True, "is_file": os.path.isfile(path), "is_dir": os.path.isdir(path),
           "size": info.st_size, "resolved": os.path.realpath(path)}
    if op == "read" and row["is_file"]:
        with open(path, "rb") as source:
            row["text"] = source.read(limit + 1).decode("utf-8")
except FileNotFoundError:
    pass
except NotADirectoryError:
    pass
sys.stdout.write(json.dumps(row))
"""


class Unreachable(Exception):
    """A filesystem probe did not return within its timeout."""


def probe(path, op="stat", limit=0, timeout=None):
    """Stat (or read up to `limit` bytes of) one exact path in a child process."""
    timeout = PROBE_TIMEOUT_SECONDS if timeout is None else timeout
    child = subprocess.Popen([sys.executable, "-I", "-c", PROBE_CHILD, op, str(path), str(limit)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL)
    try:
        out, err = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass  # A child stuck in the kernel cannot be reaped; never block on it.
        raise Unreachable(f"{op} {path} did not return within {timeout:g} s")
    if child.returncode != 0:
        raise ValueError(f"{op} probe failed for {path}: {err.decode('utf-8', 'replace').strip()}")
    return json.loads(out.decode("utf-8"))
