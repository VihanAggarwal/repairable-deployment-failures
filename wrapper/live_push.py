"""LOCAL side of the live-fix folder (Vihan's machine, needs rclone remote `gdrive`). After committing a fix:

    uv run --no-project python wrapper/live_push.py "what the fix is"

Uploads every wrapper/scripts/*.py, wrapper/live_sync.py (as _sync.py) and a MANIFEST.json to
restoring_organs/plus/live_scripts. The manifest lists, per script, the hash of the committed version and the hashes of
every EARLIER committed version, which is what lets the Colab side replace old copies and nothing else.
Refuses to run with uncommitted script changes: the live folder must equal a commit. Crash fixes only while a method's
run is in progress (agent guide, rule 4).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from live_sync import digest  # noqa: E402

RCLONE = glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Rclone.Rclone_*\rclone-*\rclone.exe"))[0]
REMOTE = "gdrive:restoring_organs/plus/live_scripts"


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def main():
    note = sys.argv[1] if len(sys.argv) > 1 else ""
    dirty = [l for l in git("status", "--porcelain", "--", "wrapper/scripts", "wrapper/live_sync.py").splitlines() if l.strip()]
    assert not dirty, f"commit first, the live folder must equal a commit: {dirty}"
    commit = git("rev-parse", "--short", "HEAD").strip()
    scripts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for p in sorted(glob.glob(os.path.join(HERE, "scripts", "*.py"))):
            name = os.path.basename(p)
            rel = f"wrapper/scripts/{name}"
            older = {digest(git("show", f"{c}:{rel}")) for c in git("log", "--format=%H", "--", rel).split()}
            sha = digest(open(p, encoding="utf-8").read())
            scripts[name] = dict(sha=sha, older=sorted(older - {sha}))
            shutil.copyfile(p, os.path.join(tmp, name))
        shutil.copyfile(os.path.join(HERE, "live_sync.py"), os.path.join(tmp, "_sync.py"))
        json.dump(dict(commit=commit, note=note, scripts=scripts), open(os.path.join(tmp, "MANIFEST.json"), "w"))
        r = subprocess.run([RCLONE, "sync", tmp, REMOTE], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
    print(f"live folder = {commit}: {len(scripts)} scripts, {sum(len(v['older']) for v in scripts.values())} older versions known")


if __name__ == "__main__":
    main()
