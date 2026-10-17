"""COLAB side of the live-fix folder. Every run line starts with
    [ -f <Drive>/plus/live_scripts/_sync.py ] && python3 <Drive>/plus/live_scripts/_sync.py
so a plain re-run of a cell picks up a fix pushed after the notebook was opened.

A script in /content/scripts is replaced ONLY when its content is a known OLDER version of the live one (its hash is in
the manifest's `older` list, built from the git history by wrapper/live_push.py). A copy the manifest does not know (a
newer notebook, a hand edit) is never touched, so a stale live folder cannot downgrade anything.
"""
import hashlib
import json
import os
import shutil

LIVE = os.path.dirname(os.path.abspath(__file__))
DST = os.environ.get("LIVE_DST", "/content/scripts")


def digest(text):
    return hashlib.sha256(text.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()


def main():
    man = json.load(open(os.path.join(LIVE, "MANIFEST.json")))
    for name, info in sorted(man["scripts"].items()):
        dst, src = os.path.join(DST, name), os.path.join(LIVE, name)
        if not (os.path.exists(dst) and os.path.exists(src)):
            continue
        have = digest(open(dst, encoding="utf-8").read())
        if have != info["sha"] and have in info["older"]:
            shutil.copyfile(src, dst)
            print(f"live fix applied: {name} -> {man['commit']} ({man['note']})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                                   # a broken live folder must never stop a run
        print("live_sync skipped:", e, flush=True)
