"""Snapshot every raw result file into the repo under results/ (rows only: .jsonl / .json under 20 MB; no images, masks,
checkpoints or logs). RESULTS.md stays the readable record; this is the data behind it, so an analysis can be re-run from a
clone without Drive access.

    uv run --no-project python wrapper/snapshot_results.py            (pulls both Drive folders with rclone first)
    uv run --no-project python wrapper/snapshot_results.py --no_pull
"""
import argparse
import glob
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.expanduser("~"), "Downloads", "local_data")
OUT = os.path.join(ROOT, "results")
FILTER = ["--filter", "- live_scripts/**", "--filter", "+ *.jsonl", "--filter", "+ *.json", "--filter", "- *", "--max-size", "20M", "-q"]


def rclone():
    hits = glob.glob(os.path.join(os.path.expanduser("~"), "AppData", "Local", "Microsoft", "WinGet", "Packages", "Rclone.Rclone_*", "rclone-*", "rclone.exe"))
    return hits[0] if hits else "rclone"


def copy_tree(src, dst, keep=(".jsonl", ".json")):
    n = 0
    for base, _, files in os.walk(src):
        for f in files:
            p = os.path.join(base, f)
            if f.endswith(keep) and not f.startswith("_") and os.path.getsize(p) < 20e6 and "live_scripts" not in p:
                q = os.path.join(dst, os.path.relpath(p, src))
                os.makedirs(os.path.dirname(q), exist_ok=True)
                shutil.copy2(p, q)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_pull", action="store_true")
    a = ap.parse_args()
    if not a.no_pull:
        r = rclone()
        subprocess.run([r, "copy", "gdrive:restoring_organs/plus", os.path.join(DATA, "plus")] + FILTER, check=False)
        subprocess.run([r, "copy", "--drive-shared-with-me", "gdrive:restoring_organs/plus", os.path.join(DATA, "plus_subhi")] + FILTER, check=False)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    n = {"drive_vihan": copy_tree(os.path.join(DATA, "plus"), os.path.join(OUT, "drive_vihan")),
         "drive_subhi": copy_tree(os.path.join(DATA, "plus_subhi"), os.path.join(OUT, "drive_subhi"))}
    rec = os.path.join(DATA, "recovered_file8")
    os.makedirs(os.path.join(OUT, "recovered_file8"), exist_ok=True)
    for f in ("episodes_from_cell_output.json", "sweep_run_full_paste.log", "sweep_analysis.d412e2.log", "gate_report.d412e2.log"):
        if os.path.exists(os.path.join(rec, f)):
            shutil.copy2(os.path.join(rec, f), os.path.join(OUT, "recovered_file8", f.replace(".log", ".txt")))
    shutil.copy2(os.path.join(rec, "shard_test", "episodes_openvla_oft.recovered_file8.jsonl"), os.path.join(OUT, "recovered_file8"))
    os.makedirs(os.path.join(OUT, "local"), exist_ok=True)
    loc = 0
    for f in glob.glob(os.path.join(ROOT, "vendor", "*.jsonl")) + [os.path.join(ROOT, "vendor", "novelty_scour_final.md")]:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(OUT, "local"))
            loc += 1
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Raw result rows (snapshot)\n\nWritten by `wrapper/snapshot_results.py`; do not edit by hand. `RESULTS.md` is the readable record.\n\n"
                "- `drive_vihan/`: `restoring_organs/plus` on Vihan's Drive (OpenVLA-OFT runs, scores, DROID, ladder, tolerance).\n"
                "- `drive_subhi/`: the same folder on Subhi's Drive (pi0.5 runs).\n"
                "- `recovered_file8/`: the 600 OFT sweep episodes rebuilt from the notebook cell output (2026-09-20) and the text they were rebuilt from.\n"
                "- `local/`: policy-free experiments run on Vihan's machine (envelope, measurability, unmeasurable, fidelity, passive calibration, ...).\n\n"
                "Only `.jsonl` / `.json` under 20 MB: no images, masks, checkpoints or logs.\n")
    print("snapshot:", n, "| local", loc)


if __name__ == "__main__":
    main()
