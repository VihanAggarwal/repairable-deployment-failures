"""Per-session result shards (plus_common.load_shards): two sessions never share a results file, readers merge.
uv run --no-project python wrapper/_shard_tests.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from plus_common import load_shards  # noqa: E402


def write(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def row(task, cond, arm, ok, **kw):
    return dict(policy="p", task_id=task, cond=cond, arm=arm, success=ok, **kw)


with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "episodes_p.jsonl")
    assert load_shards(base) == []                                              # S1 nothing on disk
    write(base, [row(1, "a", "raw", True), row(1, "a", "full", False)])
    write(os.path.join(d, "episodes_p.abc123.jsonl"), [row(2, "a", "raw", False), row(1, "a", "full", True)])
    write(os.path.join(d, "episodes_p.rec.jsonl"), [row(3, "a", "raw", True, recovered="cell_output"), row(2, "a", "raw", True, recovered="cell_output")])
    write(os.path.join(d, "episodes_p.jsonl.bak"), [row(9, "a", "raw", True)])
    write(os.path.join(d, "episodes_pq.jsonl"), [row(8, "a", "raw", True)])
    got = {(r["task_id"], r["arm"]): r for r in load_shards(base)}
    assert set(got) == {(1, "raw"), (1, "full"), (2, "raw"), (3, "raw")}, set(got)   # S2 merged; backups and other policies ignored
    assert got[(1, "full")]["success"] is False                                  # S3 first measured row wins over a later duplicate
    assert got[(2, "raw")]["success"] is False and "recovered" not in got[(2, "raw")]   # S4 measured beats recovered
    assert got[(3, "raw")]["recovered"] == "cell_output"                         # S5 recovered fills what nothing else has
    write(os.path.join(d, "episodes_p.zzz.jsonl"), [row(1, "a", "raw", False, photo_applied=True, photo_shading=0.1)])
    write(base, [row(1, "a", "raw", True, photo_applied=True), row(1, "a", "full", False)])
    got = {(r["task_id"], r["arm"]): r for r in load_shards(base)}
    assert got[(1, "raw")]["success"] is False                                   # S7 a re-run beats the stale row it replaced
    os.remove(os.path.join(d, "episodes_p.zzz.jsonl"))
    os.remove(base)
    assert len(load_shards(base)) == 3                                          # S6 shards alone (no main file) still load
print("shard tests: 7 passed")
