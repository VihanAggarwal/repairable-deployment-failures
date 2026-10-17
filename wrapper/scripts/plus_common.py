"""Shared by every LIBERO-Plus notebook: one deterministic task subset (so all five policy families see
exactly the same perturbed tasks), perturbation categories parsed from task names, and the summary table."""
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

# LIBERO-Plus task names, checked on the real suite: token perturbations carry their own tag; camera and robot
# initial state are encoded as values, e.g. ..._view_0_0_100_0_0_initstate_0 is the unperturbed setting.
CATS = [("sensor_noise", r"_noise_\d+"), ("language", r"_language_\d+"), ("light", r"_light_\d+"),
        ("background", r"_table_\d+|_tb_\d+"), ("layout", r"_add_\d+|_level")]
DEFAULT_VIEW = ("0", "0", "100", "0", "0")
# Every run script reads the suite from the environment, so one variable switches a whole notebook:
#   LIBERO_SUITE=libero_spatial | libero_object | libero_goal | libero_10 (default; nothing changes when it is unset)
SUITE = os.environ.get("LIBERO_SUITE", "libero_10")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}     # OFT's own horizons
OFT_CKPT = {s: "moojink/openvla-7b-oft-finetuned-" + s.replace("_", "-") for s in MAX_STEPS}


def categorize(name):
    for cat, pat in CATS:
        if re.search(pat, name):
            return cat
    m = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", name)
    if m and m.groups() != DEFAULT_VIEW:
        return "camera"
    s = re.search(r"_initstate_(\d+)", name)
    if s and s.group(1) != "0":
        return "robot_init"
    return "other"


def build_subset(suite_name="libero_10", per_cat=30, seed=0):
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[suite_name]()
    names = [suite.get_task(i).name for i in range(suite.n_tasks)]
    langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
    by = defaultdict(list)
    for i, n in enumerate(names):
        by[categorize(n)].append(i)
    rng = random.Random(seed)
    pick = []
    for cat in sorted(by):
        ids = sorted(by[cat])
        rng.shuffle(ids)
        pick += ids[:per_cat]
    pick.sort()
    tokens = Counter(tok for n in names for tok in re.findall(r"_([a-z]+)_\d+", n))
    return {"suite": suite_name, "n_tasks": suite.n_tasks, "per_cat": per_cat, "seed": seed,
            "counts": {c: len(v) for c, v in sorted(by.items())},
            "other_examples": [names[i] for i in by.get("other", [])[:5]],
            "name_tokens": dict(tokens.most_common(20)),
            "tasks": [{"task_id": i, "name": names[i], "language": langs[i], "category": categorize(names[i])}
                      for i in pick]}


def load_or_build(path, per_cat=30):
    if os.path.exists(path):
        return json.load(open(path))
    sub = build_subset(suite_name=SUITE, per_cat=per_cat)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(sub, open(path, "w"), indent=1)
    return sub


def load_rows(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def append_row(path, row):
    """Append one result row and CLOSE the file. Drive uploads a file only when it is closed: a runner that keeps its
    results file open for a whole cell has nothing on Drive until the cell ends, and a killed runtime loses all of it."""
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def load_shards(path):
    """Rows of `<stem>.jsonl` plus every per-session shard `<stem>.<shard>.jsonl` next to it, one row per
    (task, condition, arm, rep). Two Colab sessions that append to ONE Drive file overwrite each other (each uploads its
    own whole copy, the last upload wins; file 8 lost 576 rows that way on 2026-09-20), so every session writes its own
    shard and readers merge. A measured row beats one recovered from a log (`recovered`), which lacks the detail fields."""
    import glob
    best = {}
    # 0 = measured, 1 = measured before the shading gate (re-queued by the runner, its re-run lives in a later shard),
    # 2 = rebuilt from a log. The lowest rank wins; on a tie the first file does.
    rank = lambda r: 2 if r.get("recovered") else int(bool(r.get("photo_applied")) and "photo_shading" not in r)
    for p in [path] + sorted(glob.glob(path[:-len(".jsonl")] + ".*.jsonl")):
        for r in load_rows(p):
            k = (r.get("policy"), r["task_id"], r.get("cond"), r["arm"], r.get("rep", 0))
            if k not in best or rank(r) < rank(best[k]):
                best[k] = r
    return list(best.values())


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(rows, label=""):
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(bool(r["success"]))
    out = {}
    print(f"\n{label}  ({len(rows)} episodes)")
    print(f"{'category':<14}{'n':>5}{'success':>9}{'95% CI':>16}")
    for cat in sorted(by) + ["ALL"]:
        v = [s for r in by.values() for s in r] if cat == "ALL" else by[cat]
        k, n = sum(v), len(v)
        lo, hi = wilson(k, n)
        out[cat] = {"n": n, "success": k / max(1, n), "ci": [lo, hi]}
        print(f"{cat:<14}{n:>5}{k / max(1, n):>9.3f}   [{lo:.2f}, {hi:.2f}]")
    return out
