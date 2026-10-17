"""LeRobot side: resumable lerobot-eval batches over the LIBERO-Plus subset (pi0, pi0.5, Diffusion Policy),
result parsing, and selection of the LIBERO-10 episodes inside lerobot/libero for Diffusion Policy training."""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import summarize  # noqa: E402


def _walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def parse_eval_info(path):
    """{task_id: success}. Handles the per_task layout of eval_policy_all and falls back to any dict
    carrying a task_id and metrics."""
    info = json.load(open(path))
    out = {}
    for d in _walk(info):
        if "task_id" in d and isinstance(d.get("metrics"), dict):
            m = d["metrics"]
            if "successes" in m and isinstance(m["successes"], list) and m["successes"]:
                ok = any(bool(s) for s in m["successes"])
            elif "pc_success" in m:
                ok = float(m["pc_success"]) > 50.0
            else:
                continue
            out[int(d["task_id"])] = ok
    return out


def run_eval(a):
    sub = json.load(open(a.subset))
    tasks = sub["tasks"][:a.limit] if a.limit else sub["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    ids = [t["task_id"] for t in tasks]
    os.makedirs(a.out, exist_ok=True)
    for b in range(0, len(ids), a.batch):
        chunk = ids[b:b + a.batch]
        keep = os.path.join(a.out, f"batch_{b // a.batch:03d}.json")
        if os.path.exists(keep):
            continue
        od = f"/content/lerobot_eval/{a.name}/batch_{b // a.batch:03d}"
        if os.path.isdir(od):
            shutil.rmtree(od)
        cmd = ["lerobot-eval", f"--policy.path={a.policy}", "--env.type=libero", "--env.task=libero_10",
               "--env.is_libero_plus=true", "--env.task_ids=[" + ",".join(map(str, chunk)) + "]",
               "--eval.batch_size=1", "--eval.n_episodes=1", "--env.max_parallel_tasks=1", f"--output_dir={od}"]
        if "/pi0" in a.policy:
            cmd.append("--policy.compile_model=false")   # ~11 calls per episode: compiling costs more than it saves
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env={**os.environ, "MUJOCO_GL": "egl"})
        found = glob.glob(f"{od}/**/eval_info.json", recursive=True)
        assert found, f"no eval_info.json under {od}"
        res = parse_eval_info(found[0])
        missing = [i for i in chunk if i not in res]
        assert not missing, f"eval_info.json lacks task ids {missing[:5]}; keys seen {sorted(res)[:5]}"
        json.dump({str(i): res[i] for i in chunk}, open(keep, "w"))
        print(f"batch {b // a.batch}: {sum(res[i] for i in chunk)}/{len(chunk)} success", flush=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.out, "batch_*.json"))):
        for k, v in json.load(open(p)).items():
            if int(k) in by_id:
                rows.append(dict(family=a.name, **by_id[int(k)], success=bool(v)))
    with open(os.path.join(a.out, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summarize(rows, a.name)


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower().rstrip("."))


def select_episodes(a):
    """Episodes of lerobot/libero whose task string is one of the 10 LIBERO-10 instructions."""
    import pandas as pd
    from huggingface_hub import snapshot_download
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    langs = {_norm(suite.get_task(i).language) for i in range(suite.n_tasks)}
    assert len(langs) == 10, f"expected 10 LIBERO-10 instructions, got {len(langs)}"
    root = snapshot_download("lerobot/libero", repo_type="dataset", allow_patterns=["meta/*", "meta/**/*"])
    ep_files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    assert ep_files, f"no episode parquet under {root}/meta/episodes"
    eps = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    task_col = "tasks" if "tasks" in eps.columns else None
    if task_col is None:
        tf = glob.glob(os.path.join(root, "meta", "tasks.parquet"))
        assert tf and "task_index" in eps.columns, f"episode columns: {list(eps.columns)}"
        tt = pd.read_parquet(tf[0]).reset_index()
        tcol = [c for c in tt.columns if tt[c].dtype == object][0]
        imap = dict(zip(tt["task_index"], tt[tcol]))
        eps["tasks"] = eps["task_index"].map(lambda i: [imap[i]])
        task_col = "tasks"

    def ep_langs(v):
        return [_norm(x) for x in (list(v) if not isinstance(v, str) else [v])]

    chosen = sorted(int(r["episode_index"]) for _, r in eps.iterrows() if any(l in langs for l in ep_langs(r[task_col])))
    matched = {l for _, r in eps.iterrows() for l in ep_langs(r[task_col]) if l in langs}
    print(f"lerobot/libero: {len(eps)} episodes, {len(chosen)} are LIBERO-10, instructions matched {len(matched)}/10")
    assert len(matched) == 10 and len(chosen) >= 300, "LIBERO-10 episode match failed; paste this output"
    json.dump({"episodes": chosen, "instructions": sorted(langs)}, open(a.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "select"])
    ap.add_argument("--subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy")
    ap.add_argument("--name")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run_eval(a) if a.mode == "eval" else select_episodes(a)
