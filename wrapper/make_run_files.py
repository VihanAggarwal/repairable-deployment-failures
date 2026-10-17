"""Slice the generated notebooks into short, numbered RUN FILES: the setup cells plus only the cells one person has to
run for one task, in order. Output goes to ~/Downloads by default (not committed; regenerate any time):

    python wrapper/make_plus_notebooks.py && python wrapper/make_run_files.py [out_dir]

A run file = every cell of the source notebook up to and including its last setup cell, then the cells whose first
comment line starts with one of the given prefixes (or literal extra cells), under a header that says what to do.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "notebooks")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Downloads")
DRIVE = "/content/drive/MyDrive/restoring_organs/plus"
SWEEP, NOM, PY1 = f"{DRIVE}/sweep", f"{DRIVE}/wrapper_oft/nominals_v3.json", "/content/venv_oft/bin/python"
DSC = f"{DRIVE}/droid_selfcal"


def first_line(cell):
    lines = "".join(cell["source"]).strip().split("\n")
    return lines[1] if lines[0].startswith("%%bash") and len(lines) > 1 else lines[0]


def code(src):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src.splitlines(keepends=True))


def log_to_drive(src, logs):
    """Every `!... /content/scripts/<name>.py ...` line also appends its output (stderr included, so tracebacks too) to
    <logs>/<name>.log on Drive, where it can be read while the cell is still running. Output still shows in the cell.
    It first runs the live-fix sync (wrapper/live_sync.py) when the Drive folder has one."""
    import re
    out = []
    for line in src.split("\n"):
        m = re.search(r"/content/scripts/(\w+)\.py", line)
        if line.startswith("!") and m and "| tee" not in line and " > " not in line:
            live = logs.rsplit("/", 1)[0] + "/live_scripts/_sync.py"     # fixes pushed after the notebook was opened
            line = f"![ -f {live} ] && python3 {live}; {line[1:]} 2>&1 | tee -a /content/logs/{m.group(1)}.log"
        out.append(line)
    return "\n".join(out)


# Finished or dead run files: not written again (the definitions stay below as the record of what each one ran).
RETIRED = {"2_", "3_", "6_", "8_", "9_", "10_", "11_", "12_", "14_", "17_", "1_", "7_", "13_"}       # 17 was merged into 4


def build(name, source, setup_until, header, picks):
    if name.split("_")[0] + "_" in RETIRED:
        return
    nb = json.load(open(os.path.join(NB, source), encoding="utf-8"))
    cells = nb["cells"]
    end = max(i for i, c in enumerate(cells) if first_line(c).startswith(setup_until))
    out = [dict(cell_type="markdown", metadata={}, source=header.splitlines(keepends=True))] + cells[1:end + 1]
    for p in picks:
        if isinstance(p, dict):
            p["source"] = log_to_drive("".join(p["source"]), f"{DRIVE}/logs").splitlines(keepends=True)
            out.append(p)
            continue
        hit = [c for c in cells[end + 1:] if first_line(c).startswith(p)]
        assert len(hit) == 1, (name, p, len(hit))
        out.append(hit[0])
    assert "—" not in json.dumps(out, ensure_ascii=False), name
    nb["cells"] = out
    path = os.path.join(OUT, name + ".ipynb")
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
    print(f"wrote {path} ({len(out)} cells, {len(out) - end - 1} to run after setup)")


SUBSET = "# the SAME task subset"
RULE = ("\n\nRun the cells top to bottom. Every run cell is resumable: after a disconnect, run the setup cells again and "
        "re-run the same cell. Before leaving, run `from google.colab import drive; drive.flush_and_unmount()`.")

build("1_VIHAN_NOW_learned_sensing_root_cause_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 1 (Vihan, A100, NOW): why learned sensing loses tasks simulator sensing solves\n\n"
      "Step 1 (~1 h): six tasks x five arms in one harness. Step 2: paste the table. **Do NOT run step 3 until the "
      "table has been read**: it is only worth its 1.5 h if `full_hybrid_truepose` beats `full_plane_truepose`." + RULE,
      [code('# scripts current and segmenter on Drive: three True\nimport os\n'
            f'print("seg_net", os.path.exists("{DRIVE}/wrapper_oft/seg_net.pt"), '
            '"hybrid", "hybrid_depth" in open("/content/scripts/wrapper_libero.py").read(), '
            '"redo", "--redo" in open("/content/scripts/wrap_run.py").read())'),
       "# learned sensing root cause, step 1", "# learned sensing root cause, step 2", "# learned sensing root cause, step 3"])

build("2_VIHAN_NOW_droid_selfcal_v6_T4", "droid_selfcal.ipynb", "set -e",
      "# 2 (Vihan, T4, NOW, runs alongside file 1): DROID self-calibration v6\n\n"
      "Export and segmentation are already on Drive and are NOT repeated. Park the older results, calibrate (~2.5 h, "
      "resumable), report. Paste the first 10 calibrate lines early: `est` IoU should now reach or beat `gt` IoU and the rotation error should drop below ~15 deg. Stop rule: reliable-reference median above 30 cm means a negative result." + RULE,
      [code("\n".join([
          "# park results made by an older search so v6 starts fresh (safe to re-run; v6 rows are kept and resumed)",
          "import os, glob, json",
          f'd = "{DSC}"',
          'src = d + "/selfcal_results.jsonl"',
          "rows = [json.loads(l) for l in open(src) if l.strip()] if os.path.exists(src) else []",
          'if rows and rows[0].get("search") != "v6":',
          '    free = [t for t in ("v4", "v5_partial", "old2", "old3") if not os.path.exists(d + "/selfcal_results_" + t + ".jsonl")]',
          '    dst = d + "/selfcal_results_" + free[0] + ".jsonl"',
          '    os.rename(src, dst); print("parked", len(rows), "old rows ->", os.path.basename(dst))',
          'print("cameras on Drive:", len(glob.glob(d + "/*.npz")), "| v6 script:", "best_shift" in open("/content/scripts/droid_selfcal.py").read())',
      ])),
       "# 3. self-calibrate every camera", "# 4. report"])

build("3_VIHAN_AFTER_oracle_canonicalizer_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 3 (Vihan, A100, after file 1): learned canonicalizer, oracle target (claim 3)\n\n"
      "Skip the training cell if `canon_net_oracle.pt` is already on Drive (the check cell says so). Then the priority "
      "run picks up the `learned` rows and the re-queued light rows." + RULE,
      [code('# shards and scripts in place: three True; the fourth says whether training can be skipped\nimport glob, os\n'
            f'print("shards", len(glob.glob("{DRIVE}/wrapper_oft/canon_data_v2/task_*.npz")) == 150, '
            '"train", "--input" in open("/content/scripts/train_canon.py").read(), '
            '"eval", "in_ch" in open("/content/scripts/eval_oft_wrapper.py").read(), '
            f'"| already trained:", os.path.exists("{DRIVE}/wrapper_oft/canon_net_oracle.pt"))'),
       "# claim 3, step 2", "# wrapper run in priority order"])

build("4_VIHAN_pi05_takeover_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 4 (Vihan, A100, taken over from Subhi 2026-09-21): everything that is left on pi0.5, in priority order\n\n"
      "BEFORE the first run, on YOUR Colab account: add the secret `HF_TOKEN` (key icon, notebook access ON) with a Hugging "
      "Face read token, and accept the PaliGemma license on that Hugging Face account "
      "(huggingface.co/google/paligemma-3b-pt-224), or the pi0.5 download fails. Subhi's rows are already on your Drive "
      "(421 sweep episodes, 306 wrapper episodes), so every cell resumes. Every cell is resumable and skips "
      "finished work, so nothing already on Drive is run again. The setup cells carry the fog fix, the robust asset "
      "download and the per-session results files, so none of the old paste cells are needed.\n\n"
      "1. **Scale-up of the held-out test** (resumes at about 156 of 504, ~9 h). The single most important run left.\n"
      "2. **Ladder on pi0.5** (~1.5 h): true camera pose and a perfect render on the 17 camera tasks pi0.5 fails with and "
      "without the wrapper. Says whether pi0.5 loses to the measurement, the lost view, or itself.\n"
      "3. **Tolerance probe** (~40 min, no rollouts).\n"
      "4. **Settle + lazy-measurement check** (~3 h): the two fixes for the harm on pi0.5. NOW THE MOST IMPORTANT CELL LEFT.\n"
      "5. **Replicates** (~3 h): new seeds, own folders.\n"
      "6. **Baselines on pi0.5** (~1.2 h).\n\n"
      "TWO SESSIONS AT ONCE: cell 1 in one session, and cells 2 then 3 in a second one (they write to their own folders). "
      "Cells 4, 5 and 6 share the `wraps` folder logic: run them ONE AFTER ANOTHER, never two at a time. Before closing "
      "any session: run `!ls -la` on its results folder from a new cell and check the time stamp." + RULE,
      ["# scale-up of the held-out test: pi0.5", "# ladder on pi0.5", "# tolerance probe, pi0.5", "# settle check on pi0.5",
       code("# replicates on pi0.5 (~3 h, resumable): 2 more rollouts per arm on light then camera, NEW seeds, each replicate in its OWN folder. Measures how often a single pi0.5 rollout flips with nothing changed\n"
            f"!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {DRIVE}/subset_libero10.json --out_dir {DRIVE}/wraps_rep1 --nominals {NOM} --arms raw,full --categories light,camera --rep 1\n"
            f"!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {DRIVE}/subset_libero10.json --out_dir {DRIVE}/wraps_rep2 --nominals {NOM} --arms raw,full --categories light,camera --rep 2"),
       "# baselines on pi0.5"])

build("5_SUBHI_w9_openvla_base_gate_test_A100_24h", "plus_2_openvla_base.ipynb", SUBSET,
      "# 5 (Subhi, A100, after file 4): W9, OpenVLA base, the third policy family (~24 h in total, NOT cut)\n\n"
      "**This family is the pre-registered test of the do-no-harm gate (RESULTS.md 2026-09-18): nothing in the project "
      "has been tuned on it. Do not change any script between cells.** Smoke first (2 tasks); stop and report if it "
      "fails. Then the baseline, the graded sweep (4 base tasks, raw / full / gated, ~12 h), the wrapper table (raw / "
      "full / full_gated, ~8 h) and the gate report. Every cell resumes, so split it over sessions. If time is short: "
      "sweep first, then the report, then the wrapper table." + RULE,
      ["# smoke: 2 tasks", "# full run (resumable)", "# graded identifiability sweep, OpenVLA", "# wrapper on OpenVLA",
       "# self-motion consistency, OpenVLA probe", "# do-no-harm gate on OpenVLA"])

build("6_VIHAN_AFTER_oft_boundary_sweep_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 6 (Vihan, A100, after file 3): boundary conditions on OFT, then the pooled analysis" + RULE,
      [code("# boundary conditions, policy-free scores (8 conditions x 3 tasks + references, ~25 min, resumable)\n"
            f"!MUJOCO_GL=egl PYTHONPATH=/content/LIBERO-plus MPLBACKEND=Agg {PY1} /content/scripts/sweep_run.py score --n_tasks 3 --out_dir {SWEEP} --nominals {NOM} --extended"),
       code("# boundary conditions, OFT episodes (48 episodes, ~1.3 h, resumable)\n"
            f"!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/sweep_run.py episodes --family oft --n_tasks 3 --out_dir {SWEEP} --nominals {NOM} --extended"),
       "# graded sweep, step 3"])

build("7_VIHAN_HOLD_learned_sensing_canonicalizer_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 7 (Vihan, A100, ON HOLD): canonicalizer trained on learned-sensing shards\n\n"
      "**Do not run until file 1 has been read.** These cells use plane depth; if file 1 shows hybrid depth is the fix, "
      "the chain is re-pointed at hybrid first." + RULE,
      ["# measurement plus learning, step 1", "# measurement plus learning, step 2", "# measurement plus learning, step 3"])

build("8_VIHAN_AFTER_scaleup_and_gate_oft_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 8 (Vihan, A100, after file 6): scale the score test to all 10 base tasks, gate on the development family\n\n"
      "Step 1 scores 7 new base tasks (~2 h, no policy; the 10-task pi0.5 analysis needs these too). Step 2 runs "
      "OFT on them with raw / full / gated (~11 h, resumable). Then the gate report and the pooled analysis: paste both." + RULE,
      ["# scale-up, step 1", "# scale-up, step 2", "# do-no-harm gate on OFT", "# graded sweep, step 3"])

build("9_VIHAN_NOW_smc_probe_oft_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 9 (Vihan, A100, NOW, ~1.5 h): self-motion consistency probe on OFT (development family)\n\n"
      "No rollouts. The probe asks the frozen policy for its action at 7 known end-effector positions, on the raw and on "
      "the corrected frame, and checks whether the commanded motions agree on one sub-goal. The report joins with the "
      "raw / full outcomes already on Drive. **Paste the whole report.** If the signal is real here, Subhi runs the same "
      "probe on pi0.5 (held-out, harm known) from run file 4." + RULE,
      ["# self-motion consistency, OFT probe", "# self-motion consistency: report"])

build("10_VIHAN_baselines_oft_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 10 (Vihan, A100): baselines reviewers will ask for (OFT, libero_10)\n\n"
      "Cell 1 (~7 h, resumable): camera baselines on the 30 extra camera tasks: test-time augmentation, generic "
      "corrections, a depth-free homography with the measured pose, and the wrapper without the reference fill. "
      "Cell 2 (~3 h): generic corrections on tasks where OFT is already robust. Cell 3: the report, paste it." + RULE,
      ["# baselines, camera (OFT", "# baselines, generic corrections where OFT", "# baselines: report"])

build("11_VIHAN_four_suites_oft_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 11 (Vihan, A100): the wrapper on the three other LIBERO-Plus suites (libero_spatial, libero_object, libero_goal)\n\n"
      "One cell per suite (~3 to 3.5 h each, resumable, independent: run them in any order or in separate sessions). "
      "Each downloads that suite's OFT checkpoint (~15 GB), builds the task subset and the nominal camera on first "
      "use, runs 20 robot_init + 20 camera tasks paired raw / full, and prints the paired table: paste it. "
      "The three CAMERA EXTENSION cells (~2 h each) add 40 more camera tasks per suite: run each after that suite's first "
      "cell. Verified locally: task naming parses with no leftovers, each suite has ONE training camera, horizons 220 / 280 / 300." + RULE,
      ["# four suites, libero_spatial", "# four suites, libero_object", "# four suites, libero_goal",
       "# camera extension, libero_spatial", "# camera extension, libero_object", "# camera extension, libero_goal"])

build("12_VIHAN_droid_vlm_masks_T4", "droid_selfcal.ipynb", "set -e",
      "# 12 (Vihan, T4): DROID self-calibration with robot masks chosen by a VLM\n\n"
      "Reopens W6 as EXPLORATORY (the v6 stop rule already fired; this is logged as post hoc). Before the first run: "
      "Colab left bar > key icon > add a secret named `ANTHROPIC_API_KEY` with notebook access ON. The key never goes in "
      "a cell. Cell 1 needs no key. Cell 2 is a pilot on 6 cameras (about 2 dollars): paste its table and STOP if the "
      "VLM masks do not beat the old ones. Cells 3 to 6: all 54 cameras (about 25 to 45 dollars of API), new masks, "
      "calibration (~2.5 h), report. Success = reliable-reference median under 30 cm." + RULE,
      ["# VLM masks 1", "# VLM masks 2", "# VLM masks 3", "# VLM masks 4", "# VLM masks 5", "# VLM masks 6"])

build("13_VIHAN_camera_power_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 13 (Vihan, A100): fixes for the weak spots, tested on camera tasks\n\n"
      "Three new arms built 2026-09-20: `full_settle` (the measurement hands the arm over at rest, where the episode "
      "began), `full_verify` (full_settle, plus a second 7-pose measurement when the robot's own self-check rejects the "
      "calibration) and `full_tta` (wrapper + test-time augmentation on the corrected frame). Cell 1: libero_10, 30 "
      "tasks, ~2.5 h. Cell 2: libero_object, 60 tasks, also adds the plain tta baseline there, ~6 h. Cell 3: 40 more "
      "libero_goal camera tasks, ~5 h. Existing rows are reused; every cell prints its paired table." + RULE,
      ["# camera power 1", "# camera power 2", "# camera power 3"])

build("14_VIHAN_camera_extension_spatial_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 14 (Vihan, A100): 40 more camera tasks on libero_spatial\n\n"
      "One cell, ~2 h, resumable, paired raw / full. The spatial camera result is null on the first 20 tasks because that "
      "checkpoint survives most camera shifts; 40 more tasks give it enough raw failures to test. Writes ONLY to "
      "`plus/wraps_libero_spatial`, so it is safe next to run file 13 (libero_10, object, goal). The object and goal "
      "extensions are NOT here on purpose: run file 13 cells 2 and 3 already run those 40 extra tasks with more arms, and "
      "two sessions must never write the same results folder. Paste the paired table it prints." + RULE,
      ["# camera extension, libero_spatial"])

build("15_VIHAN_droid_robotseg_baseline_T4", "droid_selfcal.ipynb", "set -e",
      "# 15 (Vihan, T4 or L4): the RobotSeg baseline for the DROID self-calibration\n\n"
      "A reviewer will ask why the masks come from a VLM choosing SAM regions when a robot segmenter exists (RobotSeg, CVPR "
      "2026). This runs RobotSeg on the same 54 cameras and the same frames, then the SAME calibration search. No API key, "
      "no policy. BEFORE cell 1: download `robotseg.pt` once (link in the Checkpoint section of github.com/showlab/RobotSeg) "
      "and put it on Drive at `restoring_organs/plus/robotseg/robotseg.pt`. Cell 2 is a one-minute smoke test on two "
      "cameras: paste it and STOP if it errors (this code could not be tested without the weights). Cells 3 to 5: masks "
      "(~10 min), calibration (~2 h), report. Writes only to `plus/droid_selfcal_robotseg`, safe next to everything else." + RULE,
      ["# RobotSeg baseline 1", "# RobotSeg baseline 2", "# RobotSeg baseline 3", "# RobotSeg baseline 4", "# RobotSeg baseline 5"])

build("16_VIHAN_attribution_ladder_oft_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 16 (Vihan, A100): where the deficit lives (the attribution ladder)\n\n"
      "The first 30 libero_10 camera tasks already have all four rungs: raw 13, measured pose 19, TRUE pose 21, perfect "
      "training-view render 27, and of the 17 raw failures 7 are repaired by measurement, 2 only by the true pose, 8 only "
      "by the perfect render, 0 by nothing (the policy). These cells run the two upper rungs on the 29 other camera tasks "
      "that fail under both raw and full (libero_10 extra 9, object 9, goal 8, spatial 3; about 2 h in total, each cell "
      "resumable). Every cell writes to its OWN folder `plus/wraps_ladder_<suite>` and only reads the finished wrap "
      "files, so it is safe next to run file 13. LAST CELL (optional, ~25 min, no rollouts): the tolerance probe, the "
      "policy's own calibration tolerance in cm and degrees. Run file 13 cells 2 and 3 may still add failures: re-running a ladder "
      "cell later picks up only the new ones. Last cell: the report, paste all of it." + RULE,
      ["# ladder 1", "# ladder 2", "# ladder 3", "# ladder 4", "# ladder: report", "# tolerance probe, OFT"])

build("17_SUBHI_attribution_ladder_pi05_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 17 (Subhi, A100): the attribution ladder on pi0.5\n\n"
      "One cell, ~1.5 h, resumable. pi0.5 fails 17 camera tasks under both raw and the wrapper. This runs them with the "
      "TRUE camera pose and with a perfect training-view render, which says whether the held-out family loses to the "
      "measurement, to the lost view, or to itself. Writes to its OWN folder `plus/wraps_ladder`, safe next to the scale-up "
      "sweep and to any other wrapper cell. Paste the table it prints. Second cell (optional, ~40 min, no rollouts): the "
      "tolerance probe on pi0.5, own folder `plus/tolerance`." + RULE,
      ["# ladder on pi0.5", "# tolerance probe, pi0.5"])

LZ = f"{DRIVE}/wraps_lazy"
build("18_VIHAN_lazy_where_it_must_measure_pi05_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 18 (Vihan, A100, ~1.5 h): the motion-free test on pi0.5 where it has to decide (camera and robot_init)\n\n"
      "The motion-free arm `full_lazy` (post hoc, the paper's motion-free proposition) has so far been scored on pi0.5 "
      "only on units where the camera did not move, i.e. where it declines to measure. This runs it on the 30 camera "
      "tasks (it must measure) and the 30 robot_init tasks (arm moved, camera did not): 60 episodes. The raw and full "
      "rows for these tasks already exist in `plus/wraps` and are NOT re-run: seeds depend only on the task, so the new "
      "rows pair with them exactly.\n\n"
      "BEFORE the first run, on YOUR Colab account: the secret `HF_TOKEN` (key icon, notebook access ON) with a Hugging "
      "Face read token, and the PaliGemma license accepted (huggingface.co/google/paligemma-3b-pt-224), exactly as in "
      "run file 4, or the pi0.5 download fails.\n\n"
      "Cell 1 writes to its OWN folder `plus/wraps_lazy`, so it is safe next to run file 4 or any other session. "
      "Cell 2 (seconds, no rollouts) merges the two files in /content and prints full_lazy against raw and against "
      "full per category with the exact McNemar p, plus how often the test skipped the sweep. Paste all of it." + RULE,
      [code("# lazy where it must measure, pi0.5 (60 episodes, ~1.5 h on A100, resumable): full_lazy on the 30 camera and 30 robot_init tasks. Own folder; raw and full are read from plus/wraps in cell 2\n"
            f"!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {DRIVE}/subset_libero10.json --out_dir {LZ} --nominals {NOM} --arms full_lazy --categories camera,robot_init"),
       code("# lazy where it must measure: report (paste all of it)\n"
            f"!cat {DRIVE}/wraps/wrap_pi05.jsonl {LZ}/wrap_pi05.jsonl > /content/lazy_merged.jsonl\n"
            "!python /content/scripts/arms_report.py --rows /content/lazy_merged.jsonl --arms raw,full,full_lazy --ref raw\n"
            "!python /content/scripts/arms_report.py --rows /content/lazy_merged.jsonl --arms raw,full,full_lazy --ref full\n"
            f"!python -c \"import json,collections;c=collections.Counter((r['category'],r.get('lazy_skipped'),r.get('state_applied',False)) for r in map(json.loads,open('{LZ}/wrap_pi05.jsonl')));[print('category',k[0],'sweep skipped',k[1],'arm homed',k[2],'n',v) for k,v in sorted(c.items(),key=str)]\"")])

build("19_VIHAN_nominal_base_rate_oft_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 19 (Vihan, A100, ~4 h): the unperturbed base rate on the exact tasks and seeds of the joint table and the ladder sets\n\n"
      "POST HOC (reviewer point: wrapped success is bounded by nominal success, which the paper never measured on these "
      "seeds). No method code changes: `nominal_start` (arm PLACED at the training start, then the raw policy) and "
      "`nominal_view` (every frame rendered from the training camera, nothing measured) are diagnostic arms that already "
      "exist. Cell 1 (~2.5 h): robot_init on all four suites; libero_10 also re-runs raw and full here because its Table 2 row "
      "came from the older harness. Cell 2 (~1.5 h): camera, set A in full plus the ladder units of set B, object, goal and "
      "spatial. The spatial, object and goal cells download that suite's OFT checkpoint (~15 GB) on first use. Cells 3 and 4 "
      "(seconds): the reports, paste all of both. Every cell writes ONLY to `plus/wraps_nominal_<suite>`, so it is safe next "
      "to run files 16, 18, 20, 21 and 22; run cells 1 and 2 in ONE session (they share those folders)." + RULE,
      ["# nominal base rate 1", "# nominal base rate 2", "# nominal base rate 3", "# nominal base rate 4"])

build("20_VIHAN_3d_renders_for_figures_T4", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 20 (Vihan, any GPU, T4 is enough, ~35 min with setup): policy-free 3D renders for the paper figures\n\n"
      "No policy is loaded and nothing is scored; the policy checkpoint is never used. Cell 1 is a 3-minute smoke test at "
      "256 px into /content: it must print `anchor check` of 0.90 or more (the projected robot geometry lands on the rendered "
      "robot). Paste that line and STOP on an error or a WARNING. Cell 2 writes three .npz files (512 px, clean, no text) to "
      "`plus/paper_renders`: the god's-eye scene with the training camera and the four Fig. 1 cameras as frusta, the "
      "joint-offset start against the training start, and the 7 fingerprint probe poses: `scene_frusta.npz`, "
      "`joint_offset_ghost.npz`, `probe_ghosts.npz`. Frames carry no text and no arrows: the probe action arrows are written "
      "offline as their own transparent layer from the logged fingerprint rows. Afterwards copy the three files to "
      "`wrapper/paper/iclr/figs/renders/` (or `vendor/drive_watch.py pull`) and run make_image_figures.py. If an older run "
      "already left `joint_offset.npz` in `plus/paper_renders`, it is still read; nothing else is re-rendered." + RULE,
      ["# 3D renders 1", "# 3D renders 2"])

build("21_VIHAN_openvla_heldout_extension_A100", "plus_2_openvla_base.ipynb", SUBSET,
      "# 21 (Vihan, A100, ~0.5 h + ~3 h per solved base task, at most ~19 h): more OpenVLA base tasks for the held-out score test\n\n"
      "POST HOC extension of the OpenVLA (fine-tuned on LIBERO-10) graded sweep: the frozen policy-free score and constants, "
      "the 6 base tasks the registered sweep did not use. The rule is fixed in TASKS.md before this runs: a new base task "
      "enters iff its unperturbed raw episode succeeds (cell 1, ~25 min). Cell 2 runs all 24 conditions x raw / full on those "
      "tasks only; if it prints `solved: []`, stop, there is nothing to extend. Cell 3 (seconds): the analysis on the new units "
      "alone and pooled with the registered sweep; paste all of it. The registered 60-unit result is not replaced. Everything "
      "writes to its OWN folder `plus/sweep_openvla_ext` (per-session shards), so it is safe next to every other run file." + RULE,
      ["# OpenVLA extension 1", "# OpenVLA extension 2", "# OpenVLA extension 3"])

build("22_VIHAN_metric_depth_set_A_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 22 (Vihan, A100, ~2 h): learned masks + monocular METRIC depth on set A (does self-calibration reach ~1 cm?)\n\n"
      "POST HOC arm `full_metric`: the trained segmenter plus Depth Anything V2 metric (used as is, served from Colab's system "
      "python because the OFT venv pins transformers 4.40). Cell 1 is a 10-minute smoke on 2 tasks in its own folder: paste "
      "it and STOP on an error. Cell 2 (~1.5 h, resumable): all 30 set A camera tasks, own folder `plus/wraps_metric`, same "
      "harness and seeds as full_plane / full_hybrid. Cell 3 (seconds): calibration error against the simulator's pose and "
      "success for plane, hybrid and metric on the same tasks; paste all of it. Needs `plus/wrapper_oft/seg_net.pt` (exists)." + RULE,
      ["# metric depth 1", "# metric depth 2", "# metric depth 3"])

build("23_VIHAN_pose_source_ablation_set_B_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 23 (Vihan, A100, ~4 h): the pose-source ablation on set B (TASKS.md P47, POST HOC)\n\n"
      "The same reprojection driven by three camera-pose sources on the 30 set B camera tasks (Table 6), same sweep, "
      "state channel, hand-over and seeds: (i) the paper's silhouette search (the existing `full` rows in `plus/wraps`, NOT "
      "re-run), (ii) `full_refine`, EasyHeC-style dense-mask refinement from the same initial pose, (iii) `full_fiducial`, "
      "a fiducial PnP pose with oracle corners (the AnyCamVLA pose source). New script `pose_sources.py`; no registered "
      "rule or existing arm changes. Cell 1 is a 10-minute smoke on 2 tasks in its own folder: paste it and STOP on an "
      "error. Cell 2 (~3.5 h, resumable): all 30 tasks, own folder `plus/wraps_posesource`, safe next to every other run "
      "file. Cell 3 (seconds): success, calibration error and the paired test against the silhouette pose per source; "
      "paste all of it (it also writes `posesource_rows.jsonl`: task id, arm, cam_est_err_cm_deg, success)." + RULE,
      ["# pose source 1", "# pose source 2", "# pose source 3"])

build("25_VIHAN_learned_sensing_refined_pose_set_A_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 25 (Vihan, A100, ~2.5 h): LEARNED sensing with the refined pose on set A (tests pre-registered)\n\n"
      "Learned segmenter, depth from the training table plane (no depth sensor), and the camera pose refined EasyHeC-style "
      "on the LEARNED masks from two starts (the silhouette pose and the training pose), keeping the lower dense loss. "
      "Locally, policy-free, this took learned calibration from centimetres toward ~1 cm; this run measures success. "
      "Cell 1 is a 10-minute smoke on 2 tasks: paste it and STOP on an error. Cell 2 (~2.5 h, resumable): raw and "
      "`full_hybrid_refine2` on all 30 set A camera tasks, own folder `plus/wraps_refine2`. Cell 3 (seconds): the report, "
      "paste all of it. Safe next to run files 24 and 26 (different folders)." + RULE,
      ["# learned refine 1", "# learned refine 2", "# learned refine 3"])

build("26_VIHAN_pi05_camera_refined_pose_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 26 (Vihan, A100, ~3.5 h): pi0.5 camera with the refined pose, 60 tasks (tests pre-registered)\n\n"
      "Cell 1 (~1 h): `full_refine2` on the 30 set A camera tasks (raw and full exist in plus/wraps). Cell 2 (~2.5 h): 30 "
      "FURTHER camera tasks with raw, full and full_refine2. Both write to their own folder `plus/wraps_refine2_pi05`; run "
      "them in ONE session. Cell 3 (seconds): paired McNemar tests; paste all of it.\n\n"
      "BEFORE the first run, on YOUR Colab account: the secret `HF_TOKEN` (key icon, notebook access ON) and the PaliGemma "
      "license accepted, exactly as in run file 4." + RULE,
      ["# refined pose on pi0.5 1", "# refined pose on pi0.5 2", "# refined pose on pi0.5 3"])

build("27_VIHAN_learned_refine_set_B_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 27 (Vihan, A100, ~1.5 h): learned sensing + refined pose on set B (replication, pre-registered)\n\n"
      "The same arm as run file 25 on the 30 set B camera tasks (raw and full already exist in plus/wraps, same seeds). "
      "Cell 1 (~1.5 h, resumable), own folder `plus/wraps_refine2_setB`, safe next to runs 25, 26 and 28. Cell 2 "
      "(seconds): the report; paste all of it." + RULE,
      ["# learned refine set B 1", "# learned refine set B 2"])

build("28_VIHAN_learned_refine_pi05_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 28 (Vihan, A100, ~1.5 h): pi0.5 with LEARNED sensing + refined pose (pre-registered)\n\n"
      "The deployable wrapper (learned segmenter, table-plane depth, refined pose; no simulator masks or depth) on the 30 "
      "set A camera tasks for pi0.5 (raw and full exist). Cell 1 (~1.5 h, resumable), own folder "
      "`plus/wraps_refine2_learned_pi05`; it reads the segmenter from plus/wrapper_oft/seg_net.pt. Cell 2: the report.\n\n"
      "BEFORE the first run: `HF_TOKEN` secret and the PaliGemma license, as in run file 4." + RULE,
      ["# learned refine on pi0.5 1", "# learned refine on pi0.5 2"])

def reverse_cell(source, prefix, flag="--reverse", note="SECOND RUNTIME: --reverse, shares the folder with the first"):
    """The notebook cell starting with `prefix`, with --reverse added to its wrap_run line: the SECOND runtime on a
    shared --out_dir (wrap_run shards per session and skips what the other session has done or claimed)."""
    nb = json.load(open(os.path.join(NB, source), encoding="utf-8"))
    hit = [c for c in nb["cells"] if first_line(c).startswith(prefix)]
    assert len(hit) == 1, (source, prefix, len(hit))
    lines = "".join(hit[0]["source"]).split("\n")
    lines[0] = lines[0] + f" [{note}]"
    lines = [l.replace("/content/scripts/wrap_run.py ", f"/content/scripts/wrap_run.py {flag} ") for l in lines]
    assert any(flag in l for l in lines[1:]), prefix
    return code("\n".join(lines))


SECOND = ("\n\nThis is the SECOND-RUNTIME copy: it writes to the SAME folder as Vihan's run file, walks the task list from the "
          "END (`--reverse`), and skips every episode the other runtime has done or claimed (per-session shards + "
          "`claims/`, wrap_run since commit 4230f32). The first line of the output must say `results shard: "
          "wrap_...<suffix>.jsonl`; `taken by another session, skipped` lines are expected when the two meet. Needs "
          "`restoring_organs` in My Drive to be the SHORTCUT to Vihan's shared folder (CLAUDE.md, Colab setup).")

build("25S_SUBHI_learned_refine_set_A_second_runtime_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 25S (Subhi, A100): run file 25 cell 2 from the other end (learned sensing + refined pose, set A)" + SECOND + RULE,
      [reverse_cell("plus_6_wrapper_oft.ipynb", "# learned refine 2"), "# learned refine 3"])

build("26S_SUBHI_pi05_refined_pose_further_tasks_second_runtime_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 26S (Subhi, A100): run file 26 cell 2 from the other end (pi0.5, 30 further camera tasks, raw / full / "
      "full_refine2)" + SECOND + "\n\nBEFORE the first run: your `HF_TOKEN` secret and the PaliGemma license, as in run "
      "file 4." + RULE,
      [reverse_cell("plus_3_pi0_pi05.ipynb", "# refined pose on pi0.5 2"), "# refined pose on pi0.5 3"])

THIRD = SECOND.replace("SECOND-RUNTIME", "THIRD-RUNTIME").replace("walks the task list from the END (`--reverse`)",
                                                                   "starts in the MIDDLE of the task list (`--rotate 0.5`)")
build("26M_VIHAN_pi05_refined_pose_further_tasks_third_runtime_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 26M (Vihan, A100): run file 26 cell 2 from the MIDDLE (third runtime on the same folder)" + THIRD +
      "\n\nBEFORE the first run: `HF_TOKEN` and the PaliGemma license, as in run file 4." + RULE,
      [reverse_cell("plus_3_pi0_pi05.ipynb", "# refined pose on pi0.5 2", "--rotate 0.5",
                    "THIRD RUNTIME: --rotate 0.5, shares the folder with two others")])
build("25M_learned_refine_set_A_third_runtime_A100", "plus_6_wrapper_oft.ipynb", SUBSET,
      "# 25M (A100): run file 25 cell 2 from the MIDDLE (third runtime on the same folder)" + THIRD + RULE,
      [reverse_cell("plus_6_wrapper_oft.ipynb", "# learned refine 2", "--rotate 0.5",
                    "THIRD RUNTIME: --rotate 0.5, shares the folder with two others")])
build("28S_SUBHI_learned_refine_pi05_second_runtime_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 28S (Subhi, A100): run file 28 cell 1 from the other end (pi0.5, learned sensing + refined pose)" + SECOND +
      "\n\nBEFORE the first run: your `HF_TOKEN` and the PaliGemma license, as in run file 4." + RULE,
      [reverse_cell("plus_3_pi0_pi05.ipynb", "# learned refine on pi0.5 1"), "# learned refine on pi0.5 2"])

build("24_VIHAN_render_rung_net_of_nominal_pi05_A100", "plus_3_pi0_pi05.ipynb", "# REQUIRED for pi0 / pi0.5",
      "# 24 (Vihan, A100, ~1 h): the perfect-render rung net of nominal success (POST HOC)\n\n"
      "The ladder's top rung is the nominal observation, so a render-only unit may just be a task the policy solves "
      "unperturbed. Cell 1 runs `nominal_view` (the training camera's own render, nothing measured) on EXACTLY the 24 "
      "pi0.5 ladder units, same seeds, own folder `plus/wraps_nominal_pi05`. The OFT half (the 21 render-only units among "
      "all its ladder units) is run file 19 cell 2; run that first or in another session (different folders). Cell 2 "
      "(seconds): per family and bucket, how many units the unperturbed view solves; paste all of it.\n\n"
      "BEFORE the first run, on YOUR Colab account: the secret `HF_TOKEN` (key icon, notebook access ON) with a Hugging "
      "Face read token, and the PaliGemma license accepted (huggingface.co/google/paligemma-3b-pt-224), exactly as in "
      "run file 4, or the pi0.5 download fails." + RULE,
      ["# render rung net of nominal 1", "# render rung net of nominal 2"])
