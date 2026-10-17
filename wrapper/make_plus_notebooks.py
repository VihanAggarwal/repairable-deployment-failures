"""Build the seven Colab notebooks for the five-family LIBERO-Plus study, SIMPLER, and DROID.
Every code cell is validated: bash cells with `bash -n`, Python cells with ast (shell lines masked),
%%writefile sources are the py_compiled scripts. No em-dashes anywhere."""
import ast, json, pathlib, re, shutil, subprocess, tempfile

HERE = pathlib.Path(__file__).parent
SCRIPTS = HERE / "scripts"
DL = HERE / "notebooks"
DL.mkdir(parents=True, exist_ok=True)
EM = "—"
DRIVE_OUT = "/content/drive/MyDrive/restoring_organs/plus"
SUBSET = f"{DRIVE_OUT}/subset_libero10.json"
SWEEP_OUT = f"{DRIVE_OUT}/sweep"
NOMINALS = f"{DRIVE_OUT}/wrapper_oft/nominals_v3.json"
SWEEP_SCRIPTS = ["arms_report.py", "smc.py", "smc_probe.py", "gate_report.py", "wrapper_libero.py", "calib_check.py", "identifiability.py", "sweep_conditions.py",
                 "policy_adapters.py", "sweep_run.py", "sweep_analysis.py", "wrap_run.py", "canon_net.py",
                 "train_seg.py", "invertibility_report.py", "attribution_ladder.py", "tolerance_probe.py", "pose_sources.py"]
WRAPS = f"{DRIVE_OUT}/wraps"
SEG_CKPT = f"{DRIVE_OUT}/wrapper_oft/seg_net.pt"
# notebooks nobody should run now (done, dead, superseded, or optional); written to notebooks/archive/ so the live
# folder holds only plus_2 (W9), plus_3 (W5), plus_6 (W7, W7b, W8) and droid_selfcal (W6)
ARCHIVE = {"plus_1_openvla_oft.ipynb", "plus_4_diffusion_policy.ipynb", "plus_5_octo.ipynb",
           "simpler_octo_openvla.ipynb", "droid_real_signal.ipynb"}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip()}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()}


def drive_cell():
    return code(f"""
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['MPLBACKEND'] = 'Agg'   # Colab's inline matplotlib backend crashes every venv python
os.environ['PYTHONUNBUFFERED'] = '1'   # run cells are piped into a log on Drive; without this their output arrives in bursts
os.makedirs('{DRIVE_OUT}/logs', exist_ok=True)
os.makedirs('/content/logs', exist_ok=True)
print('outputs ->', '{DRIVE_OUT}')
# Drive uploads a file only after it is CLOSED, so a log held open by a running cell never arrives. Run cells log to
# /content/logs; this thread rewrites the last 64 KB of each log on Drive once a minute as a fresh, closed file.
import builtins, glob, threading, time
def _mirror_logs():
    while True:
        for p in glob.glob('/content/logs/*.log'):
            try:
                with open(p, 'rb') as f:
                    f.seek(max(0, os.path.getsize(p) - 65536))
                    tail = f.read()
                with open('{DRIVE_OUT}/logs/' + os.path.basename(p)[:-4] + '.' + os.uname().nodename[-6:] + '.log', 'wb') as f:
                    f.write(tail)
            except Exception:
                pass
        time.sleep(60)
if not getattr(builtins, '_log_mirror_on', False):
    builtins._log_mirror_on = True
    threading.Thread(target=_mirror_logs, daemon=True).start()
# Hugging Face login from the Colab secret HF_TOKEN (key icon, left sidebar), if you added one; optional here
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'])
    print('Hugging Face login ok')
except Exception as e:
    print('no HF_TOKEN secret, continuing without Hugging Face login:', type(e).__name__)
""")


def write_scripts(names):
    return [code(f"%%writefile /content/scripts/{n}\n" + (SCRIPTS / n).read_text(encoding="utf-8")) for n in names]


def mkdir_scripts():
    return code("import os; os.makedirs('/content/scripts', exist_ok=True); print('ok')")


LP_BLOCK = r'''
apt-get -qq update > /dev/null
apt-get -qq install -y libexpat1 libfontconfig1-dev libmagickwand-dev unzip > /dev/null
cd /content
[ -d LIBERO-plus ] || git clone -q --depth 1 https://github.com/sylvestf/LIBERO-plus.git
@@PIP@@ -e /content/LIBERO-plus @@NODEPS@@
@@PIP@@ wand scikit-image
# LIBERO's setup.py does not register the top-level package, so the editable install alone is not importable
SITE=$(@@PY@@ -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo /content/LIBERO-plus > "$SITE/libero_plus.pth"
@@PY@@ -c "import importlib.util as u; assert u.find_spec('libero') is not None, 'libero still not importable'; print('libero importable')"
if [ ! -f /content/LIBERO-plus/.assets_ok ]; then
  echo "downloading LIBERO-Plus assets (6.4 GB)..."
  # Hugging Face answers a rate-limited or interrupted download with an error page or a short file; plain `curl -s`
  # saved that as the zip and unzip then failed with no message (2026-09-19). Fail on HTTP errors, resume, retry, send
  # the HF_TOKEN when there is one (anonymous downloads are limited first), and test the archive before unpacking.
  ZIP=/content/lp_assets.zip
  URL=https://huggingface.co/datasets/sii-research/LIBERO_plus_assets/resolve/main/assets.zip
  ok=0
  for try in 1 2 3 4 5 6; do
    if [ -f $ZIP ] && unzip -tq $ZIP > /dev/null 2>&1; then ok=1; break; fi
    size=$(stat -c %s $ZIP 2>/dev/null || echo 0)
    if [ $try -gt 1 ]; then echo "asset download attempt $((try-1)) incomplete ($size bytes), retrying in 30 s"; sleep 30; fi
    if [ "$size" -lt 1000000 ]; then rm -f $ZIP; fi                 # an error page cannot be resumed
    curl -fL -C - --retry 3 --retry-delay 15 -sS ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o $ZIP $URL || true
  done
  if [ $ok != 1 ]; then echo "LIBERO-Plus assets could not be downloaded (Hugging Face rate limit or outage). Wait a few minutes and re-run this cell; it resumes."; exit 1; fi
  rm -rf /content/lp_assets && mkdir -p /content/lp_assets
  unzip -q /content/lp_assets.zip -d /content/lp_assets && rm -f /content/lp_assets.zip
  python3 - <<'EOF'
import os, shutil
root, dst = "/content/lp_assets", "/content/LIBERO-plus/libero/libero/assets"
cands = [os.path.join(d, n) for d, ns, _ in os.walk(root) for n in ns if n == "assets"]
src = min(cands, key=len) if cands else root
os.makedirs(dst, exist_ok=True)
shutil.copytree(src, dst, dirs_exist_ok=True)
print("assets:", src, "->", dst, "|", len(os.listdir(dst)), "entries")
EOF
  touch /content/LIBERO-plus/.assets_ok
fi
'''


def lp_block(pip, py, nodeps=""):
    return LP_BLOCK.replace("@@PIP@@", pip).replace("@@PY@@", py).replace("@@NODEPS@@", nodeps)


CONFIG_CELL = code(r'''
# LIBERO-Plus asks for its paths on first import; write them up front
import os
root = '/content/LIBERO-plus/libero/libero'
os.makedirs(os.path.expanduser('~/.libero'), exist_ok=True)
open(os.path.expanduser('~/.libero/config.yaml'), 'w').write(
    f'benchmark_root: {root}\nbddl_files: {root}/bddl_files\ninit_states: {root}/init_files\n'
    f'datasets: {root}/../datasets\nassets: {root}/assets\n')
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MPLBACKEND'] = 'Agg'   # Colab's inline backend is not valid inside the venvs
os.environ['PYTHONPATH'] = '/content/LIBERO-plus'   # Colab's system python ignores the .pth file
print(open(os.path.expanduser('~/.libero/config.yaml')).read())
''')


def subset_cell(py):
    return code(f"""
# the SAME task subset for every family (built once, then read from Drive)
!MUJOCO_GL=egl {py} -c "import sys; sys.path.insert(0, '/content/scripts'); import plus_common as pc; s = pc.load_or_build('{SUBSET}'); print('suite tasks', s['n_tasks']); print('per category', s['counts']); print('name tokens', s['name_tokens']); print('other examples', s['other_examples']); print(len(s['tasks']), 'tasks in subset')"
""")


def log_to_drive(src, logs):
    """Every `!... /content/scripts/<name>.py ...` line also appends its output (stderr included, so tracebacks too) to
    /content/logs/<name>.log, whose tail the Drive cell's mirror thread copies to <logs>/<name>.log once a minute, where
    it can be read while the cell is still running. Output still shows in the cell.
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


def notebook(cells, name):
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    for i, c in enumerate(cells):
        src = c["source"]
        assert EM not in src, f"{name} cell {i}: em-dash"
        if c["cell_type"] != "code" or src.startswith("%%writefile"):
            continue
        if not src.startswith("%%"):
            src = c["source"] = log_to_drive(src, f"{DRIVE_OUT}/logs")
        if src.startswith("%%bash"):
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, newline="\n") as f:
                f.write(src.split("\n", 1)[1])
            r = subprocess.run(["bash", "-n", f.name], capture_output=True, text=True)
            assert r.returncode == 0, f"{name} cell {i}: bash syntax\n{r.stderr}"
            continue
        masked = "\n".join((l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l)
                           for l in src.split("\n"))
        try:
            ast.parse(masked)
        except SyntaxError as e:
            raise AssertionError(f"{name} cell {i}: {e}\n{masked}")
    path = (DL / "archive" / name) if name in ARCHIVE else (DL / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(path, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"wrote {path} ({len(cells)} cells)")


VENV = "pip -q install uv\n[ -x @@V@@/bin/python ] || uv venv -q @@V@@ -p 3.10\n"

# ------------------------------------------------------------------ 1. OpenVLA-OFT
V1, PY1 = "/content/venv_oft", "/content/venv_oft/bin/python"
notebook([
    md("""
# LIBERO-Plus, family 1: OpenVLA-OFT (L1-regression chunks)
**GPU: A100** (L4 works, slower). Setup ~20 min (includes 6.4 GB LIBERO-Plus assets and the 15 GB checkpoint).
Full run: ~210 perturbed LIBERO-10 tasks, 1 episode each, about 1.5 to 2 h. Resumable: rerun the last cells after a disconnect.
Paste back the printed table of the last cell.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["plus_common.py", "eval_oft.py"]),
    code("%%bash\nset -e\n" + VENV.replace("@@V@@", V1) + f"""cd /content
[ -d openvla-oft ] || git clone -q https://github.com/moojink/openvla-oft.git
uv pip install -q --python {PY1} -e /content/openvla-oft
""" + lp_block(f"uv pip install -q --python {PY1}", PY1) + f"""
uv pip install -q --python {PY1} -r /content/openvla-oft/experiments/robot/libero/libero_requirements.txt
uv pip install -q --python {PY1} "numpy<2" "mujoco==2.3.7"
uv pip install -q --python {PY1} "protobuf==4.25.3"
uv pip install -q --python {PY1} --no-deps "tensorflow-metadata==1.16.1"
{PY1} -c "import torch, robosuite, mujoco; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mujoco', mujoco.__version__)"
"""),
    CONFIG_CELL, subset_cell(PY1),
    code(f"# smoke: 2 tasks\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/eval_oft.py --subset {SUBSET} --out {DRIVE_OUT}/smoke_oft.jsonl --limit 2"),
    code(f"# full run (resumable)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/eval_oft.py --subset {SUBSET} --out {DRIVE_OUT}/rows_oft.jsonl"),
], "plus_1_openvla_oft.ipynb")

# ------------------------------------------------------------------ 2. OpenVLA base
V2, PY2 = "/content/venv_vla", "/content/venv_vla/bin/python"
notebook([
    md("""
# LIBERO-Plus, family 2: OpenVLA (discrete action tokens)
**GPU: L4 or A100.** Official openvla eval loop. One forward pass per action, so this is the slowest family:
~210 episodes, about 4 to 5 h on an A100. Resumable. Paste back the last table.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["plus_common.py", "eval_openvla.py"] + SWEEP_SCRIPTS),
    code("%%bash\nset -e\n" + VENV.replace("@@V@@", V2) + f"""cd /content
[ -d openvla ] || git clone -q https://github.com/openvla/openvla.git
uv pip install -q --python {PY2} -e /content/openvla
# no flash-attn build on Colab: fall back to eager attention if the loader asks for flash attention
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' /content/openvla/experiments/robot/openvla_utils.py
""" + lp_block(f"uv pip install -q --python {PY2}", PY2) + f"""
if [ -f /content/openvla/experiments/robot/libero/libero_requirements.txt ]; then
  uv pip install -q --python {PY2} -r /content/openvla/experiments/robot/libero/libero_requirements.txt
else
  uv pip install -q --python {PY2} "imageio[ffmpeg]" robosuite==1.4.1 bddl easydict cloudpickle gym
fi
uv pip install -q --python {PY2} "numpy<2" "mujoco==2.3.7"
uv pip install -q --python {PY2} "protobuf==4.25.3"
uv pip install -q --python {PY2} --no-deps "tensorflow-metadata==1.16.1"
{PY2} -c "import torch, transformers, robosuite; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
"""),
    CONFIG_CELL, subset_cell(PY2),
    code(f"# smoke: 2 tasks\n!cd /content/openvla && MUJOCO_GL=egl {PY2} /content/scripts/eval_openvla.py --subset {SUBSET} --out {DRIVE_OUT}/smoke_openvla.jsonl --limit 2"),
    code(f"# full run (resumable)\n!cd /content/openvla && MUJOCO_GL=egl {PY2} /content/scripts/eval_openvla.py --subset {SUBSET} --out {DRIVE_OUT}/rows_openvla.jsonl"),
    code(f"# graded identifiability sweep, OpenVLA episodes (slowest family: 4 base tasks x 24 conditions x raw/full, ~12 h on A100, resumable; the `gated` arm re-runs only episodes where the do-no-harm gate vetoes a channel, the rest are copied from `full`)\n!cd /content/openvla && MUJOCO_GL=egl {PY2} /content/scripts/sweep_run.py episodes --family openvla --n_tasks 4 --arms raw,full,gated --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# wrapper on OpenVLA, all 7 categories, paired raw/full (a third of the subset per category: ~7 h on A100, resumable)\n!cd /content/openvla && MUJOCO_GL=egl {PY2} /content/scripts/wrap_run.py --family openvla --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full,full_gated --per_cat_scale 0.34\n!{PY2} /content/scripts/invertibility_report.py --family openvla:{WRAPS}/wrap_openvla.jsonl:{WRAPS}/wrap_openvla.jsonl --out {WRAPS}/report_openvla.json"),
    code(f"# self-motion consistency, OpenVLA probe (NO rollouts, 4 base tasks x 24 conditions, ~2.5 h, resumable; run AFTER the sweep so outcomes exist to join)\n!cd /content/openvla && MUJOCO_GL=egl {PY2} /content/scripts/smc_probe.py run --family openvla --n_tasks 4 --out_dir {SWEEP_OUT} --nominals {NOMINALS}\n!{PY2} /content/scripts/smc_probe.py report --out_dir {SWEEP_OUT}"),
    code(f"# do-no-harm gate on OpenVLA: the PRE-REGISTERED test (RESULTS.md 2026-09-18), paste all of it\n!{PY2} /content/scripts/gate_report.py --rows {WRAPS}/wrap_openvla.jsonl --rows {SWEEP_OUT}/episodes_openvla.jsonl"),
    # ---- run file 21 (POST HOC 2026-09-24): more OpenVLA base tasks for the held-out score test; frozen score and constants, own folder
    code(f"# OpenVLA extension 1 (~25 min, resumable): the unperturbed task (condition nominal, raw) on the 6 base tasks the registered sweep did not use (all 10 have policy-free scores). Rule written in TASKS.md BEFORE this runs: a new base task enters the extension iff this raw nominal episode succeeds (seed = task id), exactly the label rule of the registered sweep. Own folder plus/sweep_openvla_ext\n!cd /content/openvla && MUJOCO_GL=egl {PY2} /content/scripts/sweep_run.py episodes --family openvla --n_tasks 10 --task_ids 1101,1146,1255,1297,1325,1371 --conditions nominal --arms raw --out_dir {SWEEP_OUT}_openvla_ext --nominals {NOMINALS}"),
    code(f"# OpenVLA extension 2 (~3 h per solved base task on A100, resumable): all 24 conditions x raw/full on the base tasks cell 1 found solved. If it prints 'solved: []' STOP: the extension has no units\n"
         f"import glob, json\nE = '{SWEEP_OUT}_openvla_ext'\nrows = [json.loads(l) for p in glob.glob(E + '/episodes_openvla*.jsonl') for l in open(p) if l.strip()]\n"
         "solved = sorted({r['task_id'] for r in rows if r.get('cond') == 'nominal' and r['arm'] == 'raw' and r['success']})\nprint('solved:', solved)\nids = ','.join(map(str, solved)) or '0'\n"
         f"!cd /content/openvla && MUJOCO_GL=egl {PY2} /content/scripts/sweep_run.py episodes --family openvla --n_tasks 10 --task_ids {{ids}} --arms raw,full --out_dir {SWEEP_OUT}_openvla_ext --nominals {NOMINALS}"),
    code(f"# OpenVLA extension 3: report (paste all of it). (a) the NEW units alone, (b) pooled with the registered 4-task sweep; both are POST HOC, the registered AUROC stays the headline\n"
         f"import glob, os, shutil\nS, E = '{SWEEP_OUT}', '{SWEEP_OUT}_openvla_ext'\n"
         "for d in ('/content/ovla_ext', '/content/ovla_pooled'):\n    os.makedirs(d, exist_ok=True); shutil.copy(S + '/scores_v2.jsonl', d)\n"
         "reg = glob.glob(S + '/episodes_openvla.jsonl') + glob.glob(S + '/episodes_openvla.*.jsonl')      # OpenVLA only, never episodes_openvla_oft\n"
         "new = glob.glob(E + '/episodes_openvla.jsonl') + glob.glob(E + '/episodes_openvla.*.jsonl')\n"
         "[shutil.copy(p, '/content/ovla_pooled') for p in reg]; [shutil.copy(p, '/content/ovla_ext') for p in new]\n"
         "[shutil.copy(p, '/content/ovla_pooled/episodes_openvla.ext%d.jsonl' % i) for i, p in enumerate(new)]\nprint('registered files', len(reg), '| new files', len(new))\n"
         f"!echo '(a) new units only'; {PY2} /content/scripts/sweep_analysis.py --out_dir /content/ovla_ext\n"
         f"!echo '(b) pooled with the registered sweep'; {PY2} /content/scripts/sweep_analysis.py --out_dir /content/ovla_pooled"),
], "plus_2_openvla_base.ipynb")

# ------------------------------------------------------------------ 3. pi0 + pi0.5 (LeRobot)
LEROBOT_INSTALL = r'''
pip -q install "lerobot[@@EXTRAS@@] @ git+https://github.com/huggingface/lerobot.git@v0.6.1"
if ! python -c "import dataclasses; from lerobot.envs.configs import LiberoEnv; assert 'is_libero_plus' in {f.name for f in dataclasses.fields(LiberoEnv)}"; then
  echo "v0.6.1 lacks is_libero_plus, installing lerobot main"
  pip -q install "lerobot[@@EXTRAS@@] @ git+https://github.com/huggingface/lerobot.git"
fi
python -c "import lerobot; print('lerobot', lerobot.__version__ if hasattr(lerobot, '__version__') else 'ok')"
'''
# robosuite's segmentation renderer overflows uint8 under NumPy 2 (system python only; the venvs pin numpy<2)
SWAP = ("pip -q uninstall -y hf-libero || true\n" + lp_block("pip -q install", "python", "--no-deps")
        + "python /content/scripts/robosuite_np2.py\n")
notebook([
    md("""
# LIBERO-Plus, family 3: pi0 and pi0.5 (flow matching, LeRobot)
**GPU: L4** (A100 also fine). LeRobot's built-in LIBERO env with `is_libero_plus=true`, same task subset as every other family.
Setup ~15 min. Each policy ~210 episodes, about 1 to 2 h. Resumable in batches of 30. Paste back both tables.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["plus_common.py", "lerobot_tools.py", "eval_octo.py", "robosuite_np2.py"] + SWEEP_SCRIPTS),
    code("%%bash\nset -e\n" + LEROBOT_INSTALL.replace("@@EXTRAS@@", "libero,pi") + SWAP),
    CONFIG_CELL, subset_cell("python"),
    code("# REQUIRED for pi0 / pi0.5: PaliGemma's tokenizer is gated by Google.\n"
         "# 1) accept the license at https://huggingface.co/google/paligemma-3b-pt-224\n"
         "# 2) make a READ token at https://huggingface.co/settings/tokens\n"
         "# 3) Colab key icon (left sidebar) -> Add new secret: name HF_TOKEN, value = token, notebook access ON\n"
         "import os\n"
         "from google.colab import userdata\n"
         "from huggingface_hub import login, whoami, hf_hub_download\n"
         "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
         "login(token=os.environ['HF_TOKEN'])\n"
         "print('logged in as', whoami()['name'])\n"
         "print('PaliGemma access ok:', hf_hub_download('google/paligemma-3b-pt-224', 'config.json'))"),
    code(f"# smoke: pi0.5 on 2 tasks\n!MUJOCO_GL=egl python /content/scripts/lerobot_tools.py eval --subset {SUBSET} --out {DRIVE_OUT}/smoke_pi05 --policy lerobot/pi05_libero_finetuned_v044 --name smoke_pi05 --limit 2 --batch 2"),
    code(f"# pi0.5 full (resumable)\n!MUJOCO_GL=egl python /content/scripts/lerobot_tools.py eval --subset {SUBSET} --out {DRIVE_OUT}/lerobot_pi05 --policy lerobot/pi05_libero_finetuned_v044 --name pi05"),
    code(f"# pi0 full (resumable)\n!MUJOCO_GL=egl python /content/scripts/lerobot_tools.py eval --subset {SUBSET} --out {DRIVE_OUT}/lerobot_pi0 --policy lerobot/pi0_libero_finetuned_v044 --name pi0"),
    code(f"# wrapper on pi0.5, all 7 categories, paired raw/full through LeRobot's own processors (~5 h on A100, resumable)\n!pip -q install opencv-python-headless scipy\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full\n!python /content/scripts/invertibility_report.py --family pi05:{WRAPS}/wrap_pi05.jsonl:{WRAPS}/wrap_pi05.jsonl --out {WRAPS}/report_pi05.json"),
    code(f"# graded identifiability sweep, pi0.5 episodes (3 base tasks x 24 conditions x raw/full, ~3 h on A100, resumable)\n!MUJOCO_GL=egl python /content/scripts/sweep_run.py episodes --family pi05 --n_tasks 3 --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# settle check on pi0.5 (~3 h, resumable): TWO fixes for the harm the wrapper does when it has nothing to correct (45 of 158 raw successes broken on the ten-task sweep, 40 of them where no camera correction was applied). full_lazy = one motion-free look at the robot's silhouette first, and NO measurement sweep when it sits where the training camera puts it (the arm never moves, so the episode starts exactly like raw). the arm full_settle ends the measurement sweep with the arm at the training pose AND at rest (plain full hands over 2.6 mm off and still moving at 0.19 rad/s, measured locally). First line: raw / full / full_settle on the categories where the wrapper mostly applies nothing. Second line: the raw arm AGAIN with the same seeds, which measures how often pi0.5 flips with no wrapper involved at all\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full,full_settle,full_lazy --categories layout,background,light,sensor_noise\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {WRAPS}_same_seed --nominals {NOMINALS} --arms raw --categories layout,background\n!python /content/scripts/arms_report.py --rows {WRAPS}/wrap_pi05.jsonl --arms raw,full,full_settle,full_lazy"),
    code(f"# tolerance probe, pi0.5 (~40 min, resumable, NO rollouts): how wrong may the camera pose be before pi0.5's opening action leaves what it does under a perfect calibration? True pose plus a known residual on 10 base tasks x 2 orbit conditions; paste the report. Own folder\n!MUJOCO_GL=egl python /content/scripts/tolerance_probe.py run --family pi05 --n_tasks 10 --out_dir {DRIVE_OUT}/tolerance --nominals {NOMINALS}\n!python /content/scripts/tolerance_probe.py report --out_dir {DRIVE_OUT}/tolerance"),
    code(f"# ladder on pi0.5 (~1.5 h, resumable): the two UPPER RUNGS of the attribution ladder on the camera tasks pi0.5 fails under both raw and full. full_truepose = the true camera pose in the same correction, full_oracle = a perfect render from the training camera (what still fails there is pi0.5 itself, not the camera). Writes to its OWN folder; truepose rows that exist already are not repeated\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_ladder --nominals {NOMINALS} --arms full_truepose,full_oracle --categories camera --only_failed_in {WRAPS}/wrap_pi05.jsonl\n!python /content/scripts/attribution_ladder.py --set pi05 {WRAPS}/wrap_pi05.jsonl {DRIVE_OUT}/wraps_ladder/wrap_pi05.jsonl"),
    code(f"# camera diagnostic on pi0.5: the 8 camera tasks with a SMALL estimated shift (under 35 cm) that raw pi0.5 fails, of which the wrapper recovered 1. full_truepose = the true camera pose instead of our estimate (is it calibration?), full_nofill = holes inpainted instead of filled from the training view (is it the fill?). 16 episodes, ~45 min, resumable; the existing raw / full rows are kept\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full,full_truepose,full_nofill --categories camera --tasks 699,712,716,717,722,730,784,1060\n!python /content/scripts/arms_report.py --rows {WRAPS}/wrap_pi05.jsonl --arms raw,full,full_truepose,full_nofill"),
    code(f"# baselines on pi0.5: generic corrections on light / sensor_noise (raw rows exist; 20 episodes, ~1.2 h). If an UNMEASURED correction breaks pi0.5 too, the harm is a property of correcting a robust policy, not of our wrapper\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full,generic --categories light,sensor_noise\n!python /content/scripts/arms_report.py --rows {WRAPS}/wrap_pi05.jsonl --arms raw,full,generic"),
    code(f"# self-motion consistency, pi0.5 probe (NO rollouts, 72 units, ~1.5 h, resumable)\n!MUJOCO_GL=egl python /content/scripts/smc_probe.py run --family pi05 --n_tasks 3 --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# self-motion consistency: report (paste all of it)\n!python /content/scripts/smc_probe.py report --out_dir {SWEEP_OUT}"),
    code(f"# do-no-harm gate on pi0.5, sweep (adds the `gated` arm to the finished sweep: ~1 min per episode where nothing is vetoed, a full episode where something is; ~2 h, resumable)\n!MUJOCO_GL=egl python /content/scripts/sweep_run.py episodes --family pi05 --n_tasks 3 --arms raw,full,gated --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# do-no-harm gate on pi0.5, LIBERO-Plus light / sensor_noise / camera tasks (where the wrapper broke pi0.5; ~2 h, resumable; run AFTER the wrapper cell has finished)\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full,full_gated --categories light,sensor_noise,camera"),
    code(f"# do-no-harm gate on pi0.5: report (paste all of it)\n!python /content/scripts/gate_report.py --rows {WRAPS}/wrap_pi05.jsonl --rows {SWEEP_OUT}/episodes_pi05.jsonl"),
    code(f"# scale-up of the held-out test: pi0.5 on all 10 base tasks (7 new tasks x 24 conditions x raw/full + gated, ~8 h on A100, resumable). Needs the 10-task scores from plus_6 'scale-up, step 1' before the analysis, not before this cell\n!MUJOCO_GL=egl python /content/scripts/sweep_run.py episodes --family pi05 --n_tasks 10 --arms raw,full,gated --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    # ---- run file 26 (2026-09-25, tests pre-registered in wrapper/prereg/2026-09-25_refine2.md): pi0.5 camera with the refined pose
    code(f"# refined pose on pi0.5 1 (~1 h on A100, resumable): full_refine2 (the silhouette pose refined EasyHeC-style from two starts, the lower dense loss kept) on the 30 set A camera tasks; raw and full for these tasks already exist in plus/wraps. Own folder plus/wraps_refine2_pi05\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_refine2_pi05 --nominals {NOMINALS} --arms full_refine2 --categories camera"),
    code(f"# refined pose on pi0.5 2 (~2.5 h on A100, resumable): 30 FURTHER camera tasks outside every evaluation subset, raw, full and full_refine2 on each, same folder\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_refine2_pi05 --nominals {NOMINALS} --arms raw,full,full_refine2 --categories camera --per_cat_scale 0 --extra_from_suite 30"),
    code("# refined pose on pi0.5 3: report (seconds; paste all of it). Paired against raw, exact McNemar, on set A, the new tasks and both\n"
         "import json, os\nfrom math import comb\nD = '@D@'\nimport glob; rows = lambda p: [json.loads(l) for q in [p] + sorted(glob.glob(p[:-6] + '.*.jsonl')) if os.path.exists(q) for l in open(q) if l.strip()]\n"
         "R = [r for r in rows(f'{D}/wraps_refine2_pi05/wrap_pi05.jsonl') + rows(f'{D}/wraps/wrap_pi05.jsonl') if r.get('category') == 'camera' and r.get('rep', 0) == 0]\n"
         "by = {}\nfor r in R: by.setdefault(r['task_id'], {}).setdefault(r['arm'], bool(r['success']))\n"
         "mc = lambda b, c: min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0\n"
         "new = {r['task_id'] for r in rows(f'{D}/wraps_refine2_pi05/wrap_pi05.jsonl') if r['arm'] == 'raw'}\n"
         "for name, ids in (('set A', set(by) - new), ('new tasks', new), ('both', set(by))):\n"
         "    for arm in ('full', 'full_refine2'):\n"
         "        t = [i for i in ids if 'raw' in by[i] and arm in by[i]]\n"
         "        b = sum(by[i][arm] and not by[i]['raw'] for i in t); c = sum(by[i]['raw'] and not by[i][arm] for i in t)\n"
         "        print(f'{name:<10} {arm:<13} n {len(t):3d} | raw {sum(by[i][\"raw\"] for i in t):3d} -> {sum(by[i][arm] for i in t):3d} | repaired {b} broken {c} | exact McNemar p {mc(b, c):.4f}')\n".replace("@D@", DRIVE_OUT)),
    # ---- run file 28 (2026-09-26, pre-registered in wrapper/prereg/2026-09-26_learned_refine_replication.md): pi0.5, LEARNED sensing
    code(f"# learned refine on pi0.5 1 (~1.5 h on A100, resumable): full_hybrid_refine2 (learned segmenter, table-plane depth, refined pose) on the 30 set A camera tasks; raw and full exist in plus/wraps. Needs the segmenter at {SEG_CKPT}. Own folder plus/wraps_refine2_learned_pi05\n!pip -q install opencv-python-headless scipy\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_refine2_learned_pi05 --nominals {NOMINALS} --arms full_hybrid_refine2 --categories camera --seg_ckpt {SEG_CKPT}"),
    code("# learned refine on pi0.5 2: report (seconds; paste all of it)\n"
         "import json, os, numpy as np\nfrom math import comb\nD = '@D@'\nimport glob; rows = lambda p: [json.loads(l) for q in [p] + sorted(glob.glob(p[:-6] + '.*.jsonl')) if os.path.exists(q) for l in open(q) if l.strip()]\n"
         "new = {r['task_id']: r for r in rows(f'{D}/wraps_refine2_learned_pi05/wrap_pi05.jsonl') if r['arm'] == 'full_hybrid_refine2'}\n"
         "old = {}\nfor r in rows(f'{D}/wraps/wrap_pi05.jsonl'):\n    if r['task_id'] in new and r.get('rep', 0) == 0 and r.get('category') == 'camera': old.setdefault(r['task_id'], {}).setdefault(r['arm'], bool(r['success']))\n"
         "mc = lambda b, c: min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0\n"
         "e = np.array([r['cam_est_err_cm_deg'][0] for r in new.values() if r.get('cam_est_err_cm_deg')])\n"
         "print('learned refine2 n', len(new), 'success', sum(bool(r['success']) for r in new.values()), '| cal err median %.2f cm, within 2 cm %d' % (np.median(e), (e <= 2).sum()))\n"
         "for ref in ('raw', 'full'):\n    t = [i for i in new if ref in old.get(i, {})]\n"
         "    b = sum(new[i]['success'] and not old[i][ref] for i in t); c = sum(old[i][ref] and not new[i]['success'] for i in t)\n"
         "    print(f'learned refine2 vs {ref}: n {len(t)} {ref} {sum(old[i][ref] for i in t)} -> {sum(bool(new[i][\"success\"]) for i in t)} | wins {b} losses {c} | exact McNemar p {mc(b, c):.4f}')\n".replace("@D@", DRIVE_OUT)),
    # ---- run file 24 (POST HOC 2026-09-24): the render rung net of nominal (pi0.5 rollouts; the OFT rows come from run file 19)
    code(f"# render rung net of nominal 1, pi0.5 (~1 h on A100, resumable): nominal_view (every frame the simulator's render from the TRAINING camera, nothing measured, arm untouched) on EXACTLY the pi0.5 ladder units (camera tasks that fail under both raw and full; same seeds). Own folder plus/wraps_nominal_pi05\n!MUJOCO_GL=egl python /content/scripts/wrap_run.py --family pi05 --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_nominal_pi05 --nominals {NOMINALS} --arms nominal_view --categories camera --only_failed_in {WRAPS}/wrap_pi05.jsonl"),
    code("# render rung net of nominal 2: report (seconds; paste all of it). For every ladder bucket, how many units the UNPERTURBED view solves; the render-only bucket is the one that matters. OFT needs run file 19 cell 2 on Drive (plus/wraps_nominal_<suite>)\n"
         "import sys; sys.path.insert(0, '/content/scripts')\nimport attribution_ladder as AL\nfrom plus_common import load_rows\nD = '@D@'\n"
         "S = [('OFT set A', [f'{D}/wrapper_oft/rows.jsonl', f'{D}/rows_oft.jsonl'], f'{D}/wraps_nominal_libero_10/wrap_openvla_oft.jsonl'),\n"
         "     ('OFT set B', [f'{D}/wraps/wrap_openvla_oft.jsonl', f'{D}/wraps_ladder_libero_10/wrap_openvla_oft.jsonl'], f'{D}/wraps_nominal_libero_10/wrap_openvla_oft.jsonl')]\n"
         "S += [(f'OFT {s}', [f'{D}/wraps_libero_{s}/wrap_openvla_oft.jsonl', f'{D}/wraps_ladder_libero_{s}/wrap_openvla_oft.jsonl'], f'{D}/wraps_nominal_libero_{s}/wrap_openvla_oft.jsonl') for s in ('object', 'goal', 'spatial')]\n"
         "S += [('pi0.5 set A', [f'{D}/wraps/wrap_pi05.jsonl', f'{D}/wraps_ladder/wrap_pi05.jsonl'], f'{D}/wraps_nominal_pi05/wrap_pi05.jsonl')]\n"
         "tot = {}\nfor name, files, nomf in S:\n"
         "    u = AL.units(files)\n    nv = {r['task_id']: bool(r['success']) for r in load_rows(nomf) if r.get('arm') == 'nominal_view' and r.get('rep', 0) == 0}\n"
         "    fam = name.split()[0]\n"
         "    for t, v in u.items():\n        if 'raw' in v and 'full' in v and not v['raw']['success']:\n"
         "            k = (fam, AL.bucket(v)); c = tot.setdefault(k, [0, 0, 0])\n            c[0] += 1\n"
         "            if t in nv:\n                c[1] += 1; c[2] += nv[t]\n"
         "for (fam, b), (n, m, s) in sorted(tot.items()):\n"
         "    print(f'{fam:<6} {b:<26} units {n:3d} | with a nominal_view row {m:3d} | unperturbed view solves {s:3d}' + (f' | the unperturbed task fails {m - s} of the units the perfect render solved (single-rollout noise, or the render is not the nominal frame)' if b == 'not invertible' else ''))\n".replace("@D@", DRIVE_OUT)),
], "plus_3_pi0_pi05.ipynb")

# ------------------------------------------------------------------ 4. Diffusion Policy (train + eval)
TRAIN = r'''
# ---- train Diffusion Policy on the LIBERO-10 episodes of lerobot/libero (resumable via Drive) ----
import glob, json, os, shutil, subprocess, threading, time
STEPS = 60000
LOCAL, DRV = '/content/dp_out', '@@DRIVE@@/dp_libero10_ckpts'
os.makedirs(DRV, exist_ok=True)
eps = json.load(open('@@DRIVE@@/dp_libero10_episodes.json'))['episodes']
steps_on_drive = sorted(int(d) for d in os.listdir(DRV) if d.isdigit())
if steps_on_drive and steps_on_drive[-1] >= STEPS:
    print('training already finished at step', steps_on_drive[-1])
else:
    base = ['lerobot-train', '--policy.type=diffusion', '--policy.push_to_hub=false', '--policy.crop_shape=[224,224]',
            '--policy.horizon=16', '--policy.n_action_steps=8', '--dataset.repo_id=lerobot/libero',
            '--dataset.episodes=[' + ','.join(map(str, eps)) + ']', '--batch_size=64', f'--steps={STEPS}',
            '--save_freq=5000', '--num_workers=8', '--wandb.enable=false', '--job_name=dp_libero10', f'--output_dir={LOCAL}']
    if steps_on_drive:
        last = steps_on_drive[-1]
        src = f'{DRV}/{last}'
        name = [d for d in os.listdir(src)][0] if len(os.listdir(src)) == 1 and not os.path.isdir(f'{src}/pretrained_model') else None
        shutil.rmtree(LOCAL, ignore_errors=True)
        ck = f'{LOCAL}/checkpoints'
        os.makedirs(ck, exist_ok=True)
        shutil.copytree(src if name is None else f'{src}/{name}', f'{ck}/{name or str(last)}')
        os.symlink(f'{ck}/{name or str(last)}', f'{ck}/last')
        cmd = ['lerobot-train', '--resume=true', f'--config_path={ck}/last/pretrained_model/train_config.json']
        print('resuming from step', last)
    else:
        shutil.rmtree(LOCAL, ignore_errors=True)
        cmd = base
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def mirror():
        while proc.poll() is None:
            time.sleep(120)
            for d in glob.glob(f'{LOCAL}/checkpoints/*'):
                n = os.path.basename(d)
                if n.isdigit() and not os.path.exists(f'{DRV}/{int(n)}') and os.path.exists(f'{d}/pretrained_model') and os.path.exists(f'{d}/training_state'):
                    time.sleep(30)
                    shutil.copytree(d, f'{DRV}/{int(n)}/{n}')
                    print('mirrored checkpoint', n, 'to Drive', flush=True)
    th = threading.Thread(target=mirror, daemon=True); th.start()
    for line in proc.stdout:
        if 'step' in line or 'Error' in line or 'error' in line:
            print(line.rstrip(), flush=True)
    proc.wait(); th.join(timeout=5)
    for d in glob.glob(f'{LOCAL}/checkpoints/*'):
        n = os.path.basename(d)
        if n.isdigit() and not os.path.exists(f'{DRV}/{int(n)}'):
            shutil.copytree(d, f'{DRV}/{int(n)}/{n}')
    assert proc.returncode == 0, 'training failed, paste the output above'
print('checkpoints on Drive:', sorted(int(d) for d in os.listdir(DRV) if d.isdigit()))
'''.replace("@@DRIVE@@", DRIVE_OUT)

DP_EVAL = r'''
# ---- evaluate the final Diffusion Policy checkpoint on the LIBERO-Plus subset ----
import glob, os, subprocess
DRV = '@@DRIVE@@/dp_libero10_ckpts'
last = max(int(d) for d in os.listdir(DRV) if d.isdigit())
pm = glob.glob(f'{DRV}/{last}/**/pretrained_model', recursive=True)
assert pm, f'no pretrained_model under {DRV}/{last}'
print('policy:', pm[0])
subprocess.run(['python', '/content/scripts/lerobot_tools.py', 'eval', '--subset', '@@SUBSET@@', '--out', '@@DRIVE@@/lerobot_dp',
                '--policy', pm[0], '--name', 'diffusion_policy'], check=True, env={**os.environ, 'MUJOCO_GL': 'egl'})
'''.replace("@@DRIVE@@", DRIVE_OUT).replace("@@SUBSET@@", SUBSET)

notebook([
    md("""
# LIBERO-Plus, family 4: Diffusion Policy (trained here, LeRobot)
**GPU: A100** (training). Steps: pick the LIBERO-10 episodes of `lerobot/libero`, train 60k steps (about 4 to 6 h; checkpoints
mirrored to Drive every 5k, so a disconnect resumes), then swap in LIBERO-Plus and evaluate on the shared subset (~1 h).
Diffusion Policy has no language input, so it is expected to be the weakest family; that is fine, it is the non-VLA baseline.
Paste back the episode-selection line, the final training lines, and the last table.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["plus_common.py", "lerobot_tools.py"]),
    code("%%bash\nset -e\n" + LEROBOT_INSTALL.replace("@@EXTRAS@@", "libero,diffusion")),
    code(f"""
# LIBERO-10 episodes (uses the stock LIBERO package, before the LIBERO-Plus swap)
import os
if not os.path.exists('{DRIVE_OUT}/dp_libero10_episodes.json'):
    !python /content/scripts/lerobot_tools.py select --out {DRIVE_OUT}/dp_libero10_episodes.json
else:
    print('episodes already selected')
"""),
    code(TRAIN),
    code("%%bash\nset -e\n" + SWAP),
    CONFIG_CELL, subset_cell("python"),
    code(DP_EVAL),
], "plus_4_diffusion_policy.ipynb")

# ------------------------------------------------------------------ 5. Octo
V5, PY5 = "/content/venv_octo", "/content/venv_octo/bin/python"
notebook([
    md("""
# LIBERO-Plus, family 5: Octo (diffusion head, JAX)
**GPU: L4** (A100 also fine). Uses the community LIBERO fine-tune `akuramshin/octo-libero-100-finetuned` (step 30000).
It has no model card, so a short calibration first tries both image orientations and both gripper conventions on 3 tasks
(~10 min) and keeps the best. If calibration prints a WARNING, stop and paste it. Full run ~210 episodes, about 1 to 1.5 h.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["plus_common.py", "eval_octo.py"] + SWEEP_SCRIPTS),
    code("%%bash\nset -e\n" + VENV.replace("@@V@@", V5) + f"""cd /content
[ -d octo ] || git clone -q https://github.com/octo-models/octo.git
uv pip install -q --python {PY5} -e /content/octo
uv pip install -q --python {PY5} -r /content/octo/requirements.txt
uv pip install -q --python {PY5} "jax[cuda11_pip]==0.4.20" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
""" + lp_block(f"uv pip install -q --python {PY5}", PY5) + f"""
uv pip install -q --python {PY5} "imageio[ffmpeg]" robosuite==1.4.1 bddl easydict cloudpickle gym huggingface_hub pillow
uv pip install -q --python {PY5} "numpy<2" "mujoco==2.3.7"
uv pip install -q --python {PY5} "protobuf==4.25.3"
uv pip install -q --python {PY5} --no-deps "tensorflow-metadata==1.16.1"
{PY5} -c "import jax; print('jax', jax.__version__, jax.devices())"
"""),
    CONFIG_CELL, subset_cell(PY5),
    code(f"# calibration (orientation x gripper convention)\n!MUJOCO_GL=egl {PY5} /content/scripts/eval_octo.py calib --subset {SUBSET} --out {DRIVE_OUT}/rows_octo.jsonl --calib_file {DRIVE_OUT}/octo_calib.json"),
    code(f"# full run (resumable)\n!MUJOCO_GL=egl {PY5} /content/scripts/eval_octo.py run --subset {SUBSET} --out {DRIVE_OUT}/rows_octo.jsonl --calib_file {DRIVE_OUT}/octo_calib.json"),
    code(f"# graded identifiability sweep, Octo episodes (3 base tasks x 24 conditions x raw/full, ~1.5 h on L4, resumable)\n!MUJOCO_GL=egl {PY5} /content/scripts/sweep_run.py episodes --family octo --octo_calib {DRIVE_OUT}/octo_calib.json --n_tasks 3 --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# wrapper on Octo, all 7 categories, paired raw/full (~3 h on L4, resumable)\n!MUJOCO_GL=egl {PY5} /content/scripts/wrap_run.py --family octo --octo_calib {DRIVE_OUT}/octo_calib.json --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full\n!{PY5} /content/scripts/invertibility_report.py --family octo:{WRAPS}/wrap_octo.jsonl:{WRAPS}/wrap_octo.jsonl --out {WRAPS}/report_octo.json"),
], "plus_5_octo.ipynb")

# ------------------------------------------------------------------ 6. SIMPLER
VS_O, PYS_O = "/content/venv_simpler_octo", "/content/venv_simpler_octo/bin/python"
VS_V, PYS_V = "/content/venv_simpler_vla", "/content/venv_simpler_vla/bin/python"
SIMPLER_BASE = """
uv pip install -q --python @@PY@@ numpy==1.24.4
uv pip install -q --python @@PY@@ -e /content/SimplerEnv-OpenVLA/ManiSkill2_real2sim
uv pip install -q --python @@PY@@ -e /content/SimplerEnv-OpenVLA
uv pip install -q --python @@PY@@ opencv-python-headless transforms3d matplotlib mediapy
"""
notebook([
    md("""
# SIMPLER (real-to-sim): Octo and OpenVLA, clean vs camera bump vs lighting
**GPU: L4** (needs Vulkan; A100 also works). WidowX visual-matching tasks, whose success tracks the real robot.
5 conditions x 4 tasks x 24 episodes per policy. Octo ~1 h, OpenVLA ~2 h.
The smoke cell checks Vulkan rendering first; if it fails, stop and paste the error.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["plus_common.py", "simpler_eval.py"]),
    code("""%%bash
set -e
apt-get -qq update > /dev/null
apt-get -qq install -y libvulkan1 vulkan-tools libglvnd-dev > /dev/null
mkdir -p /usr/share/vulkan/icd.d
cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.2.155"}}
EOF
cd /content
[ -d SimplerEnv-OpenVLA ] || git clone -q --recurse-submodules --depth 1 https://github.com/DelinQu/SimplerEnv-OpenVLA.git
pip -q install uv
[ -x /content/venv_simpler_octo/bin/python ] || uv venv -q /content/venv_simpler_octo -p 3.10
[ -x /content/venv_simpler_vla/bin/python ] || uv venv -q /content/venv_simpler_vla -p 3.10
""" + SIMPLER_BASE.replace("@@PY@@", PYS_O) + f"""
uv pip install -q --python {PYS_O} tensorflow==2.15.0 "git+https://github.com/octo-models/octo.git"
uv pip install -q --python {PYS_O} "jax[cuda11_pip]==0.4.20" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install -q --python {PYS_O} "protobuf==4.25.3"
uv pip install -q --python {PYS_O} --no-deps "tensorflow-metadata==1.16.1"
""" + SIMPLER_BASE.replace("@@PY@@", PYS_V) + f"""
uv pip install -q --python {PYS_V} torch==2.2.0 torchvision==0.17.0 transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 accelerate
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' /content/SimplerEnv-OpenVLA/simpler_env/policies/openvla/openvla_model.py || true
echo done
"""),
    code(f"""
# smoke: Vulkan rendering + env reset in both venvs
!{PYS_O} -c "import simpler_env; from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict as g; e = simpler_env.make('widowx_spoon_on_towel'); o, _ = e.reset(); print('octo venv render', g(e, o).shape)"
!{PYS_V} -c "import simpler_env; from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict as g; e = simpler_env.make('widowx_spoon_on_towel'); o, _ = e.reset(); print('openvla venv render', g(e, o).shape)"
"""),
    code(f"# Octo (octo-small): full grid, resumable\n!{PYS_O} /content/scripts/simpler_eval.py --policy octo-small --out {DRIVE_OUT}/simpler_octo.jsonl"),
    code(f"# OpenVLA: full grid, resumable\n!{PYS_V} /content/scripts/simpler_eval.py --policy openvla --out {DRIVE_OUT}/simpler_openvla.jsonl"),
], "simpler_octo_openvla.ipynb")

# ------------------------------------------------------------------ 7. DROID real data
notebook([
    md("""
# Real robot data without a robot: pi0.5-DROID on DROID-100
**GPU: L4** (A100 also fine). Public DROID sample (2 GB, no login) and the public pi0.5-DROID checkpoint.
E1 cross-camera disagreement vs real action error, E2 noise disagreement vs real error, E3 known camera and lighting damage on
real frames, E4 the pixel correction fitted only against the robot's own recorded motion and tested on later frames.
Setup ~20 min, E1/E2 ~15 min, E3/E4 ~1 to 1.5 h. Resumable. Paste back the final summary block.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["droid_export.py", "droid_pi05.py"]),
    code("""%%bash
cd /content
if [ ! -d /content/droid_100 ]; then
  gsutil -m -q cp -r gs://gresearch/robotics/droid_100 /content/ || echo "gsutil copy failed; the export will read directly from GCS"
fi
"""),
    code("!python /content/scripts/droid_export.py --episodes 40 --stride 5"),
    code("""%%bash
set -e
cd /content
[ -d openpi ] || git clone -q --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
pip -q install uv
cd /content/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync -q
GIT_LFS_SKIP_SMUDGE=1 uv pip install -q -e .
uv pip install -q --python /content/openpi/.venv/bin/python opencv-python-headless
/content/openpi/.venv/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
"""),
    code(f"# smoke: one real frame through pi0.5-DROID (downloads the checkpoint)\n!cd /content/openpi && .venv/bin/python /content/scripts/droid_pi05.py --out {DRIVE_OUT}/droid_results.json --smoke"),
    code(f"# E1-E4 (resumable)\n!cd /content/openpi && .venv/bin/python /content/scripts/droid_pi05.py --out {DRIVE_OUT}/droid_results.json"),
], "droid_real_signal.ipynb")

# ------------------------------------------------------------------ 6. wrapper on OpenVLA-OFT
WRAP_OUT = f"{DRIVE_OUT}/wrapper_oft"
notebook([
    md("""
# Canonicalization wrapper on OpenVLA-OFT: camera and robot initial state
**GPU: A100.** Frozen OFT, no retraining. Two channels, both using only what a deployment knows:
the **state** channel drives the arm back to its nominal start pose; the **camera** channel calibrates the real camera
pose from the robot's own silhouette after a short arm sweep, then reprojects the RGB-D image into the nominal view.

1. Setup (~20 min, same install as plus_1).
2. **Validate** (~5 min): calibration error vs ground truth and reprojection quality on 3 camera tasks, plus
   side-by-side PNGs on Drive. Paste this output before running the rest.
3. **Calibration check** (~30 min, no policy): old vs new camera calibration. Paste the table.
4. **Run** (~4 h, resumable): robot_init tasks (state, full2), camera tasks (true-pose and oracle upper bounds,
   full2), light tasks (full2, no-harm control). The nominal camera is read per scene (LIBERO-10 spans several
   rooms). Rows made with the old single nominal are backed up and re-queued automatically. Paste the final table.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["plus_common.py", "wrapper_libero.py", "eval_oft_wrapper.py", "calib_check.py",
                                                  "canon_net.py", "collect_canon.py", "train_canon.py",
                                                  "invertibility_report.py", "identifiability.py", "sweep_conditions.py",
                                                  "policy_adapters.py", "sweep_run.py", "sweep_analysis.py",
                                                  "wrap_run.py", "train_seg.py", "camera_pooled.py", "diag_learned.py", "gate_report.py", "smc.py", "smc_probe.py", "arms_report.py", "attribution_ladder.py", "tolerance_probe.py",
                                                  "depth_server.py", "render_paper_views.py", "pose_sources.py"]),
    code("%%bash\nset -e\n" + VENV.replace("@@V@@", V1) + f"""cd /content
[ -d openvla-oft ] || git clone -q https://github.com/moojink/openvla-oft.git
uv pip install -q --python {PY1} -e /content/openvla-oft
""" + lp_block(f"uv pip install -q --python {PY1}", PY1) + f"""
uv pip install -q --python {PY1} -r /content/openvla-oft/experiments/robot/libero/libero_requirements.txt
uv pip install -q --python {PY1} "numpy<2" "mujoco==2.3.7"
uv pip install -q --python {PY1} "protobuf==4.25.3"
uv pip install -q --python {PY1} --no-deps "tensorflow-metadata==1.16.1"
{PY1} -c "import torch, robosuite, mujoco, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mujoco', mujoco.__version__)"
"""),
    CONFIG_CELL, subset_cell(PY1),
    code(f"# validate the camera channel on 3 camera tasks (paste this output)\n!mkdir -p {WRAP_OUT} && cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/eval_oft_wrapper.py validate --subset {SUBSET} --out {WRAP_OUT}/rows.jsonl --n 3"),
    code(f"# calibration-only check, no policy (~1 to 2 min per task): old vs new camera calibration (paste the table)\n!MUJOCO_GL=egl {PY1} /content/scripts/calib_check.py --subset {SUBSET} --out {WRAP_OUT}/calib_check.jsonl"),
    code(f"""%%bash
# claim 3, step 1: self-labelled pairs for the learned canonicalizer, 4 parallel workers (~1 h, resumable, no policy)
mkdir -p {WRAP_OUT}/canon_data_v2
# v2 shards also store the true nominal render (training-time target only); older canon_data shards lack it
# build the per-scene nominal cache ONCE before the workers start (four workers rebuilding it at once would race on the file)
MUJOCO_GL=egl PYTHONPATH=/content/LIBERO-plus MPLBACKEND=Agg {PY1} -c "import sys; sys.path.insert(0, '/content/scripts'); import wrapper_libero as W, calib_check as C; from libero.libero import benchmark; s = benchmark.get_benchmark_dict()['libero_10'](); W.read_nominals(s, lambda i: C.make_env(s, i), cache='{WRAP_OUT}/nominals_v3.json'); print('nominal cache ok')"
for i in 0 1 2 3; do
  MUJOCO_GL=egl PYTHONPATH=/content/LIBERO-plus MPLBACKEND=Agg {PY1} /content/scripts/collect_canon.py --subset {SUBSET} --nominals {WRAP_OUT}/nominals_v3.json --out_dir {WRAP_OUT}/canon_data_v2 --worker $i --workers 4 > /content/collect_$i.log 2>&1 &
  sleep 60
done
wait
tail -n 3 /content/collect_*.log
ls {WRAP_OUT}/canon_data_v2 | wc -l
"""),
    code(f"# claim 3, step 2: train the learned canonicalizer: input = the wrapper's own reprojection + hole mask + frame (measured geometry), target = the oracle nominal render (privileged at training time only; ~30 min on A100). Add --input rgb for the learning-alone ablation\n# needs canon_data_v2 shards: older canon_data shards lack the oracle render and the trainer refuses them\n!{PY1} /content/scripts/train_canon.py --data_dir {WRAP_OUT}/canon_data_v2 --out {WRAP_OUT}/canon_net_oracle.pt --target oracle --input reproj"),
    code(f"# no simulator shortcuts, step 1: train the robot segmenter on the same held-out shards (~15 min; skip if seg_net.pt already exists)\n!{PY1} /content/scripts/train_seg.py --data_dir {WRAP_OUT}/canon_data_v2 --out {SEG_CKPT}"),
    code(f"# wrapper run in priority order (resumable): camera full (wider calibration + reference fill), camera cam_true_ref, learned (if trained), light, noise, layout, background, language; finished robot_init / oracle / cam_true rows are kept\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/eval_oft_wrapper.py run --subset {SUBSET} --out {WRAP_OUT}/rows.jsonl --baseline {DRIVE_OUT}/rows_oft.jsonl --canon_ckpt {WRAP_OUT}/canon_net_oracle.pt"),
    code(f"""# camera diagnosis (standalone: works right after a restart, needs only Drive): true shift, calibration error, coverage, per-arm success
from google.colab import drive; drive.mount('/content/drive')
import json, collections
P = '{DRIVE_OUT}'
rows = [json.loads(l) for l in open(f'{{P}}/wrapper_oft/rows.jsonl') if l.strip()]
base = {{json.loads(l)['task_id']: json.loads(l)['success'] for l in open(f'{{P}}/rows_oft.jsonl') if l.strip()}}
by = collections.defaultdict(dict)
for r in rows:
    if r['category'] == 'camera':
        by[r['task_id']][r['arm']] = r
nan2 = [float('nan')] * 2
for t, a in sorted(by.items()):
    f = a.get('full') or a.get('full2') or {{}}
    s = f.get('cam_true_shift_cm_deg') or f.get('cam_true_err_cm_deg') or nan2
    e = f.get('cam_est_err_cm_deg') or nan2
    cov = a.get('cam_true', {{}}).get('reproj_coverage', float('nan'))
    ok = lambda arm: int(a.get(arm, {{}}).get('success', 0)) if arm in a else '-'
    print(t, (f.get('name') or '')[:26], '| shift %.0fcm %.0fdeg' % tuple(s[:2]), '| est err %.0fcm %.0fdeg' % tuple(e[:2]),
          '| cov %.2f' % cov, '| base', int(base.get(t, 0)), 'oracle', ok('oracle'), 'true', ok('cam_true'),
          'true_ref', ok('cam_true_ref'), 'full', int(f.get('success', 0)) if f else '-')
"""),
    code(f"# camera significance: 30 MORE camera tasks outside every subset, paired raw vs full (~2.5 h on A100, resumable)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full --categories camera --per_cat_scale 0 --extra_from_suite 30"),
    code(f"# pooled paired test for the deployable camera fix (paste this)\n!{PY1} /content/scripts/camera_pooled.py --baseline {DRIVE_OUT}/rows_oft.jsonl --wrapper {WRAP_OUT}/rows.jsonl --ext {WRAPS}/wrap_openvla_oft.jsonl"),
    code(f"# self-identifiability, policy-free: local sensitivity per scene (~10 min), then per-task index (~45 min, resumable)\n!MUJOCO_GL=egl {PY1} /content/scripts/identifiability.py local --subset {SUBSET} --nominals {WRAP_OUT}/nominals_v3.json --out {WRAP_OUT}/identifiability_local.json\n!MUJOCO_GL=egl {PY1} /content/scripts/identifiability.py tasks --subset {SUBSET} --nominals {WRAP_OUT}/nominals_v3.json --out {WRAP_OUT}/identifiability_tasks.jsonl --wrapper {WRAP_OUT}/rows.jsonl --baseline {DRIVE_OUT}/rows_oft.jsonl"),
    code(f"# no simulator shortcuts, step 2: OFT with LEARNED sensing (segmenter + Depth Anything scaled on the robot), camera / robot_init / light / noise (~5 h on A100, resumable)\n# OFT pins its own transformers fork (4.40): do not upgrade it; Depth Anything V1 loads there (V2 falls back automatically)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms full_learned --categories camera,robot_init,light,sensor_noise --seg_ckpt {SEG_CKPT}"),
    code(f"""%%bash
# measurement plus learning, step 1: shards with LEARNED sensing (segmenter + table plane), so the canonicalizer learns to repair the deployable sensing's artefacts (~1 h, 4 workers, resumable)
D={DRIVE_OUT}
mkdir -p $D/wrapper_oft/canon_data_v3_plane
for i in 0 1 2 3; do
  MUJOCO_GL=egl PYTHONPATH=/content/LIBERO-plus MPLBACKEND=Agg {PY1} /content/scripts/collect_canon.py --subset {SUBSET} --nominals {NOMINALS} --out_dir $D/wrapper_oft/canon_data_v3_plane --seg_ckpt {SEG_CKPT} --depth plane --worker $i --workers 4 > /content/collect_plane_$i.log 2>&1 &
  sleep 60
done
wait
tail -n 3 /content/collect_plane_*.log
ls $D/wrapper_oft/canon_data_v3_plane | wc -l
"""),
    code(f"# measurement plus learning, step 2: train on the learned-sensing shards, oracle target (~30 min)\n!{PY1} /content/scripts/train_canon.py --data_dir {WRAP_OUT}/canon_data_v3_plane --out {WRAP_OUT}/canon_net_plane.pt --target oracle --input reproj"),
    code(f"# measurement plus learning, step 3: the learned arm with LEARNED sensing (segmenter + plane depth + canonicalizer), camera tasks (~1 h)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/eval_oft_wrapper.py run --subset {SUBSET} --out {WRAP_OUT}/rows.jsonl --baseline {DRIVE_OUT}/rows_oft.jsonl --canon_ckpt {WRAP_OUT}/canon_net_plane.pt --seg_ckpt {SEG_CKPT} --depth plane --skip_arms full,cam_true_ref,oracle,cam_true,state"),
    code(f"# no simulator shortcuts, step 3: OFT with the learned segmenter and NO depth network (training-setup table plane), camera tasks (~1.5 h on A100, resumable)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms full_plane --categories camera --seg_ckpt {SEG_CKPT}"),
    code(f"# learned sensing root cause, step 1 (DIAGNOSTIC, ~1 h on A100, resumable): the six tasks simulator sensing solves and learned sensing loses, every arm in the SAME harness. full = simulator sensing control; *_truepose = same sensing with the simulator's camera pose (upper bounds, never reported as deployable)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --categories camera --seg_ckpt {SEG_CKPT} --tasks 768,800,803,956,958,1060 --redo --arms full,full_plane,full_plane_truepose,full_hybrid,full_hybrid_truepose"),
    code(f"# learned sensing root cause, step 2: the table (paste this)\n!{PY1} /content/scripts/diag_learned.py --rows {WRAPS}/wrap_openvla_oft.jsonl --tasks 768,800,803,956,958,1060"),
    code(f"# learned sensing root cause, step 3 (only after step 2 is read): hybrid depth on all 30 camera tasks (~1.5 h on A100, resumable)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms full_hybrid --categories camera --seg_ckpt {SEG_CKPT}"),
    code(f"# graded sweep, step 1: policy-free recoverability scores, scores_v2 (own-block reference + coverage / shading / denoiser components; 3 base tasks x 24 conditions + 21 reference instances, ~45 min, no policy). The v1 scores.jsonl normalised every family by the camera block and is kept only for the record\n!MUJOCO_GL=egl {PY1} /content/scripts/sweep_run.py score --n_tasks 3 --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# graded sweep, step 2: OpenVLA-OFT episodes (3 base tasks x 24 conditions x raw/full, ~4 h on A100, resumable)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/sweep_run.py episodes --family oft --n_tasks 3 --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# four suites, libero_spatial: OFT checkpoint for this suite, 20 robot_init + 20 camera tasks, paired raw / full (~3 h on A100, resumable). The subset and the nominal camera for the suite are built on first use\n!cd /content/openvla-oft && LIBERO_SUITE=libero_spatial MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_spatial.json --out_dir {DRIVE_OUT}/wraps_libero_spatial --nominals {WRAP_OUT}/nominals_libero_spatial.json --arms raw,full --categories robot_init,camera --per_cat_scale 0.67\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_spatial/wrap_openvla_oft.jsonl --arms raw,full"),
    code(f"# four suites, libero_object: OFT checkpoint for this suite, 20 robot_init + 20 camera tasks, paired raw / full (~3.5 h on A100, resumable). The subset and the nominal camera for the suite are built on first use\n!cd /content/openvla-oft && LIBERO_SUITE=libero_object MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_object.json --out_dir {DRIVE_OUT}/wraps_libero_object --nominals {WRAP_OUT}/nominals_libero_object.json --arms raw,full --categories robot_init,camera --per_cat_scale 0.67\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_object/wrap_openvla_oft.jsonl --arms raw,full"),
    code(f"# four suites, libero_goal: OFT checkpoint for this suite, 20 robot_init + 20 camera tasks, paired raw / full (~3.5 h on A100, resumable). The subset and the nominal camera for the suite are built on first use\n!cd /content/openvla-oft && LIBERO_SUITE=libero_goal MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_goal.json --out_dir {DRIVE_OUT}/wraps_libero_goal --nominals {WRAP_OUT}/nominals_libero_goal.json --arms raw,full --categories robot_init,camera --per_cat_scale 0.67\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_goal/wrap_openvla_oft.jsonl --arms raw,full"),
    code(f"# robot_init base rate (reviewer: is the joint result just a reset?), ~1.5 h on A100 for all three suites, resumable. nominal_start = the arm PLACED at the training start pose (no servo, no correction), then the raw policy, on the SAME 20 robot_init tasks and seeds per suite whose raw / full rows exist. Paste the three reports\n" + "\n".join(f"!cd /content/openvla-oft && LIBERO_SUITE=libero_{s} MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_{s}.json --out_dir {DRIVE_OUT}/wraps_libero_{s} --nominals {WRAP_OUT}/nominals_libero_{s}.json --arms nominal_start --categories robot_init --per_cat_scale 0.67\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_{s}/wrap_openvla_oft.jsonl --arms raw,full,nominal_start" for s in ("spatial", "object", "goal"))),
    code(f"# camera extension, libero_spatial: 40 MORE camera tasks from the full suite, paired raw / full (~2 h on A100, resumable). These checkpoints survive most camera shifts, so the camera claim needs more failures to recover; the report pools them with the first 20\n!cd /content/openvla-oft && LIBERO_SUITE=libero_spatial MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_spatial.json --out_dir {DRIVE_OUT}/wraps_libero_spatial --nominals {WRAP_OUT}/nominals_libero_spatial.json --arms raw,full --categories camera --per_cat_scale 0 --extra_from_suite 40\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_spatial/wrap_openvla_oft.jsonl --arms raw,full"),
    code(f"# camera extension, libero_object: 40 MORE camera tasks from the full suite, paired raw / full (~2 h on A100, resumable). These checkpoints survive most camera shifts, so the camera claim needs more failures to recover; the report pools them with the first 20\n!cd /content/openvla-oft && LIBERO_SUITE=libero_object MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_object.json --out_dir {DRIVE_OUT}/wraps_libero_object --nominals {WRAP_OUT}/nominals_libero_object.json --arms raw,full --categories camera --per_cat_scale 0 --extra_from_suite 40\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_object/wrap_openvla_oft.jsonl --arms raw,full"),
    code(f"# camera extension, libero_goal: 40 MORE camera tasks from the full suite, paired raw / full (~2 h on A100, resumable). These checkpoints survive most camera shifts, so the camera claim needs more failures to recover; the report pools them with the first 20\n!cd /content/openvla-oft && LIBERO_SUITE=libero_goal MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_goal.json --out_dir {DRIVE_OUT}/wraps_libero_goal --nominals {WRAP_OUT}/nominals_libero_goal.json --arms raw,full --categories camera --per_cat_scale 0 --extra_from_suite 40\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_goal/wrap_openvla_oft.jsonl --arms raw,full"),
    code(f"# tolerance probe, OFT (~25 min, resumable, NO rollouts): how wrong may the camera pose be before the policy's opening action leaves what it does under a perfect calibration? True pose plus a known residual (0.5 to 16 cm, 0.5 to 8 deg per camera axis) on 10 base tasks x 2 orbit conditions; prints the dose-response table and where the robot's own calibration error sits on it (paste the report). Own folder\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/tolerance_probe.py run --family oft --n_tasks 10 --out_dir {DRIVE_OUT}/tolerance --nominals {NOMINALS}\n!{PY1} /content/scripts/tolerance_probe.py report --out_dir {DRIVE_OUT}/tolerance"),
    code(f"# ladder 1, libero_10 (~35 min, resumable): the two UPPER RUNGS of the attribution ladder, only on camera tasks that fail under both raw and full. full_truepose = the same correction with the simulator's true camera pose (was it the measurement?), full_oracle = a perfect render from the training camera (was it the lost view? what still fails here is the policy). Writes to its OWN folder\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_ladder_libero_10 --nominals {NOMINALS} --arms full_truepose,full_oracle --categories camera --per_cat_scale 0 --extra_from_suite 30 --only_failed_in {WRAPS}/wrap_openvla_oft.jsonl"),
    code(f"# ladder 2, libero_object (~35 min, resumable): upper rungs on the object camera tasks that fail under both raw and full\n!cd /content/openvla-oft && LIBERO_SUITE=libero_object MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_object.json --out_dir {DRIVE_OUT}/wraps_ladder_libero_object --nominals {WRAP_OUT}/nominals_libero_object.json --arms full_truepose,full_oracle --categories camera --per_cat_scale 0.67 --extra_from_suite 40 --only_failed_in {DRIVE_OUT}/wraps_libero_object/wrap_openvla_oft.jsonl"),
    code(f"# ladder 3, libero_goal (~30 min, resumable): upper rungs on the goal camera tasks that fail under both raw and full\n!cd /content/openvla-oft && LIBERO_SUITE=libero_goal MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_goal.json --out_dir {DRIVE_OUT}/wraps_ladder_libero_goal --nominals {WRAP_OUT}/nominals_libero_goal.json --arms full_truepose,full_oracle --categories camera --per_cat_scale 0.67 --extra_from_suite 40 --only_failed_in {DRIVE_OUT}/wraps_libero_goal/wrap_openvla_oft.jsonl"),
    code(f"# ladder 4, libero_spatial (~15 min, resumable): upper rungs on the spatial camera tasks that fail under both raw and full\n!cd /content/openvla-oft && LIBERO_SUITE=libero_spatial MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_spatial.json --out_dir {DRIVE_OUT}/wraps_ladder_libero_spatial --nominals {WRAP_OUT}/nominals_libero_spatial.json --arms full_truepose,full_oracle --categories camera --per_cat_scale 0.67 --extra_from_suite 40 --only_failed_in {DRIVE_OUT}/wraps_libero_spatial/wrap_openvla_oft.jsonl"),
    code(f"# ladder: report (paste all of it). Attributes every raw camera failure to the first rung that repairs it\n!{PY1} /content/scripts/attribution_ladder.py --set libero_10_first30 {WRAP_OUT}/rows.jsonl {DRIVE_OUT}/rows_oft.jsonl --set libero_10_extra30 {WRAPS}/wrap_openvla_oft.jsonl {DRIVE_OUT}/wraps_ladder_libero_10/wrap_openvla_oft.jsonl --set object {DRIVE_OUT}/wraps_libero_object/wrap_openvla_oft.jsonl {DRIVE_OUT}/wraps_ladder_libero_object/wrap_openvla_oft.jsonl --set goal {DRIVE_OUT}/wraps_libero_goal/wrap_openvla_oft.jsonl {DRIVE_OUT}/wraps_ladder_libero_goal/wrap_openvla_oft.jsonl --set spatial {DRIVE_OUT}/wraps_libero_spatial/wrap_openvla_oft.jsonl {DRIVE_OUT}/wraps_ladder_libero_spatial/wrap_openvla_oft.jsonl"),
    code(f"# camera power 1, libero_10 (~2.5 h, resumable): three NEW arms on the 30 extra camera tasks whose raw / full / tta rows exist: full_settle (hand the arm over at rest, where the episode began), full_verify (full_settle + measure again with 7 poses when the robot's own self-check rejects the calibration), full_tta (the wrapper with the policy averaged over jitters of the CORRECTED frame)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full,tta,full_settle,full_verify,full_tta --categories camera --per_cat_scale 0 --extra_from_suite 30\n!{PY1} /content/scripts/arms_report.py --rows {WRAPS}/wrap_openvla_oft.jsonl --arms raw,full,tta,full_settle,full_verify,full_tta"),
    code(f"# camera power 2, libero_object (~6 h, resumable): tta and the three new arms on all 60 object camera tasks (raw / full exist), so the wrapper is compared with test-time augmentation on a second suite\n!cd /content/openvla-oft && LIBERO_SUITE=libero_object MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_object.json --out_dir {DRIVE_OUT}/wraps_libero_object --nominals {WRAP_OUT}/nominals_libero_object.json --arms raw,full,tta,full_settle,full_verify,full_tta --categories camera --per_cat_scale 0.67 --extra_from_suite 40\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_object/wrap_openvla_oft.jsonl --arms raw,full,tta,full_settle,full_verify,full_tta"),
    code(f"# camera power 3, libero_goal (~5 h, resumable): 40 more goal camera tasks, all arms\n!cd /content/openvla-oft && LIBERO_SUITE=libero_goal MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_goal.json --out_dir {DRIVE_OUT}/wraps_libero_goal --nominals {WRAP_OUT}/nominals_libero_goal.json --arms raw,full,tta,full_verify --categories camera --per_cat_scale 0.67 --extra_from_suite 40\n!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_libero_goal/wrap_openvla_oft.jsonl --arms raw,full,tta,full_verify"),
    code(f"# baselines, camera (OFT, libero_10): test-time augmentation, generic corrections, a depth-free homography with the measured pose, and the wrapper without the reference fill, on the 30 extra camera tasks whose raw / full rows already exist in this harness (~7 h on A100, resumable)\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,full,tta,generic,full_homog,full_nofill --categories camera --per_cat_scale 0 --extra_from_suite 30"),
    code(f"# baselines, generic corrections where OFT is already robust (light / sensor_noise / background, 30 tasks, raw vs generic, ~3 h): does an unmeasured correction hurt a policy that was coping?\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {WRAPS} --nominals {NOMINALS} --arms raw,generic --categories light,sensor_noise,background"),
    code(f"# baselines: report (paste all of it)\n!{PY1} /content/scripts/arms_report.py --rows {WRAPS}/wrap_openvla_oft.jsonl --arms raw,full,tta,generic,full_homog,full_nofill"),
    code(f"# self-motion consistency, OFT probe (NO rollouts: ~1 min per unit, 72 units, ~1.5 h on A100, resumable). At 7 known end-effector positions the frozen policy is asked for its action on the raw and on the corrected frame; rays that stop meeting at one sub-goal mean the policy is not perceiving correctly\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/smc_probe.py run --family oft --n_tasks 3 --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# self-motion consistency: report (joins the probe with the raw / full outcomes already on Drive; paste all of it)\n!{PY1} /content/scripts/smc_probe.py report --out_dir {SWEEP_OUT}"),
    code(f"# scale-up, step 1: policy-free recoverability scores for all 10 base tasks (7 new, ~2 h, no policy, resumable)\n!MUJOCO_GL=egl {PY1} /content/scripts/sweep_run.py score --n_tasks 10 --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# scale-up, step 2: OFT on all 10 base tasks with the gated arm (7 new tasks x 24 conditions x raw/full + gated on all 10, ~11 h on A100, resumable). OFT is the DEVELOPMENT family: its gate numbers are descriptive, the test is OpenVLA\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/sweep_run.py episodes --family oft --n_tasks 10 --arms raw,full,gated --out_dir {SWEEP_OUT} --nominals {NOMINALS}"),
    code(f"# do-no-harm gate on OFT: report (paste all of it)\n!{PY1} /content/scripts/gate_report.py --rows {SWEEP_OUT}/episodes_openvla_oft.jsonl"),
    code(f"# graded sweep, step 3: does the score predict recovery? (reads every family's episodes on Drive; paste this)\n!{PY1} /content/scripts/sweep_analysis.py --out_dir {SWEEP_OUT}"),
    code(f"# claims 1 and 2: invertibility report (paste this)\n!{PY1} /content/scripts/invertibility_report.py --family openvla_oft:{DRIVE_OUT}/rows_oft.jsonl:{WRAP_OUT}/rows.jsonl --out {WRAP_OUT}/invertibility_report.json"),
    # ---- POST HOC 2026-09-24 (run file 19): the unperturbed base rate on the exact tasks and seeds of the joint table and the ladder sets
    code(f"# nominal base rate 1, robot_init (~2.5 h on A100, resumable). nominal_start = the arm PLACED at the training start (no servo, nothing measured), then the raw policy: the unperturbed task on the SAME tasks and seeds. libero_10: raw, full AND nominal_start in this harness, because the Table 2 libero_10 row came from the older eval_oft_wrapper harness and single rollouts are harness-sensitive. spatial / object / goal: nominal_start only, paired with the raw / full rows of the four-suites cells (same harness, seed = task id). Own folders plus/wraps_nominal_<suite>\n"
         f"!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_nominal_libero_10 --nominals {NOMINALS} --arms raw,full,nominal_start --categories robot_init\n"
         + "\n".join(f"!cd /content/openvla-oft && LIBERO_SUITE=libero_{s} MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_{s}.json --out_dir {DRIVE_OUT}/wraps_nominal_libero_{s} --nominals {WRAP_OUT}/nominals_libero_{s}.json --arms nominal_start --categories robot_init --per_cat_scale 0.67" for s in ("spatial", "object", "goal"))),
    code(f"# nominal base rate 2, ladder sets (~1.5 h on A100, resumable). nominal_view = every frame is the simulator's render from the TRAINING camera, nothing measured, the arm untouched: the unperturbed task at the same layout, start state and seed. Set A (first 30 libero_10 camera tasks) in full; set B and the other suites only on the ladder units (fail under both raw and full), with EXACTLY the ladder cells' task selection. Same own folders as cell 1\n"
         f"!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_nominal_libero_10 --nominals {NOMINALS} --arms nominal_view --categories camera\n"
         f"!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_nominal_libero_10 --nominals {NOMINALS} --arms nominal_view --categories camera --per_cat_scale 0 --extra_from_suite 30 --only_failed_in {WRAPS}/wrap_openvla_oft.jsonl\n"
         + "\n".join(f"!cd /content/openvla-oft && LIBERO_SUITE=libero_{s} MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {DRIVE_OUT}/subset_libero_{s}.json --out_dir {DRIVE_OUT}/wraps_nominal_libero_{s} --nominals {WRAP_OUT}/nominals_libero_{s}.json --arms nominal_view --categories camera --per_cat_scale 0.67 --extra_from_suite 40 --only_failed_in {DRIVE_OUT}/wraps_libero_{s}/wrap_openvla_oft.jsonl" for s in ("object", "goal", "spatial"))),
    code(f"# nominal base rate 3: report (paste all of it). robot_init: nominal_start against raw and full, same tasks and seeds (McNemar vs nominal_start)\n"
         f"!{PY1} /content/scripts/arms_report.py --rows {DRIVE_OUT}/wraps_nominal_libero_10/wrap_openvla_oft.jsonl --arms raw,full,nominal_start --ref nominal_start\n"
         + "\n".join(f"!cat {DRIVE_OUT}/wraps_libero_{s}/wrap_openvla_oft.jsonl {DRIVE_OUT}/wraps_nominal_libero_{s}/wrap_openvla_oft.jsonl > /content/nominal_{s}.jsonl\n"
                     f"!echo libero_{s}; {PY1} /content/scripts/arms_report.py --rows /content/nominal_{s}.jsonl --arms raw,full,nominal_start --ref nominal_start" for s in ("spatial", "object", "goal"))),
    code("# nominal base rate 4: the ladder sets, unperturbed view against the perfect-render rung (seconds; paste all of it)\n"
         "import json, os\nD = '@D@'\nimport glob; rows = lambda p: [json.loads(l) for q in [p] + sorted(glob.glob(p[:-6] + '.*.jsonl')) if os.path.exists(q) for l in open(q) if l.strip()]\n"
         "for s in ('libero_10', 'libero_object', 'libero_goal', 'libero_spatial'):\n"
         "    nv = {r['task_id']: bool(r['success']) for r in rows(f'{D}/wraps_nominal_{s}/wrap_openvla_oft.jsonl') if r['arm'] == 'nominal_view'}\n"
         "    orc = {r['task_id']: bool(r['success']) for r in rows(f'{D}/wraps_ladder_{s}/wrap_openvla_oft.jsonl') if r['arm'] == 'full_oracle'}\n"
         "    if s == 'libero_10':   # set A's perfect-render rung is in the older harness (cross-harness pairing, said in the paper if used)\n"
         "        orc.update({r['task_id']: bool(r['success']) for r in rows(f'{D}/wrapper_oft/rows.jsonl') if r.get('arm') == 'oracle' and r['category'] == 'camera'})\n"
         "    b = sorted(set(nv) & set(orc))\n"
         "    print(s, '| nominal_view', sum(nv.values()), 'of', len(nv), '| paired with the perfect render:', len(b), 'units: both', sum(nv[t] and orc[t] for t in b),\n"
         "          'unperturbed only', sum(nv[t] and not orc[t] for t in b), 'render only', sum(orc[t] and not nv[t] for t in b), 'neither', sum(not (nv[t] or orc[t]) for t in b))\n".replace("@D@", DRIVE_OUT)),
    # ---- run file 20: policy-free 3D renders for the paper figures (no policy, no rollouts)
    code(f"# 3D renders 1, smoke (~3 min, no policy): one job at 256 px into /content only. It must print 'anchor check' >= 0.90; paste the line and STOP on an error\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/render_paper_views.py --smoke --nominals {NOMINALS} --out_dir /content/render_smoke"),
    code(f"# 3D renders 2 (~15 min, no policy, 512 px): god's-eye scene with the training camera and the four Fig. 1 cameras as frusta, the joint-offset start against the training start, the 7 fingerprint probe poses. Own folder plus/paper_renders; paste the three 'wrote' lines\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/render_paper_views.py --nominals {NOMINALS} --out_dir {DRIVE_OUT}/paper_renders\n!ls -la {DRIVE_OUT}/paper_renders"),
    # ---- run file 22 (POST HOC 2026-09-24): learned masks + a monocular METRIC depth network on set A
    code(f"# metric depth 1, smoke (~10 min): full_metric on 2 set A tasks, own folder. It must print 'learned sensing: segmenter ... metric depth ...' and two episode lines; paste them and STOP on an error\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_metric_smoke --nominals {NOMINALS} --arms full_metric --categories camera --seg_ckpt {SEG_CKPT} --tasks 699,707"),
    code(f"# metric depth 2 (~1.5 h on A100, resumable): full_metric (learned segmenter + Depth Anything V2 metric, used as is) on all 30 set A camera tasks. Same harness and seeds as full_plane / full_hybrid in plus/wraps; own folder plus/wraps_metric\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_metric --nominals {NOMINALS} --arms full_metric --categories camera --seg_ckpt {SEG_CKPT}"),
    code("# metric depth 3: report (seconds; paste all of it). Calibration error against the simulator's pose and success, per learned-sensing arm on the same 30 tasks\n"
         "import json, os, numpy as np\nD = '@D@'\nimport glob; rows = lambda p: [json.loads(l) for q in [p] + sorted(glob.glob(p[:-6] + '.*.jsonl')) if os.path.exists(q) for l in open(q) if l.strip()]\n"
         "R = rows(f'{D}/wraps_metric/wrap_openvla_oft.jsonl') + [r for r in rows(f'{D}/wraps/wrap_openvla_oft.jsonl') if r['arm'] in ('full_plane', 'full_hybrid')]\n"
         "ids = {r['task_id'] for r in R if r['arm'] == 'full_metric' and r['category'] == 'camera'}\n"
         "for arm in ('full_plane', 'full_hybrid', 'full_metric'):\n"
         "    a = {r['task_id']: r for r in R if r['arm'] == arm and r['task_id'] in ids}\n"
         "    e = np.array([a[t]['cam_est_err_cm_deg'][:2] for t in a if a[t].get('cam_est_err_cm_deg')], float).reshape(-1, 2)\n"
         "    print(arm, 'n', len(a), 'success', sum(bool(r['success']) for r in a.values()), '| calib err median cm %.2f deg %.2f' % tuple(np.median(e, 0)) if len(e) else '', '| within 1 cm', int((e[:, 0] <= 1.0).sum()) if len(e) else 0)\n"
         "m = {r['task_id']: bool(r['success']) for r in R if r['arm'] == 'full_metric'}; h = {r['task_id']: bool(r['success']) for r in R if r['arm'] == 'full_hybrid'}\n"
         "b = sorted(set(m) & set(h)); print('metric vs hybrid, paired:', len(b), 'tasks, metric only', sum(m[t] and not h[t] for t in b), 'hybrid only', sum(h[t] and not m[t] for t in b))\n".replace("@D@", DRIVE_OUT)),
    # ---- run file 25 (2026-09-25, tests pre-registered in wrapper/prereg/2026-09-25_refine2.md): LEARNED sensing + refined pose
    code(f"# learned refine 1, smoke (~10 min): full_hybrid_refine2 (learned segmenter, table-plane depth, the pose refined EasyHeC-style from two starts on the LEARNED masks) on 2 set A tasks, own smoke folder. Paste the two episode lines and STOP on an error\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_refine2_smoke --nominals {NOMINALS} --arms full_hybrid_refine2 --categories camera --seg_ckpt {SEG_CKPT} --tasks 699,707"),
    code(f"# learned refine 2 (~2.5 h on A100, resumable): raw and full_hybrid_refine2 on all 30 set A camera tasks, same harness and seeds as full_hybrid in plus/wraps. Own folder plus/wraps_refine2\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_refine2 --nominals {NOMINALS} --arms raw,full_hybrid_refine2 --categories camera --seg_ckpt {SEG_CKPT}"),
    code("# learned refine 3: report (seconds; paste all of it). Success and calibration error per arm on the same 30 tasks, paired against raw and against full_hybrid\n"
         "import json, os, numpy as np\nfrom math import comb\nD = '@D@'\nimport glob; rows = lambda p: [json.loads(l) for q in [p] + sorted(glob.glob(p[:-6] + '.*.jsonl')) if os.path.exists(q) for l in open(q) if l.strip()]\n"
         "R = rows(f'{D}/wraps_refine2/wrap_openvla_oft.jsonl') + [r for r in rows(f'{D}/wraps/wrap_openvla_oft.jsonl') if r['arm'] in ('full_plane', 'full_hybrid')]\n"
         "ids = {r['task_id'] for r in R if r['arm'] == 'full_hybrid_refine2'}\nby = {}\n"
         "for r in R:\n    if r['task_id'] in ids and r.get('category') == 'camera' and r.get('rep', 0) == 0: by.setdefault(r['task_id'], {}).setdefault(r['arm'], r)\n"
         "mc = lambda b, c: min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0\n"
         "for arm in ('raw', 'full_hybrid', 'full_hybrid_refine2'):\n"
         "    a = [by[t][arm] for t in by if arm in by[t]]\n"
         "    e = np.array([r['cam_est_err_cm_deg'][0] for r in a if r.get('cam_est_err_cm_deg')])\n"
         "    print(arm, 'n', len(a), 'success', sum(bool(r['success']) for r in a), ('| cal err median %.2f cm, within 1 cm %d, within 2 cm %d' % (np.median(e), (e <= 1).sum(), (e <= 2).sum())) if len(e) else '')\n"
         "for ref in ('raw', 'full_hybrid'):\n"
         "    t = [i for i in by if ref in by[i] and 'full_hybrid_refine2' in by[i]]\n"
         "    b = sum(by[i]['full_hybrid_refine2']['success'] and not by[i][ref]['success'] for i in t); c = sum(by[i][ref]['success'] and not by[i]['full_hybrid_refine2']['success'] for i in t)\n"
         "    print(f'full_hybrid_refine2 vs {ref}: n {len(t)}, wins {b}, losses {c}, exact McNemar p {mc(b, c):.4f}')\n".replace("@D@", DRIVE_OUT)),
    # ---- run file 27 (2026-09-26, pre-registered in wrapper/prereg/2026-09-26_learned_refine_replication.md): set B
    code(f"# learned refine set B 1 (~1.5 h on A100, resumable): full_hybrid_refine2 on the 30 set B camera tasks (raw and full exist in plus/wraps, same seeds). Own folder plus/wraps_refine2_setB\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_refine2_setB --nominals {NOMINALS} --arms full_hybrid_refine2 --categories camera --per_cat_scale 0 --extra_from_suite 30 --seg_ckpt {SEG_CKPT}"),
    code("# learned refine set B 2: report (seconds; paste all of it)\n"
         "import json, os, numpy as np\nfrom math import comb\nD = '@D@'\nimport glob; rows = lambda p: [json.loads(l) for q in [p] + sorted(glob.glob(p[:-6] + '.*.jsonl')) if os.path.exists(q) for l in open(q) if l.strip()]\n"
         "new = {r['task_id']: r for r in rows(f'{D}/wraps_refine2_setB/wrap_openvla_oft.jsonl') if r['arm'] == 'full_hybrid_refine2'}\n"
         "old = {}\nfor r in rows(f'{D}/wraps/wrap_openvla_oft.jsonl'):\n    if r['task_id'] in new and r.get('rep', 0) == 0: old.setdefault(r['task_id'], {}).setdefault(r['arm'], bool(r['success']))\n"
         "mc = lambda b, c: min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0\n"
         "e = np.array([r['cam_est_err_cm_deg'][0] for r in new.values() if r.get('cam_est_err_cm_deg')])\n"
         "print('refine2 n', len(new), 'success', sum(bool(r['success']) for r in new.values()), '| cal err median %.2f cm, within 2 cm %d' % (np.median(e), (e <= 2).sum()))\n"
         "for ref in ('raw', 'full'):\n    t = [i for i in new if ref in old.get(i, {})]\n"
         "    b = sum(new[i]['success'] and not old[i][ref] for i in t); c = sum(old[i][ref] and not new[i]['success'] for i in t)\n"
         "    print(f'refine2 vs {ref}: n {len(t)} {ref} {sum(old[i][ref] for i in t)} -> refine2 {sum(bool(new[i][\"success\"]) for i in t)} | wins {b} losses {c} | exact McNemar p {mc(b, c):.4f}')\n".replace("@D@", DRIVE_OUT)),
    # ---- run file 23 (POST HOC 2026-09-24, TASKS.md P47): the pose-source ablation on set B (pose_sources.py)
    code(f"# pose source 1, smoke (~10 min): full_refine and full_fiducial on 2 set B tasks into their own smoke folder. Each episode line must print; paste them and STOP on an error\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_posesource_smoke --nominals {NOMINALS} --arms full_refine,full_fiducial --categories camera --per_cat_scale 0 --extra_from_suite 30 --tasks 827,855"),
    code(f"# pose source 2 (~3.5 h on A100, resumable): the same reprojection driven by (ii) EasyHeC-style dense refinement from the silhouette pose and (iii) a fiducial (oracle-corner) PnP pose, on all 30 set B camera tasks, same harness and seeds as the raw / full rows in plus/wraps. Own folder plus/wraps_posesource\n!cd /content/openvla-oft && MUJOCO_GL=egl {PY1} /content/scripts/wrap_run.py --family oft --subset {SUBSET} --out_dir {DRIVE_OUT}/wraps_posesource --nominals {NOMINALS} --arms full_refine,full_fiducial --categories camera --per_cat_scale 0 --extra_from_suite 30"),
    code("# pose source 3: report (seconds; paste all of it). Per pose source on the same set B tasks: success, calibration error, paired against the silhouette pose (full); also writes the compact rows file posesource_rows.jsonl\n"
         "import json, os, numpy as np\nfrom math import comb\nD = '@D@'\nimport glob; rows = lambda p: [json.loads(l) for q in [p] + sorted(glob.glob(p[:-6] + '.*.jsonl')) if os.path.exists(q) for l in open(q) if l.strip()]\n"
         "new = rows(f'{D}/wraps_posesource/wrap_openvla_oft.jsonl')\nids = {r['task_id'] for r in new if r['category'] == 'camera'}\n"
         "old = [r for r in rows(f'{D}/wraps/wrap_openvla_oft.jsonl') if r['task_id'] in ids and r['category'] == 'camera' and r.get('rep', 0) == 0 and r['arm'] in ('raw', 'full')]\n"
         "lad = [r for r in rows(f'{D}/wraps_ladder_libero_10/wrap_openvla_oft.jsonl') if r['task_id'] in ids and r['arm'] == 'full_truepose']\n"
         "A = {}\nfor r in old + lad + new:\n    A.setdefault(r['arm'], {})[r['task_id']] = r\n"
         "with open(f'{D}/wraps_posesource/posesource_rows.jsonl', 'w') as f:\n"
         "    for arm, a in A.items():\n        for t, r in sorted(a.items()):\n"
         "            f.write(json.dumps(dict(task_id=t, arm=arm, cam_est_err_cm_deg=r.get('cam_est_err_cm_deg'), success=bool(r['success']), pose_applied=r.get('cam_pose_applied', r.get('cam_applied')), fid_markers_visible=r.get('fid_markers_visible'))) + chr(10))\n"
         "mc = lambda b, c: 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))\n"
         "raw = {t: bool(r['success']) for t, r in A.get('raw', {}).items()}\nfull = {t: bool(r['success']) for t, r in A.get('full', {}).items()}\n"
         "for arm in ('raw', 'full', 'full_refine', 'full_fiducial', 'full_truepose'):\n"
         "    a = A.get(arm, {})\n    if not a:\n        continue\n"
         "    e = np.array([r['cam_est_err_cm_deg'][:2] for r in a.values() if r.get('cam_est_err_cm_deg')], float).reshape(-1, 2)\n"
         "    s = {t: bool(r['success']) for t, r in a.items()}\n"
         "    rf = [t for t in s if t in raw and not raw[t]]\n    b = sorted(set(s) & set(full))\n"
         "    print(f'{arm:<14} n {len(a):2d} success {sum(s.values()):2d} | repaired {sum(s[t] for t in rf)} of {len(rf)} raw failures'\n"
         "          + (' | calib err median %.2f cm %.2f deg, within 2 cm %d of %d' % (*np.median(e, 0), int((e[:, 0] <= 2).sum()), len(e)) if len(e) else '')\n"
         "          + (f' | vs full: only this {sum(s[t] and not full[t] for t in b)}, only full {sum(full[t] and not s[t] for t in b)}, p {mc(sum(s[t] and not full[t] for t in b), sum(full[t] and not s[t] for t in b)):.3g}' if arm not in ('raw', 'full') else ''))\n"
         "fv = [r.get('fid_markers_visible') for r in A.get('full_fiducial', {}).values()]\nprint('fiducial: markers visible per task', sorted(fv))\n".replace("@D@", DRIVE_OUT)),
], "plus_6_wrapper_oft.ipynb")

# ------------------------------------------------------------------ 8. DROID real-image self-calibration
DSC = f"{DRIVE_OUT}/droid_selfcal"
notebook([
    md("""
# Real-image self-calibration on DROID (no robot needed)
**GPU: T4 or L4.** Estimates each exterior camera's pose from the Franka's own body in REAL DROID frames and scores it
against DROID's post-hoc calibrated extrinsics (`KarlP/droid`, ungated). The demo's own arm motion is the sweep.

1. Export (~20 min, CPU): droid_100 episodes that have calibrated extrinsics; 8 frames per camera with joint angles.
2. Segment (~10 min): text-prompted robot masks (OWLv2 box -> SAM).
3. Calibrate (~1 to 2 min per camera): first checks the extrinsics convention by rendering the robot at the reference
   pose (paste that line), then grid + CEM over the 6-DoF camera pose.
4. Report: error vs the reference, and how often the estimate explains the silhouette as well as the reference.
The reference extrinsics are themselves estimates, so this measures agreement with DROID's calibration, not truth.
"""),
    drive_cell(), mkdir_scripts(), *write_scripts(["droid_selfcal.py", "droid_vlm_seg.py", "droid_robotseg.py"]),
    code("""%%bash
set -e
pip -q install "mujoco>=3.1" opencv-python-headless anthropic
# Colab's tensorflow_datasets links a tensorflow-metadata generated with a newer protobuf than the runtime; this pin is loadable
pip -q install --no-deps "tensorflow-metadata==1.16.1"
[ -d /content/mujoco_menagerie ] || git clone -q --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /content/mujoco_menagerie
ls /content/mujoco_menagerie/franka_emika_panda/panda_nohand.xml
python -c "import mujoco, tensorflow_datasets, transformers; print('mujoco', mujoco.__version__, '| transformers', transformers.__version__)"
"""),
    code(f"# 1. export (droid_100 from GCS, joined to the calibration files)\n!python /content/scripts/droid_selfcal.py export --out_dir {DSC} --max_cameras 60"),
    code(f"# 2. segment the robot in the real frames\n!python /content/scripts/droid_selfcal.py segment --out_dir {DSC}"),
    code(f"# 3. self-calibrate every camera (resumable); the first line printed is the convention check\n!MUJOCO_GL=egl python /content/scripts/droid_selfcal.py calibrate --out_dir {DSC}"),
    code(f"# 4. report (paste this)\n!python /content/scripts/droid_selfcal.py report --out_dir {DSC}"),
    code(f"# VLM masks 1: SAM cuts every frame into numbered regions (~15 min on T4, resumable, no API)\n!python /content/scripts/droid_vlm_seg.py propose --in_dir {DSC} --work_dir {DSC}/vlm_work"),
    code(f"# VLM masks 2, PILOT: the VLM picks the robot regions on the first 6 cameras (Colab secret ANTHROPIC_API_KEY; about 2 dollars, prints the real cost), then masks old vs new against the reference render (paste the table)\nimport os\nfrom google.colab import userdata\nos.environ[\"ANTHROPIC_API_KEY\"] = userdata.get(\"ANTHROPIC_API_KEY\")\n!python /content/scripts/droid_vlm_seg.py select --in_dir {DSC} --work_dir {DSC}/vlm_work --limit 6\n!python /content/scripts/droid_vlm_seg.py assemble --in_dir {DSC} --work_dir {DSC}/vlm_work --out_dir {DSC}_vlm_pilot --limit 6\n!MUJOCO_GL=egl python /content/scripts/droid_vlm_seg.py eval --out_dir {DSC}_vlm_pilot"),
    code(f"# VLM masks 3: every camera (resumable; answered frames are reused and never paid for twice). The zoom round is NOT run: the robot's own motion splits the part-robot regions for free\nimport os\nfrom google.colab import userdata\nos.environ[\"ANTHROPIC_API_KEY\"] = userdata.get(\"ANTHROPIC_API_KEY\")\n!python /content/scripts/droid_vlm_seg.py select --in_dir {DSC} --work_dir {DSC}/vlm_work"),
    code(f"# VLM masks 4: build the masks, score them against the reference render (paste the table)\n!python /content/scripts/droid_vlm_seg.py assemble --in_dir {DSC} --work_dir {DSC}/vlm_work --out_dir {DSC}_vlm\n!MUJOCO_GL=egl python /content/scripts/droid_vlm_seg.py eval --out_dir {DSC}_vlm"),
    code(f"# VLM masks 5: self-calibrate on the new masks (same v6 search, ~2 min per answered camera, resumable). Motion completion OFF (it is what re-attached the background) and every large mask component kept\n!MUJOCO_GL=egl python /content/scripts/droid_selfcal.py calibrate --out_dir {DSC}_vlm --convention cam2base --motion 0 --keep_frac 0.15"),
    code(f"# VLM masks 6: report (paste this)\n!python /content/scripts/droid_selfcal.py report --out_dir {DSC}_vlm"),
    code(f"# RobotSeg baseline 1: install RobotSeg (CVPR 2026, a SAM 2 model trained to segment robots) and find its checkpoint (~6 min). The authors host the weights on OneDrive only, which a notebook cannot fetch: download robotseg.pt ONCE from the link in github.com/showlab/RobotSeg (Checkpoint section) and put it on Drive at restoring_organs/plus/robotseg/robotseg.pt\nimport os\nCK = '{DRIVE_OUT}/robotseg/robotseg.pt'\nassert os.path.exists(CK), 'put robotseg.pt on Drive at ' + CK + ' (link: github.com/showlab/RobotSeg, Checkpoint section), then re-run this cell'\nassert os.path.getsize(CK) > 100e6, 'robotseg.pt on Drive is only ' + str(os.path.getsize(CK) // 1000000) + ' MB: the download was cut short (the model is a SAM 2.1 network, well over 100 MB). Download it again, wait until the browser says it is complete, replace the file on Drive, then re-run'\n![ -d /content/RobotSeg ] || git clone -q --depth 1 https://github.com/showlab/RobotSeg /content/RobotSeg\n!cd /content/RobotSeg && pip install -q --no-deps -e . && pip install -q hydra-core iopath natsort omegaconf kornia && (python setup.py build_ext --inplace > /content/robotseg_build.log 2>&1 || echo 'optional CUDA extension did not build (only post-processing uses it): continuing')\n!mkdir -p /content/RobotSeg/checkpoints && cp -f {DRIVE_OUT}/robotseg/robotseg.pt /content/RobotSeg/checkpoints/ && ls -la /content/RobotSeg/checkpoints"),
    code(f"# RobotSeg baseline 2, SMOKE: two cameras only (~1 min). Paste the two lines it prints: a robot pixel fraction between about 0.02 and 0.30 means it works; an error here costs nothing, send it to the agent\n!cd /content/RobotSeg/test && python /content/scripts/droid_robotseg.py --in_dir {DSC} --out_dir /content/robotseg_smoke --limit 2"),
    code(f"# RobotSeg baseline 3: every camera (~10 min on T4, resumable), then masks scored against the reference render (paste the table)\n!cd /content/RobotSeg/test && python /content/scripts/droid_robotseg.py --in_dir {DSC} --out_dir {DSC}_robotseg\n!MUJOCO_GL=egl python /content/scripts/droid_vlm_seg.py eval --out_dir {DSC}_robotseg"),
    code(f"# RobotSeg baseline 4: the SAME self-calibration search on RobotSeg's masks (~2 min per camera, ~2 h, resumable), same settings as the VLM masks\n!MUJOCO_GL=egl python /content/scripts/droid_selfcal.py calibrate --out_dir {DSC}_robotseg --convention cam2base --motion 0 --keep_frac 0.15"),
    code(f"# RobotSeg baseline 5: report (paste this)\n!python /content/scripts/droid_selfcal.py report --out_dir {DSC}_robotseg"),
], "droid_selfcal.ipynb")
