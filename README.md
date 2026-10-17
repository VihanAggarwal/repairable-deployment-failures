# Distinguishing Repairable Deployment Failures from Vision-Language-Action Failures Before Rollout

Code, notebooks and result files for the paper by V. Aggarwal and S. Stephan (under review at ICLR 2027).

A frozen vision-language-action policy often fails after deployment because the deployment changed (camera pose,
start pose, lighting, sensor noise), not because the policy cannot do the task. This repo measures, before rollout and
from the robot's own body, whether a deployment shift is one a wrapper can repair, and evaluates the wrapper on
LIBERO-Plus with OpenVLA-OFT and other policy families, plus real DROID camera data.

## Layout

| Path | Contents |
|---|---|
| `wrapper/scripts/` | All experiment code: calibration, coverage and residual scores, sweep runners, analysis |
| `wrapper/notebooks/` | Colab notebooks, generated from the scripts by `wrapper/make_plus_notebooks.py` (do not edit by hand) |
| `wrapper/prereg/` | Pre-registered tests and their decision rules |
| `wrapper/paper/iclr/` | LaTeX source and figures of the submission |
| `wrapper/_*_tests.py` | Local tests that need no simulator or GPU |
| `results/` | Result files (JSONL/CSV) behind every table and figure |

## Running

Policy runs need a GPU and were run on Google Colab; open a notebook from `wrapper/notebooks/` and run the cells it
names. Policy-free parts (calibration, scores, tests) run locally:

```
uv run --no-project --with numpy --with scipy --with opencv-python-headless python wrapper/_wrapper_tests.py
```

Regenerate the notebooks after changing a script with `python wrapper/make_plus_notebooks.py`.
