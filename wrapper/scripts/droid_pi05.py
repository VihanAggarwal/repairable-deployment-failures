"""Real-data evidence without a robot, with pi0.5-DROID (openpi) on DROID-100.

E1 cross-view: the same real moment seen by exterior camera 1 vs 2 (a real camera change). Does the
   policy's disagreement between views track its error against the recorded real action?
E2 self-disagreement: spread of the policy's action over sampler noise, vs its real error.
E3 nuisance: known camera bump (rotation + shift) and lighting change applied to real frames; error rise.
E4 pixel channel: fit a 6-parameter image correction (similarity warp + gain + gamma) on the first third of
   each episode, anchored ONLY to the robot's own recorded motion, then test on held-out later frames.
Run with openpi's venv from the openpi repo root."""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

NUIS = {"camera": np.array([8.0, 12.0, -8.0, 0.0, 0.0, 0.0]),     # rot deg, tx px, ty px, log scale, log gain, log gamma
        "light": np.array([0.0, 0.0, 0.0, 0.0, np.log(0.55), np.log(1.3)])}
STD0 = np.array([6.0, 15.0, 15.0, 0.08, 0.4, 0.3])


def apply(img, p):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(p[0]), float(np.exp(p[3])))
    M[0, 2] += p[1]
    M[1, 2] += p[2]
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
    out = np.clip(out ** float(np.exp(p[5])) * float(np.exp(p[4])), 0, 1)
    return (out * 255).astype(np.uint8)


def pad224(img):
    h, w = img.shape[:2]
    s = 224.0 / max(h, w)
    r = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((224, 224, 3), np.uint8)
    y, x = (224 - r.shape[0]) // 2, (224 - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/content/droid_npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--e12_episodes", type=int, default=30)
    ap.add_argument("--e34_episodes", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config
    cfg = _config.get_config("pi05_droid")
    policy = policy_config.create_trained_policy(cfg, download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid"))
    H, D = cfg.model.action_horizon, cfg.model.action_dim

    def act(ext, wrist, q, g, prompt, seed):
        noise = np.random.default_rng(seed).standard_normal((H, D)).astype(np.float32)
        out = policy.infer({"observation/exterior_image_1_left": pad224(ext),
                            "observation/wrist_image_left": pad224(wrist),
                            "observation/joint_position": np.asarray(q, np.float64),
                            "observation/gripper_position": np.asarray(g, np.float64).reshape(1),
                            "prompt": prompt}, noise=noise)
        return np.asarray(out["actions"])

    def err(a0, jv):
        return float(np.linalg.norm(a0[0, :7] - jv))

    files = sorted(glob.glob(os.path.join(a.npz, "ep*.npz")))
    assert files, f"no npz in {a.npz}"
    if a.smoke:
        z = np.load(files[0])
        t0 = time.time()
        A = act(z["ext1"][5], z["wrist"][5], z["q"][5], z["g"][5], str(z["lang"]), 0)
        print("action chunk", A.shape, "| infer s", round(time.time() - t0, 2),
              "| pred jv", np.round(A[0, :7], 3), "| recorded jv", np.round(z["jv"][5], 3))
        return

    res = json.load(open(a.out)) if os.path.exists(a.out) else {"e12": [], "e34": []}
    done12 = {r["ep"] for r in res["e12"]}
    for f in files[:a.e12_episodes]:
        ep = os.path.basename(f)
        if ep in done12:
            continue
        z = np.load(f)
        prompt = str(z["lang"])
        n = len(z["q"])
        for t in np.linspace(3, n - 4, 6).astype(int):
            a1 = act(z["ext1"][t], z["wrist"][t], z["q"][t], z["g"][t], prompt, 0)
            a2 = act(z["ext2"][t], z["wrist"][t], z["q"][t], z["g"][t], prompt, 0)
            extra = [act(z["ext1"][t], z["wrist"][t], z["q"][t], z["g"][t], prompt, s)[0, :7] for s in (1, 2, 3)]
            disp = float(np.mean(np.var(np.stack([a1[0, :7]] + extra), 0)))
            res["e12"].append(dict(ep=ep, t=int(t), err1=err(a1, z["jv"][t]), err2=err(a2, z["jv"][t]),
                                   cross=float(np.linalg.norm(a1[0, :7] - a2[0, :7])), disp=disp))
        json.dump(res, open(a.out, "w"))
        print(f"E1/E2 {ep} done ({len(res['e12'])} frames)", flush=True)

    done34 = {(r["ep"], r["nuisance"]) for r in res["e34"]}
    rng = np.random.default_rng(0)
    for f in files[:a.e34_episodes]:
        ep = os.path.basename(f)
        z = np.load(f)
        prompt, n = str(z["lang"]), len(z["q"])
        calib = np.linspace(2, n // 3, 8).astype(int)
        test = np.linspace(n // 3 + 2, n - 3, 8).astype(int)
        for name, p_true in NUIS.items():
            if (ep, name) in done34:
                continue
            seen = lambda t, corr=None: apply(z["ext1"][t], p_true) if corr is None else apply(apply(z["ext1"][t], p_true), corr)

            def objective(corr, frames):
                return float(np.mean([err(act(seen(t, corr), z["wrist"][t], z["q"][t], z["g"][t], prompt, 100 + i), z["jv"][t])
                                      for i, t in enumerate(frames)]))

            mu, sd = np.zeros(6), STD0.copy()
            for it in range(6):                       # CEM, 12 candidates, 4 elites
                cand = mu + sd * rng.standard_normal((12, 6))
                cand[0] = mu
                scores = np.array([objective(c, calib) for c in cand])
                elite = cand[np.argsort(scores)[:4]]
                mu, sd = elite.mean(0), elite.std(0) + 1e-3
            clean = float(np.mean([err(act(z["ext1"][t], z["wrist"][t], z["q"][t], z["g"][t], prompt, 100 + i), z["jv"][t])
                                   for i, t in enumerate(test)]))
            res["e34"].append(dict(ep=ep, nuisance=name, err_clean=clean,
                                   err_nuisance=objective(np.zeros(6), test),
                                   err_corrected=objective(mu, test), fitted=mu.tolist(), true=p_true.tolist()))
            json.dump(res, open(a.out, "w"))
            r = res["e34"][-1]
            print(f"E3/E4 {ep} {name}: clean {r['err_clean']:.3f} | nuisance {r['err_nuisance']:.3f} | corrected {r['err_corrected']:.3f}", flush=True)

    e12, e34 = res["e12"], res["e34"]
    print("\n== E1 cross-view (real camera change) ==")
    print(f"frames {len(e12)} | mean err view1 {np.mean([r['err1'] for r in e12]):.3f} view2 {np.mean([r['err2'] for r in e12]):.3f}")
    print(f"Spearman(cross-view disagreement, view-2 error) = {spearman([r['cross'] for r in e12], [r['err2'] for r in e12]):.3f}")
    print("== E2 self-disagreement ==")
    print(f"Spearman(noise dispersion, view-1 error) = {spearman([r['disp'] for r in e12], [r['err1'] for r in e12]):.3f}")
    for name in NUIS:
        rr = [r for r in e34 if r["nuisance"] == name]
        if rr:
            c, u, k = (np.mean([r[x] for r in rr]) for x in ("err_clean", "err_nuisance", "err_corrected"))
            rec = (u - k) / max(u - c, 1e-9)
            print(f"== E3/E4 {name}: episodes {len(rr)} | clean {c:.3f} -> nuisance {u:.3f} -> corrected {k:.3f} | fraction of damage recovered {rec:.2f}")


if __name__ == "__main__":
    main()
