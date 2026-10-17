"""Robot masks for the DROID self-calibration chosen by a vision-language model (set-of-marks), replacing the zero-shot
OWLv2 box -> SAM masks whose background pixels capped the real-image result (mask precision 0.46 to 0.57).

A VLM is poor at drawing a mask and good at saying what a region is, so the work is split:
  propose   SAM's automatic masks cut every frame into a COMPLETE partition of numbered regions (GPU helps, CPU works)
  select    the VLM sees the frame and the numbered frame and returns the region numbers that are robot (API only)
  refine    every region the VLM called part robot, part background is cropped, zoomed and cut again (optional)
  select --round 2   the same question on the zoomed crops
  assemble  the chosen regions become the mask; the camera files are COPIED to --out_dir with the new `mask`
            (the old one is kept as `mask_owl`), so `droid_selfcal.py calibrate --out_dir <out_dir>` runs unchanged
  eval      old and new masks against the robot rendered at DROID's reference pose (needs mujoco + menagerie)

    python droid_vlm_seg.py propose  --in_dir <droid_selfcal> --work_dir <work>
    python droid_vlm_seg.py select   --in_dir <droid_selfcal> --work_dir <work>          # key in ANTHROPIC_API_KEY
    python droid_vlm_seg.py refine   --in_dir <droid_selfcal> --work_dir <work>          # optional, then select --round 2
    python droid_vlm_seg.py assemble --in_dir <droid_selfcal> --work_dir <work> --out_dir <droid_selfcal_vlm>
    python droid_vlm_seg.py eval     --in_dir <droid_selfcal> --out_dir <droid_selfcal_vlm> --menagerie <path>

Every stage is resumable. The key is read from the environment only and is never written anywhere.
"""
import argparse
import base64
import glob
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UP = 3                      # SAM runs on the frame upscaled 3x (the stored frames are 320 x 180)
SEND_W = 1280               # width of the two images the VLM sees
MIN_FRAC = 0.0015           # regions smaller than this share of the frame are not numbered
MAX_REGIONS = 80
PRICE = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
         "claude-fable-5-1": (10.0, 50.0)}          # USD per million tokens (input, output)
PALETTE = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0), (255, 0, 255), (0, 220, 220), (255, 128, 0),
           (160, 96, 255), (0, 160, 96), (255, 128, 192)]

SYSTEM = (
    "You label image regions for a robot self-calibration pipeline. Each camera frame shows a lab or home scene with a "
    "Franka Emika Panda arm (white links, black joints and wrist) carrying a Robotiq two-finger gripper (black). The arm "
    "may be cut by the image border, may be small, or may be absent. You receive the plain frame and the same frame cut "
    "into numbered regions (tinted, outlined, number inside the region). Your answer is compared against a render of "
    "the arm and gripper ALONE, so a region counts as robot only if it is part of the arm's links, joints, wrist or the "
    "gripper and its fingers. NOT robot: cables and hoses (also those hanging from the arm), the table, stand, pedestal "
    "or control box the arm is bolted to, cameras and their mounts or tripods (also a camera on the wrist when it is its "
    "own region), any object held or touched, shadows, reflections, other equipment, and people.")

QUESTION = (
    "Image 1 is the camera frame. Image 2 is the same frame with numbered regions. Return:\n"
    "robot_visible: whether any part of the arm or gripper is in view.\n"
    "robot_ids: every region number of which at least about 80 percent is arm or gripper.\n"
    "mixed_ids: region numbers that contain a substantial part of the arm or gripper AND a substantial part of "
    "something else (for example the arm merged with the wall behind it).\n"
    "Check every number near the arm, including small regions on the gripper and regions where the arm leaves the frame. "
    "Use only numbers that appear in image 2.")

SCHEMA = {
    "type": "object",
    "properties": {
        "robot_visible": {"type": "boolean"},
        "robot_ids": {"type": "array", "items": {"type": "integer"}},
        "mixed_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["robot_visible", "robot_ids", "mixed_ids"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------ regions
def regions_from_masks(masks, shape, min_frac=MIN_FRAC, max_regions=MAX_REGIONS):
    """A complete partition of the frame from overlapping SAM masks: larger masks are painted first so every pixel ends
    in the SMALLEST mask that contains it; what no mask covers is split into connected components. Regions under
    min_frac of the frame, and those beyond the max_regions largest, get label 0 (not selectable).
    Returns (labels int16, share of the frame left unlabelled)."""
    import cv2
    H, W = shape
    lab = np.zeros((H, W), np.int32)
    order = sorted(range(len(masks)), key=lambda i: -int(np.asarray(masks[i]).sum()))
    for n, i in enumerate(order, 1):
        lab[np.asarray(masks[i]) > 0] = n
    n_cc, cc = cv2.connectedComponents((lab == 0).astype(np.uint8), connectivity=4)
    top = int(lab.max())
    for k in range(1, n_cc):
        lab[cc == k] = top + k
    area = np.bincount(lab.ravel())
    keep = [i for i in np.argsort(-area) if i > 0 and area[i] >= min_frac * H * W][:max_regions]
    out = np.zeros((H, W), np.int16)
    for new, old in enumerate(keep, 1):
        out[lab == old] = new
    return out, float((out == 0).mean())


def draw_marks(img, lab, font=0.45):
    """The frame with every region tinted, outlined and numbered at its deepest interior point."""
    import cv2
    out = img.astype(np.float32)
    ids = [int(i) for i in np.unique(lab) if i > 0]
    for i in ids:
        m = lab == i
        out[m] = 0.84 * out[m] + 0.16 * np.array(PALETTE[i % len(PALETTE)], np.float32)   # light: the object stays readable
    out = np.clip(out, 0, 255).astype(np.uint8)
    for i in ids:
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, PALETTE[i % len(PALETTE)], 1)
    for i in ids:
        m = np.pad((lab == i).astype(np.uint8), 1)                       # the pad makes the image border count as an edge
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
        x0 = int(np.clip(x - tw // 2, 1, out.shape[1] - tw - 2))
        y0 = int(np.clip(y + th // 2, th + 2, out.shape[0] - 2))
        cv2.rectangle(out, (x0 - 1, y0 - th - 2), (x0 + tw + 1, y0 + 2), (0, 0, 0), -1)
        cv2.putText(out, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def views(img, lab, inside=None):
    """(plain, marked) RGB images at the size the VLM sees (long side SEND_W); lab is the label map at any resolution.
    inside: optional mask of the area in question; the rest of the marked image is dimmed."""
    import cv2
    k = SEND_W / max(img.shape[:2])
    w, h = int(round(img.shape[1] * k)), int(round(img.shape[0] * k))
    plain = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    marked = draw_marks(plain, cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST), font=0.45 if inside is None else 0.6)
    if inside is not None:
        out = cv2.resize(inside.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) == 0
        marked[out] = (0.35 * marked[out]).astype(np.uint8)
    return plain, marked


def frames_of(a, n):
    return list(range(n)) if a.frames == "all" else [int(x) for x in a.frames.split(",") if int(x) < n]


def camera_files(a):
    ps = sorted(glob.glob(os.path.join(a.in_dir, "*.npz")))
    return ps[:a.limit] if a.limit else ps


def props_path(a, p, kind="props"):
    return os.path.join(a.work_dir, os.path.basename(p).replace(".npz", f".{kind}.npz"))


def upscaled(img):
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((img.shape[1] * UP, img.shape[0] * UP), Image.BICUBIC))


def sam_generator(a):
    import torch
    from transformers import pipeline
    gpu = torch.cuda.is_available()
    name = a.sam or ("facebook/sam-vit-huge" if gpu else "facebook/sam-vit-base")
    print("SAM:", name, "on", "cuda" if gpu else "cpu", flush=True)
    return pipeline("mask-generation", model=name, device=0 if gpu else -1)


# ------------------------------------------------------------------ propose (transformers)
def propose(a):
    import cv2
    import torch
    from PIL import Image
    os.makedirs(a.work_dir, exist_ok=True)
    gen = sam_generator(a)
    for p in camera_files(a):
        out = props_path(a, p)
        if os.path.exists(out):
            continue
        d = np.load(p, allow_pickle=True)
        labs, lost = [], []
        for t, img in enumerate(d["img"]):
            if t not in frames_of(a, len(d["img"])):
                labs.append(np.zeros((img.shape[0] * UP, img.shape[1] * UP), np.int16))
                continue
            big = Image.fromarray(img).resize((img.shape[1] * UP, img.shape[0] * UP), Image.BICUBIC)
            with torch.no_grad():
                res = gen(big, points_per_batch=a.points_per_batch, pred_iou_thresh=0.86, stability_score_thresh=0.90)
            lab, miss = regions_from_masks([np.asarray(m) for m in res["masks"]], (big.size[1], big.size[0]))
            labs.append(lab)
            lost.append(miss)
        np.savez_compressed(out, labels=np.stack(labs))
        print(f"{os.path.basename(p)}: {int(np.mean([l.max() for l in labs]))} regions per frame, "
              f"{100 * float(np.mean(lost)):.1f}% of the frame in regions too small to number", flush=True)
        if a.dump_marked:
            t = frames_of(a, len(d["img"]))[0]
            _, marked = views(d["img"][t], labs[t])
            cv2.imwrite(out.replace(".props.npz", f".f{t}.marked.jpg"), cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------ refine (transformers)
def refine(a):
    """Zoom into every region the VLM called mixed (the arm merged with the wall behind it, say): crop it, upscale the
    crop, cut it again with SAM. The crops are asked about with `select --round 2`."""
    import cv2
    import torch
    from PIL import Image
    _, sel = read_selection(a)
    gen = None
    for p in camera_files(a):
        out = props_path(a, p, "refine")
        todo = [(t, rid) for (f, t, reg), r in sel.items() if f == os.path.basename(p) and reg == 0 and r["robot_visible"]
                for rid in r["mixed_ids"][:a.max_mixed]]
        if os.path.exists(out) or not todo or not os.path.exists(props_path(a, p)):
            continue
        gen = gen or sam_generator(a)
        d = np.load(p, allow_pickle=True)
        labs = np.load(props_path(a, p))["labels"]
        store = {}
        for t, rid in sorted(todo):
            parent = labs[t] == rid
            near = np.isin(labs[t], sel[(os.path.basename(p), t, 0)]["robot_ids"])
            if near.any():                  # a mixed region can be the whole wall: only its part next to the robot regions
                ys, xs = np.where(near)     # found so far is worth a zoom (their box, grown by half its size)
                gx, gy = (xs.max() - xs.min()) // 2 + 30, (ys.max() - ys.min()) // 2 + 30
                box = np.zeros_like(parent)
                box[max(ys.min() - gy, 0):ys.max() + gy, max(xs.min() - gx, 0):xs.max() + gx] = True
                parent = parent & box
            ys, xs = np.where(parent)
            if len(ys) == 0:
                continue
            padx, pady = int(0.08 * (xs.max() - xs.min() + 1)) + 4, int(0.08 * (ys.max() - ys.min() + 1)) + 4
            x0, x1 = max(xs.min() - padx, 0), min(xs.max() + padx + 1, parent.shape[1])
            y0, y1 = max(ys.min() - pady, 0), min(ys.max() + pady + 1, parent.shape[0])
            crop = upscaled(d["img"][t])[y0:y1, x0:x1]
            k = min(4.0, 1024.0 / max(crop.shape[:2]))
            big = Image.fromarray(crop).resize((max(int(crop.shape[1] * k), 8), max(int(crop.shape[0] * k), 8)), Image.BICUBIC)
            with torch.no_grad():
                res = gen(big, points_per_batch=a.points_per_batch, pred_iou_thresh=0.86, stability_score_thresh=0.90)
            lab, _ = regions_from_masks([np.asarray(m) for m in res["masks"]], (big.size[1], big.size[0]), min_frac=0.004, max_regions=40)
            inside = cv2.resize(parent[y0:y1, x0:x1].astype(np.uint8), big.size, interpolation=cv2.INTER_NEAREST) > 0
            lab[~inside] = 0                                  # only the parent region is in question
            store[f"lab_{t}_{rid}"] = lab
            store[f"box_{t}_{rid}"] = np.array([x0, y0, x1, y1])
        np.savez_compressed(out, **store)
        print(f"{os.path.basename(p)}: {len(store) // 2} mixed regions cut again", flush=True)


# ------------------------------------------------------------------ select (Anthropic API)
def jpeg_block(img):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.tobytes()).decode("utf-8")}}


ZOOM_NOTE = ("\n\nThis pair is a ZOOMED CROP of one region of a larger frame that holds part of the arm or gripper merged "
             "with something else. Only the numbered, undimmed area is in question; the dimmed surround is context.")


def ask(client, a, img, lab, inside=None):
    """One frame (or one zoomed crop) -> dict(robot_visible, robot_ids, mixed_ids, tokens) or dict(error=...)."""
    plain, marked = views(img, lab, inside)
    cfg = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if "haiku" not in a.model:                              # Haiku 4.5 rejects the effort field
        cfg["effort"] = a.effort
    resp = client.messages.create(
        model=a.model, max_tokens=8000, system=SYSTEM, output_config=cfg,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Image 1 (plain frame):"}, jpeg_block(plain),
                                               {"type": "text", "text": "Image 2 (numbered regions):"}, jpeg_block(marked),
                                               {"type": "text", "text": QUESTION + (ZOOM_NOTE if inside is not None else "")}]}])
    usage = dict(input=resp.usage.input_tokens, output=resp.usage.output_tokens)
    if resp.stop_reason != "end_turn":                      # refusal or max_tokens: no usable answer for this frame
        return dict(error=str(resp.stop_reason), tokens=usage)
    ans = json.loads(next(b.text for b in resp.content if b.type == "text"))
    valid = {int(i) for i in np.unique(lab) if i > 0}
    return dict(robot_visible=bool(ans["robot_visible"]),
                robot_ids=sorted({int(i) for i in ans["robot_ids"]} & valid),
                mixed_ids=sorted(({int(i) for i in ans["mixed_ids"]} & valid) - {int(i) for i in ans["robot_ids"]}),
                unknown_ids=sorted(({int(i) for i in ans["robot_ids"]} | {int(i) for i in ans["mixed_ids"]}) - valid),
                tokens=usage)


def read_selection(a):
    path = os.path.join(a.work_dir, "vlm_select.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    return path, {(r["file"], r["frame"], r.get("region", 0)): r for r in rows if "error" not in r}      # the last good row wins


def select(a):
    import anthropic
    from concurrent.futures import ThreadPoolExecutor
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("no API key: set ANTHROPIC_API_KEY in the environment (Colab: a secret of that name)")
    client = anthropic.Anthropic(max_retries=6, timeout=180.0)
    path, done = read_selection(a)
    jobs = []
    for p in camera_files(a):
        if a.round == 2 and os.path.exists(props_path(a, p, "refine")):
            keys = [k for k in np.load(props_path(a, p, "refine")).files if k.startswith("lab_")]
            units = [(int(k.split("_")[1]), int(k.split("_")[2])) for k in keys]
            jobs += [(p, t, rid) for t, rid in sorted(units) if (os.path.basename(p), t, rid) not in done]
        elif a.round == 1 and os.path.exists(props_path(a, p)):
            n = len(np.load(p, allow_pickle=True)["img"])
            jobs += [(p, t, 0) for t in frames_of(a, n) if (os.path.basename(p), t, 0) not in done]
    print(f"{len(done)} frames already answered, {len(jobs)} to ask ({a.model}, effort {a.effort})", flush=True)
    lock, spent, halt = threading.Lock(), dict(input=0, output=0, n=0), threading.Event()

    def work(job):
        p, t, rid = job
        if halt.is_set():                                   # the key stopped working: do not burn through the queue
            return None
        d = np.load(p, allow_pickle=True)
        if rid:
            z = np.load(props_path(a, p, "refine"))
            lab, (x0, y0, x1, y1) = z[f"lab_{t}_{rid}"], z[f"box_{t}_{rid}"]
            img, inside = upscaled(d["img"][t])[y0:y1, x0:x1], lab > 0
        else:
            lab, img, inside = np.load(props_path(a, p))["labels"][t], d["img"][t], None
        try:
            r = ask(client, a, img, lab, inside)
        except anthropic.AuthenticationError as e:
            r = dict(error=f"authentication: {e}")
        except anthropic.RateLimitError as e:
            r = dict(error=f"rate limit after retries: {e}")
        except anthropic.APIStatusError as e:
            r = dict(error=f"status {e.status_code}: {e}")
        except anthropic.APIConnectionError as e:
            r = dict(error=f"connection: {e}")
        if any(w in str(r.get("error", "")) for w in ("usage limits", "authentication", "credit balance", "status 401", "status 403")):
            if not halt.is_set():
                print("STOPPING: the API key is refused (" + r["error"][:160] + "). Fix the key or its spend limit, then re-run this "
                      "cell: answered frames are kept, only the rest is asked.", flush=True)
            halt.set()
            return None
        r.update(file=os.path.basename(p), frame=t, region=rid, model=a.model)
        with lock:
            for attempt in range(6):                        # the Drive mount drops out for a moment now and then
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    break
                except OSError:
                    if attempt == 5:
                        raise
                    time.sleep(5)
            for k in ("input", "output"):
                spent[k] += r.get("tokens", {}).get(k, 0)
            spent["n"] += 1
            tail = r.get("error") or f"robot {r['robot_ids']} mixed {r['mixed_ids']}" + ("" if r["robot_visible"] else " (not visible)")
            print(f"[{spent['n']}/{len(jobs)}] {r['file']} f{t}" + (f" region {rid}" if rid else "") + f": {tail}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(work, jobs) if r is not None]
    pin, pout = PRICE.get(a.model, (0.0, 0.0))
    bad = sum("error" in r for r in rows)
    print(f"asked {len(rows)} frames, {bad} without an answer (re-run to retry them) | tokens in {spent['input']} out "
          f"{spent['output']} | about ${(spent['input'] * pin + spent['output'] * pout) / 1e6:.2f} at list price")


# ------------------------------------------------------------------ assemble
def motion_px(imgs, masks, thr=28.0):
    """Pixels that differ from the static background (per-pixel median over the frames in which the mask saw no robot
    there): the robot's own motion. uint8 0/1 per frame, at the stored frame size."""
    import cv2
    import warnings
    imgs = np.asarray(imgs, np.float32)
    free = np.where((np.asarray(masks) > 0)[..., None], np.nan, imgs)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")                        # all-NaN columns are expected and handled on the next line
        bg = np.nanmedian(free, axis=0)
    bg = np.where(np.isnan(bg), np.median(imgs, axis=0), bg)
    return [(cv2.GaussianBlur(np.abs(im - bg).max(-1), (5, 5), 0) > thr).astype(np.uint8) for im in imgs]


def mask_from_ids(lab, ids, shape, extra=()):
    """Chosen regions -> uint8 mask at the stored frame size; a small close bridges the unnumbered slivers between them.
    extra: (box, sub_mask) pairs from the zoomed crops of mixed regions, pasted in at the label-map resolution."""
    import cv2
    m = np.isin(lab, list(ids)).astype(np.uint8)
    for (x0, y0, x1, y1), sm in extra:
        m[y0:y1, x0:x1] |= cv2.resize(sm.astype(np.uint8), (int(x1 - x0), int(y1 - y0)), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2 * UP + 1, 2 * UP + 1), np.uint8))
    return (cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) if m.any() else np.zeros(shape, np.uint8)


def assemble(a):
    os.makedirs(a.out_dir, exist_ok=True)
    assert os.path.abspath(a.out_dir) != os.path.abspath(a.in_dir), "--out_dir must differ from --in_dir: the old masks are kept"
    _, sel = read_selection(a)
    for p in camera_files(a):
        if not os.path.exists(props_path(a, p)):
            continue
        d = dict(np.load(p, allow_pickle=True))
        labs = np.load(props_path(a, p))["labels"]
        z = np.load(props_path(a, p, "refine")) if os.path.exists(props_path(a, p, "refine")) else None
        masks, used, zoomed = [], 0, 0
        for t, img in enumerate(d["img"]):
            name = os.path.basename(p)
            r = sel.get((name, t, 0))
            ids = [] if r is None or not r["robot_visible"] else r["robot_ids"] + (r["mixed_ids"] if a.mixed == 1 else [])
            extra = []
            for rid in ([] if r is None or z is None or not r["robot_visible"] else r["mixed_ids"]):
                r2 = sel.get((name, t, rid))
                if r2 is not None and r2["robot_visible"] and f"lab_{t}_{rid}" in z.files:
                    extra.append((z[f"box_{t}_{rid}"], np.isin(z[f"lab_{t}_{rid}"], r2["robot_ids"])))
            zoomed += len(extra)
            masks.append(mask_from_ids(labs[t], ids, img.shape[:2], extra))
            used += r is not None
        if a.mixed == 2:
            # a region the VLM called part robot, part background is split by the robot's own motion: inside it, the
            # pixels that moved are robot. No extra API call; motion is trusted ONLY where the VLM saw robot.
            import cv2
            mv = motion_px(d["img"], masks)
            for t, img in enumerate(d["img"]):
                r = sel.get((os.path.basename(p), t, 0))
                if r is None or not r["robot_visible"] or not r["mixed_ids"]:
                    continue
                mixed = cv2.resize(np.isin(labs[t], r["mixed_ids"]).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                masks[t] = masks[t] | cv2.morphologyEx(mv[t] & mixed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if "mask" in d:
            d["mask_owl"] = d["mask"]
        d["mask"] = np.stack(masks).astype(np.uint8)
        np.savez_compressed(os.path.join(a.out_dir, os.path.basename(p)), **d)
        print(f"{os.path.basename(p)}: {used}/{len(masks)} frames answered, {zoomed} zoomed regions, robot pixel fraction {d['mask'].mean():.3f}"
              + (f" (was {d['mask_owl'].mean():.3f})" if "mask_owl" in d else ""), flush=True)


# ------------------------------------------------------------------ eval (mujoco)
def evaluate(a):
    import cv2
    import droid_selfcal as ds
    rob = ds.Franka(a.menagerie, hand=True)
    files = sorted(glob.glob(os.path.join(a.out_dir, "*.npz")))
    valid = [float(np.load(p, allow_pickle=True)["intr"][2]) for p in files]
    valid = [v for v in valid if np.isfinite(v) and v > 0]
    median_fy = float(np.median(valid)) if valid else 530.0
    rows = []
    for p in files:
        d = np.load(p, allow_pickle=True)
        if "mask_owl" not in d:
            continue
        fx, _, fy, _ = d["intr"]
        fy = fy if (np.isfinite(fy) and fy > 0) else (fx if (np.isfinite(fx) and fx > 0) else median_fy)
        wh_y = float(d["intr_wh"][1]) if (np.isfinite(d["intr_wh"][1]) and d["intr_wh"][1] > 0) else 720.0
        fy_px = float(fy) * rob.H / wh_y
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = ds.euler_xyz_to_mat(d["ext"][3:6]), d["ext"][:3]
        Tg = T if a.convention == "cam2base" else np.linalg.inv(T)
        small = lambda ms: [cv2.resize(np.asarray(m, np.uint8), (rob.W, rob.H), interpolation=cv2.INTER_NEAREST) for m in ms]
        arms = {"owl": small(ds.motion_masks(d["img"], d["mask_owl"])), "vlm": small(ds.motion_masks(d["img"], d["mask"])),
                "vlm_nomotion": small([ds.clean_mask(m, keep_frac=0.15) for m in d["mask"]])}     # what calibrate --motion 0 --keep_frac 0.15 fits
        row = dict(file=os.path.basename(p), reliable=bool(float(d["quality"]) < 20))
        for k, ms in arms.items():
            pr, rc, io = [], [], []
            for i, m in enumerate(ms):
                ren = rob.mask(d["q"][i], Tg[:3, 3], Tg[:3, :3], fy_px)
                if ren.sum() < 20 or m.sum() == 0:            # the reference puts no robot in view, or no mask: not scored
                    continue
                pr.append((ren & m).sum() / m.sum()); rc.append((ren & m).sum() / ren.sum()); io.append(ds.iou(ren, m))
            row.update({f"{k}_precision": float(np.mean(pr)) if pr else None, f"{k}_recall": float(np.mean(rc)) if rc else None,
                        f"{k}_iou": float(np.mean(io)) if io else None, f"{k}_frames": len(pr)})
        rows.append(row)
    with open(os.path.join(a.out_dir, "mask_eval.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    med = lambda rs, k: float(np.median([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else float("nan")
    print("masks against the robot rendered at DROID's reference pose (the reference is itself an estimate):")
    for name, rs in (("all cameras", rows), ("reliable reference", [r for r in rows if r["reliable"]])):
        print(f"\n{name}: {len(rs)}")
        print(f"  {'mask':<14}{'precision':>11}{'recall':>9}{'IoU':>8}{'frames scored':>15}")
        for k in ("owl", "vlm", "vlm_nomotion"):
            print(f"  {k:<14}{med(rs, k + '_precision'):>11.3f}{med(rs, k + '_recall'):>9.3f}{med(rs, k + '_iou'):>8.3f}"
                  f"{sum(r[k + '_frames'] for r in rs):>15}")
    better = sum(1 for r in rows if r["vlm_iou"] is not None and r["owl_iou"] is not None and r["vlm_iou"] > r["owl_iou"])
    both = sum(1 for r in rows if r["vlm_iou"] is not None and r["owl_iou"] is not None)
    print(f"\nIoU higher with the VLM masks on {better} of {both} cameras scored under both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["propose", "select", "refine", "assemble", "eval"])
    ap.add_argument("--in_dir", default="")
    ap.add_argument("--work_dir", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--frames", default="all", help="all, or frame indices such as 0,4,7")
    ap.add_argument("--limit", type=int, default=0, help="first N cameras only (0 = all)")
    ap.add_argument("--points_per_batch", type=int, default=64)
    ap.add_argument("--dump_marked", type=int, default=1, help="save one numbered frame per camera next to the proposals")
    ap.add_argument("--sam", default="", help="SAM checkpoint; default sam-vit-huge on GPU, sam-vit-base on CPU")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2], help="select: 1 = whole frames, 2 = the zoomed crops from refine")
    ap.add_argument("--max_mixed", type=int, default=3, help="refine: at most this many mixed regions per frame")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mixed", type=int, default=2, help="regions the VLM calls part robot, part background: 0 = leave out, "
                    "1 = take whole, 2 = take the pixels inside them that moved between frames (default)")
    ap.add_argument("--menagerie", default="/content/mujoco_menagerie")
    ap.add_argument("--convention", default="cam2base", choices=["cam2base", "base2cam"])
    a = ap.parse_args()
    a.work_dir = a.work_dir or os.path.join(a.in_dir or a.out_dir, "vlm_work")
    dict(propose=propose, select=select, refine=refine, assemble=assemble, eval=evaluate)[a.mode](a)


if __name__ == "__main__":
    main()
