"""NumPy 2 fixes for the system python used by the LeRobot (pi0.5) notebook; the venvs pin numpy<2 and never hit them.

1. robosuite's segmentation renderer computes `rgb[:, :, 1] * (2**8)` on a uint8 array, which NumPy 2 refuses
   (OverflowError: Python integer 256 out of bounds for uint8). Cast to int64 first.
2. LIBERO-plus uses aliases NumPy 2 removed. The one that actually bites is `np.float_` in `plasma_fractal`, which
   the fog corruption calls for sensor_noise tasks of severity 31 to 40, so `env.reset()` dies on those tasks only.
   The trailing underscore matters: `np.float_` is a different name from `np.float`, and a pattern written for the
   bare alias silently leaves it in place, which is how this reached a paid run twice.
Idempotent; run once after installing robosuite and LIBERO-plus.
"""
import glob
import os
import re
import sys

# name as it appears after "np." -> the replacement text. Bare aliases first, then the trailing-underscore forms.
ALIASES = {"float": "np.float64", "int": "np.int64", "bool": "np.bool_", "object": "np.object_", "str": "np.str_",
           "float_": "np.float64", "complex_": "np.complex128", "unicode_": "np.str_", "string_": "np.bytes_",
           "bool8": "np.bool_", "object0": "np.object_", "int0": "np.intp", "uint0": "np.uintp",
           "longfloat": "np.longdouble", "singlecomplex": "np.complex64", "cfloat": "np.complex128",
           "clongfloat": "np.clongdouble", "longcomplex": "np.clongdouble",
           "NaN": "np.nan", "NAN": "np.nan", "Inf": "np.inf", "Infinity": "np.inf", "infty": "np.inf",
           "PINF": "np.inf", "NINF": "(-np.inf)"}


def patch_aliases(text):
    """`np.float_` and `np.float)` become `np.float64`; `np.float64` and `np.float32` are left alone. The lookahead
    is what draws that line, so every key must be matched by its own full name."""
    for name, repl in ALIASES.items():
        text = re.sub(r"\bnp\." + re.escape(name) + r"(?![\w.])", repl, text)
    return text


def remaining(text):
    return sorted({m for m in re.findall(r"\bnp\.\w+", text) if m[3:] in ALIASES})


def main():
    import robosuite

    old = "seg_img = rgb_img[:, :, 0] + rgb_img[:, :, 1] * (2**8) + rgb_img[:, :, 2] * (2**16)"
    new = "rgb_img = rgb_img.astype(np.int64); " + old
    path = os.path.join(os.path.dirname(robosuite.__file__), "utils", "binding_utils.py")
    src = open(path).read()
    if new in src:
        print("robosuite segmentation already NumPy 2 safe")
    elif old in src:
        open(path, "w").write(src.replace(old, new))
        print("robosuite segmentation patched for NumPy 2")
    else:
        raise SystemExit(f"expected line not found in {path}; robosuite version changed, inspect read_pixels")

    root = os.environ.get("LIBERO_PLUS", "/content/LIBERO-plus")
    files = glob.glob(os.path.join(root, "libero", "**", "*.py"), recursive=True)
    if not files:
        raise SystemExit(f"LIBERO-plus not found at {root}; set LIBERO_PLUS or clone it first")
    touched, left = 0, {}
    for fp in files:
        s = open(fp).read()
        t = patch_aliases(s)
        if t != s:
            open(fp, "w").write(t)
            touched += 1
            print("patched legacy NumPy aliases in", os.path.relpath(fp, root))
        rest = remaining(t)
        if rest:
            left[os.path.relpath(fp, root)] = rest
    # 3. fog on frames larger than 256 px: `plasma_fractal` builds a 256 x 256 map and `fog` crops it to the frame, so a
    #    360 x 360 frame (what lerobot-eval renders) cannot be broadcast and `env.reset()` dies on every fog task
    #    (2026-09-19, pi0.5 baseline batch 004). The map grows to the next power of two ONLY when the frame needs it,
    #    so 256 px frames, which every wrapper run uses, get bit-identical fog.
    fog_old = "plasma_fractal(wibbledecay=c[1])[:height_x, :weight_x]"
    fog_new = ("plasma_fractal(mapsize=int(2 ** np.ceil(np.log2(max(256, height_x, weight_x)))), "
               "wibbledecay=c[1])[:height_x, :weight_x]")
    for fp in files:
        s = open(fp).read()
        if fog_old in s:
            open(fp, "w").write(s.replace(fog_old, fog_new))
            print("fog map sized to the frame in", os.path.relpath(fp, root))
    print(f"LIBERO-plus: {len(files)} files scanned, {touched} patched")
    if left:                                    # silent no-ops are the failure mode this whole file exists for
        for fp, names in sorted(left.items()):
            print("STILL BROKEN", fp, names, file=sys.stderr)
        raise SystemExit("removed NumPy aliases survived the patch; fix ALIASES before running episodes")
    print("no removed NumPy aliases left in LIBERO-plus")


if __name__ == "__main__":
    main()
