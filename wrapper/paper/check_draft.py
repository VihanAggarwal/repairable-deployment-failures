"""Sanity checks on the LaTeX draft when no local LaTeX is installed: every cited key exists, braces and environments
balance, every figure exists, no em-dash.  uv run --no-project python wrapper/paper/check_draft.py"""
import io
import os
import re

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iclr")
t = io.open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
b = io.open(os.path.join(HERE, "refs.bib"), encoding="utf-8").read()
BS = chr(92)
keys = set(re.findall(r"@\w+\{([^,]+),", b))
cited = set(k.strip() for m in re.findall(re.escape(BS) + r"cite[tp]?\{([^}]+)\}", t) for k in m.split(","))
print("cited:", len(cited), "| missing from the bibliography:", sorted(cited - keys))
print("braces balanced:", t.count("{") == t.count("}"), "| begin / end:", t.count(BS + "begin{"), t.count(BS + "end{"))
figs = re.findall(r"figs/[\w.]+", t)
print("figures:", [(f, os.path.exists(os.path.join(HERE, f))) for f in figs])
print("em-dash present:", chr(0x2014) in t or chr(0x2014) in b, "| pending slots:", t.count(BS + "pending{") - 1)
labels = set(re.findall(re.escape(BS) + r"label\{([^}]+)\}", t)); refs = set(re.findall(re.escape(BS) + r"ref\{([^}]+)\}", t))
print("refs without a label:", sorted(refs - labels))
unverified = re.findall(r"@\w+\{([^,]+),\s*author = \{\{\(authors to verify\)\}\}", b)
print("bibliography entries with unverified authors:", len(unverified), unverified)
