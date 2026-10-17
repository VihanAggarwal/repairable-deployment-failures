"""Black-and-white box diagrams in the draw.io idiom (white boxes, 1 pt black borders, black arrows, one sans-serif face).
Each diagram is written twice from the same spec: a vector PDF (+ PNG preview) for the paper, and a .drawio file that
opens in diagrams.net for hand editing.

    uv run --no-project --with matplotlib python wrapper/paper/make_bw_diagrams.py
"""
import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "iclr", "figs")
DIO = os.path.join(HERE, "drawio")
os.makedirs(DIO, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7.5,
                     "text.color": "#000000", "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
U = 10  # 1 spec unit = 10 draw.io px


class D:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.nodes, self.edges, self.texts = name, w, h, {}, [], []

    def box(self, key, x, y, w, h, text, shape="box", bold=False, dashed=False, fill="white"):
        self.nodes[key] = dict(x=x, y=y, w=w, h=h, text=text, shape=shape, bold=bold, dashed=dashed, fill=fill)

    def edge(self, a, b, label="", side=("r", "l"), via=None, dashed=False, p0=None):
        self.edges.append(dict(a=a, b=b, label=label, side=side, via=via, dashed=dashed, p0=p0))

    def text(self, x, y, s, size=7, ha="left", italic=False):
        self.texts.append(dict(x=x, y=y, s=s, size=size, ha=ha, italic=italic))

    def port(self, key, side):
        n = self.nodes[key]
        return {"l": (n["x"], n["y"] + n["h"] / 2), "r": (n["x"] + n["w"], n["y"] + n["h"] / 2),
                "t": (n["x"] + n["w"] / 2, n["y"]), "b": (n["x"] + n["w"] / 2, n["y"] + n["h"])}[side]

    # ---- matplotlib
    def render(self, width_in=9.0):
        fig, ax = plt.subplots(figsize=(width_in, width_in * self.h / self.w))
        ax.set_xlim(0, self.w); ax.set_ylim(self.h, 0); ax.axis("off")
        for n in self.nodes.values():
            ls = (0, (3, 2)) if n["dashed"] else "-"
            if n["shape"] == "diamond":
                cx, cy = n["x"] + n["w"] / 2, n["y"] + n["h"] / 2
                ax.add_patch(patches.Polygon([(cx, n["y"]), (n["x"] + n["w"], cy), (cx, n["y"] + n["h"]), (n["x"], cy)], closed=True, fc=n["fill"], ec="black", lw=0.9, ls=ls, zorder=2))
            elif n["shape"] == "rounded":
                ax.add_patch(patches.FancyBboxPatch((n["x"], n["y"]), n["w"], n["h"], boxstyle="round,pad=0,rounding_size=0.8", fc=n["fill"], ec="black", lw=0.9, ls=ls, zorder=2))
            elif n["shape"] == "doc":
                ax.add_patch(patches.Rectangle((n["x"], n["y"]), n["w"], n["h"], fc=n["fill"], ec="black", lw=0.9, ls=ls, zorder=2))
                ax.plot([n["x"], n["x"] + n["w"]], [n["y"] + n["h"] - 1.2] * 2, color="black", lw=0.5, zorder=3)
            else:
                ax.add_patch(patches.Rectangle((n["x"], n["y"]), n["w"], n["h"], fc=n["fill"], ec="black", lw=0.9, ls=ls, zorder=2))
            ax.text(n["x"] + n["w"] / 2, n["y"] + n["h"] / 2, n["text"], ha="center", va="center", fontsize=7.5, fontweight="bold" if n["bold"] else "normal", zorder=4, linespacing=1.25)
        for e in self.edges:
            p0, p1 = e["p0"] or self.port(e["a"], e["side"][0]), self.port(e["b"], e["side"][1])
            pts = [p0] + (e["via"] or []) + [p1]
            ls = (0, (3, 2)) if e["dashed"] else "-"
            for u, v in zip(pts[:-2], pts[1:-1]):
                ax.plot([u[0], v[0]], [u[1], v[1]], color="black", lw=0.9, ls=ls, zorder=1, solid_capstyle="round")
            ax.annotate("", pts[-1], pts[-2], arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=8), zorder=3)
            if e["label"]:
                mx, my = (pts[len(pts) // 2 - 1][0] + pts[len(pts) // 2][0]) / 2, (pts[len(pts) // 2 - 1][1] + pts[len(pts) // 2][1]) / 2
                ax.text(mx, my - 0.6, e["label"], ha="center", va="bottom", fontsize=6.3, zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))
        for t in self.texts:
            ax.text(t["x"], t["y"], t["s"], ha=t["ha"], va="center", fontsize=t["size"], style="italic" if t["italic"] else "normal", linespacing=1.3)
        fig.savefig(os.path.join(FIGS, self.name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(os.path.join(FIGS, self.name + ".png"), bbox_inches="tight", pad_inches=0.04, dpi=220)
        plt.close(fig)

    # ---- draw.io XML
    def drawio(self):
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        ids = {}
        for i, (k, n) in enumerate(self.nodes.items(), start=2):
            ids[k] = str(i)
            style = {"box": "rounded=0;whiteSpace=wrap;html=1;", "rounded": "rounded=1;arcSize=12;whiteSpace=wrap;html=1;",
                     "diamond": "rhombus;whiteSpace=wrap;html=1;", "doc": "shape=note;whiteSpace=wrap;html=1;size=8;"}[n["shape"]]
            style += "strokeColor=#000000;fillColor=%s;fontSize=10;fontFamily=Helvetica;" % ("#ffffff" if n["fill"] == "white" else "#f2f2f2")
            if n["bold"]:
                style += "fontStyle=1;"
            if n["dashed"]:
                style += "dashed=1;"
            cells.append('<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/></mxCell>'
                         % (ids[k], html.escape(n["text"]).replace("\n", "&lt;br&gt;"), style, n["x"] * U, n["y"] * U, n["w"] * U, n["h"] * U))
        for j, e in enumerate(self.edges, start=len(cells) + 1):
            style = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#000000;endArrow=block;endFill=1;fontSize=9;fontFamily=Helvetica;"
            if e["dashed"]:
                style += "dashed=1;"
            pts = "".join('<mxPoint x="%d" y="%d"/>' % (v[0] * U, v[1] * U) for v in (e["via"] or []))
            cells.append('<mxCell id="e%d" value="%s" style="%s" edge="1" parent="1" source="%s" target="%s"><mxGeometry relative="1" as="geometry"><Array as="points">%s</Array></mxGeometry></mxCell>'
                         % (j, html.escape(e["label"]), style, ids[e["a"]], ids[e["b"]], pts))
        for k, t in enumerate(self.texts, start=1):
            cells.append('<mxCell id="t%d" value="%s" style="text;html=1;strokeColor=none;fillColor=none;align=%s;verticalAlign=middle;fontSize=9;fontFamily=Helvetica;%s" vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="20" as="geometry"/></mxCell>'
                         % (k, html.escape(t["s"]).replace("\n", "&lt;br&gt;"), t["ha"], "fontStyle=2;" if t["italic"] else "", t["x"] * U, t["y"] * U - 10, 300))
        xml = ('<mxfile host="app.diagrams.net"><diagram name="%s"><mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d">'
               '<root>%s</root></mxGraphModel></diagram></mxfile>') % (self.name, self.w * U + 40, self.h * U + 40, "".join(cells))
        open(os.path.join(DIO, self.name + ".drawio"), "w", encoding="utf-8").write(xml)


def pipeline():
    d = D("bw_pipeline", 100, 33)
    d.box("body", 2, 2, 96, 7, "the robot's own body: joint encoders, kinematic model, renderer, silhouette in the image", "rounded", dashed=True)
    for k, x, t in (("m", 2, "MEASURE\n3 arm poses\ncamera pose: silhouette fit\njoint offset: encoders\nlight, noise: body pixels"),
                    ("p", 27, "PREDICT\nrecover = ident x invert\nident: can the body\nmeasure the change\ninvert: view kept after undo"),
                    ("u", 52, "UNDO\nservo to the training pose\nre-render into the\ntraining camera\nlight map, denoise"),
                    ("v", 77, "VERIFY\nwarp the observed\nsilhouette into the\ntraining view; kinematics\nsay where it must land")):
        d.box(k, x, 12, 21, 13, t)
    d.edge("m", "p"); d.edge("p", "u"); d.edge("u", "v")
    for k in ("m", "p", "u", "v"):
        cx = d.nodes[k]["x"] + d.nodes[k]["w"] / 2
        d.edge("body", k, side=("b", "t"), dashed=True, p0=(cx, 9))
    d.box("pol", 52, 29, 21, 3.5, "frozen policy: no weights change", "rounded", bold=True)
    d.edge("u", "pol", side=("b", "t"), label="corrected frame")
    return d


def ladder():
    d = D("bw_ladder", 100, 36)
    d.box("start", 1, 13, 16, 8, "raw camera failure\n(the policy fails on\nthe deployed frame)", "rounded", bold=True)
    for k, x, t in (("q1", 19, "measured\npose\nrepairs?"), ("q2", 43, "true\npose\nrepairs?"), ("q3", 67, "perfect\nrender\nrepairs?")):
        d.box(k, x, 9, 20, 16, t, "diamond")
    for k, x, t in (("b1", 19, "recovered by\nmeasurement\n(OFT 61 / 100)"), ("b2", 43, "not identified:\ncalibration failed\n(17 / 100)"),
                    ("b3", 67, "not invertible:\nview lost\n(21 / 100)")):
        d.box(k, x, 28, 20, 7, t)
    d.box("b4", 89, 12.5, 10, 9, "the policy\nitself\n(1 / 100)")
    d.edge("start", "q1"); d.edge("q1", "q2", label="no"); d.edge("q2", "q3", label="no"); d.edge("q3", "b4", label="no")
    d.edge("q1", "b1", side=("b", "t"), label="yes"); d.edge("q2", "b2", side=("b", "t"), label="yes"); d.edge("q3", "b3", side=("b", "t"), label="yes")
    d.text(2, 4, "same task, same seed, one more cause removed at each step (left to right)", 7, italic=True)
    d.text(2, 7, "arms: raw, wrapper with the pose the robot measured, wrapper with the simulator's true pose, the training camera's own render", 6.5)
    return d


def lazy():
    d = D("bw_lazy", 100, 33)
    d.box("obs", 2, 11, 16, 8, "one frame, arm at rest\n(no motion)", "rounded", bold=True)
    d.box("pred", 22, 11, 18, 8, "render the silhouette the\nTRAINING camera predicts\nfor the current joints")
    d.box("res", 44, 9, 16, 12, "residual\nr > 1.5 ?", "diamond")
    d.box("sweep", 66, 2, 20, 8, "measure: 3-pose sweep,\ncalibrate, settle at rest")
    d.box("skip", 66, 20, 20, 8, "skip: arm never moves,\nepisode starts as raw")
    d.box("pol", 90, 11, 8, 8, "frozen\npolicy")
    d.edge("obs", "pred"); d.edge("pred", "res")
    d.edge("res", "sweep", side=("t", "l"), via=[(52, 6)], label="yes: camera moved")
    d.edge("res", "skip", side=("b", "l"), via=[(52, 24)], label="no: camera unmoved")
    d.edge("sweep", "pol", side=("r", "t"), via=[(94, 6)]); d.edge("skip", "pol", side=("r", "b"), via=[(94, 24)])
    d.text(2, 31.5, "on every logged unit: r = 0.00 when the camera is unmoved (n = 60), r > 1.9 when it moved (n = 70); harm on pi0.5 drops from 10 to 3 of 30", 6.5)
    return d


def score():
    d = D("bw_score", 100, 36)
    d.box("eta", 2, 14, 14, 8, "nuisance η\n(camera pose, light,\njoint offset, noise)", "rounded")
    d.box("z", 19, 14, 18, 8, "self-observations\nz(q_k, η), k = 1..3\n(silhouette, colour, joints)")
    d.box("J", 40, 4, 15, 8, "Jacobian\nJ = dz / dη\n(finite differences)")
    d.box("id", 58, 3, 19, 10, "ident =\nσ_min(J) / σ_min(J at training)\nclipped to 1")
    d.box("inv", 58, 23, 19, 10, "invert =\ncamera: view kept (coverage)\nlight: shading consistency\nnoise: denoiser gain\njoint: 1, scene content: 0")
    d.box("undo", 40, 24, 15, 8, "the undo, applied to\nthe robot's own view")
    d.box("rec", 81, 13, 18, 10, "recover = ident × invert\npredicts repair:\nAUROC 0.82 (OFT)\n0.81 (π0.5, held out)", bold=True)
    d.edge("eta", "z"); d.edge("z", "J", side=("r", "l"), via=[(38, 18), (38, 8)]); d.edge("z", "undo", side=("r", "l"), via=[(38, 18), (38, 28)])
    d.edge("J", "id"); d.edge("undo", "inv"); d.edge("id", "rec", side=("r", "l"), via=[(78.5, 8), (78.5, 18)]); d.edge("inv", "rec", side=("r", "l"), via=[(78.5, 28), (78.5, 18)])
    d.text(2, 34, "Prop. 1: calibration error ≥ noise / σ_min(J).   Prop. 2: restoration error ≥ (1 - coverage) × prior error.   No policy query anywhere.", 6.5)
    return d


def loop():
    d = D("bw_loop", 100, 35.5)
    d.box("cam", 2, 4, 14, 7, "camera\n(moved)")
    d.box("enc", 2, 22, 14, 7, "joint encoders")
    d.box("kin", 20, 22, 16, 7, "kinematic model\n+ renderer")
    d.box("cal", 40, 13, 18, 8, "silhouette fit:\ncamera pose, error\nabout 0.3 cm")
    d.box("rep", 62, 4, 16, 7, "re-render into\nthe training camera")
    d.box("pol", 82, 4, 16, 7, "frozen policy", bold=True)
    d.box("act", 82, 22, 16, 7, "robot executes\nthe action")
    d.box("chk", 62, 22, 16, 7, "self-check: F1 of\nsilhouette vs kinematics")
    d.edge("cam", "rep", label="deployed frame"); d.edge("cam", "cal", side=("b", "l"), via=[(9, 17)])
    d.edge("enc", "kin"); d.edge("kin", "cal", side=("r", "b"), via=[(49, 25.5)])
    d.edge("cal", "rep", side=("r", "b"), via=[(70, 17)], label="pose")
    d.edge("rep", "pol", label="corrected frame"); d.edge("pol", "act", side=("b", "t"), label="action")
    d.edge("act", "enc", side=("b", "b"), via=[(90, 33), (9, 33)], label="new joint state")
    d.edge("cal", "chk", side=("r", "l"), via=[(60, 17), (60, 25.5)], dashed=True)
    d.text(2, 15, "measure", 6.5, italic=True); d.text(62, 14, "undo", 6.5, italic=True); d.text(62, 30.8, "verify (logging and re-measure trigger only)", 6.5, italic=True)
    return d


if __name__ == "__main__":
    import sys
    if "--force-unused" not in sys.argv:
        sys.exit("make_bw_diagrams.py (bw_*): its outputs are not in the paper (dropped 2026-09-24); pass --force-unused to rebuild them")
    for mk in (pipeline, ladder, lazy, score, loop):
        d = mk()
        d.render()
        d.drawio()
        print("wrote", d.name)
