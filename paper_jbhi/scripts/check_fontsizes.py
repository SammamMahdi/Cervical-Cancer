"""Audit rendered font sizes: authored pt x (displayed width / authored figure width).

Reports the smallest text in each figure as it appears in the compiled PDF, so labels that end up
below ~7 pt can be found and fixed.  Run:  python paper_jbhi/scripts/check_fontsizes.py
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
COL, TEXT = 3.40, 7.16          # IEEEtran journal: column and text width in inches

tex = open(os.path.join(HERE, "..", "main.tex"), encoding="utf-8").read()
src = open(os.path.join(HERE, "make_figures.py"), encoding="utf-8").read()
src += open(os.path.join(HERE, "gradcam.py"), encoding="utf-8").read()
src += open(os.path.join(HERE, "make_figures2.py"), encoding="utf-8").read()

wide = [(m.start(), tex.index(r"\end{figure*}", m.start())) for m in re.finditer(re.escape(r"\begin{figure*}"), tex)]


def is_wide(pos):
    return any(a < pos < b for a, b in wide)


# authored figure width + smallest explicit fontsize, per figure stem (parsed from the figure functions)
blocks = re.split(r"\ndef ", src)
authored = {}
for b in blocks:
    for stem in re.findall(r'save\(fig, f?"([a-z_0-9{}.\'()\\ ]+)"', b):
        stem = stem.split("{")[0].rstrip("_")
        # figures are authored through pubstyle.size(), which maps them onto the printed width
        fs = [(TEXT if float(w) > 8 else COL, h)
              for w, h in re.findall(r"figsize=size\(([0-9.]+),\s*([0-9.]+)\)", b)]
        fs += re.findall(r"figsize=\(([0-9.]+),\s*([0-9.]+)\)", b)
        pts = [float(x) for x in re.findall(r"fontsize=([0-9.]+)", b)]
        if fs:
            authored[stem] = (float(fs[0][0]), min(pts) if pts else None)

RC = dict(tick=16.0, legend=14.5, label=19.0, title=18.0)   # keep in sync with pubstyle.py
rows = []
for w, f in re.findall(r"includegraphics\[width=([^\]]+)\]\{([^}]+)\}", tex):
    pos = tex.index("includegraphics[width=" + w + "]{" + f + "}")
    base = TEXT if is_wide(pos) else COL
    m = re.match(r"([0-9.]*)\\\\?(columnwidth|textwidth)", w)
    frac = float(m.group(1)) if m and m.group(1) else 1.0
    disp = base * frac
    key = next((k for k in authored if f.startswith(k)), None)
    if key is None:
        rows.append((f, disp, None, None, None)); continue
    aw, minpt = authored[key]
    scale = disp / aw
    smallest = min([RC["tick"], RC["legend"]] + ([minpt] if minpt else []))
    rows.append((f, disp, aw, scale, smallest * scale))

print(f"{'figure':34s} {'disp_in':>7s} {'auth_in':>7s} {'scale':>6s} {'min_pt':>7s}")
bad = 0
for f, disp, aw, scale, eff in sorted(rows, key=lambda r: (r[4] is not None, r[4])):
    if eff is None:
        print(f"{f:34s} {disp:7.2f} {'?':>7s} {'?':>6s} {'?':>7s}"); continue
    flag = "  <-- too small" if eff < 7 else ""
    bad += eff < 7
    print(f"{f:34s} {disp:7.2f} {aw:7.2f} {scale:6.2f} {eff:7.1f}{flag}")
print(f"\n{bad} figures with text below 7 pt")
