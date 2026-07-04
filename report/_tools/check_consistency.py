import re, glob, os
os.chdir(os.path.join(os.path.dirname(__file__), "..", "cvpr_paper", "sections"))

print("=== caption openings (should all be \\textbf{) ===")
bad = 0
for fn in sorted(glob.glob("*.tex")):
    txt = open(fn, encoding="utf-8").read()
    for m in re.finditer(r"\\caption\{", txt):
        seg = txt[m.end():m.end()+12].lstrip()
        if not seg.startswith("\\textbf{"):
            bad += 1
            print(f"  NOT BOLD {fn}: ...{seg!r}")
print("  ALL BOLD" if bad == 0 else f"  {bad} non-bold captions")

print("=== stray \\best in prose (lines without a tabular &) ===")
n = 0
for fn in sorted(glob.glob("*.tex")):
    for i, line in enumerate(open(fn, encoding="utf-8"), 1):
        if "\\best{" in line and "&" not in line:
            n += 1
            print(f"  {fn}:{i}: {line.strip()[:80]}")
print("  none" if n == 0 else f"  {n} found")

print("=== raw unicode dashes ===")
n = 0
for fn in sorted(glob.glob("*.tex")):
    txt = open(fn, encoding="utf-8").read()
    for ch, name in [("–", "en-dash"), ("—", "em-dash")]:
        if ch in txt:
            n += 1
            print(f"  {fn}: {name}")
print("  none" if n == 0 else f"  {n} files")

print("=== raw dataset names typed instead of macro ===")
n = 0
names = re.compile(r"(?<![\\A-Za-z])(Herlev|SIPaKMeD|APACC|HiCervix)\b")
for fn in sorted(glob.glob("*.tex")):
    for i, line in enumerate(open(fn, encoding="utf-8"), 1):
        if "cite" in line or "includegraphics" in line:
            continue
        for m in names.finditer(line):
            n += 1
            print(f"  {fn}:{i}: {m.group()} | {line.strip()[:65]}")
print("  none" if n == 0 else f"  {n} found")
