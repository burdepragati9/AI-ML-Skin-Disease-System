path = r"c:\Updated_Project\backend\services\detect_face.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

# Fix the line that got mangled to column 0
for i, line in enumerate(lines):
    if line.strip() == 'result["detector_ok"] = True' and not line.startswith(" "):
        lines[i] = "        " + line
        print("Fixed line", i)

with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("COMPILE OK")
except py_compile.PyCompileError as e:
    print("COMPILE ERROR:", e)
