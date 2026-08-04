import pathlib, sys

def rewrite(path, reps):
    p = pathlib.Path(path)
    c = p.read_text(encoding="utf-8")
    changed = False
    for old, new in reps:
        if old in c and old != new:
            c = c.replace(old, new)
            changed = True
    if changed:
        p.write_text(c, encoding="utf-8")

if __name__ == "__main__":
    f = sys.argv[1]
    reps = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|||", 1)
        if len(parts) == 2:
            reps.append((parts[0], parts[1]))
    rewrite(f, reps)