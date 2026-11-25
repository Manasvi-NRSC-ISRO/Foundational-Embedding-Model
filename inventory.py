"""What data do we have?  Counts zips per sensor and month, and which are not in the stack yet.

    python inventory.py

Reads bhoonidhi_year/*/*.zip (names only, nothing is opened) and outputs/scene_coverage.csv.
Also writes outputs/inventory.csv (one row per zip).
"""
import csv
import glob
import os
import re
import shutil
from collections import defaultdict

DATA = "bhoonidhi_year"
DATE_RE = re.compile(r"(\d{2})([A-Z]{3})(20\d{2})", re.I)
MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def sensor_of(folder):
    f = folder.upper()
    return "AWIFS" if "AWIFS" in f else ("LISS3" if "LISS3" in f else ("EOS04" if "EOS-04" in f else f))


def main():
    done = set()
    p = os.path.join("outputs", "scene_coverage.csv")
    if os.path.exists(p):
        done = {r["scene"] for r in csv.DictReader(open(p)) if r["status"] == "ok"}
    rows = []
    for z in sorted(glob.glob(os.path.join(DATA, "*", "*.zip"))):
        name = os.path.splitext(os.path.basename(z))[0]
        m = DATE_RE.search(name)
        date = "%s-%02d-%s" % (m.group(3), MON[m.group(2).upper()], m.group(1)) if m else "?"
        rows.append(dict(sensor=sensor_of(os.path.basename(os.path.dirname(z))), date=date, month=date[:7],
                         scene=name, gb=os.path.getsize(z) / 1e9, in_stack=name in done))
    if not rows:
        raise SystemExit("No zips found in %s/*/" % DATA)
    g = defaultdict(lambda: [0, 0.0, 0, set()])
    for r in rows:
        x = g[(r["sensor"], r["month"])]
        x[0] += 1; x[1] += r["gb"]; x[2] += (not r["in_stack"]); x[3].add(r["date"])
    print("%-6s %-8s %5s %6s %6s %s" % ("sensor", "month", "zips", "dates", "GB", "NOT in stack yet"))
    for (s, mo), (n, gb, new, dates) in sorted(g.items()):
        print("%-6s %-8s %5d %6d %6.1f %s" % (s, mo, n, len(dates), gb, new if new else "-"))
    tot = sum(r["gb"] for r in rows)
    print("\ntotal: %d zips, %.1f GB; not in stack yet: %d zips (%.1f GB)" % (
        len(rows), tot, sum(not r["in_stack"] for r in rows), sum(r["gb"] for r in rows if not r["in_stack"])))
    for s in sorted({r["sensor"] for r in rows}):
        d = sorted({r["date"] for r in rows if r["sensor"] == s})
        print("  %-6s %d dates, %s -> %s" % (s, len(d), d[0], d[-1]))
    free = shutil.disk_usage(".").free / 1e9
    print("free disk here: %.1f GB" % free)
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "inventory.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "date", "scene", "gb", "in_stack"])
        for r in rows:
            w.writerow([r["sensor"], r["date"], r["scene"], "%.2f" % r["gb"], r["in_stack"]])


if __name__ == "__main__":
    main()
