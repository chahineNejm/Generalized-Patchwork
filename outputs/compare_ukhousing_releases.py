import csv
import hashlib
import json
from pathlib import Path


root = Path("data/ukhousing_verified_20261112")
previous = {}
with Path("data/ukhousing/pp-2018.csv").open(encoding="utf-8", newline="") as stream:
    for row in csv.reader(stream, strict=True):
        assert len(row) == 16
        assert row[0] not in previous
        previous[row[0]] = hashlib.sha256(json.dumps(row).encode()).digest()
old_count = len(previous)
added = changed = rows = 0
seen = set()
with (root / "pp-2018.csv").open(encoding="utf-8", newline="") as stream:
    for row in csv.reader(stream, strict=True):
        assert len(row) == 16
        assert row[2].startswith("2018-")
        assert int(row[1]) > 0
        assert row[0] not in seen
        seen.add(row[0])
        rows += 1
        if row[0] not in previous:
            added += 1
        else:
            changed += previous.pop(row[0]) != hashlib.sha256(json.dumps(row).encode()).digest()
report = {"old_rows": old_count, "new_rows": rows, "added_ids": added,
          "removed_ids": len(previous), "changed_existing_records": changed,
          "checks": "Unique IDs, exact 16 columns, 2018 sale dates, positive prices"}
(root / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
