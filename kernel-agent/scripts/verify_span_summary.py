"""Verify compact published counts, not absent server-only raw model/GPU evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify(path):
    s = json.loads(Path(path).read_text())
    if digest({k: v for k, v in s.items() if k != "summary_sha256"}) != s["summary_sha256"]:
        raise ValueError("compact summary changed")
    if s["scheduled"] != 66 or len(s["rows"]) != 66 or s["dispatched_requests"] > 140:
        raise ValueError("compact denominator/budget changed")
    keys = {(r["block"], r["arm"]) for r in s["rows"]}
    if len(keys) != 66 or len({b for b, _ in keys}) != 22:
        raise ValueError("duplicate/missing paired outcomes")
    for st in s["strata"]:
        for cell in st["cells"]:
            rows = [r for r in s["rows"] if r["stratum"] == st["stratum"] and r["arm"] == cell["arm"]]
            if len(rows) != cell["scheduled"] or sum(r["correct"] for r in rows) != cell["correct"]:
                raise ValueError("compact cell differs from recorded rows")
    if s["status"] == "complete" and (s["resolved"] != 66 or any(r["status"] not in {"complete", "failed", "interrupted_unknown"} for r in s["rows"])):
        raise ValueError("compact completion changed")
    return {"status": "verified", "resolved": s["resolved"], "summary_sha256": s["summary_sha256"],
            "scope": "Published summary self-hash and count consistency only; full raw reconstruction runs on the server",
            "remote_raw_manifest_sha256": s["provenance"]["raw_manifest_sha256"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    print(json.dumps(verify(p.parse_args().path)))
