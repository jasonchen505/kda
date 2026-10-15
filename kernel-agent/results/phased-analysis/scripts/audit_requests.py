"""Summarize local request traces without treating them as generation outcomes."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from kernel_agent.transport import PROPOSER_PREFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    files = {}
    for directory in args.directories:
        for path in sorted(directory.glob("*.json")):
            raw = path.read_bytes()
            request = json.loads(raw)
            proposer = any(m.get("role") == "system" and isinstance(m.get("content"), str)
                           and m["content"].startswith(PROPOSER_PREFIX) for m in request.get("messages", []))
            policy = request.get("policy", "legacy-passthrough")
            groups[(policy, "proposer" if proposer else "background")].append(request)
            files[str(path)] = hashlib.sha256(raw).hexdigest()
    rows = []
    for (policy, scope), requests in sorted(groups.items()):
        rows.append({"policy": policy, "scope": scope, "requests": len(requests),
                     "offered_tools_before": dict(Counter(r.get("tools_before", len(r.get("tools", []))) for r in requests)),
                     "forwarded_tools_after": dict(Counter(len(r.get("tools", [])) for r in requests)),
                     "schema_requests": sum("response_format" in r for r in requests),
                     "models": sorted({r["model"] for r in requests}),
                     "temperatures": sorted({r["temperature"] for r in requests}),
                     "seeds": sorted({r["seed"] for r in requests})})
    args.output.write_text(json.dumps({"scope": "request audit includes background titles and diagnostics; not outcome counts",
                                      "groups": rows, "trace_sha256": files}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
