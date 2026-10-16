"""Read the public installed API inventory; never invoke a candidate or model."""
import json
import sys

import torch
import triton
import triton.language as tl

if __name__ == "__main__":
    value = {"schema_version": 1, "torch": torch.__version__, "triton": triton.__version__,
             "python": sys.version.split()[0], "module": tl.__name__,
             "language_symbols": sorted(n for n in dir(tl) if not n.startswith("__")),
             "scope": "Installed direct language attributes; no model execution or CUDA evaluation"}
    assert value["torch"] == "2.6.0+cu124" and value["triton"] == "3.2.0"
    print(json.dumps(value, indent=2))
