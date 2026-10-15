"""Only the existing local provider is enabled. No paid API fallback."""
import json
import os
from pathlib import Path

from hmz.sdk import Hmz

ROOT = Path(__file__).resolve().parents[1]
os.environ["OPENCODE_CONFIG_CONTENT"] = (ROOT / "data/opencode-local.json").read_text()
os.environ["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
os.environ["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "true"
result = Hmz(ROOT).run(str(ROOT / "flows/smoke.py"),
                       'Do not call tools. Reply with exactly {"local_kernel_agent":"ready"}.',
                       agents={"builder": "opencode/kernel-local/kernel-qwen3-4b"},
                       budget={"duration": 180}).run()
(ROOT / "results/smoke.json").write_text(json.dumps({"reply": result}, indent=2))
print(result)
