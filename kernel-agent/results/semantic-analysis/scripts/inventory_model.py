"""Identify a cached model without claiming an unverified upstream revision."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.directory / "config.json").read_text())
    text = config.get("text_config", config)
    metadata_files = ["config.json", "generation_config.json", "tokenizer_config.json",
                      "tokenizer.json", "model.safetensors.index.json"]
    metadata_hashes = {name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest()
                       for name in metadata_files if (args.directory / name).exists()}
    shards = [{"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
              for path in sorted(args.directory.glob("*.safetensors"))]
    result = {"directory_alias": args.directory.name, "directory": str(args.directory.resolve()),
              "upstream_revision": None, "weight_hashes_verified": False,
              "model_type": config.get("model_type"), "architectures": config.get("architectures"),
              "text_config": {key: text.get(key) for key in ("num_hidden_layers", "hidden_size", "intermediate_size")},
              "quantization_method": config.get("quantization_config", {}).get("quant_method"),
              "metadata_sha256": metadata_hashes, "weight_shards": shards}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "weight_shards"}, indent=2))


if __name__ == "__main__":
    main()
