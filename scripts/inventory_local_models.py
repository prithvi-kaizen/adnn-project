"""Read installed model metadata only; never pull, generate or load tensors."""

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def ollama_request(endpoint, body=None):
    if endpoint not in {"/api/tags", "/api/show", "/api/ps", "/api/version"}:
        raise ValueError("Only read-only Ollama metadata endpoints are allowed")
    request = urllib.request.Request(
        "http://127.0.0.1:11434" + endpoint,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def ollama_inventory():
    result = []
    for model in ollama_request("/api/tags")["models"]:
        detail = ollama_request("/api/show", {"model": model["name"]})
        info = detail.get("model_info", {})
        result.append({
            "name": model["name"], "digest": model["digest"], "size_bytes": model["size"],
            "details": detail.get("details", {}), "capabilities": detail.get("capabilities", []),
            "architecture_metadata": {k: v for k, v in info.items() if k in {
                "general.architecture", "general.name", "general.parameter_count",
                "general.file_type", "general.quantization_version",
            } or k.endswith(".block_count") or k.endswith(".embedding_length")},
            "license_present": bool(detail.get("license")),
        })
    residents = [{k: v for k, v in model.items() if k in {
        "name", "model", "size", "size_vram", "digest", "context_length",
    }} for model in ollama_request("/api/ps").get("models", [])]
    return {"models": result, "resident_models": residents,
            "ollama_version": ollama_request("/api/version")}


def hf_inventory():
    result = []
    root = Path.home() / ".cache/huggingface/hub"
    for repo in sorted(root.glob("models--*")):
        item = {"model": repo.name.removeprefix("models--").replace("--", "/"), "snapshots": []}
        for snapshot in sorted((repo / "snapshots").glob("*")):
            if not snapshot.is_dir():
                continue
            files = [{"path": str(p.relative_to(snapshot)), "bytes": p.stat().st_size}
                     for p in snapshot.rglob("*") if p.is_file()]
            configuration = {}
            config = snapshot / "config.json"
            if config.is_file():
                obj = json.loads(config.read_text())
                configuration = {k: obj[k] for k in
                                 ("model_type", "architectures", "quantization", "quantization_config")
                                 if k in obj}
            item["snapshots"].append({"revision": snapshot.name, "files": files,
                                      "configuration_summary": configuration})
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Preserve the previous inventory; choose a new directory")
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "standard_Mac_model_cache_metadata_only",
              "ollama": ollama_inventory(), "hf_model_repositories": hf_inventory(),
              "weights_downloaded": False, "inference_run": False,
              "custom_prompts_and_credentials_recorded": False}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ollama_tags": len(report["ollama"]["models"]),
                      "unique_manifest_digests": len({m["digest"] for m in report["ollama"]["models"]}),
                      "hf_repositories": len(report["hf_model_repositories"]),
                      "weights_downloaded": False, "inference_run": False}, indent=2))


if __name__ == "__main__":
    main()
