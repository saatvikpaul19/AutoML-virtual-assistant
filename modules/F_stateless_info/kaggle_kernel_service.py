from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path


import os
import sys

def _run_kaggle_command(args: list[str]) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf8"
    kaggle_bin = os.path.join(os.path.dirname(sys.executable), "kaggle")
    if not os.path.exists(kaggle_bin):
        kaggle_bin = "kaggle"  # Fallback to PATH if not in a venv

    try:
        proc = subprocess.run(
            [kaggle_bin, *args],
            capture_output=True,
            check=False,
            shell=False,
            env=env,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return {"success": False, "error": stderr or stdout or "Kaggle command failed"}
        return {"success": True, "output": stdout}
    except Exception as e:
        return {"success": False, "error": f"Failed to run Kaggle command: {e}"}


def _parse_csv_output(text: str) -> list[dict]:
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text)))


def search_kernels(query: str, page_size: int = 10) -> dict:
    result = _run_kaggle_command(["kernels", "list", "-s", query, "-v"])
    if not result["success"]:
        return result
    rows = _parse_csv_output(result["output"])
    return {"success": True, "query": query, "results": rows}


def resolve_best_kernel_ref(query: str) -> dict:
    search = search_kernels(query, page_size=10)
    if not search["success"]:
        return search
    rows = search["results"]
    if not rows:
        return {"success": False, "error": f"No kernels found for '{query}'."}
    row = rows[0]
    ref = row.get("ref") or row.get("id")
    if not ref:
        return {"success": False, "error": f"Could not resolve kernel ref for '{query}'."}
    return {"success": True, "kernel_ref": ref, "top_result": row, "results": rows}


def pull_kernel_code(kernel_ref: str, output_dir: str) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = _run_kaggle_command(["kernels", "pull", "-p", output_dir, "-k", kernel_ref])
    if not result["success"]:
        return result

    out_dir = Path(output_dir)
    files = list(out_dir.rglob("*"))
    candidates = [p for p in files if p.is_file() and p.suffix.lower() in {".py", ".ipynb"}]
    if not candidates:
        return {"success": False, "error": f"Kernel pulled but no .py or .ipynb file found for '{kernel_ref}'."}

    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".py" else 1, len(p.name)))
    code_file = candidates[0]
    code_text = code_file.read_text(encoding="utf-8", errors="ignore")
    return {
        "success": True,
        "kernel_ref": kernel_ref,
        "code_path": str(code_file.resolve()),
        "code_format": code_file.suffix.lower().lstrip("."),
        "code_text": code_text,
    }


def get_kernel_code(query: str, base_dir: str = "artifacts/kernels") -> dict:
    resolved = resolve_best_kernel_ref(query)
    if not resolved["success"]:
        return resolved
    safe = resolved["kernel_ref"].replace("/", "__")
    pulled = pull_kernel_code(resolved["kernel_ref"], str(Path(base_dir) / safe))
    if not pulled["success"]:
        return pulled
    return {
        "success": True,
        "query": query,
        "kernel_ref": resolved["kernel_ref"],
        "top_result": resolved["top_result"],
        "code_format": pulled["code_format"],
        "code_text": pulled["code_text"],
        "code_path": pulled["code_path"],
    }