"""Push a Kaggle kernel without starting a run.

The normal ``kaggle kernels push`` queues execution immediately.  This uses
Kaggle's quick-save API so the notebook source and metadata update, but no run
slot is consumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import kaggle
from kagglesdk.kernels.types.kernels_api_service import (
    ApiSaveKernelRequest,
    KernelExecutionType,
)


def push(folder: str) -> None:
    meta_path = Path(folder) / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    code_file = Path(folder) / meta["code_file"]
    script_body = code_file.read_text(encoding="utf-8")
    if meta.get("kernel_type") == "notebook":
        nb = json.loads(script_body)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
            if isinstance(cell.get("source"), list):
                cell["source"] = "".join(cell["source"])
        script_body = json.dumps(nb)

    with kaggle.api.build_kaggle_client() as client:
        req = ApiSaveKernelRequest()
        req.slug = meta.get("id")
        req.new_title = meta.get("title")
        req.text = script_body
        req.language = meta.get("language", "python")
        req.kernel_type = meta.get("kernel_type", "notebook")
        req.is_private = meta.get("is_private", True)
        req.enable_gpu = meta.get("enable_gpu", False)
        req.enable_tpu = meta.get("enable_tpu", False)
        req.enable_internet = meta.get("enable_internet", False)
        req.dataset_data_sources = meta.get("dataset_sources", [])
        req.competition_data_sources = meta.get("competition_sources", [])
        req.kernel_data_sources = meta.get("kernel_sources", [])
        req.model_data_sources = meta.get("model_sources", [])
        req.machine_shape = meta.get("machine_shape")
        req.kernel_execution_type = KernelExecutionType.QUICK_SAVE
        resp = client.kernels.kernels_api_client.save_kernel(req)

    if resp.error:
        print(f"push failed: {resp.error}", file=sys.stderr)
        sys.exit(1)
    print(f"pushed (no run): version {resp.version_number}  {resp.url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="folder containing kernel-metadata.json")
    args = parser.parse_args()
    push(args.folder)


if __name__ == "__main__":
    main()
