# b2_io.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import Any, Dict


def read_b1_result(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_version") != "b1_result_v1":
        raise ValueError("Unsupported b1_result schema_version")
    if "cameras" not in data or "cam1" not in data["cameras"] or "cam2" not in data["cameras"]:
        raise KeyError("b1_result must contain cameras.cam1 and cameras.cam2")
    return data


def ensure_b2_output_dir(run_id: str, out_root: str = "outputs", overwrite: bool = False) -> str:
    base = os.path.join(out_root, str(run_id), "b2")
    os.makedirs(base, exist_ok=True)
    if not overwrite:
        expected_files = [
            os.path.join(base, "b2_calib_bundle.json"),
            os.path.join(base, "grid_overlay.png"),
            os.path.join(base, "cam1_overlay.png"),
            os.path.join(base, "cam2_overlay.png"),
        ]
        for f in expected_files:
            if os.path.exists(f):
                raise FileExistsError(f"Output already exists: {f} (use --overwrite to replace)")
    return base


def write_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
