# b2_runner.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os

from b2_config import load_b2_config
from b2_core import assemble_bundle, compute_bev_grid, compute_world_bbox
from b2_io import ensure_b2_output_dir, read_b1_result, write_json
from b2_vis import draw_bev_observed, draw_cam_overlay, draw_grid_overlay


def main():
    ap = argparse.ArgumentParser("b2_runner.py (build B2 calib bundle + visualization)")
    ap.add_argument("--cfg", type=str, required=True, help="path to b2_config.json")
    ap.add_argument("--no-vis", action="store_true", help="disable all visualizations")
    ap.add_argument("--vis-only", action="store_true", help="only emit visualizations, skip JSON")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--out-root", type=str, default="outputs")
    args = ap.parse_args()

    cfg = load_b2_config(args.cfg)
    b1_res = read_b1_result(cfg.b1_result_path)

    if str(cfg.run_id) != str(b1_res.get("run_id")):
        raise ValueError("run_id mismatch between b2_config and b1_result")

    margins = {"x_margin_m": cfg.b2_grid.x_margin_m, "y_margin_m": cfg.b2_grid.y_margin_m}

    world_bbox = compute_world_bbox(b1_res["layout"], margins)
    bev_grid, _, _ = compute_bev_grid(b1_res["layout"], margins, cfg.b2_grid)

    out_dir = ensure_b2_output_dir(cfg.run_id, out_root=args.out_root, overwrite=args.overwrite)

    if not args.vis_only:
        bundle = assemble_bundle(
            cfg.run_id,
            b1_res,
            cfg.b1_result_path,
            cfg.b2_grid,
            bev_grid,
            world_bbox,
            margins,
        )
        bundle_path = os.path.join(out_dir, "b2_calib_bundle.json")
        write_json(bundle, bundle_path)
        print(f"b2_calib_bundle.json written to {bundle_path}")

    if cfg.vis.enable and (not args.no_vis):
        grid_path = os.path.join(out_dir, "grid_overlay.png")
        draw_grid_overlay(grid_path, b1_res["layout"], cfg.b2_grid, bev_grid)
        print(f"grid_overlay.png written to {grid_path}")

        if cfg.videos is None or cfg.videos.cam1_path is None or cfg.videos.cam2_path is None:
            raise KeyError("videos.cam1_path/cam2_path required for camera overlays")

        cam1_path = os.path.join(out_dir, "cam1_overlay.png")
        cam2_path = os.path.join(out_dir, "cam2_overlay.png")
        draw_cam_overlay(b1_res["cameras"]["cam1"], cfg.videos.cam1_path, cam1_path, cfg.vis)
        draw_cam_overlay(b1_res["cameras"]["cam2"], cfg.videos.cam2_path, cam2_path, cfg.vis)
        print(f"Camera overlays written to {cam1_path} and {cam2_path}")

        if cfg.vis.emit_bev_observed:
            bev_path = os.path.join(out_dir, "bev_observed.png")
            draw_bev_observed(bev_path, b1_res, bev_grid)
            print(f"bev_observed.png written to {bev_path}")
if __name__ == "__main__":
    main()
