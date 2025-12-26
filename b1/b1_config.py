# b1_config.py
# python b1_config.py --init b1_config.json
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge patch into base (returns new dict)."""
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_set_kv(items: Optional[list[str]]) -> Dict[str, Any]:
    """
    Parse overrides like:
      --set pole_detector.init_secs=8
      --set multicam_resolver.min_coverage=7
    """
    if not items:
        return {}
    patch: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid --set '{it}', expected key=value")
        key, raw = it.split("=", 1)
        key = key.strip()
        raw = raw.strip()

        # guess type
        if raw.lower() in ("true", "false"):
            val: Any = (raw.lower() == "true")
        else:
            try:
                if "." in raw:
                    val = float(raw)
                    if val.is_integer():
                        # keep integer-ish floats as float; caller can decide
                        pass
                else:
                    val = int(raw)
            except Exception:
                val = raw  # string fallback

        # set into nested dict by dot path
        cur = patch
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return patch


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass(frozen=True)
class LayoutConfig:
    """
    场地/世界坐标几何参数（单位：米）。
    注意：世界坐标约定见 layout_fitter.py 顶部注释（y=0 中线，列在 ±lat_gap/2）。
    """
    n_poles: int = 7               # 场地立杆总数（当前固定 7）
    same_col_step_m: float = 10.0  # 同一列相邻立杆的纵向间距（P1->P3 等）
    stagger_m: float = 5.0         # 两列交错的纵向偏移（半步）
    lat_gap_m: float = 2.5         # 两列的横向间距（最终列坐标为 ±lat_gap_m/2）


@dataclass(frozen=True)
class YoloConfig:
    """
    YOLOv5 推理相关参数。
    只负责“怎么跑 YOLO”，不涉及“如何聚类成 7 根杆”。
    """
    weights: str = "../yolov5-master/best.pt"              # 训练好的权重 .pt 路径
    yolov5_repo: str = "../yolov5-master"          # 本地 yolov5 仓库路径（source=local 时必须）
    source: str = "local"          # "local" 优先；"remote" 会走 torch.hub 下载
    class_name: str = "pole"       # 只保留该类别的检测框（与你训练的 name 对齐）
    conf: float = 0.25             # 置信度阈值（越低召回越高，误检也更多）
    iou: float = 0.45              # NMS IoU 阈值
    imgsz: int = 960               # 推理输入尺寸（越大越利于远处小目标，但更慢）


@dataclass(frozen=True)
class PoleDetectorConfig:
    """
    B1：从视频中提取“杆底点（像素）”的参数。
    典型流程：读前 init_secs 秒 -> 每 frame_stride 帧跑一次 YOLO -> 收集底点 -> 聚类 -> 输出 M 个点（<=n_poles）。
    """
    n_poles: int = 7               # 最大杆数（用于聚类上限/候选上限）
    init_secs: float = 15.0         # 仅使用视频开头多少秒进行杆位置估计（越大越稳，但更慢）
    frame_stride: int = 1         # 帧采样步长（60fps 下 stride=2 约等于 30fps 采样）
    seed: int = 2025               # 随机种子（kmeans 初始化等）
    kmeans_iters: int = 60         # kmeans 迭代次数上限

    # 下面两项用于“不要硬凑 7 根杆”：对每个簇做有效性筛选
    min_points_per_pole: int = 15  # 认为“真的检测到一根杆”至少需要多少个检测点支撑（点数越少越不可靠）
    max_mad_px: float = 18.0       # 簇内离散度阈值（MAD，像素；越小越严格，误检更难通过）

    window_mode: str = "custom"   # "start" | "center" | "custom"
    start_sec: float = 5.0       # window_mode="custom" 时使用
    sample_secs: float = 10.0     # 0 表示沿用 init_secs


@dataclass(frozen=True)
class LayoutFitterConfig:
    """
    单机位：将像素点集合拟合到已知世界布局，产生候选（编号子集 + H_pix2world）。
    """
    ransac_th_px: float = 3.0      # cv2.findHomography 的 RANSAC 像素阈值
    topk_candidates: int = 40      # 单机位保留多少个候选给双机位联合决策用
    try_reverse: bool = True       # 是否尝试输入序列反向（cam2 常见需要）
    try_swap_columns: bool = True  # 是否尝试两列互换（镜像/左右不确定时有用）


@dataclass(frozen=True)
class MultiCamResolverConfig:
    """
    双机位联合：从 cam1/cam2 各自候选里选一对最一致的解（并尽量覆盖 7 根杆）。
    """
    topk_single: int = 40          # 每个机位最多取前 K 个候选参与联合搜索
    ransac_th_px: float = 3.0      # 传递给单机位候选生成（保持一致即可）

    coverage_penalty_m: float = 0.15   # 覆盖度惩罚：每缺 1 根杆，score 增加多少（“米等价”权重）
    shared_consistency_w: float = 0.30 # 共享杆一致性权重：两机位同编号杆的世界坐标差（均值）乘这个系数加到 score
    min_coverage: int = 4              # 联合解最少覆盖多少根杆才算可用（一般>=4；想严格可设 7）

    cam1_role: str = "start"       # 固定：cam1 在起点端（用于方向先验）
    cam2_role: str = "end"         # 固定：cam2 在终点端（用于方向先验）


@dataclass(frozen=True)
class B1Config:
    """
    B1 总配置：按模块分块管理，所有模块都从这里取参数，不在模块内部写“隐式默认值”。
    """
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    pole_detector: PoleDetectorConfig = field(default_factory=PoleDetectorConfig)
    layout_fitter: LayoutFitterConfig = field(default_factory=LayoutFitterConfig)
    multicam_resolver: MultiCamResolverConfig = field(default_factory=MultiCamResolverConfig)



# -----------------------------
# Serialization / validation
# -----------------------------
def default_config_dict() -> Dict[str, Any]:
    return asdict(B1Config())


def _validate(d: Dict[str, Any]) -> None:
    # minimal sanity checks
    L = d["layout"]
    if int(L["n_poles"]) != 7:
        raise ValueError("This project currently expects layout.n_poles == 7")
    if float(L["lat_gap_m"]) <= 0:
        raise ValueError("layout.lat_gap_m must be > 0")
    if float(L["same_col_step_m"]) <= 0 or float(L["stagger_m"]) <= 0:
        raise ValueError("layout distances must be > 0")

    if d["multicam_resolver"]["cam1_role"] != "start" or d["multicam_resolver"]["cam2_role"] != "end":
        # you can relax this later, but you said these are guaranteed
        raise ValueError("multicam_resolver.cam1_role must be 'start' and cam2_role must be 'end'")


def load_b1_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> B1Config:
    base = default_config_dict()
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    merged = _deep_update(base, user)
    if overrides:
        merged = _deep_update(merged, overrides)

    _validate(merged)

    # construct dataclasses (manual to keep strict)
    return B1Config(
        layout=LayoutConfig(**merged["layout"]),
        yolo=YoloConfig(**merged["yolo"]),
        pole_detector=PoleDetectorConfig(**merged["pole_detector"]),
        layout_fitter=LayoutFitterConfig(**merged["layout_fitter"]),
        multicam_resolver=MultiCamResolverConfig(**merged["multicam_resolver"]),
    )


def save_default_b1_config(path: str) -> None:
    d = default_config_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# -----------------------------
# Self-test
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("b1_config.py")
    ap.add_argument("--init", type=str, default="", help="Write a default config json to this path")
    ap.add_argument("--load", type=str, default="", help="Load config json from this path and print")
    ap.add_argument("--set", action="append", default=[], help="Override like a.b.c=123 (repeatable)")
    args = ap.parse_args()

    if args.init:
        save_default_b1_config(args.init)
        print("Wrote default config:", args.init)
        raise SystemExit(0)

    if not args.load:
        print("Usage examples:")
        print("  python b1_config.py --init b1_config.json")
        print("  python b1_config.py --load b1_config.json")
        print("  python b1_config.py --load b1_config.json --set pole_detector.init_secs=8 --set yolo.conf=0.2")
        raise SystemExit(0)

    patch = _parse_set_kv(args.set)
    cfg = load_b1_config(args.load, overrides=patch)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
