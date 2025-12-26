#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FFmpeg-based video augmentation for mp4, preserving audio and folder structure.
- Recursive scan input_root/*.mp4
- Each recipe is a standalone function
- Keep directory structure (relative path preserved)
- Preserve audio (try copy, fallback to AAC)
- Progress bar via tqdm

Usage:
  python augmentation.py
    --input_root D:\graduate\项目\体能实训\体能实训数据\original
    --output_root D:\graduate\项目\体能实训\体能实训数据\augment
    --recipe shadow_highlight
    --seed 32
    --overwrite

If --recipe all:
  output_root/
    codec_like/<relative_path>.mp4
    geom_affine/<relative_path>.mp4
    geom_perspective/<relative_path>.mp4
    shadow_highlight/<relative_path>.mp4
"""

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit("tqdm 未安装：请先执行 `pip install tqdm`") from e


# ----------------------------
# subprocess helpers
# ----------------------------
def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check
    )



# ----------------------------
# ffprobe helpers
# ----------------------------
def ffprobe_video(ffprobe: str, video_path: Path) -> Dict:
    """
    Returns basic metadata: width, height, fps, has_audio
    """
    cmd = [
        ffprobe, "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(video_path)
    ]
    p = run_cmd(cmd)
    info = json.loads(p.stdout)
    v = info["streams"][0]
    width = int(v.get("width", 0))
    height = int(v.get("height", 0))

    def parse_fps(fr: str) -> float:
        if not fr or fr == "0/0":
            return 0.0
        num, den = fr.split("/")
        num_f = float(num)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return num_f / den_f

    fps = parse_fps(v.get("avg_frame_rate", "")) or parse_fps(v.get("r_frame_rate", "")) or 30.0

    # audio presence
    cmd_a = [
        ffprobe, "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        str(video_path)
    ]
    p2 = run_cmd(cmd_a, check=False)
    has_audio = False
    if p2.returncode == 0:
        try:
            ainfo = json.loads(p2.stdout)
            has_audio = len(ainfo.get("streams", [])) > 0
        except Exception:
            has_audio = False

    return {"width": width, "height": height, "fps": fps, "has_audio": has_audio}


# ----------------------------
# deterministic RNG per file
# ----------------------------
def rng_for_file(seed: int, rel_path: str) -> random.Random:
    h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return random.Random((seed ^ x) & 0xFFFFFFFF)


# ----------------------------
# Recipes (each returns filter string + is_complex + encode params)
# ----------------------------
def recipe_codec_like(meta: Dict, rng: random.Random) -> Tuple[str, bool, Dict]:
    gamma = rng.uniform(0.90, 1.10)
    contrast = rng.uniform(0.85, 1.15)
    brightness = rng.uniform(-0.05, 0.05)
    saturation = rng.uniform(0.85, 1.20)

    noise_s = rng.uniform(2.0, 12.0)
    sigma = rng.uniform(0.2, 0.8)

    vf = (
        f"eq=gamma={gamma:.4f}:contrast={contrast:.4f}:brightness={brightness:.4f}:saturation={saturation:.4f},"
        f"gblur=sigma={sigma:.3f},"
        f"noise=alls={noise_s:.2f}:allf=t+u,"
        f"format=yuv420p"
    )
    crf = int(rng.uniform(26, 34))
    enc = {"crf": crf, "preset": "veryfast"}
    return vf, False, enc


def recipe_geom_affine(meta: Dict, rng: random.Random) -> Tuple[str, bool, Dict]:
    w, h = meta["width"], meta["height"]

    deg = rng.uniform(-5.0, 5.0)
    rad = deg * 3.141592653589793 / 180.0
    scale = rng.uniform(0.92, 1.10)
    tx = rng.uniform(-0.04, 0.04) * w
    ty = rng.uniform(-0.04, 0.04) * h

    k1 = rng.uniform(-0.10, 0.10)
    k2 = rng.uniform(-0.05, 0.05)

    rotate = f"rotate={rad:.6f}:ow=iw:oh=ih:fillcolor=black@0"

    if scale >= 1.0:
        sw = int(round(w * scale))
        sh = int(round(h * scale))
        base_x = (sw - w) / 2.0 + tx
        base_y = (sh - h) / 2.0 + ty
        base_x = max(0.0, min(base_x, sw - w))
        base_y = max(0.0, min(base_y, sh - h))
        vf = (
            f"{rotate},"
            f"scale={sw}:{sh}:flags=bicubic,"
            f"crop={w}:{h}:{base_x:.2f}:{base_y:.2f},"
            f"lenscorrection=cx=0.5:cy=0.5:k1={k1:.4f}:k2={k2:.4f},"
            f"format=yuv420p"
        )
    else:
        sw = int(round(w * scale))
        sh = int(round(h * scale))
        px = (w - sw) / 2.0 + tx
        py = (h - sh) / 2.0 + ty
        px = max(0.0, min(px, w - sw))
        py = max(0.0, min(py, h - sh))
        vf = (
            f"{rotate},"
            f"scale={sw}:{sh}:flags=bicubic,"
            f"pad={w}:{h}:{px:.2f}:{py:.2f}:color=black,"
            f"lenscorrection=cx=0.5:cy=0.5:k1={k1:.4f}:k2={k2:.4f},"
            f"format=yuv420p"
        )

    enc = {"crf": 20, "preset": "veryfast"}
    return vf, False, enc


def recipe_geom_perspective(meta: Dict, rng: random.Random) -> Tuple[str, bool, Dict]:
    w, h = meta["width"], meta["height"]

    jx = 0.06 * w
    jy = 0.06 * h

    def clamp(v, lo, hi):
        return max(lo, min(v, hi))

    x0 = clamp(rng.uniform(-jx, jx), -0.10 * w, 0.10 * w)
    y0 = clamp(rng.uniform(-jy, jy), -0.10 * h, 0.10 * h)

    x1 = clamp(w + rng.uniform(-jx, jx), 0.90 * w, 1.10 * w)
    y1 = clamp(rng.uniform(-jy, jy), -0.10 * h, 0.10 * h)

    x2 = clamp(rng.uniform(-jx, jx), -0.10 * w, 0.10 * w)
    y2 = clamp(h + rng.uniform(-jy, jy), 0.90 * h, 1.10 * h)

    x3 = clamp(w + rng.uniform(-jx, jx), 0.90 * w, 1.10 * w)
    y3 = clamp(h + rng.uniform(-jy, jy), 0.90 * h, 1.10 * h)

    scale = rng.uniform(1.02, 1.10)
    sw = int(round(w * scale))
    sh = int(round(h * scale))
    cx = (sw - w) / 2.0
    cy = (sh - h) / 2.0

    vf = (
        f"perspective="
        f"x0={x0:.2f}:y0={y0:.2f}:x1={x1:.2f}:y1={y1:.2f}:"
        f"x2={x2:.2f}:y2={y2:.2f}:x3={x3:.2f}:y3={y3:.2f}:"
        f"sense=destination:interpolation=cubic,"
        f"scale={sw}:{sh}:flags=bicubic,"
        f"crop={w}:{h}:{cx:.2f}:{cy:.2f},"
        f"format=yuv420p"
    )

    enc = {"crf": 20, "preset": "veryfast"}
    return vf, False, enc


def recipe_shadow_highlight(meta: Dict, rng: random.Random) -> Tuple[str, bool, Dict]:
    w, h = meta["width"], meta["height"]
    fps = meta["fps"]

    # shadow params
    sw = int(rng.uniform(0.35, 0.70) * w)
    sh = int(rng.uniform(0.20, 0.45) * h)
    sx = int(rng.uniform(0.00, 1.00) * max(1, w - sw))
    sy = int(rng.uniform(0.00, 1.00) * max(1, h - sh))
    s_alpha = rng.uniform(0.30, 0.60)
    s_sigma = rng.uniform(12.0, 28.0)
    margin = int(rng.uniform(0.08, 0.16) * min(sw, sh))

    # highlight params
    hw = int(rng.uniform(0.18, 0.45) * w)
    hh = int(rng.uniform(0.12, 0.30) * h)
    hx = int(rng.uniform(0.00, 1.00) * max(1, w - hw))
    hy = int(rng.uniform(0.00, 1.00) * max(1, h - hh))
    h_alpha = rng.uniform(0.18, 0.40)
    h_sigma = rng.uniform(10.0, 22.0)
    margin2 = int(rng.uniform(0.08, 0.16) * min(hw, hh))

    sh_cw = sw + 2 * margin
    sh_ch = sh + 2 * margin
    sh_x_in = margin
    sh_y_in = margin

    hi_cw = hw + 2 * margin2
    hi_ch = hh + 2 * margin2
    hi_x_in = margin2
    hi_y_in = margin2

    filter_complex = (
        f"[0:v]format=rgba[base];"
        f"color=c=black@0.0:s={sh_cw}x{sh_ch}:r={fps:.3f},format=rgba,"
        f"drawbox=x={sh_x_in}:y={sh_y_in}:w={sw}:h={sh}:color=black@{s_alpha:.3f}:t=fill,"
        f"gblur=sigma={s_sigma:.2f}[sh];"
        f"[base][sh]overlay=x={sx - margin}:y={sy - margin}:format=auto:shortest=1[tmp];"
        f"color=c=white@0.0:s={hi_cw}x{hi_ch}:r={fps:.3f},format=rgba,"
        f"drawbox=x={hi_x_in}:y={hi_y_in}:w={hw}:h={hh}:color=white@{h_alpha:.3f}:t=fill,"
        f"gblur=sigma={h_sigma:.2f}[hi];"
        f"[tmp][hi]overlay=x={hx - margin2}:y={hy - margin2}:format=auto:shortest=1,"
        f"format=yuv420p[vout]"
    )

    enc = {"crf": 20, "preset": "veryfast"}
    return filter_complex, True, enc


RECIPE_TABLE = {
    "codec_like": recipe_codec_like,
    "geom_affine": recipe_geom_affine,
    "geom_perspective": recipe_geom_perspective,
    "shadow_highlight": recipe_shadow_highlight,
}


# ----------------------------
# Core processing
# ----------------------------
def iter_mp4(input_root: Path) -> List[Path]:
    return sorted([p for p in input_root.rglob("*.mp4") if p.is_file()])


def build_output_path(input_root: Path, output_root: Path, video_path: Path, recipe_name: str, multi: bool) -> Path:
    rel = video_path.relative_to(input_root)
    return (output_root / recipe_name / rel) if multi else (output_root / rel)


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def ffmpeg_process_one(
    ffmpeg: str,
    ffprobe: str,
    input_root: Path,
    output_root: Path,
    video_path: Path,
    recipe_name: str,
    seed: int,
    overwrite: bool,
    multi: bool,
) -> Tuple[bool, str]:
    rel = str(video_path.relative_to(input_root)).replace("\\", "/")
    rng = rng_for_file(seed, f"{recipe_name}|{rel}")

    meta = ffprobe_video(ffprobe, video_path)
    if meta["width"] <= 0 or meta["height"] <= 0:
        return False, "ffprobe failed"

    out_path = build_output_path(input_root, output_root, video_path, recipe_name, multi)
    ensure_parent(out_path)
    if out_path.exists() and not overwrite:
        return True, "exists-skip"

    filt, is_complex, enc = RECIPE_TABLE[recipe_name](meta, rng)

    cmd = [ffmpeg, "-hide_banner"]
    cmd += (["-y"] if overwrite else ["-n"])
    cmd += ["-i", str(video_path)]
    cmd += ["-map_metadata", "0"]
    cmd += ["-map", "0:v:0", "-map", "0:a?"]

    if is_complex:
        cmd += ["-filter_complex", filt, "-map", "[vout]"]
    else:
        cmd += ["-vf", filt]

    cmd += ["-c:v", "libx264", "-preset", enc.get("preset", "veryfast"), "-crf", str(enc.get("crf", 20))]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    # audio preserve: try copy first
    cmd += ["-c:a", "copy"]
    cmd += ["-shortest"]
    cmd += [str(out_path)]

    p = run_cmd(cmd, check=False)
    if p.returncode == 0:
        return True, "ok"

    # fallback: re-encode audio
    cmd2 = cmd[:]
    for i in range(len(cmd2) - 1):
        if cmd2[i] == "-c:a":
            cmd2[i + 1] = "aac"
            cmd2.insert(i + 2, "-b:a")
            cmd2.insert(i + 3, "192k")
            break

    p2 = run_cmd(cmd2, check=False)
    if p2.returncode == 0:
        return True, "ok(audio->aac)"

    err = (p2.stderr or p.stderr or "").strip()
    if len(err) > 500:
        err = err[-500:]
    return False, err or "ffmpeg failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, type=str)
    ap.add_argument("--output_root", required=True, type=str)
    ap.add_argument("--recipe", required=True, type=str, choices=list(RECIPE_TABLE.keys()) + ["all"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ffmpeg", type=str, default="ffmpeg")
    ap.add_argument("--ffprobe", type=str, default="ffprobe")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    videos = iter_mp4(input_root)
    if not videos:
        print(f"[INFO] no mp4 found under: {input_root}")
        return

    recipes = list(RECIPE_TABLE.keys()) if args.recipe == "all" else [args.recipe]
    multi = (args.recipe == "all")

    total_jobs = len(videos) * len(recipes)

    ok = 0
    fail = 0

    with tqdm(total=total_jobs, dynamic_ncols=True, smoothing=0.1) as pbar:
        for vp in videos:
            rel = vp.relative_to(input_root)
            for r in recipes:
                pbar.set_description(f"{r}")
                pbar.set_postfix_str(str(rel))

                succ, info = ffmpeg_process_one(
                    ffmpeg=args.ffmpeg,
                    ffprobe=args.ffprobe,
                    input_root=input_root,
                    output_root=output_root,
                    video_path=vp,
                    recipe_name=r,
                    seed=args.seed,
                    overwrite=args.overwrite,
                    multi=multi,
                )
                if succ:
                    ok += 1
                else:
                    fail += 1
                    tqdm.write(f"[ERROR] {r}  {rel}\n  {info}")

                pbar.update(1)

    print(f"[DONE] ok={ok}, fail={fail}, videos={len(videos)}, recipes={len(recipes)}")


if __name__ == "__main__":
    main()
