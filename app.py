from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

_cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_cjk_font).exists():
    font_manager.fontManager.addfont(_cjk_font)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)
MODEL_PATH = os.getenv("TENNIS_MODEL", "yolo11m.pt")
BALL_MODEL_PATH = os.getenv("TENNIS_BALL_MODEL", "/opt/Good-Tennis/weights/tennis-ball.pt")
DEVICE = os.getenv("TENNIS_DEVICE", "0")
CONFIDENCE = float(os.getenv("TENNIS_CONF", "0.14"))
BALL_CONFIDENCE = float(os.getenv("TENNIS_BALL_CONF", "0.05"))
BALL_CLASS, PERSON_CLASS, RACKET_CLASS = 32, 0, 38

_model: YOLO | None = None
_ball_model: YOLO | None = None


def model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def ball_model() -> YOLO:
    global _ball_model
    if _ball_model is None:
        _ball_model = YOLO(BALL_MODEL_PATH)
    return _ball_model


def center(box: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def nearest_candidate(candidates: list[tuple[int, int, float]], previous, max_jump: float):
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda p: p[2])
    valid = [p for p in candidates if math.dist(p[:2], previous[:2]) <= max_jump]
    # A missing nearby candidate means a missed frame, not permission to jump to
    # a logo, spectator or scoreboard elsewhere in the image.
    return min(valid, key=lambda p: math.dist(p[:2], previous[:2])) if valid else None


def local_motion(gray: np.ndarray, previous: np.ndarray | None, point: tuple[int, int]) -> tuple[float, float]:
    """Measure frame change near a detected ball, independent of confidence."""
    if previous is None:
        return 0.0, 0.0
    x, y = point
    radius = max(5, round(gray.shape[1] / 160))
    x0, x1 = max(0, x - radius), min(gray.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(gray.shape[0], y + radius + 1)
    delta = cv2.absdiff(gray[y0:y1, x0:x1], previous[y0:y1, x0:x1])
    return float(delta.mean()), float((delta > 15).mean())


def moving_ball_candidates(candidates: list[tuple[int, int, float]], gray: np.ndarray,
                           previous_frames: deque, active_position=None):
    moving = []
    for candidate in candidates:
        scores = [local_motion(gray, previous, candidate[:2]) for previous in previous_frames]
        motion, changed_fraction = max(scores, default=(0.0, 0.0))
        # A softer threshold near an established track preserves balls around
        # the apex of a shot, where displacement can briefly become small.
        near_active = active_position is not None and math.dist(candidate[:2], active_position[:2]) <= max(gray.shape) * .03
        threshold = 1.3 if near_active else 2.5
        fraction_threshold = .015 if near_active else .035
        if motion >= threshold and changed_fraction >= fraction_threshold:
            moving.append(candidate)
    return moving


def court_polygon(width: int, height: int, expanded: bool = False) -> np.ndarray:
    """Broadcast-camera court corridor; deliberately includes both baselines."""
    if expanded:
        points = ((.25, .10), (.75, .10), (1.0, 1.0), (0.0, 1.0))
    else:
        points = ((.32, .15), (.68, .15), (.98, 1.0), (.02, 1.0))
    return np.array([(int(x * width), int(y * height)) for x, y in points], dtype=np.int32)


def inside(point: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_label(frame, text, origin, color):
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 3), color, -1)
    cv2.putText(frame, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 15, 24), 1, cv2.LINE_AA)


def add_trajectory_segments(records: pd.DataFrame, width: int, height: int, max_gap: int = 3) -> pd.DataFrame:
    """Split detections at missing spans or physically implausible jumps."""
    records = records.copy()
    records["trajectory_segment"] = pd.Series(pd.NA, index=records.index, dtype="Int64")
    valid_idx = records.index[records.ball_x.notna() & records.ball_y.notna()].tolist()
    segment = 0
    previous_idx = None
    previous_point = None
    diagonal = math.hypot(width, height)
    for idx in valid_idx:
        point = (float(records.at[idx, "ball_x"]), float(records.at[idx, "ball_y"]))
        if previous_idx is None:
            segment += 1
        else:
            gap = int(records.at[idx, "frame"] - records.at[previous_idx, "frame"])
            distance = math.dist(point, previous_point)
            # Permit normal fast motion and interpolation over at most two missing
            # frames, but never bridge a long absence or a reacquisition jump.
            max_distance = diagonal * .035 * max(1, gap)
            if gap > max_gap or distance > max_distance:
                segment += 1
        records.at[idx, "trajectory_segment"] = segment
        previous_idx, previous_point = idx, point
    # A continuous detection can contain several strokes. Split sustained
    # vertical reversals (the ball changes court direction), while limiting
    # very long spans to readable time windows.
    fps = 1 / max(.001, float(records.time_s.diff().median()))
    next_segment = 0
    refined = pd.Series(pd.NA, index=records.index, dtype="Int64")
    for _, group in records.dropna(subset=["trajectory_segment"]).groupby("trajectory_segment", sort=False):
        indexes = group.index.to_list()
        y = group.ball_y.astype(float).rolling(5, center=True, min_periods=1).median().to_numpy()
        cuts = [0]
        min_frames = max(12, int(fps * .7))
        max_frames = max(min_frames * 2, int(fps * 3))
        for i in range(5, len(indexes) - 5):
            if i - cuts[-1] < min_frames:
                continue
            before, after = y[i] - y[i - 5], y[i + 5] - y[i]
            turn = before * after < 0 and min(abs(before), abs(after)) >= height * .035
            if turn or i - cuts[-1] >= max_frames:
                cuts.append(i)
        cuts.append(len(indexes))
        for start, end in zip(cuts, cuts[1:]):
            next_segment += 1
            refined.loc[indexes[start:end]] = next_segment
    records["trajectory_segment"] = refined
    return records


def interpolate_short_gaps(records: pd.DataFrame, width: int, height: int, max_gap: int = 3) -> pd.DataFrame:
    """Fill only short, bounded gaps whose endpoints imply plausible motion."""
    records = records.copy()
    records["ball_source"] = np.where(records.ball_x.notna(), "detected", "missing")
    diagonal = math.hypot(width, height)
    missing = records.ball_x.isna().to_numpy()
    start = None
    gaps = []
    for i, is_missing in enumerate(np.append(missing, False)):
        if is_missing and start is None:
            start = i
        elif not is_missing and start is not None:
            gaps.append((start, i - 1))
            start = None
    for start, end in gaps:
        length = end - start + 1
        left, right = start - 1, end + 1
        if length > max_gap or left < 0 or right >= len(records):
            continue
        p0 = (float(records.at[left, "ball_x"]), float(records.at[left, "ball_y"]))
        p1 = (float(records.at[right, "ball_x"]), float(records.at[right, "ball_y"]))
        if math.dist(p0, p1) > diagonal * .035 * (length + 1):
            continue
        for offset, idx in enumerate(range(start, end + 1), 1):
            ratio = offset / (length + 1)
            records.at[idx, "ball_x"] = p0[0] + (p1[0] - p0[0]) * ratio
            records.at[idx, "ball_y"] = p0[1] + (p1[1] - p0[1]) * ratio
            records.at[idx, "ball_source"] = "interpolated"
    return records


def configure_chart_style():
    plt.style.use("dark_background")
    plt.rcParams.update({"axes.facecolor": "#101c2b", "figure.facecolor": "#0a1422", "savefig.facecolor": "#0a1422", "axes.edgecolor": "#506274", "text.color": "#edf5fb", "axes.labelcolor": "#c3d3df", "xtick.color": "#9eb2c4", "ytick.color": "#9eb2c4"})


def trajectory_choices(records: pd.DataFrame):
    valid = records.dropna(subset=["ball_x", "ball_y", "trajectory_segment"])
    groups = [(int(sid), group.sort_values("frame")) for sid, group in valid.groupby("trajectory_segment") if len(group) >= 5]
    longest = sorted(groups, key=lambda pair: len(pair[1]), reverse=True)[:60]
    default_id = longest[0][0] if longest else None
    options = []
    for segment_id, segment in sorted(longest, key=lambda pair: pair[1].time_s.iloc[0]):
        start, end = float(segment.time_s.iloc[0]), float(segment.time_s.iloc[-1])
        def fmt(seconds):
            return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
        options.append((f"{fmt(start)}–{fmt(end)} · {len(segment)} 帧", segment_id))
    return options, default_id


def frame_at(video_path: str, frame_number: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def render_trajectory_chart(records: pd.DataFrame, out_dir: Path, width: int, height: int,
                            selected_segment: int | None, background: np.ndarray | None = None):
    configure_chart_style()
    segment = records[records.trajectory_segment == selected_segment].sort_values("frame") if selected_segment is not None else records.iloc[0:0]
    fig, ax = plt.subplots(figsize=(13, 7.2), layout="constrained")
    if background is not None:
        backdrop = cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        ax.imshow(np.clip(backdrop * .44, 0, 1), extent=(0, width, height, 0), zorder=0)
    else:
        ax.set_facecolor("#123327")
    if len(segment):
        color = "#2dd4bf"
        ax.plot(segment.ball_x, segment.ball_y, color="#031018", lw=5.5, alpha=.8, zorder=2)
        ax.plot(segment.ball_x, segment.ball_y, color=color, lw=3.2, zorder=3)
        detected = segment[segment.ball_source == "detected"]
        filled = segment[segment.ball_source == "interpolated"]
        marker_step = max(1, len(detected) // 80)
        detected = detected.iloc[::marker_step]
        ax.scatter(detected.ball_x, detected.ball_y, s=19, c=color, edgecolors="#06111c", linewidths=.5, zorder=4)
        if len(filled):
            ax.scatter(filled.ball_x, filled.ball_y, s=38, facecolors="none", edgecolors="#ffffff", linewidths=1.2, zorder=5)
        start, end = segment.iloc[0], segment.iloc[-1]
        ax.scatter([start.ball_x], [start.ball_y], s=115, marker="o", c="#34d399", edgecolors="white", linewidths=1.8, zorder=6)
        ax.scatter([end.ball_x], [end.ball_y], s=115, marker="X", c="#fb7185", edgecolors="white", linewidths=1.2, zorder=6)
        subtitle = f"{start.time_s:.1f}–{end.time_s:.1f} 秒 · {len(segment)} 帧"
    else:
        subtitle = "没有足够长的运动轨迹"
    ax.set(xlim=(0, width), ylim=(height, 0), title=f"单段网球轨迹 · {subtitle}", xlabel="画面横坐标（像素）", ylabel="画面纵坐标（像素）")
    ax.grid(alpha=.07)
    p = out_dir / f"trajectory_segment_{selected_segment if selected_segment is not None else 'none'}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return str(p)


def save_charts(records: pd.DataFrame, out_dir: Path, width: int, height: int,
                selected_segment: int | None, background: np.ndarray | None = None):
    configure_chart_style()
    paths = [render_trajectory_chart(records, out_dir, width, height, selected_segment, background)]
    time_formatter = FuncFormatter(lambda value, _: f"{int(value // 60):02d}:{int(value % 60):02d}")

    fig, ax = plt.subplots(figsize=(13, 5.8), layout="constrained")
    speed = records.ball_speed_px_s.astype(float).copy()
    speed[records.ball_source == "missing"] = np.nan
    first_in_segment = records.trajectory_segment.ne(records.trajectory_segment.shift()).fillna(True)
    speed[first_in_segment] = np.nan
    smooth = speed.groupby(records.trajectory_segment).transform(lambda series: series.rolling(7, center=True, min_periods=2).median())
    ax.plot(records.time_s, speed, color="#fbbf24", alpha=.24, lw=.9, label="逐帧速度")
    ax.plot(records.time_s, smooth, color="#38bdf8", lw=2.3, label="7 帧中位趋势")
    missing = records.ball_source.eq("missing").to_numpy()
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    dt = float(records.time_s.diff().median()) if len(records) > 1 else .04
    for start, end in zip(starts, ends):
        if end - start + 1 >= 2:
            ax.axvspan(float(records.time_s.iloc[start]), float(records.time_s.iloc[end]) + dt,
                       color="#fb7185", alpha=.17, lw=0)
    ax.plot([], [], color="#fb7185", lw=7, alpha=.35, label="未跟踪区间")
    ax.set(xlim=(0, float(records.time_s.max()) if len(records) else 1), ylim=(0, None), title="网球运动速度趋势", xlabel="视频时间（分:秒）", ylabel="画面速度（像素/秒）")
    ax.xaxis.set_major_formatter(time_formatter)
    ax.grid(alpha=.13)
    ax.legend(loc="upper right", framealpha=.7, facecolor="#102235")
    p = out_dir / "speed_timeline.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))

    fig, (band, ax) = plt.subplots(2, 1, figsize=(13, 5.8), height_ratios=(1, 3), sharex=True, layout="constrained")
    source = records.ball_source.fillna("missing")
    colors = {"missing": "#fb7185", "interpolated": "#fbbf24", "detected": "#2dd4bf"}
    states = source.map({"missing": 0, "interpolated": 1, "detected": 2}).to_numpy(dtype=int)
    duration = float(records.time_s.max()) if len(records) else 1
    band.imshow(states[np.newaxis, :], aspect="auto", interpolation="nearest", extent=(0, duration, 0, 1), cmap=ListedColormap([colors[key] for key in ("missing", "interpolated", "detected")]), vmin=0, vmax=2)
    band.set_yticks([])
    band.set_ylabel("逐帧状态", rotation=0, labelpad=42, va="center")
    bin_seconds = 1 if duration <= 90 else 5
    bins = (records.time_s / bin_seconds).astype(int)
    fractions = pd.crosstab(bins, source, normalize="index").reindex(columns=["detected", "interpolated", "missing"], fill_value=0)
    x = fractions.index.to_numpy(dtype=float) * bin_seconds
    y0 = np.zeros(len(fractions))
    for key, label in (("detected", "真实检测"), ("interpolated", "短时补点"), ("missing", "未跟踪")):
        y1 = y0 + fractions[key].to_numpy(dtype=float)
        ax.fill_between(x, y0 * 100, y1 * 100, step="post", color=colors[key], alpha=.85, label=label)
        y0 = y1
    ax.set(xlim=(0, duration), ylim=(0, 100), title=f"跟踪质量 · 每 {bin_seconds} 秒占比", xlabel="视频时间（分:秒）", ylabel="帧占比（%）")
    ax.xaxis.set_major_formatter(time_formatter)
    ax.grid(alpha=.13)
    ax.legend(loc="lower left", ncol=3, framealpha=.7, facecolor="#102235")
    p = out_dir / "tracking_quality.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    return paths


def process_video(video_path: str, conf: float, ball_conf: float, trail_seconds: float, match_mode: str = "单打（最多 2 名球员）", progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video first.")
    run_dir = RUNS_DIR / time.strftime("%Y%m%d") / uuid.uuid4().hex[:10]
    run_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("The video could not be opened.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_path = run_dir / "annotated_raw.mp4"
    out = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    trail = deque(maxlen=max(12, int(fps * trail_seconds)))
    last_ball = None
    last_seen_frame = -999
    previous_grays = deque(maxlen=2)
    player_zone = court_polygon(width, height, expanded=False)
    ball_zone = court_polygon(width, height, expanded=True)
    player_limit = 4 if "4" in match_mode else 2
    records = []
    frame_idx = 0
    chart_background = None
    progress(0, desc="Loading detector")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if chart_background is None:
            chart_background = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = model().predict(frame, classes=[PERSON_CLASS, RACKET_CLASS], conf=conf, iou=.45, device=DEVICE, verbose=False)[0]
        ball_result = ball_model().predict(frame, conf=ball_conf, iou=.40, imgsz=1280, device=DEVICE, half=True, verbose=False)[0]
        players, rackets, balls = [], [], []
        if result.boxes is not None:
            for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                cls = int(cls)
                if cls == PERSON_CLASS:
                    foot = (int((box[0] + box[2]) / 2), int(box[3]))
                    if inside(foot, player_zone):
                        players.append((box, float(score)))
                elif cls == RACKET_CLASS: rackets.append((box, float(score)))

        # The dedicated tennis-ball model runs at 1280px instead of the generic
        # detector's 640px input, preserving the 2–6 pixel ball at the far end.
        if ball_result.boxes is not None:
            for box, score in zip(ball_result.boxes.xyxy.cpu().numpy(), ball_result.boxes.conf.cpu().numpy()):
                cx, cy = center(box)
                bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
                if inside((cx, cy), ball_zone) and max(bw, bh) <= max(width, height) * .045 and min(bw, bh) >= 1 and .25 <= bw / max(bh, 1) <= 4.0:
                    balls.append((cx, cy, float(score)))
        raw_ball_candidates = len(balls)

        # Rank court candidates by detector confidence and closeness to the
        # court's centre axis. In singles mode, prefer one player at each end.
        def player_rank(item):
            box, score = item
            px = float((box[0] + box[2]) / 2) / width
            return score + .35 * (1 - min(1.0, abs(px - .5) * 2))

        if player_limit == 2 and len(players) > 2:
            far = [p for p in players if p[0][3] / height < .52]
            near = [p for p in players if p[0][3] / height >= .52]
            selected = []
            if far: selected.append(max(far, key=player_rank))
            if near: selected.append(max(near, key=player_rank))
            if len(selected) < 2:
                selected_ids = {id(p) for p in selected}
                remaining = [p for p in players if id(p) not in selected_ids]
                selected.extend(sorted(remaining, key=player_rank, reverse=True)[:2-len(selected)])
            players = selected
        else:
            players = sorted(players, key=player_rank, reverse=True)[:player_limit]

        gap = frame_idx - last_seen_frame
        if gap > max(3, int(fps * .12)):
            last_ball = None
            trail.clear()
        balls = moving_ball_candidates(balls, gray, previous_grays, last_ball)
        # A real tennis ball cannot teleport across a broadcast frame. Allow a
        # larger first acquisition, then use a strict per-frame motion gate.
        ball = nearest_candidate(balls, last_ball, max(width, height) * min(.10, .025 * max(1, gap)))
        ball_speed = 0.0
        if ball:
            if last_ball is not None and frame_idx - last_seen_frame <= 3:
                dt = max(1, frame_idx - last_seen_frame) / fps
                ball_speed = math.dist(ball[:2], last_ball[:2]) / dt
            trail.append((ball[0], ball[1]))
            last_ball, last_seen_frame = ball, frame_idx

        for i, (box, score) in enumerate(players, 1):
            x1, y1, x2, y2 = map(int, box); cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 212, 191), 2)
            draw_label(frame, f"Player {i}  {score:.2f}", (x1, max(22, y1)), (45, 212, 191))
        for box, score in rackets:
            x1, y1, x2, y2 = map(int, box); cv2.rectangle(frame, (x1, y1), (x2, y2), (251, 113, 133), 2)
            draw_label(frame, f"Racket  {score:.2f}", (x1, max(22, y1)), (251, 113, 133))
        pts = list(trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (34, 211, 238), max(1, int(5 * i / len(pts))), cv2.LINE_AA)
        if ball:
            cv2.circle(frame, ball[:2], 8, (20, 20, 20), 3); cv2.circle(frame, ball[:2], 6, (34, 238, 168), -1)
            draw_label(frame, f"Ball {ball[2]:.2f}", (ball[0] + 9, max(22, ball[1] - 8)), (34, 238, 168))
        cv2.rectangle(frame, (0, 0), (width, 48), (12, 18, 30), -1)
        cv2.putText(frame, f"TENNIS VISION  |  {frame_idx/fps:06.2f}s  |  Players {len(players)}  |  Ball speed {ball_speed:,.0f} px/s", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, .67, (235, 245, 255), 2, cv2.LINE_AA)
        out.write(frame)
        records.append({"frame": frame_idx, "time_s": round(frame_idx / fps, 3), "ball_x": ball[0] if ball else np.nan, "ball_y": ball[1] if ball else np.nan, "ball_confidence": ball[2] if ball else np.nan, "ball_speed_px_s": round(ball_speed, 2), "ball_candidates": raw_ball_candidates, "moving_candidates": len(balls), "players": len(players), "rackets": len(rackets)})
        previous_grays.append(gray)
        frame_idx += 1
        if frame_count and frame_idx % 10 == 0:
            progress(min(.96, frame_idx / frame_count), desc=f"Analyzing frame {frame_idx:,}/{frame_count:,}")
    cap.release(); out.release()

    if not records:
        raise gr.Error("No decodable frames were found.")
    final_path = run_dir / "annotated.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_path)]
    try:
        subprocess.run(cmd, check=True); raw_path.unlink(missing_ok=True)
    except Exception:
        shutil.move(raw_path, final_path)

    df = interpolate_short_gaps(pd.DataFrame(records), width, height)
    df = add_trajectory_segments(df, width, height)
    # Recalculate image-plane speed for the cleaned trajectory, including safe
    # short-gap interpolation, while keeping breaks between trajectory segments.
    for _, indexes in df.dropna(subset=["trajectory_segment"]).groupby("trajectory_segment").groups.items():
        ordered = list(indexes)
        for previous, current in zip(ordered, ordered[1:]):
            dt = (df.at[current, "frame"] - df.at[previous, "frame"]) / fps
            if dt > 0:
                df.at[current, "ball_speed_px_s"] = math.dist(
                    (df.at[current, "ball_x"], df.at[current, "ball_y"]),
                    (df.at[previous, "ball_x"], df.at[previous, "ball_y"]),
                ) / dt
    csv_path = run_dir / "frame_metrics.csv"; df.to_csv(csv_path, index=False)
    choices, default_segment = trajectory_choices(df)
    selected_frames = df[df.trajectory_segment == default_segment] if default_segment is not None else df.iloc[0:0]
    selected_background = frame_at(video_path, int(selected_frames.frame.iloc[0])) if len(selected_frames) else chart_background
    charts = save_charts(df, run_dir, width, height, default_segment, selected_background)
    (run_dir / "source_path.txt").write_text(video_path, encoding="utf-8")
    detected = df.ball_source.eq("detected")
    interpolated = df.ball_source.eq("interpolated")
    valid = df.ball_x.notna()
    segment_sizes = df.dropna(subset=["trajectory_segment"]).groupby("trajectory_segment").size()
    useful_segments = int((segment_sizes >= 2).sum())
    detection_rate = float(detected.mean() * 100)
    coverage_rate = float(valid.mean() * 100)
    active_speed = df.loc[df.ball_speed_px_s > 0, "ball_speed_px_s"]
    summary = {
        "视频信息": {"时长（秒）": round(len(df) / fps, 2), "总帧数": len(df), "帧率": round(fps, 2), "分辨率": f"{width}×{height}"},
        "网球跟踪": {"模型候选帧数": int((df.ball_candidates > 0).sum()), "运动球跟踪帧数": int(detected.sum()), "运动球跟踪率（%）": round(detection_rate, 1), "短时补点帧数": int(interpolated.sum()), "轨迹覆盖帧数": int(valid.sum()), "轨迹覆盖率（%）": round(coverage_rate, 1), "有效轨迹段数": useful_segments, "最高画面速度（像素/秒）": round(float(active_speed.max()), 1) if len(active_speed) else 0, "平均画面速度（像素/秒）": round(float(active_speed.mean()), 1) if len(active_speed) else 0},
        "目标统计": {"平均球员数/帧": round(float(df.players.mean()), 2), "单帧最多球员数": int(df.players.max()), "检测到球拍的帧数": int((df.rackets > 0).sum())},
        "说明": "运动球跟踪率以全部视频帧为分母；静止球、发球准备和暂停时间可能没有轨迹点。速度为画面像素速度，真实 km/h 需要摄像机和球场标定。"
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress(1, desc="Complete")
    default_index = next((i for i, (_, sid) in enumerate(choices) if sid == default_segment), 0)
    page_label = f"第 {default_index + 1} / {len(choices)} 段 · {choices[default_index][0]}" if choices else "没有可展示的轨迹段"
    return str(final_path), *charts, summary, str(csv_path), str(run_dir), choices, default_index, page_label


def render_selected_trajectory(selected_segment: int | None, run_dir_value: str):
    if selected_segment is None or not run_dir_value:
        return None
    run_dir = Path(run_dir_value).resolve()
    if not run_dir.is_relative_to(RUNS_DIR.resolve()):
        raise gr.Error("轨迹数据目录无效。")
    selected_segment = int(selected_segment)
    cached = run_dir / f"trajectory_segment_{selected_segment}.png"
    if cached.is_file():
        return str(cached)
    csv_path = run_dir / "frame_metrics.csv"
    source_file = run_dir / "source_path.txt"
    if not csv_path.is_file() or not source_file.is_file():
        raise gr.Error("轨迹数据已失效，请重新分析视频。")
    records = pd.read_csv(csv_path)
    segment = records[records.trajectory_segment == selected_segment]
    if len(segment) < 5:
        raise gr.Error("所选轨迹段不存在。")
    source_path = source_file.read_text(encoding="utf-8")
    cap = cv2.VideoCapture(source_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise gr.Error("原视频已失效，请重新上传。")
    background = frame_at(source_path, int(segment.frame.iloc[0]))
    return render_trajectory_chart(records, run_dir, width, height, selected_segment, background)


def turn_trajectory_page(current_index: int, choices: list, run_dir_value: str, direction: int):
    if not choices or not run_dir_value:
        return gr.skip(), current_index, "没有可展示的轨迹段"
    index = max(0, min(len(choices) - 1, int(current_index) + direction))
    label, segment_id = choices[index]
    if index == current_index:
        return gr.skip(), index, f"第 {index + 1} / {len(choices)} 段 · {label}"
    image_path = render_selected_trajectory(segment_id, run_dir_value)
    return image_path, index, f"第 {index + 1} / {len(choices)} 段 · {label}"


def previous_trajectory(current_index: int, choices: list, run_dir_value: str):
    return turn_trajectory_page(current_index, choices, run_dir_value, -1)


def next_trajectory(current_index: int, choices: list, run_dir_value: str):
    return turn_trajectory_page(current_index, choices, run_dir_value, 1)


CSS = """
.gradio-container{max-width:1400px!important;background:radial-gradient(circle at 15% 0,#10243b 0,#07111f 42%,#050a12 100%)!important}
.hero{padding:26px 28px;border:1px solid #21364c;border-radius:20px;background:linear-gradient(135deg,rgba(21,47,73,.9),rgba(8,19,33,.9));margin-bottom:18px}
.hero h1{font-size:34px;margin:0;color:#eef8ff}.hero p{color:#9fc4dc;margin:8px 0 0}.accent{color:#35e6b0}
.panel{border:1px solid #20354b!important;border-radius:16px!important;background:rgba(9,20,34,.82)!important}
#analysis-tabs button[role="tab"],#analysis-tabs .tab-nav button{color:#fff!important;background:#25415d!important;font-weight:800!important;font-size:17px!important;border:1px solid #84aac4!important;opacity:1!important}
#analysis-tabs button[role="tab"] *,#analysis-tabs .tab-nav button *{color:inherit!important;opacity:1!important}
#analysis-tabs button[role="tab"][aria-selected="true"],#analysis-tabs .tab-nav button.selected{color:#07211a!important;background:#55f0bf!important;border-color:#55f0bf!important}
#analysis-tabs button[role="tab"]:hover,#analysis-tabs .tab-nav button:hover{background:#416b8a!important;color:#fff!important}
.trajectory-page-label,.trajectory-page-label *{color:#eef8ff!important;font-weight:700!important;font-size:16px!important}
.chart-note,.chart-note *{color:#d8e7f2!important}
footer{display:none!important}
"""

with gr.Blocks(title="网球视觉分析平台") as demo:
    gr.HTML("<div class='hero'><h1>网球<span class='accent'>视觉分析平台</span></h1><p>上传比赛视频，自动识别场内球员、球拍和网球，重建网球轨迹并生成逐帧分析数据。</p></div>")
    with gr.Row():
        with gr.Column(scale=5, elem_classes="panel"):
            inp = gr.Video(label="输入视频", sources=["upload"])
            with gr.Row():
                conf = gr.Slider(.05, .60, value=CONFIDENCE, step=.01, label="检测置信度")
                trail_s = gr.Slider(.3, 5, value=2, step=.1, label="轨迹保留时长（秒）")
            ball_conf = gr.Slider(.01, .50, value=BALL_CONFIDENCE, step=.01, label="网球专用模型置信度")
            match_mode = gr.Radio(["单打（最多 2 名球员）", "双打（最多 4 名球员）"], value="单打（最多 2 名球员）", label="比赛模式")
            run = gr.Button("开始分析", variant="primary", size="lg")
            gr.Markdown("建议使用固定机位、球场完整可见且分辨率不低于 720P 的视频。系统只统计脚点位于球场区域内的球员。")
        with gr.Column(scale=7, elem_classes="panel"):
            output_video = gr.Video(label="检测结果视频", autoplay=False)
    with gr.Tabs(elem_id="analysis-tabs"):
        with gr.Tab("球场轨迹"):
            with gr.Row():
                previous_page = gr.Button("← 上一段", variant="secondary")
                page_label = gr.Markdown("分析后可逐段翻页查看轨迹", elem_classes="trajectory-page-label")
                next_page = gr.Button("下一段 →", variant="secondary")
            trajectory_chart = gr.Image(label="网球主要轨迹", interactive=False, height=560)
            gr.Markdown("一次只显示一段轨迹；点击上一段或下一段翻页。明显转向或约 3 秒时会分段；绿色圆点是起点，粉色 × 是终点，空心点是短时补点。", elem_classes="chart-note")
        with gr.Tab("速度趋势"):
            speed_chart = gr.Image(label="画面速度变化", interactive=False, height=500)
            gr.Markdown("粉色背景表示至少连续 2 帧未跟踪到运动球；重新捕获或切换轨迹段时也会断线。这里是画面像素速度，不能直接当作真实 km/h。", elem_classes="chart-note")
        with gr.Tab("跟踪质量"):
            quality_chart = gr.Image(label="逐帧跟踪状态", interactive=False, height=500)
            gr.Markdown("上方色带显示每一帧状态，下方显示各时间段真实检测、短时补点和未跟踪的比例。", elem_classes="chart-note")
    with gr.Row():
        summary_out = gr.JSON(label="分析摘要")
        csv_out = gr.File(label="逐帧数据（CSV）")
    run_state = gr.State(value="")
    page_choices = gr.State(value=[])
    page_index = gr.State(value=0)
    run.click(process_video, [inp, conf, ball_conf, trail_s, match_mode], [output_video, trajectory_chart, speed_chart, quality_chart, summary_out, csv_out, run_state, page_choices, page_index, page_label], concurrency_limit=1)
    previous_page.click(previous_trajectory, [page_index, page_choices, run_state], [trajectory_chart, page_index, page_label], concurrency_limit=1)
    next_page.click(next_trajectory, [page_index, page_choices, run_state], [trajectory_chart, page_index, page_label], concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7861, show_error=True, allowed_paths=[str(RUNS_DIR)], css=CSS, theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"))
