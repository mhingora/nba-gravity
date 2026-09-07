"""Pipeline testing & debugging viewer.

Run with:
    streamlit run app.py

One tab per pipeline stage (docs/08-ui-design.md). Each tab is a read-only view
over that stage's cached output under `outputs/` — the UI never re-runs
processing and never imports stage logic, so it cannot drift from what the
pipeline actually produced. Run stages from the command line, then refresh.

Tabs for stages that aren't built yet render a placeholder rather than being
hidden, so the viewer grows alongside the roadmap.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

# Streamlit only puts the app's directory on sys.path when launched a specific
# way; make `import pipeline` work regardless of where this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv
from PIL import Image

from pipeline.common import (
    CLASS_BALL,
    CLASS_PLAYER,
    detections_path,
    find_video,
    list_games,
    load_shots,
    read_frame,
    identity_path,
    possession_path,
    shot_diffs_path,
    tracks_path,
    video_info,
)
from pipeline.team_clustering import (
    TEAM_DARK,
    TEAM_LIGHT,
    TEAM_OTHER,
    crop_torso,
)
from pipeline.court_region import (
    DEFAULT_BROADCAST_POLYGON,
    list_profiles,
    load_profile,
    parse_polygon,
    points_in_polygon,
    save_profile,
    to_pixels,
)
from pipeline.court_geometry import (
    COURT_LANDMARKS,
    lane_orientation_problem,
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    MIN_LANDMARKS,
    compute_homography,
    known_distance_checks,
    parse_keypoints,
    to_court_feet,
    worst_landmark,
)
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # optional: the tab falls back to typing coordinates
    streamlit_image_coordinates = None

from pipeline.possession import possession_summary
from pipeline.track_quality import EXPECTED_PLAYERS, summarize

st.set_page_config(page_title="NBA Gravity — Pipeline Viewer", layout="wide")

BALL_COLOR_BGR = (40, 140, 245)
MAX_GIF_FRAMES = 400
# The click-to-annotate frame is rendered at "stretch" so it fills whatever
# container holds it. A fixed pixel width overflows a narrow column and is
# *clipped* rather than scaled, which silently hides part of the court and
# makes those landmarks unreachable. Normalising uses the rendered width and
# height the component returns with each click
# (`sendValue({x: offsetX, y: offsetY, width: img.width, height: img.height})`),
# so any rendered size is handled correctly.
CLICK_DISPLAY_WIDTH = "stretch"


# --------------------------------------------------------------------------
# Cached loaders. Keying on mtime means editing a parquet on disk and hitting
# "rerun" picks up the new data without clearing caches by hand.
# --------------------------------------------------------------------------


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_parquet(path_str: str, _mtime_key: float) -> pd.DataFrame:
    return pd.read_parquet(path_str)


@st.cache_data(show_spinner=False)
def load_shot_list(game_id: str, _mtime_key: float):
    return load_shots(game_id)


@st.cache_data(show_spinner=False, max_entries=64)
def get_frame(video_path_str: str, frame_idx: int) -> np.ndarray | None:
    from pathlib import Path

    return read_frame(Path(video_path_str), frame_idx)


@st.cache_data(show_spinner=False)
def get_video_info(video_path_str: str):
    from pathlib import Path

    return video_info(Path(video_path_str))


# --------------------------------------------------------------------------
# Annotation
# --------------------------------------------------------------------------


def to_detections(rows: pd.DataFrame, color_by: str) -> sv.Detections:
    """Build sv.Detections from parquet rows, keyed for the requested coloring."""
    if rows.empty:
        return sv.Detections.empty()

    xyxy = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    class_id = (rows["class"] == CLASS_BALL).to_numpy().astype(int)
    detections = sv.Detections(xyxy=xyxy, class_id=class_id)

    if "confidence" in rows:
        detections.confidence = rows["confidence"].to_numpy(dtype=np.float32)
    if color_by == "track" and "tracker_id" in rows:
        detections.tracker_id = rows["tracker_id"].to_numpy(dtype=int)
    return detections


def annotate(
    frame: np.ndarray, detections: sv.Detections, labels: list[str], color_by: str
) -> np.ndarray:
    lookup = sv.ColorLookup.TRACK if color_by == "track" else sv.ColorLookup.CLASS
    box = sv.BoxAnnotator(color_lookup=lookup, thickness=2)
    label = sv.LabelAnnotator(
        color_lookup=lookup, text_scale=0.4, text_padding=3, text_position=sv.Position.TOP_LEFT
    )
    out = box.annotate(frame.copy(), detections)
    return label.annotate(out, detections, labels=labels)


def draw_ball(frame: np.ndarray, ball_rows: pd.DataFrame) -> np.ndarray:
    """Ball is drawn separately — it has no tracker_id to color by."""
    for row in ball_rows.itertuples():
        cx = int((row.x1 + row.x2) / 2)
        cy = int((row.y1 + row.y2) / 2)
        radius = max(int(max(row.x2 - row.x1, row.y2 - row.y1) / 2), 4)
        cv2.circle(frame, (cx, cy), radius + 3, BALL_COLOR_BGR, 2)
        cv2.putText(
            frame, "ball", (cx - 12, cy - radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, BALL_COLOR_BGR, 1, cv2.LINE_AA,
        )
    return frame


def _draw_coordinate_grid(frame: np.ndarray, step: float) -> None:
    """Overlay labelled normalized gridlines, for reading landmark coordinates.

    Annotation means naming the pixel a court landmark sits at, and without
    a reference there is no way to tell 0.34 from 0.38 by eye. The labels are
    the whole point of the grid.
    """
    height, width = frame.shape[:2]
    ticks = int(round(1.0 / step))
    for i in range(1, ticks):
        value = i * step
        x = int(width * value)
        y = int(height * value)
        cv2.line(frame, (x, 0), (x, height), (0, 210, 210), 1)
        cv2.line(frame, (0, y), (width, y), (210, 0, 210), 1)
        # Label every other line at fine spacings, or the text becomes a wall.
        if step >= 0.05 or i % 2 == 0:
            cv2.putText(frame, f"{value:.3f}".rstrip("0"), (x + 3, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 210, 210), 1)
            cv2.putText(frame, f"{value:.3f}".rstrip("0"), (3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 0, 210), 1)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Frame scrubber
# --------------------------------------------------------------------------


def frame_scrubber(key: str, start: int, end: int) -> int:
    """Slider plus step buttons.

    Stepping one frame at a time is the point — coarse scrubbing hides exactly
    the failures worth catching, like a ball detection dropping out mid-pass.
    """
    state_key = f"{key}_frame"
    if state_key not in st.session_state or not (
        start <= st.session_state[state_key] <= end
    ):
        st.session_state[state_key] = start

    def step(delta: int):
        def _step():
            st.session_state[state_key] = int(
                np.clip(st.session_state[state_key] + delta, start, end)
            )

        return _step

    cols = st.columns([1, 1, 1, 1, 6])
    cols[0].button("◀◀ -10", key=f"{key}_b10", on_click=step(-10))
    cols[1].button("◀ -1", key=f"{key}_b1", on_click=step(-1))
    cols[2].button("+1 ▶", key=f"{key}_f1", on_click=step(1))
    cols[3].button("+10 ▶▶", key=f"{key}_f10", on_click=step(10))

    if end > start:
        st.slider("Frame", start, end, key=state_key)
    else:
        st.caption(f"Single-frame range (frame {start})")
    return int(st.session_state[state_key])


def shot_selector(key: str, shots, label: str = "Shot"):
    options = ["All shots"] + [
        f"Shot {s.shot_id}  ({s.start_frame}–{s.end_frame}, {s.n_frames}f)"
        for s in shots
    ]
    choice = st.selectbox(label, options, key=f"{key}_shot")
    if choice == "All shots":
        return None
    return shots[options.index(choice) - 1]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("NBA Gravity")
st.sidebar.caption("Pipeline debugging viewer")

games = list_games()
if not games:
    st.title("No games found")
    st.warning(
        "Drop a video at `data/raw_video/{game_id}/{game_id}.mp4`, then run "
        "`python pipeline/01_detect.py --game-id {game_id}`.\n\n"
        "To try the viewer without footage, generate a synthetic clip:\n"
        "`python tools/make_test_clip.py --game-id TESTCLIP`"
    )
    st.stop()

game_id = st.sidebar.selectbox("Game", games)

try:
    video_path = find_video(game_id)
    info = get_video_info(str(video_path))
    st.sidebar.caption(
        f"{video_path.name}\n\n{info.width}×{info.height} @ {info.fps:.1f}fps\n\n"
        f"{info.frame_count} frames"
    )
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

det_path = detections_path(game_id)
trk_path = tracks_path(game_id)
shots = load_shot_list(game_id, _mtime(det_path.parent / f"{game_id}_shots.json"))

st.sidebar.divider()
st.sidebar.write("**Stage output**")
st.sidebar.write(f"{'✅' if shots else '⬜'} Shot boundaries ({len(shots)})")
st.sidebar.write(f"{'✅' if det_path.exists() else '⬜'} Detections")
st.sidebar.write(f"{'✅' if trk_path.exists() else '⬜'} Tracks")

tab_names = [
    "1 · Detection",
    "2 · Tracking",
    "3 · Team Classification",
    "4 · Ball Possession",
    "5 · Court Calibration",
    "6 · Identity Resolution",
    "7 · Gravity Results",
]
tabs = st.tabs(tab_names)


def milestone_placeholder(stage: str, milestone: str, script: str) -> None:
    st.subheader(stage)
    st.info(
        f"Not built yet — arrives with **{milestone}** (`pipeline/{script}`).\n\n"
        "See `docs/06-roadmap.md` for the milestone definition and "
        "`docs/08-ui-design.md` for what this tab will show."
    )


# --------------------------------------------------------------------------
# Tab 1 — Detection Viewer
# --------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Detection Viewer")
    st.caption("Are the detector's boxes landing on players and the ball?")

    if not det_path.exists():
        st.info(
            f"No detections yet. Run:\n\n"
            f"`python pipeline/01_detect.py --game-id {game_id}`"
        )
    else:
        detections_df = load_parquet(str(det_path), _mtime(det_path))
        shot = shot_selector("det", shots)

        if shot is None:
            lo = int(detections_df["frame_idx"].min())
            hi = int(detections_df["frame_idx"].max())
        else:
            lo, hi = shot.start_frame, shot.end_frame

        frame_idx = frame_scrubber("det", lo, hi)
        rows = detections_df[detections_df["frame_idx"] == frame_idx]

        players = rows[rows["class"] == CLASS_PLAYER]
        balls = rows[rows["class"] == CLASS_BALL]

        stat_cols = st.columns(4)
        stat_cols[0].metric("Player detections", len(players))
        stat_cols[1].metric("Ball detections", len(balls))
        stat_cols[2].metric(
            "Min confidence",
            f"{rows['confidence'].min():.2f}" if len(rows) else "—",
        )
        stat_cols[3].metric(
            "Shot", shot.shot_id if shot else "—"
        )
        if len(balls) == 0:
            st.warning("No ball detected on this frame.")

        frame = get_frame(str(video_path), frame_idx)
        if frame is None:
            st.error(f"Could not read frame {frame_idx}.")
        else:
            labels = [
                f"{r['class']} {r['confidence']:.2f}" for _, r in rows.iterrows()
            ]
            annotated = annotate(frame, to_detections(rows, "class"), labels, "class")
            st.image(bgr_to_rgb(annotated), width="stretch")

        with st.expander("Detections on this frame"):
            st.dataframe(
                rows[["class", "confidence", "x1", "y1", "x2", "y2"]]
                .sort_values("confidence", ascending=False)
                .reset_index(drop=True),
                width="stretch",
            )

        with st.expander("Per-frame detection counts (spot systematically bad frames)"):
            counts = (
                detections_df[detections_df["frame_idx"].between(lo, hi)]
                .groupby(["frame_idx", "class"])
                .size()
                .unstack(fill_value=0)
            )
            st.line_chart(counts)

# --------------------------------------------------------------------------
# Tab 2 — Tracking Viewer
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Rendering annotated clip…")
def render_shot_gif(
    game_id: str,
    video_path_str: str,
    start_frame: int,
    end_frame: int,
    stride: int,
    fps: float,
    _mtime_key: float,
) -> bytes:
    """Render an annotated shot as a GIF for playback in the browser.

    ID stability is far easier to judge in motion than by stepping through
    stills. GIF rather than mp4 because browser mp4 playback needs an H.264
    encoder that OpenCV builds don't reliably ship.
    """
    tracks = pd.read_parquet(tracks_path(game_id))
    window = tracks[tracks["frame_idx"].between(start_frame, end_frame)]

    images: list[Image.Image] = []
    cap = cv2.VideoCapture(video_path_str)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % stride:
                continue
            rows = window[window["frame_idx"] == frame_idx]
            players = rows[rows["class"] == CLASS_PLAYER]
            labels = [f"#{int(t)}" for t in players["tracker_id"]]
            annotated = annotate(
                frame, to_detections(players, "track"), labels, "track"
            )
            annotated = draw_ball(annotated, rows[rows["class"] == CLASS_BALL])
            images.append(Image.fromarray(bgr_to_rgb(annotated)))
            if len(images) >= MAX_GIF_FRAMES:
                break
    finally:
        cap.release()

    if not images:
        return b""

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(int(1000 / (fps / stride)), 20),
        loop=0,
        optimize=True,
    )
    return buffer.getvalue()


with tabs[1]:
    st.subheader("Tracking Viewer")
    st.caption("Do tracker_ids stay stable across a shot, and where do they break?")

    if not trk_path.exists():
        st.info(
            f"No tracks yet. Run:\n\n"
            f"`python pipeline/02_track.py --game-id {game_id}`"
        )
    else:
        tracks_df = load_parquet(str(trk_path), _mtime(trk_path))
        shot = shot_selector("trk", shots)

        if tracks_df.empty:
            # A zero-row parquet is a real state an over-tight filter can
            # produce. Say so rather than dying on min() of nothing.
            st.warning(
                "The tracks file is empty — every detection was filtered out. "
                "Re-run `02_track.py` and check `--min-confidence` and any "
                "court polygon against the frame."
            )
            st.stop()

        if shot is None:
            lo = int(tracks_df["frame_idx"].min())
            hi = int(tracks_df["frame_idx"].max())
        else:
            lo, hi = shot.start_frame, shot.end_frame

        window = tracks_df[tracks_df["frame_idx"].between(lo, hi)]
        window_players = window[window["class"] == CLASS_PLAYER]

        summary_cols = st.columns(4)
        # tracker_id is only unique within a shot, so counting it alone across
        # several shots collapses distinct tracks and hides fragmentation.
        n_ids = len(window_players.groupby(["shot_id", "tracker_id"]))
        summary_cols[0].metric("Distinct tracks", n_ids)
        summary_cols[1].metric(
            "Frames", int(window["frame_idx"].nunique())
        )
        summary_cols[2].metric(
            "Median track_len",
            int(
                window_players.groupby(["shot_id", "tracker_id"])["track_len"]
                .first()
                .median()
            )
            if n_ids
            else 0,
        )
        summary_cols[3].metric("Shots in view", len(shots) if shot is None else 1)

        if shot is not None and n_ids > 12:
            st.warning(
                f"{n_ids} tracker_ids in a shot with 10 players on court — "
                "likely ID switching or track fragmentation."
            )

        # The Milestone 1 scorecard. Same function the CLI prints after every
        # `02_track.py` run, so the table here and the terminal never disagree.
        st.markdown("**Tracking quality per shot**")
        quality = summarize(window)
        if quality.empty:
            st.caption("No player tracks to score.")
        else:
            st.dataframe(
                quality.rename(
                    columns={
                        "n_frames": "frames",
                        "n_tracks": "tracks",
                        "median_per_frame": f"median/frame (want {EXPECTED_PLAYERS})",
                        "pct_frames_full": f"% frames with {EXPECTED_PLAYERS}+",
                        "longest_track": "longest",
                        "longest_pct": "longest %",
                        "tracks_over_half": "tracks >50% shot",
                        "tracks_over_90pct": "tracks >90% shot",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Milestone 1 closes when a shot shows ~10 tracks, a median of "
                f"{EXPECTED_PLAYERS}/frame, and {EXPECTED_PLAYERS} tracks "
                "spanning >90% of the shot. A low median means the detector is "
                "missing players; a high track count with a good median means "
                "crowd is being tracked or IDs are fragmenting — two different "
                "fixes, which is why both are shown."
            )

        left, right = st.columns([3, 2])

        with left:
            frame_idx = frame_scrubber("trk", lo, hi)
            rows = window[window["frame_idx"] == frame_idx]
            players = rows[rows["class"] == CLASS_PLAYER]

            frame = get_frame(str(video_path), frame_idx)
            if frame is None:
                st.error(f"Could not read frame {frame_idx}.")
            else:
                labels = [f"#{int(t)}" for t in players["tracker_id"]]
                annotated = annotate(
                    frame, to_detections(players, "track"), labels, "track"
                )
                annotated = draw_ball(annotated, rows[rows["class"] == CLASS_BALL])
                st.image(bgr_to_rgb(annotated), width="stretch")
                st.caption(
                    "Box color is keyed to tracker_id — a box changing color "
                    "mid-shot, or two players swapping colors, is an ID switch."
                )

        with right:
            st.markdown("**Tracks in view**")
            if window_players.empty:
                st.caption("No player tracks in this range.")
            else:
                per_track = (
                    window_players.groupby(["shot_id", "tracker_id"])
                    .agg(
                        track_len=("track_len", "first"),
                        first_frame=("frame_idx", "min"),
                        last_frame=("frame_idx", "max"),
                    )
                    .reset_index()
                    .sort_values("track_len")
                )
                st.dataframe(per_track, width="stretch", height=320)
                st.caption(
                    "Sorted shortest-first: brief tracks are usually detection "
                    "noise or a player lost behind another."
                )

        st.divider()
        st.markdown("**Play the shot** — ID stability reads much better in motion.")

        if shot is None:
            st.caption("Select a single shot above to render its annotated clip.")
        else:
            play_cols = st.columns([1, 1, 4])
            stride = play_cols[0].number_input(
                "Frame stride", 1, 10, 2, key="gif_stride"
            )
            if play_cols[1].button("Render clip", key="render_gif"):
                gif = render_shot_gif(
                    game_id,
                    str(video_path),
                    shot.start_frame,
                    shot.end_frame,
                    int(stride),
                    info.fps,
                    _mtime(trk_path),
                )
                if gif:
                    st.image(gif, width="stretch")
                    st.download_button(
                        "Download GIF",
                        gif,
                        file_name=f"{game_id}_shot{shot.shot_id}.gif",
                        mime="image/gif",
                    )
                else:
                    st.error("Nothing rendered — check the frame range.")

    # The polygon tuner renders whenever detections exist, tracking or not:
    # fitting the court is something you do *before* running Stage 2, and the
    # only thing it needs is somewhere to put foot points.
    if det_path.exists():
        st.divider()
        with st.expander("Court polygon tuner — fit the playing surface, copy the flag"):
            st.caption(
                "A broadcast camera sees the court as a trapezoid, so the "
                "axis-aligned `--court-roi` band cannot exclude spectators "
                "standing behind the far baseline without also cutting "
                "far-side players. Drag the numbers until green boxes are "
                "players and red boxes are crowd, then paste the flag below "
                "into `02_track.py`."
            )

            profiles = list_profiles()
            if profiles:
                load_cols = st.columns([3, 1])
                chosen = load_cols[0].selectbox(
                    "Load a saved profile", profiles, key="poly_load_name"
                )

                def _load_into_tuner() -> None:
                    # Written straight into session_state from a callback so it
                    # lands before the text area is instantiated on the rerun;
                    # assigning after the widget exists raises.
                    loaded = load_profile(st.session_state["poly_load_name"])
                    st.session_state["poly_text"] = " ".join(
                        f"{v:g}" for v in loaded["court_polygon"].reshape(-1)
                    )

                load_cols[1].button("Load", key="poly_load_btn", on_click=_load_into_tuner)

            # Seed once rather than passing `value=`: the Load button writes
            # this key from a callback, and Streamlit warns when a widget has
            # both a default and a session-state assignment.
            if "poly_text" not in st.session_state:
                st.session_state["poly_text"] = " ".join(
                    f"{v:g}" for v in DEFAULT_BROADCAST_POLYGON
                )
            raw = st.text_area(
                "Polygon — normalized `x y` pairs, clockwise around the court",
                key="poly_text",
                height=70,
            )

            polygon = None
            try:
                polygon = parse_polygon(
                    [float(v) for v in raw.replace(",", " ").split()]
                )
            except ValueError as exc:
                st.error(str(exc))

            if polygon is not None:
                det_all = load_parquet(str(det_path), _mtime(det_path))
                players_all = det_all[det_all["class"] == CLASS_PLAYER]

                if players_all.empty:
                    st.caption("No player detections to test against.")
                else:
                    p_lo = int(players_all["frame_idx"].min())
                    p_hi = int(players_all["frame_idx"].max())
                    poly_frame = frame_scrubber("poly", p_lo, p_hi)

                    frame = get_frame(str(video_path), poly_frame)
                    if frame is None:
                        st.error(f"Could not read frame {poly_frame}.")
                    else:
                        height, width = frame.shape[:2]
                        polygon_px = to_pixels(polygon, width, height)

                        rows = players_all[players_all["frame_idx"] == poly_frame]
                        foot_x = ((rows["x1"] + rows["x2"]) / 2.0).to_numpy()
                        foot_y = rows["y2"].to_numpy()
                        inside = points_in_polygon(foot_x, foot_y, polygon_px)

                        canvas = frame.copy()
                        for (_, row), keep in zip(rows.iterrows(), inside):
                            cv2.rectangle(
                                canvas,
                                (int(row["x1"]), int(row["y1"])),
                                (int(row["x2"]), int(row["y2"])),
                                (0, 200, 0) if keep else (0, 0, 220),
                                2,
                            )
                        cv2.polylines(
                            canvas, [polygon_px.astype(np.int32)], True, (255, 235, 0), 3
                        )
                        st.image(bgr_to_rgb(canvas), width="stretch")

                        # Whole-clip counts matter more than one frame: a
                        # polygon can look perfect on a still and still clip
                        # players once the camera pans.
                        all_inside = points_in_polygon(
                            ((players_all["x1"] + players_all["x2"]) / 2.0).to_numpy(),
                            players_all["y2"].to_numpy(),
                            polygon_px,
                        )
                        kept_per_frame = (
                            players_all.assign(_keep=all_inside)
                            .groupby("frame_idx")["_keep"]
                            .sum()
                        )

                        cols = st.columns(4)
                        cols[0].metric("Kept this frame", int(inside.sum()))
                        cols[1].metric("Rejected this frame", int((~inside).sum()))
                        cols[2].metric(
                            "Median kept/frame", int(kept_per_frame.median())
                        )
                        cols[3].metric(
                            f"% frames with {EXPECTED_PLAYERS}+",
                            f"{100.0 * (kept_per_frame >= EXPECTED_PLAYERS).mean():.0f}%",
                        )
                        st.caption(
                            f"Aim for a median near {EXPECTED_PLAYERS} with a high "
                            "percentage — a median far above it means crowd is "
                            "still getting through, far below means the polygon "
                            "is clipping real players."
                        )

                        st.markdown("**Save as a court profile**")
                        st.caption(
                            "A polygon belongs to a camera angle, not a game — "
                            "save it once and every game shot from the same "
                            "position can reuse it. Stored in "
                            "`data/calibration/`, alongside the tracker "
                            "settings you tuned against the same footage."
                        )
                        save_cols = st.columns(4)
                        prof_name = save_cols[0].text_input(
                            "Profile name", value="my_camera", key="poly_save_name"
                        )
                        buf = save_cols[1].number_input(
                            "lost_track_buffer", 1, 600, 150, key="poly_buf"
                        )
                        match = save_cols[2].number_input(
                            "min_matching_threshold", 0.1, 0.99, 0.95, 0.05,
                            key="poly_match",
                        )
                        activation = save_cols[3].number_input(
                            "track_activation", 0.05, 0.95, 0.30, 0.05,
                            key="poly_activation",
                        )

                        if st.button("Save profile", key="poly_save_btn"):
                            if not prof_name.strip():
                                st.error("Give the profile a name.")
                            else:
                                written = save_profile(
                                    prof_name.strip(),
                                    polygon,
                                    {
                                        "lost_track_buffer": int(buf),
                                        "minimum_matching_threshold": float(match),
                                        "track_activation_threshold": float(activation),
                                    },
                                    f"Fitted in the viewer on {game_id}, frame "
                                    f"{poly_frame}, at {width}x{height}.",
                                )
                                st.success(f"Wrote {written}")

                        st.caption("Then run Stage 2 with one flag:")
                        st.code(
                            f"python pipeline/02_track.py --game-id {game_id} "
                            f"--court-profile {prof_name.strip() or 'NAME'}",
                            language="bash",
                        )

                        flag = " ".join(f"{v:g}" for v in polygon.reshape(-1))
                        with st.popover("…or paste the raw polygon flag"):
                            st.code(
                                f"python pipeline/02_track.py --game-id {game_id} "
                                f"--court-polygon {flag}",
                                language="bash",
                            )

    # Shot boundaries render whether or not tracking has run: segmenting shots
    # needs no detector, and reading this table is how you pick a clean
    # possession to point Stage 1 at in the first place.
    if shots:
        st.divider()
        st.markdown("**Shot boundaries**")
        diff_path = shot_diffs_path(game_id)
        if diff_path.exists():
            diffs = load_parquet(str(diff_path), _mtime(diff_path))
            st.line_chart(diffs.set_index("frame_idx")["cut_distance"])
            st.caption(
                "Frame-to-frame histogram distance. Peaks are cuts — compare "
                "against `--cut-threshold` when tuning Stage 0."
            )

        shot_table = pd.DataFrame(
            [
                {
                    "shot_id": s.shot_id,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "n_frames": s.n_frames,
                    "seconds": round(s.n_frames / info.fps, 1),
                    "start_time": f"{int(s.start_frame / info.fps) // 60}:"
                    f"{int(s.start_frame / info.fps) % 60:02d}",
                    "camera_angle": s.camera_angle,
                }
                for s in shots
            ]
        )
        st.dataframe(shot_table, width="stretch")
        st.caption(
            "Pick a long, steady shot and pass its frame range to "
            "`01_detect.py --start-frame/--end-frame`."
        )

# --------------------------------------------------------------------------
# Tabs 3-7 — placeholders until their milestone lands
# --------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Team Classification")
    st.caption(
        "Do the two clusters obviously correspond to the two teams' kit "
        "colours? This is a five-second visual check, not a chart."
    )

    ident_path = identity_path(game_id)
    if not ident_path.exists():
        st.info(
            f"No identity output yet. Run:\n\n"
            f"`python pipeline/03_identify.py --game-id {game_id}`"
        )
    elif not trk_path.exists():
        st.warning("Tracks are missing, so crops cannot be located.")
    else:
        identity = load_parquet(str(ident_path), _mtime(ident_path))
        tracks_for_crops = load_parquet(str(trk_path), _mtime(trk_path))
        tracks_for_crops = tracks_for_crops[
            tracks_for_crops["class"] == CLASS_PLAYER
        ]

        counts = identity["team_id"].value_counts()
        stat_cols = st.columns(4)
        stat_cols[0].metric("Tracks clustered", len(identity))
        stat_cols[1].metric(TEAM_LIGHT, int(counts.get(TEAM_LIGHT, 0)))
        stat_cols[2].metric(TEAM_DARK, int(counts.get(TEAM_DARK, 0)))
        stat_cols[3].metric(
            TEAM_OTHER,
            int(counts.get(TEAM_OTHER, 0)),
            help="Colour evidence favoured neither team — referees, coaches, "
            "crowd. Confirm these are genuinely not players.",
        )

        light_n = int(counts.get(TEAM_LIGHT, 0))
        dark_n = int(counts.get(TEAM_DARK, 0))
        if light_n and dark_n and max(light_n, dark_n) > 3 * min(light_n, dark_n):
            st.warning(
                f"Clusters are lopsided ({light_n} vs {dark_n}). Both teams "
                "have five players on court, so a large imbalance usually "
                "means one kit is being split or the crops are contaminated."
            )

        st.markdown("**Torso crops, grouped by assigned team**")
        for team in (TEAM_LIGHT, TEAM_DARK, TEAM_OTHER):
            members = identity[identity["team_id"] == team]
            if members.empty:
                continue
            st.caption(f"{team} — {len(members)} track(s)")
            columns = st.columns(8)
            for slot, (_, row) in enumerate(members.iterrows()):
                track_rows = tracks_for_crops[
                    (tracks_for_crops["shot_id"] == row["shot_id"])
                    & (tracks_for_crops["tracker_id"] == row["tracker_id"])
                ].sort_values("frame_idx")
                if track_rows.empty:
                    continue
                # Mid-track frame: the start and end of a track are where it is
                # most likely to be entering or leaving an occlusion.
                middle = track_rows.iloc[len(track_rows) // 2]
                frame = get_frame(str(video_path), int(middle["frame_idx"]))
                if frame is None:
                    continue
                crop = crop_torso(
                    frame,
                    (middle["x1"], middle["y1"], middle["x2"], middle["y2"]),
                )
                if crop is None:
                    continue
                with columns[slot % 8]:
                    st.image(bgr_to_rgb(crop), width="stretch")
                    st.caption(
                        f"#{int(row['tracker_id'])} · "
                        f"{row['identity_confidence']:.2f}"
                    )

        st.caption(
            "Caption is tracker_id and identity_confidence. A crop that "
            "obviously belongs to the other group is a clustering error — the "
            "usual cause is a crop contaminated by background or an "
            "overlapping player rather than the kit colour itself."
        )

        with st.expander("Identity table"):
            st.dataframe(identity, width="stretch", hide_index=True)
            st.caption(
                "`jersey_number` and `player_name` stay null until Milestone 6 "
                "(OCR). That is expected: only the final per-player "
                "aggregation needs a name."
            )
with tabs[3]:
    st.subheader("Ball Possession")
    st.caption(
        "Does the flagged handler match who actually has the ball, and do "
        "possession changes line up with passes rather than flickering?"
    )

    poss_path = possession_path(game_id)
    if not poss_path.exists():
        st.info(
            "No possession output yet. Run: "
            "`python pipeline/04_ball_possession.py --game-id " + game_id + "`"
        )
    elif not trk_path.exists():
        st.warning("Tracks are missing, so boxes cannot be drawn.")
    else:
        poss = load_parquet(str(poss_path), _mtime(poss_path))
        poss_tracks = load_parquet(str(trk_path), _mtime(trk_path))

        shot = shot_selector("poss", shots)
        if shot is None:
            plo, phi = int(poss["frame_idx"].min()), int(poss["frame_idx"].max())
        else:
            plo, phi = shot.start_frame, shot.end_frame
        window = poss[poss["frame_idx"].between(plo, phi)]

        if window.empty:
            st.caption("No possession rows in this range.")
        else:
            held = window["ball_handler_tracker_id"].notna()
            resolved = window["ball_distance_px"].notna()
            pcols = st.columns(4)
            pcols[0].metric("Frames", len(window))
            pcols[1].metric(
                "With a handler", f"{100.0 * held.mean():.0f}%"
            )
            pcols[2].metric(
                "Ball located",
                f"{100.0 * resolved.mean():.0f}%",
                help="Frames where a plausible ball position survived the "
                "jump gate. Everything else is genuinely unknown.",
            )
            pcols[3].metric(
                "Possession spans",
                int(possession_summary(window)["ball_handler_tracker_id"].notna().sum()),
            )

            pframe = frame_scrubber("poss", plo, phi)
            row = window[window["frame_idx"] == pframe]
            handler = None
            if not row.empty and pd.notna(row.iloc[0]["ball_handler_tracker_id"]):
                handler = int(row.iloc[0]["ball_handler_tracker_id"])

            frame = get_frame(str(video_path), pframe)
            if frame is None:
                st.error("Could not read frame " + str(pframe) + ".")
            else:
                canvas = frame.copy()
                here = poss_tracks[
                    (poss_tracks["frame_idx"] == pframe)
                    & (poss_tracks["class"] == CLASS_PLAYER)
                ]
                for _, track_row in here.iterrows():
                    is_handler = int(track_row["tracker_id"]) == handler
                    cv2.rectangle(
                        canvas,
                        (int(track_row["x1"]), int(track_row["y1"])),
                        (int(track_row["x2"]), int(track_row["y2"])),
                        (0, 220, 0) if is_handler else (170, 170, 170),
                        4 if is_handler else 1,
                    )
                canvas = draw_ball(
                    canvas,
                    poss_tracks[
                        (poss_tracks["frame_idx"] == pframe)
                        & (poss_tracks["class"] == CLASS_BALL)
                    ],
                )
                st.image(bgr_to_rgb(canvas), width="stretch")

                distance = row.iloc[0]["ball_distance_px"] if not row.empty else float("nan")
                if handler is None:
                    st.caption(
                        "No handler this frame — the ball is in flight, or no "
                        "plausible ball position was found. Both are correct "
                        "answers, not failures."
                    )
                else:
                    st.caption(
                        "Handler #" + str(handler) + " at "
                        + ("unknown" if pd.isna(distance) else str(int(distance)) + "px")
                        + ". Watching this number spike and drop as passes "
                        "happen is the fastest check on the threshold."
                    )

            st.markdown("**Possession timeline**")
            spans = possession_summary(window)
            spans = spans[spans["ball_handler_tracker_id"].notna()]
            if spans.empty:
                st.caption("No confirmed possessions in this range.")
            else:
                st.dataframe(spans, width="stretch", hide_index=True)
                st.caption(
                    "Confirm each change lines up with a pass or rebound in "
                    "the video. Many very short spans mean the debounce is "
                    "too low; one span covering an obvious pass means it is "
                    "too high."
                )
with tabs[4]:
    st.subheader("Court Calibration")
    st.caption(
        "Annotate court landmarks, then check the homography: do the projected "
        "dots land where the players actually are?"
    )
    st.caption(
        "Until this stage exists every distance is in pixels, and a pixel is "
        "worth more feet at the far end of the court than the near end. "
        "Gravity is a distance metric, so it needs feet."
    )

    cal_profiles = list_profiles()
    if not cal_profiles:
        st.info("No court profile yet. Create one in the Tracking tab first.")
    else:
        cal_name = st.selectbox("Court profile", cal_profiles, key="cal_profile")
        cal_profile = load_profile(cal_name)
        stored_kp = cal_profile.get("court_keypoints") or {}

        # Scrubbing 7,000+ frames to find one shot is not navigation. Picking
        # the shot narrows the slider to it and starts in the middle, which is
        # also where a single homography is most accurate — its error grows
        # with distance from the annotated frame as the camera pans.
        # Landmarks describe one camera framing, but the profile is shared and
        # the landmark box survives a change of game — so it is easy to load
        # one clip's landmarks, switch game, and save them against another.
        # The stamp records where they came from; say so loudly when it
        # disagrees with the game now selected.
        stamped = re.search(
            r"\[landmarks annotated on (\S+) frame (\d+)\]",
            cal_profile.get("description", "") or "",
        )
        if stamped and stamped.group(1) != game_id:
            st.error(
                f"This profile's landmarks were annotated on **{stamped.group(1)}** "
                f"(frame {stamped.group(2)}), but **{game_id}** is selected. A "
                "homography only describes the framing it was annotated on, so "
                "these coordinates do not apply here. Clear the landmark box "
                "below and place them again on this game, or switch the game "
                f"in the sidebar back to {stamped.group(1)}."
            )

        cal_shot = shot_selector("cal", shots)
        if cal_shot is None:
            cal_lo, cal_hi = 0, max(info.frame_count - 1, 0)
        else:
            cal_lo, cal_hi = cal_shot.start_frame, cal_shot.end_frame
            middle = (cal_lo + cal_hi) // 2
            if st.session_state.get("cal_shot_span") != (cal_lo, cal_hi):
                st.session_state["cal_shot_span"] = (cal_lo, cal_hi)
                st.session_state["cal_frame"] = middle
            st.caption(
                f"Shot {cal_shot.shot_id}: frames {cal_lo}-{cal_hi} "
                f"({cal_hi - cal_lo + 1} frames). Annotating near the middle "
                f"(frame {middle}) keeps the homography's error balanced "
                "across the shot."
            )

        cal_frame_idx = frame_scrubber("cal", cal_lo, cal_hi)
        cal_frame = get_frame(str(video_path), cal_frame_idx)

        if trk_path.exists():
            _cal_tracks = load_parquet(str(trk_path), _mtime(trk_path))
            if not _cal_tracks.empty:
                t_lo = int(_cal_tracks["frame_idx"].min())
                t_hi = int(_cal_tracks["frame_idx"].max())
                if not (t_lo <= cal_frame_idx <= t_hi):
                    st.info(
                        f"Tracks only exist for frames {t_lo}-{t_hi}, so the "
                        "radar will be empty here. The radar is the check that "
                        "tells you the homography is right, so annotate inside "
                        "that range."
                    )

        st.markdown("**Landmarks**")
        st.caption(
            "One per line: landmark_name x y, with x and y normalized 0-1. At "
            "least 4, and not collinear - spread them across the court rather "
            "than along one line. Known names: " + ", ".join(sorted(COURT_LANDMARKS))
        )
        # A click is consumed here, before the text area widget exists:
        # assigning to a widget's session_state key after instantiation
        # raises, so the update has to happen on the following rerun.
        click_landmark = None
        if streamlit_image_coordinates is not None:
            click_landmark = st.selectbox(
                "Click on the frame to place this landmark",
                sorted(COURT_LANDMARKS),
                key="cal_click_target",
                help="Pick the landmark, then click where it sits in the "
                "frame below. Clicking again moves it. Left vs right: the two "
                "'left' names go on one long side of the lane and the two "
                "'right' names on the other. Which side you pick does not "
                "matter — being consistent does.",
            )
            def _normalized(payload):
                """Click position as a 0-1 fraction of the rendered image.

                The component returns the size it actually rendered at, which
                is what to divide by — the width we asked for may not have
                survived the browser fitting the image to its column.
                """
                width = float(payload.get("width") or 0.0)
                height = float(payload.get("height") or 0.0)
                if not width or not height:
                    return None
                return payload["x"] / width, payload["y"] / height

            # Clicking the overview only moves the magnifier. Placing a
            # landmark there would be guesswork: at browser scale a lane
            # corner is about two pixels across.
            overview = st.session_state.get("cal_click_overview")
            if overview and overview != st.session_state.get("cal_overview_handled"):
                st.session_state["cal_overview_handled"] = overview
                spot = _normalized(overview)
                if spot:
                    st.session_state["cal_zoom_centre"] = spot
                    st.rerun()

            pending = st.session_state.get("cal_click")
            last = st.session_state.get("cal_click_handled")
            if pending and pending != last:
                st.session_state["cal_click_handled"] = pending
                spot = _normalized(pending)
                if spot:
                    # The magnified view shows a window of the frame, so a
                    # click in it is an offset within that window, not within
                    # the whole frame.
                    zoom_x, zoom_y, zoom_w, zoom_h = st.session_state.get(
                        "cal_zoom_window", (0.0, 0.0, 1.0, 1.0)
                    )
                    nx = zoom_x + spot[0] * zoom_w
                    ny = zoom_y + spot[1] * zoom_h
                    st.session_state["cal_click_debug"] = (
                        f"click ({pending['x']:.0f}, {pending['y']:.0f}) in a "
                        f"{pending.get('width', 0):.0f}x{pending.get('height', 0):.0f} "
                        f"magnified view -> ({nx:.4f}, {ny:.4f}) in the frame"
                    )
                    target = st.session_state.get("cal_click_target")
                    existing = st.session_state.get("cal_landmarks", "")
                    kept = [
                        line
                        for line in existing.splitlines()
                        if line.strip() and line.split()[0] != target
                    ]
                    kept.append(f"{target} {nx:.4f} {ny:.4f}")
                    updated = "\n".join(kept)
                    # Both: the mirror survives a rerun that never reaches the
                    # text area, and the widget key is what the box displays.
                    st.session_state["cal_landmarks"] = updated
                    st.session_state["cal_text"] = updated
                    st.rerun()

        # `cal_landmarks` is the durable copy. It is deliberately not a widget
        # key: Streamlit garbage-collects widget state for widgets that a run
        # did not instantiate, and the click handlers above rerun before the
        # text area exists, which used to drop the edits and reload the
        # profile's landmarks over them.
        if "cal_landmarks" not in st.session_state:
            st.session_state["cal_landmarks"] = "\n".join(
                name + " " + format(pt[0], "g") + " " + format(pt[1], "g")
                for name, pt in stored_kp.items()
            )
        if "cal_text" not in st.session_state:
            st.session_state["cal_text"] = st.session_state["cal_landmarks"]

        # These write session_state from a callback, which runs before the
        # text area is instantiated on the next rerun — assigning to a live
        # widget's key raises.
        def _reload_landmarks() -> None:
            text = "\n".join(
                name + " " + format(pt[0], "g") + " " + format(pt[1], "g")
                for name, pt in stored_kp.items()
            )
            st.session_state["cal_landmarks"] = text
            st.session_state["cal_text"] = text

        def _clear_landmarks() -> None:
            st.session_state["cal_landmarks"] = ""
            st.session_state["cal_text"] = ""

        edit_cols = st.columns([1, 1, 4])
        edit_cols[0].button(
            "Reload from profile", key="cal_reload", on_click=_reload_landmarks,
            help="Discard what is in the box and load the profile's saved "
            "landmarks.",
        )
        edit_cols[1].button(
            "Clear", key="cal_clear", on_click=_clear_landmarks,
            help="Empty the box, to annotate this game from scratch.",
        )

        cal_raw = st.text_area("Landmarks", key="cal_text", height=170)
        # Typing in the box is just as valid as clicking, so the mirror
        # follows whatever the box now holds.
        st.session_state["cal_landmarks"] = cal_raw

        grid_cols = st.columns([1, 1, 2])
        show_grid = grid_cols[0].checkbox(
            "Show coordinate grid", value=True, key="cal_grid"
        )
        zoom_factor = grid_cols[1].select_slider(
            "Magnifier",
            options=[2, 3, 4, 6, 8],
            value=4,
            key="cal_zoom",
            help="How far to magnify the region you clicked in the overview.",
        )
        grid_step = grid_cols[1].select_slider(
            "Grid spacing",
            options=[0.10, 0.05, 0.025],
            value=0.05,
            key="cal_grid_step",
            help="Read a landmark's x and y straight off the labelled lines, "
            "then type them in above.",
        )

        parsed_kp = {}
        parse_error = None
        for line in cal_raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                parse_error = "Expected `name x y`, got: " + line
                break
            try:
                parsed_kp[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                parse_error = "x and y must be numbers: " + line
                break

        if parse_error:
            st.error(parse_error)

        if cal_frame is None:
            st.error("Could not read that frame.")
        else:
            cal_h, cal_w = cal_frame.shape[:2]

            # The frame is drawn whatever the landmark count. Gating it behind
            # "4 landmarks" made click-to-place unusable: you could not click
            # your way to the fourth point without an image to click on.
            canvas = cal_frame.copy()
            if show_grid:
                _draw_coordinate_grid(canvas, grid_step)
            for name, point in parsed_kp.items():
                px = int(point[0] * cal_w)
                py = int(point[1] * cal_h)
                cv2.drawMarker(
                    canvas, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 26, 2
                )
                cv2.putText(
                    canvas, name, (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )

            twisted = lane_orientation_problem(parsed_kp)
            if twisted:
                st.error(twisted)

            homography = None
            per_point = None
            kp_names: list[str] = []
            kp_image = None
            if not parse_error and len(parsed_kp) >= MIN_LANDMARKS:
                try:
                    kp_image, kp_court, kp_names = parse_keypoints(parsed_kp)
                    homography, per_point, rms = compute_homography(
                        kp_image, kp_court, cal_w, cal_h
                    )
                except ValueError as exc:
                    st.error(str(exc))

            if homography is not None:
                mcols = st.columns(3)
                mcols[0].metric("Landmarks", len(kp_names))
                mcols[1].metric(
                    "Reprojection error",
                    format(rms, ".1f") + " px",
                    help="How far annotated points land from where the "
                    "homography puts them. This is the single number that says "
                    "whether to trust this angle at all.",
                )
                mcols[2].metric(
                    "Worst landmark", format(per_point.max(), ".1f") + " px"
                )
                if len(kp_names) == MIN_LANDMARKS:
                    st.warning(
                        "With exactly 4 landmarks the reprojection error is "
                        "always 0 and means nothing: a homography has 8 degrees "
                        "of freedom and 4 points give exactly 8 equations, so "
                        "the fit is exact whether or not the points are right. "
                        "The known-distance checks are equally vacuous for the "
                        "same reason. **Add a 5th and 6th landmark** to get a "
                        "real residual, and trust the radar until you do."
                    )
                elif rms > 20:
                    culprit = worst_landmark(parsed_kp, cal_w, cal_h)
                    if culprit:
                        name, with_it, without_it = culprit
                        st.warning(
                            f"High reprojection error. Removing **{name}** drops "
                            f"it from {with_it:.0f}px to {without_it:.1f}px, so "
                            f"that landmark is very likely on the wrong spot or "
                            "carrying the wrong name — check its left/right "
                            "first. Note the per-landmark table below can point "
                            "the wrong way: a mislabelled point drags the fit "
                            "toward itself and ends up with a small error of "
                            "its own."
                        )
                    else:
                        st.warning(
                            "High reprojection error - at least one landmark is "
                            "probably misplaced. The worst offender is in the "
                            "table below."
                        )
            else:
                st.info(
                    str(len(parsed_kp)) + " landmark(s) placed; "
                    + str(MIN_LANDMARKS)
                    + " are needed for a homography. Pick a landmark above and "
                    "click it in the frame."
                )

            # Full width, not a half column: the frame is 16:9 and the whole
            # court has to be reachable by a click.
            st.markdown("**Annotated frame**")
            if streamlit_image_coordinates is None:
                st.image(bgr_to_rgb(canvas), width="stretch")
                st.caption(
                    "Install `streamlit-image-coordinates` to place landmarks "
                    "by clicking instead of typing."
                )
            else:
                centre_x, centre_y = st.session_state.get(
                    "cal_zoom_centre", (0.5, 0.5)
                )
                window = 1.0 / zoom_factor
                # Keep the window inside the frame, so the magnifier never
                # shows blank space and the mapping stays exact.
                win_x = min(max(centre_x - window / 2, 0.0), 1.0 - window)
                win_y = min(max(centre_y - window / 2, 0.0), 1.0 - window)
                st.session_state["cal_zoom_window"] = (win_x, win_y, window, window)

                overview_img = canvas.copy()
                cv2.rectangle(
                    overview_img,
                    (int(win_x * cal_w), int(win_y * cal_h)),
                    (int((win_x + window) * cal_w), int((win_y + window) * cal_h)),
                    (0, 128, 255), 3,
                )
                streamlit_image_coordinates(
                    bgr_to_rgb(overview_img),
                    width=CLICK_DISPLAY_WIDTH,
                    key="cal_click_overview",
                )
                st.caption(
                    "Overview - click to move the orange magnifier box. "
                    "Landmarks are not placed from here: at this scale a lane "
                    "corner is about two pixels across."
                )

                x0, y0 = int(win_x * cal_w), int(win_y * cal_h)
                x1, y1 = int((win_x + window) * cal_w), int((win_y + window) * cal_h)
                crop = cal_frame[y0:y1, x0:x1].copy()
                if crop.size:
                    # Upscale to a fixed width so the click target stays large
                    # whatever magnification is chosen.
                    target_w = 1100
                    crop = cv2.resize(
                        crop,
                        (target_w, int(crop.shape[0] * target_w / crop.shape[1])),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    crop_h, crop_w = crop.shape[:2]
                    cv2.line(crop, (crop_w // 2 - 18, crop_h // 2),
                             (crop_w // 2 + 18, crop_h // 2), (0, 128, 255), 1)
                    cv2.line(crop, (crop_w // 2, crop_h // 2 - 18),
                             (crop_w // 2, crop_h // 2 + 18), (0, 128, 255), 1)

                    for name, point in parsed_kp.items():
                        if not (win_x <= point[0] <= win_x + window
                                and win_y <= point[1] <= win_y + window):
                            continue
                        px = int((point[0] - win_x) / window * crop_w)
                        py = int((point[1] - win_y) / window * crop_h)
                        colour = (0, 255, 0) if name == click_landmark else (0, 255, 255)
                        cv2.drawMarker(crop, (px, py), colour, cv2.MARKER_CROSS, 34, 2)
                        # Dark backing first, so the label reads over pale wood.
                        for thickness, shade in ((4, (20, 20, 20)), (1, colour)):
                            cv2.putText(crop, name, (px + 12, py - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                        shade, thickness, cv2.LINE_AA)

                    streamlit_image_coordinates(
                        bgr_to_rgb(crop),
                        width=CLICK_DISPLAY_WIDTH,
                        key="cal_click",
                    )
                    st.caption(
                        f"Magnified {zoom_factor}x - click here to place "
                        f"`{click_landmark}`. The landmark being placed is "
                        "green; others in view are yellow."
                    )
                if st.session_state.get("cal_click_debug"):
                    # The raw payload, because a mis-scaled click is otherwise
                    # only visible much later as a wrong homography.
                    st.caption(f"last {st.session_state['cal_click_debug']}")

            radar_col, table_col = st.columns([1, 2])

            with radar_col:
                st.markdown("**Radar - players projected onto the court**")
                if homography is None:
                    st.caption(
                        "Appears once a homography can be computed. This is "
                        "the check that matters: the dots must land inside the "
                        "court, where the players actually are."
                    )
                else:
                    radar = np.full((490, 280, 3), 245, dtype=np.uint8)

                    def court_to_radar(pt):
                        return int(15 + pt[0] * 5), int(15 + pt[1] * 5)

                    cv2.rectangle(
                        radar,
                        court_to_radar((0, 0)),
                        court_to_radar((COURT_WIDTH_FT, COURT_LENGTH_FT)),
                        (60, 60, 60), 2,
                    )
                    cv2.line(
                        radar,
                        court_to_radar((0, COURT_LENGTH_FT / 2.0)),
                        court_to_radar((COURT_WIDTH_FT, COURT_LENGTH_FT / 2.0)),
                        (60, 60, 60), 1,
                    )
                    for landmark in (
                        "lane_baseline_left", "lane_baseline_right",
                        "free_throw_left", "free_throw_right",
                    ):
                        cv2.circle(
                            radar, court_to_radar(COURT_LANDMARKS[landmark]),
                            3, (150, 150, 150), -1,
                        )

                    if trk_path.exists():
                        cal_tracks = load_parquet(str(trk_path), _mtime(trk_path))
                        here = cal_tracks[
                            (cal_tracks["frame_idx"] == cal_frame_idx)
                            & (cal_tracks["class"] == CLASS_PLAYER)
                        ]
                        if here.empty:
                            st.caption(
                                "No tracks on this frame - scrub into the range "
                                "you ran Stage 2 over."
                            )
                        else:
                            feet = to_court_feet(
                                homography,
                                here[["foot_x", "foot_y"]].to_numpy(dtype=float),
                            )
                            for point in feet:
                                if not np.isfinite(point).all():
                                    continue
                                cv2.circle(
                                    radar, court_to_radar(point), 5,
                                    (200, 60, 60), -1,
                                )
                    st.image(bgr_to_rgb(radar), width="stretch")
                    st.caption(
                        "Dots should sit inside the court and match where "
                        "players stand in the frame. Dots outside the rectangle "
                        "mean the homography is wrong."
                    )

            if homography is not None:
                with table_col:
                    st.markdown("**Landmark errors**")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "landmark": kp_names,
                                "error_px": [
                                    round(float(e), 2) for e in per_point
                                ],
                            }
                        ).sort_values("error_px", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Fix the worst offender first - one badly placed "
                        "landmark drags the whole fit."
                    )

                st.markdown("**Known distance checks**")
                checks = known_distance_checks(
                    homography, kp_image, kp_names, cal_w, cal_h
                )
                if checks:
                    st.dataframe(
                        pd.DataFrame(checks), width="stretch", hide_index=True
                    )
                    st.caption(
                        "The milestone's done-when: real court distances "
                        "projected through the homography should come out near "
                        "their true values. These landmarks were used to fit "
                        "it, so this checks internal consistency, not "
                        "independent accuracy."
                    )
                else:
                    st.caption(
                        "No checkable pairs among these landmarks - add both "
                        "lane baseline corners, or both free-throw corners."
                    )

                if st.button("Save landmarks to profile", key="cal_save"):
                    note = cal_profile.get("description", "")
                    stamp = f"[landmarks annotated on {game_id} frame {cal_frame_idx}]"
                    # Strip any previous stamp so re-saving does not stack them.
                    base = note.split(" [landmarks annotated on")[0].rstrip()
                    written = save_profile(
                        cal_name,
                        cal_profile.get("court_polygon"),
                        cal_profile.get("tracker"),
                        (base + " " + stamp).strip(),
                        cal_profile.get("backend"),
                        parsed_kp,
                    )
                    st.success("Wrote " + str(written))
                    st.caption(
                        "The frame is recorded in the profile description, so "
                        "the landmarks can be checked against the picture they "
                        "were placed on."
                    )

                st.code(
                    "python pipeline/05_calibrate.py --game-id " + game_id
                    + " --court-profile " + cal_name,
                    language="bash",
                )
with tabs[5]:
    milestone_placeholder("Identity Resolution", "Milestone 6", "03_identify.py")
with tabs[6]:
    milestone_placeholder("Gravity Results", "Milestone 7", "06_aggregate.py")
