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
from pipeline.track_quality import EXPECTED_PLAYERS, summarize

st.set_page_config(page_title="NBA Gravity — Pipeline Viewer", layout="wide")

BALL_COLOR_BGR = (40, 140, 245)
MAX_GIF_FRAMES = 400


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
    milestone_placeholder("Ball Possession", "Milestone 4", "04_ball_possession.py")
with tabs[4]:
    milestone_placeholder("Court Calibration", "Milestone 5", "05_calibrate.py")
with tabs[5]:
    milestone_placeholder("Identity Resolution", "Milestone 6", "03_identify.py")
with tabs[6]:
    milestone_placeholder("Gravity Results", "Milestone 7", "06_aggregate.py")
