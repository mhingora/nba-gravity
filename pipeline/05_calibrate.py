"""Stage 5 — Court calibration (homography).

Input:  raw video frames (one representative frame per shot or angle type),
        data/calibration/ court keypoint references
Output: outputs/calibration/{game_id}_{shot_id}.json  (see docs/02-data-schemas.md)

Spec: docs/03-pipeline-stages.md (Stage 5)
Independent of stage 3; can be developed and validated in parallel.
"""


def calibrate_shot(game_id: str, shot_id: int) -> None:
    """Compute pixel -> court-feet homography for a shot.

    - Reference points: baseline corners, FT-line intersections, arc
      tangents, half-court line ends (manual per angle type to start).
    - cv2.findHomography; origin at a baseline corner. (sv.ViewTransformer,
      named in earlier drafts, was removed in supervision 0.30.)
    - Store reprojection_error_px as the calibration quality signal.
    - Angles repeat across shots — calibrate once per angle type per game
      and reuse, recomputing only when reprojection error looks off.
    """
    raise NotImplementedError


if __name__ == "__main__":
    pass
