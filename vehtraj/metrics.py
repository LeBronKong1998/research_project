import math

import numpy as np


def trajectory_metrics(prediction, truth, probabilities=None):
    """Oracle best-of-K and probability-top-1 metrics; positions are in metres."""
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.ndim != 3 or truth.ndim != 2 or prediction.shape[1] != len(truth) or not len(truth):
        raise ValueError("Expected prediction [K,T,5] and truth [T,6] with matching nonempty T")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Non-finite prediction/truth")
    error = np.linalg.norm(prediction[..., :2] - truth[None, :, :2], axis=-1)
    ade, fde = error.mean(axis=-1), error[:, -1]
    top = 0 if probabilities is None else int(np.argmax(probabilities))
    velocity = np.linalg.norm(prediction[..., 2:4] - truth[None, :, 2:4], axis=-1).mean(axis=-1)
    heading = np.arctan2(truth[:, 4], truth[:, 5])
    heading_error = np.abs(np.arctan2(np.sin(prediction[..., 4] - heading), np.cos(prediction[..., 4] - heading))).mean(axis=-1)
    result = {"minADE": float(ade.min()), "minFDE": float(fde.min()),
              "top1ADE": float(ade[top]), "top1FDE": float(fde[top]),
              "top1PositionRMSE": float(np.sqrt(np.mean(error[top] ** 2))),
              "top1VelocityError": float(velocity[top]), "top1HeadingErrorRad": float(heading_error[top]),
              "minVelocityError": float(velocity.min()), "minHeadingErrorRad": float(heading_error.min()),
              "missRate2m": float(fde.min() > 2.0),
              "top1EndpointAccuracy2m": float(fde[top] <= 2.0)}
    for second in (1, 2, 3, 4, 5):
        frame = second * 10 - 1
        if frame < len(truth):
            result[f"top1Displacement@{second}s"] = float(error[top, frame])
    return result


def nonconformity(prediction, truth):
    return float(np.linalg.norm(prediction[..., :2] - truth[None, :, :2], axis=-1).max(axis=-1).min())


def scene_calibration(records, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")
    by_scene = {}
    for record in records:
        by_scene[record["scene_id"]] = max(by_scene.get(record["scene_id"], 0.0), record["score"])
    scores = sorted(by_scene.values())
    if not scores:
        raise ValueError("Calibration split has no complete trajectories")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    radius = float(scores[rank - 1]) if rank <= len(scores) else float("inf")
    return {"alpha": alpha, "radius_m": radius, "scene_count": len(scores),
            "finite_sample_rank": rank, "coverage_scope": "all target windows in a held-out scene"}
