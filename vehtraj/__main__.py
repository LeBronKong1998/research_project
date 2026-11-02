import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DT, TrajectoryDataset, load_scenes, split_scenes
from .download import download_interaction
from .metrics import nonconformity, scene_calibration, trajectory_metrics
from .model import TrajectoryGAT, multimodal_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_ready(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_json(path, data):
    path.write_text(json.dumps(json_ready(data), indent=2, allow_nan=False), encoding="utf-8")


def device_for(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def data_manifest(args):
    from .download import sha256
    if args.source == "synthetic":
        return {"source": "synthetic_test_only", "count": args.synthetic_count}
    root = Path(args.data_dir)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.csv"))}


def constant_velocity(agents, future):
    last = agents[-1, 0]
    result = np.zeros((1, future, 5), dtype=np.float32)
    result[0, :, :2] = last[:2] + np.arange(1, future + 1)[:, None] * DT * last[2:4]
    result[0, :, 2:4] = last[2:4]
    result[0, :, 4] = np.arctan2(last[4], last[5])
    return result


def make_data(args):
    scenes = load_scenes(args.source, args.data_dir, args.synthetic_count, args.history, args.future)
    return {split: TrajectoryDataset(split_scenes(scenes, split), stride=args.stride, history=args.history, future=args.future) for split in ("train", "val", "cal", "test")}


def train(args):
    set_seed(args.seed)
    print("Loading and splitting scenes...", flush=True)
    datasets = make_data(args)
    if not datasets["train"] or not datasets["val"]:
        raise ValueError("Train/val split is empty; add independent scenes or increase --synthetic-count")
    device = device_for(args.device)
    model = TrajectoryGAT(args.hidden, args.modes, future=args.future).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    print(f"Device: {device}; samples: " + ", ".join(f"{name}={len(dataset)}" for name, dataset in datasets.items()), flush=True)
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
        train_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}: training {train_batches} batches", flush=True)
        losses = []
        for step, (agents, mask, truth, *_rest) in enumerate(loader):
            if args.max_batches and step >= args.max_batches:
                break
            prediction, logits = model(agents.to(device), mask.to(device))
            loss = multimodal_loss(prediction, logits, truth.to(device))
            group_start = (step // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, train_batches - group_start)
            (loss / group_size).backward()
            if (step + 1) % args.accumulation == 0 or step + 1 == len(loader) or (args.max_batches and step + 1 == args.max_batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == train_batches:
                elapsed = time.perf_counter() - epoch_start
                remaining = elapsed / completed * (train_batches - completed)
                print(f"  train {completed}/{train_batches} | loss {np.mean(losses):.4f} | elapsed {elapsed:.0f}s | ETA {remaining:.0f}s", flush=True)
        model.eval()
        validation = []
        validation_weights = []
        val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)
        val_batches = min(len(val_loader), args.max_batches) if args.max_batches else len(val_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}: validating {val_batches} batches", flush=True)
        with torch.no_grad():
            for step, (agents, mask, truth, *_rest) in enumerate(val_loader):
                if args.max_batches and step >= args.max_batches:
                    break
                prediction, logits = model(agents.to(device), mask.to(device))
                validation.append(float(multimodal_loss(prediction, logits, truth.to(device))))
                validation_weights.append(len(truth))
                completed = step + 1
                if completed == 1 or completed % args.log_every == 0 or completed == val_batches:
                    print(f"  validation {completed}/{val_batches} | loss {np.average(validation, weights=validation_weights):.4f}", flush=True)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": float(np.average(validation, weights=validation_weights))}
        history.append(row)
        print(f"Epoch {epoch + 1}/{args.epochs} complete in {time.perf_counter() - epoch_start:.0f}s: " + json.dumps(row), flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "modes": args.modes,
                        "source": args.source, "seed": args.seed, "epoch": epoch + 1, "history": args.history, "future": args.future,
                        "data_manifest": data_manifest(args), "stride": args.stride, "max_batches": args.max_batches}, output / "best.pt")
            print(f"  saved best checkpoint: {output / 'best.pt'}", flush=True)
        write_json(output / "training.json", {"epochs": history, "counts": {key: len(value) for key, value in datasets.items()}, "device": str(device), "max_batches": args.max_batches, "configuration": vars(args), "data_manifest": data_manifest(args),
            "scene_ids": {key: sorted({sample[3] for sample in value.samples}) for key, value in datasets.items()}})
    print(f"checkpoint={output / 'best.pt'}", flush=True)
    if args.source == "interaction":
        from .review import export_review
        best_model = load_model(output / "best.pt", device)
        export_review(args, infer(best_model, datasets["val"], device), "val", output,
                      checkpoint=output / "best.pt", training=output / "training.json")


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = TrajectoryGAT(saved["hidden"], saved["modes"], future=saved["future"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def infer(model, dataset, device, max_samples=0):
    records = []
    limit = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    with torch.no_grad():
        for index in range(limit):
            agents, mask, truth, scene_id, kind, reference = dataset[index]
            prediction, logits = model(torch.from_numpy(agents)[None].to(device), torch.from_numpy(mask)[None].to(device))
            prediction = prediction[0].cpu().numpy()
            probabilities = torch.softmax(logits[0], -1).cpu().numpy()
            records.append({"scene_id": scene_id, "kind": kind, "sample_index": index,
                            **dataset.metadata[index],
                            "history": agents, "truth": truth, "prediction": prediction,
                            "probabilities": probabilities, "reference": reference,
                            "score": nonconformity(prediction, truth),
                            "metrics": trajectory_metrics(prediction, truth, probabilities),
                            "baseline_metrics": trajectory_metrics(constant_velocity(agents, len(truth)), truth)})
    return records


def export_records(records, output, radius):
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene_id", "sample_index", "mode", "probability", "future_step", "x_m", "y_m", "vx_mps", "vy_mps", "heading_rad", "radius_m", "truth_x_m", "truth_y_m", "position_error_m"))
        for record in records:
            for mode, trajectory in enumerate(record["prediction"]):
                for step, state in enumerate(trajectory, 1):
                    writer.writerow((record["scene_id"], record["sample_index"], mode,
                                     float(record["probabilities"][mode]), step,
                                     float(state[0] + record["reference"][0]),
                                     float(state[1] + record["reference"][1]),
                                     *map(float, state[2:]), radius if np.isfinite(radius) else "",
                                     *map(float, record["truth"][step - 1, :2] + record["reference"]),
                                     float(np.linalg.norm(state[:2] - record["truth"][step - 1, :2]))))
    summary = {"sample_count": len(records), "scene_count": len({row["scene_id"] for row in records}),
               "metrics": {key: float(np.mean([row["metrics"][key] for row in records])) for key in records[0]["metrics"]} if records else {},
               "window_coverage": float(np.mean([row["score"] <= radius for row in records])) if records and np.isfinite(radius) else None,
               "scene_coverage": float(np.mean([all(row["score"] <= radius for row in records if row["scene_id"] == scene_id) for scene_id in {row["scene_id"] for row in records}])) if records and np.isfinite(radius) else None,
               "radius_m": radius,
               "constant_velocity": {key: float(np.mean([row["baseline_metrics"][key] for row in records])) for key in records[0]["baseline_metrics"]} if records else {}}
    write_json(output / "evaluation.json", summary)
    return summary


def evaluate(args):
    set_seed(args.seed)
    datasets = make_data(args)
    if not datasets["cal"] or not datasets["test"]:
        raise ValueError("Calibration/test split is empty; add independent scenes or increase --synthetic-count")
    device = device_for(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (saved["source"], saved["history"], saved["future"], saved["stride"]) != (args.source, args.history, args.future, args.stride):
        raise ValueError("Checkpoint source/history/future/stride differs from evaluation arguments")
    if saved["data_manifest"] != data_manifest(args):
        raise ValueError("Evaluation data differs from the checkpoint manifest; use the original held-out splits")
    model = load_model(args.checkpoint, device)
    calibration = scene_calibration(infer(model, datasets["cal"], device))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "calibration.json", calibration)
    records = infer(model, datasets[args.split], device, args.max_samples)
    summary = export_records(records, output, calibration["radius_m"])
    summary.update({"source": args.source, "history_frames": args.history, "future_frames": args.future,
                    "dt_seconds": DT, "data_manifest": data_manifest(args),
                    "split": args.split, "split_total_samples": len(datasets[args.split]),
                    "test_total_samples": len(datasets["test"]), "subset_evaluation": bool(args.max_samples and args.max_samples < len(datasets[args.split])),
                    "checkpoint": str(args.checkpoint), "selected_epoch": saved["epoch"], "training_max_batches": saved.get("max_batches", 0),
                    "protocol": "custom case-disjoint split; not official challenge benchmark"})
    write_json(output / "evaluation.json", summary)
    if args.source == "interaction":
        from .review import export_review
        export_review(args, records, args.split, output, checkpoint=args.checkpoint,
                      training=args.training_log or (Path(args.checkpoint).parent / "training.json"), summary=summary)
    from .visualize import animate_scene, draw_scene
    previews = set()
    for sample in records:
        key = sample["kind"] if args.source == "synthetic" else sample["scene_id"]
        if key in previews or len(previews) >= args.max_animations:
            continue
        previews.add(key)
        name = sample["kind"] if args.source == "synthetic" else f"interaction_{len(previews):02d}"
        if args.source == "interaction":
            from types import SimpleNamespace
            from .official_viewer import run_viewer
            csv_path = Path(args.data_dir) / sample["scene_id"].rsplit(":", 1)[0]
            scenario = csv_path.stem.removesuffix("_train").removesuffix("_val")
            run_viewer(SimpleNamespace(csv=csv_path, map=csv_path.parent / "maps" / (scenario + ".osm"),
                                      case_id=sample["scene_id"].rsplit(":",1)[1], review=output / f"{args.split}-review.json",
                                      sample_index=sample["sample_index"], frame=sample["future_frames"][-1],
                                      save_png=output / f"{name}.png", save_gif=output / f"{name}.gif"))
        else:
            draw_scene(output / f"{name}.png", sample["kind"], sample["history"], sample["truth"], sample["prediction"], calibration["radius_m"])
            animate_scene(output / f"{name}.gif", sample["kind"], sample["history"], sample["truth"], sample["prediction"])
        print(f"animation={output / (name + '.gif')} scene={sample['scene_id']}", flush=True)
    print(json.dumps(json_ready({"calibration": calibration, args.split: summary}), indent=2, allow_nan=False), flush=True)


def smoke(args):
    set_seed(7)
    dataset = TrajectoryDataset(load_scenes("synthetic", synthetic_count=1))
    device = device_for(args.device)
    model = TrajectoryGAT().to(device)
    agents, mask, truth, *_rest = dataset[0]
    prediction, logits = model(torch.from_numpy(agents)[None].to(device), torch.from_numpy(mask)[None].to(device))
    loss = multimodal_loss(prediction, logits, torch.from_numpy(truth)[None].to(device))
    loss.backward()
    assert prediction.shape == (1, 3, 30, 5)
    assert torch.isfinite(prediction).all() and torch.isfinite(loss)
    print(json.dumps({"device": str(device), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "loss": float(loss.detach()), "output_shape": list(prediction.shape)}), flush=True)


def download(args):
    print(json.dumps(download_interaction(args.output), indent=2), flush=True)


def visualize(args):
    from .official_viewer import run_viewer
    run_viewer(args)


def explore(args):
    from .explore import build_explorer
    print(json.dumps(build_explorer(args.csv, args.output, review_files=args.review, default_mode=args.mode), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="GRU-GAT multimodal vehicle trajectory prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name != "smoke":
            command.add_argument("--source", choices=("synthetic", "interaction"), default="interaction")
            command.add_argument("--data-dir", default="data/interaction")
            command.add_argument("--history", type=int, default=10)
            command.add_argument("--future", type=int, default=30)
            command.add_argument("--synthetic-count", type=int, default=80)
            command.add_argument("--stride", type=int, default=10)
            command.add_argument("--seed", type=int, default=7)
            command.add_argument("--output", default="runs/default")
        if name == "train":
            command.add_argument("--epochs", type=int, default=10)
            command.add_argument("--batch-size", type=int, default=2)
            command.add_argument("--accumulation", type=int, default=4)
            command.add_argument("--hidden", type=int, default=64)
            command.add_argument("--modes", type=int, default=3)
            command.add_argument("--lr", type=float, default=3e-4)
            command.add_argument("--max-batches", type=int, default=0)
            command.add_argument("--log-every", type=int, default=20)
        if name == "evaluate":
            command.add_argument("--split", choices=("val", "test"), default="test")
            command.add_argument("--training-log", help="matching training.json for the loss chart")
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--max-samples", type=int, default=0)
            command.add_argument("--max-animations", type=int, default=3)
    download_command = commands.add_parser("download")
    download_command.add_argument("--output", default="data/interaction")
    visual = commands.add_parser("visualize", help="official INTERACTION map renderer with model overlays")
    visual.add_argument("--csv", default="data/interaction/DR_USA_Intersection_GL_train.csv")
    visual.add_argument("--map", default="data/interaction/maps/DR_USA_Intersection_GL.osm")
    visual.add_argument("--case-id", default="13")
    visual.add_argument("--review", help="JSON exported by train/evaluate")
    visual.add_argument("--sample-index", type=int, default=0)
    visual.add_argument("--frame", type=int, default=10)
    visual.add_argument("--save-png")
    visual.add_argument("--save-gif")
    explorer = commands.add_parser("explore", help="build an offline raw CSV data explorer")
    explorer.add_argument("--csv", default="data/interaction/DR_USA_Intersection_GL_train.csv")
    explorer.add_argument("--output", default="docs/data-explorer.html")
    explorer.add_argument("--review", action="append", help="matching val/test-review.json, repeatable")
    explorer.add_argument("--mode", choices=("raw","val","test"), default="raw")
    args = parser.parse_args()
    for name in ("history", "future", "stride", "epochs", "batch_size", "accumulation", "hidden", "modes", "log_every"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace(chr(95), chr(45))} must be positive")
    if getattr(args, "max_batches", 0) < 0 or getattr(args, "max_samples", 0) < 0:
        parser.error("sample/batch limits cannot be negative")
    {"train": train, "evaluate": evaluate, "smoke": smoke, "download": download, "explore": explore, "visualize": visualize}[args.command](args)


if __name__ == "__main__":
    main()
