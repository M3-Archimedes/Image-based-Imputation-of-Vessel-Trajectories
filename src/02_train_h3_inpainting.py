"""
H3-Aware Inpainting Training Script with Deterministic Percentage Holdout
=========================================================================
Train a U-Net model to inpaint missing regions in H3-colored wave maps.
The holdout split is percentage-based and deterministic for the same image set.

Usage:
    python train_h3_inpainting.py --epochs 200 --batch_size 32 --lr 3e-5

Caution:
    The holdout split stays the same only if all of the following remain unchanged:

    The set of image files.
    The file names.
    The --holdout_salt value.

    Equivalent path spellings such as images, ../images, or an absolute path
    produce the same split only when they resolve to the same image files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from utils import (
    H3ColorQuantizer,
    H3InpaintDatasetAugmented,
    H3InpaintingModel,
    build_h3_color_and_position_maps,
    evaluate_holdout_images,
    list_wave_map_images,
    load_checkpoint_model_state,
    load_h3_maps_from_json,
    normalize_holdout_percentage,
    plot_training_history,
    split_holdout_paths,
    train_h3_inpainting,
    write_holdout_images_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project root: {PROJECT_ROOT}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train H3-aware inpainting model with deterministic percentage holdout"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(PROJECT_ROOT / "images"),
        help="Directory containing wave map images",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "augmented_data.csv"),
        help="Path to augmented trip data CSV. Used only when JSON maps are missing.",
    )
    parser.add_argument(
        "--color_map_json",
        type=str,
        default=str(PROJECT_ROOT / "data" / "h3_color_map.json"),
        help="Path to the saved H3 color map JSON file",
    )
    parser.add_argument(
        "--position_map_json",
        type=str,
        default=str(PROJECT_ROOT / "data" / "h3_position_map.json"),
        help="Path to the saved H3 position map JSON file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--base_ch",
        type=int,
        default=48,
        help="Base channels for U-Net",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(PROJECT_ROOT / "h3_rgb_unet_v2.pth"),
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--training_plot",
        type=str,
        default=str(PROJECT_ROOT / "training_loss.png"),
        help="Path to save the training loss plot",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation using an existing checkpoint",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--holdout_percentage",
        type=float,
        default=0.30,
        help="Holdout percentage as a fraction or percent value. Defaults to 0.30.",
    )
    parser.add_argument(
        "--holdout_salt",
        type=str,
        default="archimedes_holdout_v1",
        help="Salt used to keep the deterministic holdout split stable across runs",
    )
    parser.add_argument(
        "--holdout_file",
        type=str,
        default=str(PROJECT_ROOT / "holdout_images.txt"),
        help="Path to the holdout images manifest",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=str(PROJECT_ROOT / "holdout_metrics_summary.json"),
        help="Path to save holdout accuracy, RMSE, and runtime as JSON",
    )
    return parser.parse_args()


def resolve_h3_maps(args: argparse.Namespace) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float]]]:
    color_map_path = Path(args.color_map_json)
    position_map_path = Path(args.position_map_json)

    if color_map_path.exists() and position_map_path.exists():
        print(f"Loading H3 maps from {color_map_path} and {position_map_path}")
        return load_h3_maps_from_json(color_map_path, position_map_path)

    data_csv_path = Path(args.data_csv)
    if not data_csv_path.exists():
        raise FileNotFoundError(
            "Neither the H3 JSON maps nor the augmented dataset CSV were found. "
            "Run the preprocessing notebook first."
        )

    print(f"Loading augmented trip data from {data_csv_path}")
    augmented_trips = pd.read_csv(data_csv_path)
    _, h3_color_map, h3_position_map = build_h3_color_and_position_maps(
        augmented_trips,
        lon_col="lon",
        lat_col="lat",
        h3_col="h3",
        plot_graph=False,
    )
    return h3_color_map, h3_position_map


def main() -> None:
    args = parse_args()
    run_started_at = time.perf_counter()
    holdout_percentage = normalize_holdout_percentage(args.holdout_percentage)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("\n" + "=" * 70)
    print("H3-AWARE INPAINTING TRAINING")
    print("=" * 70)

    h3_color_map, h3_position_map = resolve_h3_maps(args)
    print(f"Total unique H3 cells: {len(h3_color_map)}")
    quantizer = H3ColorQuantizer(h3_color_map)

    all_image_paths = list_wave_map_images(args.image_dir)
    if len(all_image_paths) < 2:
        raise ValueError("At least two wave-map images are required for train/holdout splitting.")

    train_paths, holdout_paths = split_holdout_paths(
        all_image_paths,
        holdout_percentage=holdout_percentage,
        salt=args.holdout_salt,
    )

    holdout_file_path = write_holdout_images_file(
        holdout_paths,
        args.holdout_file,
        relative_to=PROJECT_ROOT,
    )

    print("\nDataset split:")
    print(f"Total images: {len(all_image_paths)}")
    print(f"Training images: {len(train_paths)}")
    print(f"Holdout images: {len(holdout_paths)} ({holdout_percentage * 100:.1f}%)")
    print(f"Holdout manifest: {holdout_file_path}")

    model = H3InpaintingModel(in_channels=4, base_ch=args.base_ch).to(device)
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    if not args.eval_only:
        dataset = H3InpaintDatasetAugmented(args.image_dir, exclude_paths=holdout_paths)
        if len(dataset) == 0:
            raise ValueError("No training samples are available after removing holdout images.")

        model, history = train_h3_inpainting(
            model,
            dataset,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_path,
            num_workers=args.num_workers,
        )
        figure = plot_training_history(history, output_path=args.training_plot, title="Training Loss")
        print(f"Training loss plot saved to: {args.training_plot}")
        figure.clf()
    else:
        checkpoint_path = Path(args.save_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        model.load_state_dict(load_checkpoint_model_state(checkpoint_path, map_location=device))
        print(f"Loaded model from: {checkpoint_path}")

    print("\n" + "=" * 70)
    print("INFERENCE PIPELINE ON HOLDOUT IMAGES")
    print("=" * 70)
    print(f"Running inference on {len(holdout_paths)} holdout images (NOT used in training)")

    all_metrics = evaluate_holdout_images(
        model,
        quantizer,
        holdout_paths,
        device,
    )

    accuracy_values = [metric["accuracy"] for metric in all_metrics if "accuracy" in metric]
    rmse_values = [metric["rmse_masked"] for metric in all_metrics if "rmse_masked" in metric]

    print("\n" + "=" * 70)
    print("HOLDOUT SET INFERENCE SUMMARY")
    print("=" * 70)
    print(f"Evaluated on {len(holdout_paths)} holdout images (never used in training)")
    if accuracy_values:
        print(
            f"H3 Classification Accuracy: Mean={np.mean(accuracy_values) * 100:.1f}%, "
            f"Std={np.std(accuracy_values) * 100:.1f}%"
        )
    if rmse_values:
        print(
            f"RMSE (masked): Mean={np.mean(rmse_values):.4f}, "
            f"Std={np.std(rmse_values):.4f}"
        )

    total_runtime_seconds = time.perf_counter() - run_started_at
    metrics_output_path = Path(args.metrics_output)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output = {
        "evaluated_holdout_images": len(holdout_paths),
        "accuracy_values": [float(value) for value in accuracy_values],
        "rmse_values": [float(value) for value in rmse_values],
        "runtime_seconds": float(total_runtime_seconds),
        "runtime_minutes": float(total_runtime_seconds / 60.0),
    }
    if accuracy_values:
        metrics_output["accuracy_mean"] = float(np.mean(accuracy_values))
        metrics_output["accuracy_std"] = float(np.std(accuracy_values))
    if rmse_values:
        metrics_output["rmse_masked_mean"] = float(np.mean(rmse_values))
        metrics_output["rmse_masked_std"] = float(np.std(rmse_values))
    metrics_output_path.write_text(json.dumps(metrics_output, indent=2), encoding="utf-8")

    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Total runtime: {total_runtime_seconds:.2f} seconds")
    print("\nAll reconstructed colors are guaranteed valid H3 cells.")
    print("These results are on truly unseen data.")


if __name__ == "__main__":
    main()
