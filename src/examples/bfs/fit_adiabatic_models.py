"""Fit and test the nested BFS adiabatic wall-observation models.

``M0`` is the interpolated frozen critical manifold.  ``M1`` adds a
coefficient function multiplying ``a_dot``.  ``M2`` additionally includes
coefficient functions multiplying ``a_ddot`` and ``a_dot**2``.  Coefficient
functions are polynomial in normalized actuation and are fitted only to runs
whose manifest declares ``split = train``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


MODEL_NAMES = ("M0", "M1", "M2")


def campaign_root(args):
    if args.input_root is not None:
        return Path(args.input_root).expanduser().resolve()
    example_dir = Path(__file__).resolve().parent
    data_root = os.environ.get("ADIABATIC_BFS_DATA_ROOT")
    if data_root:
        return Path(data_root).expanduser().resolve() / "adiabatic" / args.mesh
    return example_dir / "data_output" / args.mesh / "adiabatic"


def load_runs(root, include_smoke):
    runs = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        if manifest.get("campaign") != "deterministic_adiabatic":
            continue
        if manifest.get("smoke", False) and not include_smoke:
            continue
        observations_path = manifest_path.parent / "observations.npz"
        if not observations_path.exists():
            continue
        with np.load(observations_path) as data:
            observations = {name: data[name] for name in data.files}
        runs.append(
            {
                "path": manifest_path.parent,
                "manifest": manifest,
                "observations": observations,
            }
        )
    if not runs:
        raise FileNotFoundError(f"No deterministic adiabatic runs under {root}")
    return runs


def polynomial_basis(actuation, degree, actuation_scale):
    normalized = np.asarray(actuation) / actuation_scale
    return np.column_stack(
        [normalized**power for power in range(degree + 1)]
    )


def design_matrices(
    actuation,
    actuation_rate,
    actuation_acceleration,
    degree,
    actuation_scale,
    rate_scale,
    acceleration_scale,
):
    basis = polynomial_basis(actuation, degree, actuation_scale)
    rate_features = basis * (actuation_rate / rate_scale)[:, None]
    acceleration_features = basis * (
        actuation_acceleration / acceleration_scale
    )[:, None]
    rate_squared_features = basis * (
        actuation_rate**2 / rate_scale**2
    )[:, None]
    return rate_features, np.column_stack(
        (rate_features, acceleration_features, rate_squared_features)
    )


def filtered_observations(run):
    data = run["observations"]
    discard_time = float(run["manifest"].get("recommended_discard_time", 0.0))
    keep = data["time"] >= discard_time
    if np.count_nonzero(keep) < 2:
        raise ValueError(f"Too few samples after discard time in {run['path']}")
    return {name: values[keep] for name, values in data.items() if name in {
        "time",
        "actuation",
        "actuation_rate",
        "actuation_acceleration",
        "lower_wall_skin_friction",
    }}


def weighted_rms(error, sensor_weights):
    pointwise_norm = np.sum(error**2 * sensor_weights[None, :], axis=1)
    return float(np.sqrt(np.mean(pointwise_norm)))


def fit_models(args, runs):
    training = [run for run in runs if run["manifest"]["split"] == "train"]
    testing = [run for run in runs if run["manifest"]["split"] == "test"]
    if not training:
        raise ValueError("At least one run with split=train is required")
    if not testing:
        raise ValueError("At least one held-out run with split=test is required")
    training_seeds = {run["manifest"]["seed"] for run in training}
    testing_seeds = {run["manifest"]["seed"] for run in testing}
    if training_seeds & testing_seeds:
        raise ValueError("Training and testing runs must use disjoint phase seeds")

    reference = training[0]["observations"]
    critical_actuation = reference["critical_actuation"]
    critical_wall = reference["critical_lower_wall_skin_friction"]
    sensor_bounds = reference["wall_sensor_bounds"]
    for run in runs[1:]:
        data = run["observations"]
        if not np.allclose(data["critical_actuation"], critical_actuation):
            raise ValueError("Runs use different critical actuation grids")
        if not np.allclose(
            data["critical_lower_wall_skin_friction"], critical_wall
        ):
            raise ValueError("Runs use different critical wall tables")

    critical_interpolant = CubicSpline(
        critical_actuation, critical_wall, axis=0
    )
    train_data = [filtered_observations(run) for run in training]
    actuation_scale = float(np.max(np.abs(critical_actuation)))
    rate_scale = max(
        float(np.max(np.abs(data["actuation_rate"]))) for data in train_data
    )
    acceleration_scale = max(
        float(np.max(np.abs(data["actuation_acceleration"])))
        for data in train_data
    )
    if min(actuation_scale, rate_scale, acceleration_scale) <= 0.0:
        raise ValueError("Training trajectory does not excite every model term")

    design_1 = []
    design_2 = []
    residual = []
    for data in train_data:
        if np.min(data["actuation"]) < critical_actuation[0] - 1e-10 or np.max(
            data["actuation"]
        ) > critical_actuation[-1] + 1e-10:
            raise ValueError("Training actuation leaves the critical branch")
        frozen = critical_interpolant(data["actuation"])
        matrix_1, matrix_2 = design_matrices(
            data["actuation"],
            data["actuation_rate"],
            data["actuation_acceleration"],
            args.degree,
            actuation_scale,
            rate_scale,
            acceleration_scale,
        )
        design_1.append(matrix_1)
        design_2.append(matrix_2)
        residual.append(data["lower_wall_skin_friction"] - frozen)

    design_1 = np.vstack(design_1)
    design_2 = np.vstack(design_2)
    residual = np.vstack(residual)
    coefficients_1, _, rank_1, singular_1 = np.linalg.lstsq(
        design_1, residual, rcond=args.rcond
    )
    coefficients_2, _, rank_2, singular_2 = np.linalg.lstsq(
        design_2, residual, rcond=args.rcond
    )
    condition_1 = float(singular_1[0] / singular_1[-1])
    condition_2 = float(singular_2[0] / singular_2[-1])
    if rank_1 < design_1.shape[1] or rank_2 < design_2.shape[1]:
        raise RuntimeError(
            "Adiabatic regression is rank-deficient: "
            f"M1 {rank_1}/{design_1.shape[1]}, "
            f"M2 {rank_2}/{design_2.shape[1]}. Add independent training runs."
        )
    if max(condition_1, condition_2) > args.max_condition:
        raise RuntimeError(
            "Adiabatic regression is ill-conditioned: "
            f"cond(M1)={condition_1:.3e}, cond(M2)={condition_2:.3e}"
        )

    model = {
        "critical_actuation": critical_actuation,
        "critical_wall": critical_wall,
        "sensor_bounds": sensor_bounds,
        "actuation_scale": actuation_scale,
        "rate_scale": rate_scale,
        "acceleration_scale": acceleration_scale,
        "coefficients_1": coefficients_1,
        "coefficients_2": coefficients_2,
        "rank_1": int(rank_1),
        "rank_2": int(rank_2),
        "condition_1": condition_1,
        "condition_2": condition_2,
        "critical_interpolant": critical_interpolant,
    }
    return model, training, testing


def predict(model, data, degree):
    frozen = model["critical_interpolant"](data["actuation"])
    design_1, design_2 = design_matrices(
        data["actuation"],
        data["actuation_rate"],
        data["actuation_acceleration"],
        degree,
        model["actuation_scale"],
        model["rate_scale"],
        model["acceleration_scale"],
    )
    return {
        "M0": frozen,
        "M1": frozen
        + np.einsum("ij,jk->ik", design_1, model["coefficients_1"]),
        "M2": frozen
        + np.einsum("ij,jk->ik", design_2, model["coefficients_2"]),
    }


def evaluate_runs(args, model, runs, split, output_dir):
    lengths = model["sensor_bounds"][:, 1] - model["sensor_bounds"][:, 0]
    sensor_weights = lengths / np.sum(lengths)
    rows = []
    for run in runs:
        data = filtered_observations(run)
        predictions = predict(model, data, args.degree)
        truth = data["lower_wall_skin_friction"]
        truth_norm = weighted_rms(truth, sensor_weights)
        row = {
            "run_id": run["manifest"]["run_id"],
            "split": split,
            "seed": run["manifest"]["seed"],
            "epsilon_a": run["manifest"]["epsilon_a"],
            "samples": len(data["time"]),
        }
        for name in MODEL_NAMES:
            error = weighted_rms(truth - predictions[name], sensor_weights)
            row[f"{name}_rms"] = error
            row[f"{name}_relative_rms"] = error / truth_norm
        rows.append(row)
        if args.save_predictions:
            np.savez_compressed(
                output_dir / f"predictions_{run['manifest']['run_id']}.npz",
                time=data["time"],
                truth=truth,
                **predictions,
            )
    return rows


def error_slopes(test_rows):
    epsilons = sorted({float(row["epsilon_a"]) for row in test_rows})
    grouped = {name: [] for name in MODEL_NAMES}
    for epsilon in epsilons:
        selected = [
            row for row in test_rows if float(row["epsilon_a"]) == epsilon
        ]
        for name in MODEL_NAMES:
            grouped[name].append(
                float(
                    np.sqrt(
                        np.mean([row[f"{name}_rms"] ** 2 for row in selected])
                    )
                )
            )
    slopes = {}
    for name in MODEL_NAMES:
        if len(epsilons) < 2 or min(grouped[name]) <= 0.0:
            slopes[name] = None
        else:
            slopes[name] = float(
                np.polyfit(np.log(epsilons), np.log(grouped[name]), 1)[0]
            )
    return epsilons, grouped, slopes


def main(args):
    root = campaign_root(args)
    runs = load_runs(root, args.include_smoke)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else root / "models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, training, testing = fit_models(args, runs)
    train_rows = evaluate_runs(args, model, training, "train", output_dir)
    test_rows = evaluate_runs(args, model, testing, "test", output_dir)
    rows = train_rows + test_rows
    with (output_dir / "errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    epsilons, grouped_errors, slopes = error_slopes(test_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root),
        "training_runs": [run["manifest"]["run_id"] for run in training],
        "testing_runs": [run["manifest"]["run_id"] for run in testing],
        "observable": "segment-averaged lower-wall skin friction",
        "metric": "wall-segment-length-weighted RMS",
        "coefficient_basis": {
            "type": "polynomial in a / max(abs(critical_actuation))",
            "degree": args.degree,
        },
        "feature_scales": {
            "actuation": model["actuation_scale"],
            "actuation_rate": model["rate_scale"],
            "actuation_acceleration": model["acceleration_scale"],
        },
        "design_rank": {"M1": model["rank_1"], "M2": model["rank_2"]},
        "design_condition": {
            "M1": model["condition_1"],
            "M2": model["condition_2"],
        },
        "test_epsilon_a": epsilons,
        "grouped_test_rms": grouped_errors,
        "error_slopes": slopes,
        "target_slopes": {"M0": 1.0, "M1": 2.0, "M2": 3.0},
    }
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    np.savez_compressed(
        output_dir / "models.npz",
        critical_actuation=model["critical_actuation"],
        critical_lower_wall_skin_friction=model["critical_wall"],
        wall_sensor_bounds=model["sensor_bounds"],
        actuation_scale=model["actuation_scale"],
        actuation_rate_scale=model["rate_scale"],
        actuation_acceleration_scale=model["acceleration_scale"],
        M1_coefficients=model["coefficients_1"],
        M2_coefficients=model["coefficients_2"],
        polynomial_degree=args.degree,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--mesh", default="bfs_3")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument("--max-condition", type=float, default=1e10)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.degree < 0:
        parser.error("--degree must be nonnegative")
    if args.rcond <= 0.0:
        parser.error("--rcond must be positive")
    if args.max_condition <= 1.0:
        parser.error("--max-condition must exceed one")
    return args


if __name__ == "__main__":
    main(parse_args())
