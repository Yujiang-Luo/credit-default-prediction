from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


MODEL_PAIRS = [
    ("MLP Neural Network", "Logistic Regression"),
    ("LightGBM", "Logistic Regression"),
    ("Explainable Boosting Machine", "Logistic Regression"),
    ("MLP Neural Network", "LightGBM"),
]


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = compute_midrank(positive[r])
        ty[r] = compute_midrank(negative[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / m + sy / n
    return aucs, covariance


def paired_delong(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-y_true)
    positives = int(y_true.sum())
    preds = np.vstack([scores_a, scores_b])[:, order]
    aucs, covariance = fast_delong(preds, positives)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    variance = max(variance, 0.0)
    se = math.sqrt(variance)
    difference = float(aucs[0] - aucs[1])
    z = difference / se if se > 0 else float("nan")
    p = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else float("nan")
    return {
        "auc_model_1": float(aucs[0]),
        "auc_model_2": float(aucs[1]),
        "auc_difference": difference,
        "standard_error": se,
        "ci_95_lower": difference - 1.96 * se,
        "ci_95_upper": difference + 1.96 * se,
        "z_statistic": z,
        "p_value": p,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, name: str, details: str, checks: list[dict]) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "details": details})
    if not condition:
        raise AssertionError(f"{name}: {details}")


def audit_run(run_root: Path) -> None:
    metrics_path = run_root / "all_model_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    checks: list[dict] = []
    run_ids = metrics["run_id"].dropna().unique().tolist()
    check(len(run_ids) == 1, "single_run_id", str(run_ids), checks)
    run_id = run_ids[0]
    check(len(metrics) == 44, "metric_row_count", f"rows={len(metrics)}", checks)
    check(
        set(metrics["probability_calibration"].dropna()) == {"validation_sigmoid_on_raw_log_odds"},
        "uniform_calibration_protocol",
        str(sorted(metrics["probability_calibration"].dropna().unique())),
        checks,
    )
    check(
        bool((metrics["calibration_slope"] > 0).all()),
        "positive_calibration_slopes",
        f"min={metrics['calibration_slope'].min():.6f}",
        checks,
    )

    expected_models = {
        "Logistic Regression",
        "Probit Regression",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Explainable Boosting Machine",
        "Random Forest",
        "Linear SVM",
        "MLP Neural Network",
        "FT-Transformer",
    }
    recomputed: list[dict] = []
    for split, expected_rows in (("random", 100_000), ("temporal", 98_107)):
        split_root = run_root / f"{split}_split"
        pred_path = split_root / "tables" / "combined_test_predictions.csv"
        raw_pred_path = split_root / "tables" / "combined_test_predictions_raw.csv"
        pred = pd.read_csv(pred_path)
        raw_pred = pd.read_csv(raw_pred_path)
        model_columns = set(pred.columns) - {"test_position", "actual_default"}
        check(model_columns == expected_models, f"{split}_prediction_models", str(sorted(model_columns)), checks)
        check(len(pred) == expected_rows, f"{split}_test_rows", f"rows={len(pred)}", checks)
        check(len(raw_pred) == expected_rows, f"{split}_raw_test_rows", f"rows={len(raw_pred)}", checks)
        check(pred["actual_default"].equals(raw_pred["actual_default"]), f"{split}_label_alignment", "labels identical", checks)
        check(
            bool(((pred[list(expected_models)] >= 0) & (pred[list(expected_models)] <= 1)).all().all()),
            f"{split}_calibrated_probability_bounds",
            "all probabilities in [0,1]",
            checks,
        )
        check(
            bool(((raw_pred[list(expected_models)] >= 0) & (raw_pred[list(expected_models)] <= 1)).all().all()),
            f"{split}_raw_probability_bounds",
            "all probabilities in [0,1]",
            checks,
        )
        y = pred["actual_default"].to_numpy(dtype=int)
        split_metrics = metrics[(metrics["split"] == split) & (metrics["feature_set"] == "combined")]
        check(len(split_metrics) == 10, f"{split}_combined_metric_rows", f"rows={len(split_metrics)}", checks)
        for model in expected_models:
            score = pred[model].to_numpy(dtype=float)
            raw_score = raw_pred[model].to_numpy(dtype=float)
            row = split_metrics.loc[split_metrics["model"] == model].iloc[0]
            values = {
                "roc_auc": roc_auc_score(y, score),
                "avg_precision": average_precision_score(y, score),
                "brier_score": brier_score_loss(y, score),
                "raw_brier_score": brier_score_loss(y, raw_score),
            }
            for metric_name, value in values.items():
                reported = float(row[metric_name])
                check(
                    abs(value - reported) < 1e-12,
                    f"{split}_{model}_{metric_name}",
                    f"reported={reported:.12f}; recomputed={value:.12f}",
                    checks,
                )
            recomputed.append({"run_id": run_id, "split": split, "model": model, **values})

        composition = pd.read_csv(split_root / "tables" / "partition_composition.csv")
        expected_partitions = {"train", "validation", "test"}
        check(set(composition["partition"]) == expected_partitions, f"{split}_composition_partitions", str(expected_partitions), checks)
        test_row = composition.loc[composition["partition"] == "test"].iloc[0]
        check(int(test_row["rows"]) == expected_rows, f"{split}_composition_test_rows", str(int(test_row["rows"])), checks)
        check(int(test_row["defaults"]) == int(y.sum()), f"{split}_composition_test_defaults", str(int(y.sum())), checks)

    random_pred = pd.read_csv(run_root / "random_split" / "tables" / "combined_test_predictions.csv")
    y_random = random_pred["actual_default"].to_numpy(dtype=int)
    delong_rows = []
    for model_a, model_b in MODEL_PAIRS:
        result = paired_delong(
            y_random,
            random_pred[model_a].to_numpy(dtype=float),
            random_pred[model_b].to_numpy(dtype=float),
        )
        delong_rows.append({"run_id": run_id, "split": "random", "model_1": model_a, "model_2": model_b, **result})
    delong_path = run_root / "random_split" / "tables" / "paired_delong_comparisons.csv"
    pd.DataFrame(delong_rows).to_csv(delong_path, index=False)

    expected_files = [
        "run_complete.json",
        "all_model_metrics.csv",
        "partition_composition_all_splits.csv",
        "selected_model_configurations_all_splits.csv",
        "reproducibility_environment.csv",
        "common/tables/variable_preprocessing_mapping.csv",
        "common/tables/model_complexity_profiles.csv",
        "common/tables/model_complexity_rubric.csv",
        "random_split/figures/roc_curves_combined.png",
        "random_split/figures/precision_recall_curves_combined.png",
        "random_split/figures/gain_curves_combined.png",
        "random_split/figures/calibration_curves_combined.png",
        "random_split/figures/calibration_curves_combined_raw.png",
        "temporal_split/figures/roc_curves_combined.png",
        "temporal_split/figures/precision_recall_curves_combined.png",
        "temporal_split/figures/gain_curves_combined.png",
        "temporal_split/figures/calibration_curves_combined.png",
        "temporal_split/figures/calibration_curves_combined_raw.png",
    ]
    missing = [relative for relative in expected_files if not (run_root / relative).is_file()]
    check(not missing, "expected_artifacts_present", str(missing), checks)

    pd.DataFrame(recomputed).to_csv(run_root / "recomputed_combined_metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(run_root / "run_audit_checks.csv", index=False)
    manifest_rows = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            stat = path.stat()
            manifest_rows.append(
                {
                    "run_id": run_id,
                    "relative_path": path.relative_to(run_root).as_posix(),
                    "size_bytes": stat.st_size,
                    "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(run_root / "artifact_manifest.csv", index=False)
    summary = {
        "run_id": run_id,
        "status": "PASS",
        "checks_passed": len(checks),
        "checks_failed": 0,
        "metric_rows": len(metrics),
        "random_test_rows": 100_000,
        "temporal_test_rows": 98_107,
        "calibration_protocol": "validation_sigmoid_on_raw_log_odds",
        "paired_delong_output": delong_path.relative_to(run_root).as_posix(),
        "manifest_rows": len(manifest_rows),
    }
    (run_root / "run_audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    audit_run(args.run_root.resolve())


if __name__ == "__main__":
    main()
