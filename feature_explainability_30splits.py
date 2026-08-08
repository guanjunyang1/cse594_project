#!/usr/bin/env python3
"""
feature_explainability_30splits.py

Unified explainability analysis over the SAME 30 repeated stratified 70/30 splits:

1) Logistic Regression
   - median imputation
   - StandardScaler
   - standardized coefficients
   - aggregate signed coefficient and absolute coefficient across splits

2) Random Forest
   - permutation importance on held-out test data
   - scoring="roc_auc"
   - n_repeats=30 permutations per split

3) XGBoost
   - permutation importance on held-out test data
   - scoring="roc_auc"
   - n_repeats=30 permutations per split

Default feature set:
    full = all 28 Stage 1 + Stage 2 features

Optional:
    --feature_set stage1

Outputs:
    lr_coefficients_all_splits.csv
    rf_permutation_all_splits.csv
    xgb_permutation_all_splits.csv

    lr_feature_importance_summary.csv
    rf_feature_importance_summary.csv
    xgb_feature_importance_summary.csv

    lr_top10.csv
    rf_top10.csv
    xgb_top10.csv

    lr_top10_coefficients.png
    rf_top10_permutation_importance.png
    xgb_top10_permutation_importance.png

    model_test_performance.csv
    metadata.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "xgboost is required. Install it with: pip install xgboost"
    ) from e


STAGE1_FEATURES = [
    "questionnaire_latency_mean",
    "questionnaire_latency_sd",
    "questionnaire_interval_mean",
    "questionnaire_interval_sd",
    "social_notification_latency_mean",
    "social_notification_latency_sd",
    "ambient_light_sd",
    "fitbit_steps_active",
    "active_session_duration",
    "new_apps_added",
    "scroll_interval_sd",
    "tap_interval_sd",
    "dwell_time_sd",
    "idle_interval_sd",
    "session_fragmentation",
]

STAGE2_FEATURES = [
    "pst_fixation_duration",
    "ast_saccade_time_max",
    "mgst_saccade_degree_total",
    "mgst_saccade_degree_mean",
    "mgst_saccade_time_max",
    "mgst_fixation_time_max",
    "mgst_fixation_mean",
    "mgst_saccade_latency_total",
    "mgst_saccade_duration",
    "cdt_saccade_degree_sd",
    "stroop_saccade_degree_mean",
    "stroop_screen_duration_mean",
    "stroop_saccade_latency_total",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "30-split explainability analysis: LR standardized coefficients, "
            "RF permutation importance, and XGBoost permutation importance."
        )
    )
    p.add_argument(
        "--input",
        default="synthetic_stage1_stage2_dataset.csv",
        help="Input CSV file.",
    )
    p.add_argument(
        "--output_dir",
        default="feature_explainability_results",
        help="Output directory.",
    )
    p.add_argument(
        "--feature_set",
        choices=["full", "stage1"],
        default="full",
        help="full = 28 Stage 1+2 features; stage1 = 15 Stage 1 features.",
    )
    p.add_argument(
        "--label_col",
        default=None,
        help="Target column. If omitted, common names are detected.",
    )
    p.add_argument(
        "--positive_label",
        default=None,
        help="Positive-class label if target is not already 0/1.",
    )
    p.add_argument(
        "--n_splits",
        type=int,
        default=30,
        help="Number of repeated stratified train-test splits.",
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.30,
        help="Test fraction.",
    )
    p.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed controlling split generation.",
    )
    p.add_argument(
        "--permutation_repeats",
        type=int,
        default=30,
        help="Number of feature permutations per held-out split.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top features to save and plot.",
    )
    return p.parse_args()


def detect_label_column(df, explicit=None):
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"Label column '{explicit}' not found.")
        return explicit

    for c in [
        "label",
        "target",
        "y",
        "adhd_label",
        "class_label",
        "diagnosis",
        "group",
    ]:
        if c in df.columns:
            return c

    raise ValueError(
        "Could not detect label column. Use --label_col <column_name>."
    )


def encode_binary_target(series, positive_label=None):
    unique = list(pd.Series(series.dropna().unique()))

    if set(unique) == {0, 1}:
        return series.astype(int), 1

    normalized = series.astype(str).str.strip().str.lower()

    if positive_label is not None:
        pos = str(positive_label).strip().lower()
        y = normalized.eq(pos).astype(int)
        if y.nunique() != 2:
            raise ValueError(
                f"positive_label={positive_label!r} did not produce two classes."
            )
        return y, positive_label

    positive_names = {"adhd", "case", "positive", "1", "true", "yes"}
    y = normalized.isin(positive_names).astype(int)

    if y.nunique() == 2:
        return y, "ADHD"

    raise ValueError(
        "Could not identify positive class. Use --positive_label <value>."
    )


def infer_features(df, feature_set):
    if all(c in df.columns for c in STAGE1_FEATURES):
        s1 = STAGE1_FEATURES
    else:
        s1 = [c for c in df.columns if c.lower().startswith("s1_")]

    if all(c in df.columns for c in STAGE2_FEATURES):
        s2 = STAGE2_FEATURES
    else:
        s2 = [c for c in df.columns if c.lower().startswith("s2_")]

    if not s1:
        raise ValueError("Could not infer Stage 1 feature columns.")

    if feature_set == "stage1":
        return s1

    if not s2:
        raise ValueError("Could not infer Stage 2 feature columns.")

    return s1 + s2


def feature_stage(feature):
    if feature in STAGE1_FEATURES or feature.lower().startswith("s1_"):
        return "Stage 1"
    if feature in STAGE2_FEATURES or feature.lower().startswith("s2_"):
        return "Stage 2"
    return "Unknown"


def make_models(seed):
    lr = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return lr, rf, xgb


def evaluate_model(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_test, prob),
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


def summarize_lr(raw_df):
    grouped = raw_df.groupby(["feature", "stage"], as_index=False).agg(
        signed_coefficient_mean=("coefficient", "mean"),
        signed_coefficient_std=("coefficient", "std"),
        absolute_coefficient_mean=("absolute_coefficient", "mean"),
        absolute_coefficient_std=("absolute_coefficient", "std"),
    )

    return grouped.sort_values(
        "absolute_coefficient_mean",
        ascending=False,
    ).reset_index(drop=True)


def summarize_permutation(raw_df):
    # First average the internal permutation repeats within each split.
    # Then summarize variability across the 30 train/test splits.
    grouped = raw_df.groupby(["feature", "stage"], as_index=False).agg(
        permutation_importance_mean=("importance_mean", "mean"),
        permutation_importance_std_across_splits=("importance_mean", "std"),
        mean_within_split_permutation_std=("importance_std", "mean"),
    )

    return grouped.sort_values(
        "permutation_importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def plot_lr_top10(top_df, output_path):
    plot_df = top_df.sort_values(
        "absolute_coefficient_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        plot_df["feature"],
        plot_df["signed_coefficient_mean"],
        xerr=plot_df["signed_coefficient_std"],
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean standardized coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Logistic Regression: Top 10 Standardized Coefficients")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_top10(top_df, output_path, title):
    plot_df = top_df.sort_values(
        "permutation_importance_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        plot_df["feature"],
        plot_df["permutation_importance_mean"],
        xerr=plot_df["permutation_importance_std_across_splits"],
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Decrease in held-out ROC AUC after permutation")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    label_col = detect_label_column(df, args.label_col)
    y, positive_class = encode_binary_target(
        df[label_col],
        args.positive_label,
    )

    feature_cols = infer_features(df, args.feature_set)
    X = df[feature_cols].copy()

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    split_indices = list(
        splitter.split(np.zeros(len(y)), y)
    )

    lr_rows = []
    rf_rows = []
    xgb_rows = []
    performance_rows = []

    print("=" * 76)
    print("30-SPLIT FEATURE EXPLAINABILITY ANALYSIS")
    print("=" * 76)
    print(f"Input:                {input_path}")
    print(f"Feature set:          {args.feature_set}")
    print(f"Number of features:   {len(feature_cols)}")
    print(f"Splits:               {args.n_splits}")
    print(f"Train/Test:           {1-args.test_size:.0%}/{args.test_size:.0%}")
    print(f"Permutation repeats:  {args.permutation_repeats}")
    print(f"Permutation scoring:  roc_auc")
    print()

    for split_id, (train_idx, test_idx) in enumerate(
        split_indices,
        start=1,
    ):
        print(f"Split {split_id:02d}/{args.n_splits}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        seed = args.random_seed + split_id
        lr, rf, xgb = make_models(seed)

        # ---------------------------------------------------------
        # Logistic Regression standardized coefficients
        # ---------------------------------------------------------
        lr.fit(X_train, y_train)

        coef = lr.named_steps["model"].coef_.ravel()

        if len(coef) != len(feature_cols):
            raise RuntimeError(
                "Unexpected Logistic Regression coefficient length."
            )

        for feature, value in zip(feature_cols, coef):
            lr_rows.append(
                {
                    "split": split_id,
                    "feature": feature,
                    "stage": feature_stage(feature),
                    "coefficient": float(value),
                    "absolute_coefficient": float(abs(value)),
                }
            )

        perf = evaluate_model(lr, X_test, y_test)
        performance_rows.append(
            {
                "split": split_id,
                "model": "Logistic Regression",
                **perf,
            }
        )

        # ---------------------------------------------------------
        # Random Forest permutation importance on held-out data
        # ---------------------------------------------------------
        rf.fit(X_train, y_train)

        perf = evaluate_model(rf, X_test, y_test)
        performance_rows.append(
            {
                "split": split_id,
                "model": "Random Forest",
                **perf,
            }
        )

        rf_perm = permutation_importance(
            rf,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=args.permutation_repeats,
            random_state=seed,
            n_jobs=-1,
        )

        for idx, feature in enumerate(feature_cols):
            rf_rows.append(
                {
                    "split": split_id,
                    "feature": feature,
                    "stage": feature_stage(feature),
                    "importance_mean": float(
                        rf_perm.importances_mean[idx]
                    ),
                    "importance_std": float(
                        rf_perm.importances_std[idx]
                    ),
                }
            )

        # ---------------------------------------------------------
        # XGBoost permutation importance on held-out data
        # ---------------------------------------------------------
        xgb.fit(X_train, y_train)

        perf = evaluate_model(xgb, X_test, y_test)
        performance_rows.append(
            {
                "split": split_id,
                "model": "XGBoost",
                **perf,
            }
        )

        xgb_perm = permutation_importance(
            xgb,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=args.permutation_repeats,
            random_state=seed + 10000,
            n_jobs=-1,
        )

        for idx, feature in enumerate(feature_cols):
            xgb_rows.append(
                {
                    "split": split_id,
                    "feature": feature,
                    "stage": feature_stage(feature),
                    "importance_mean": float(
                        xgb_perm.importances_mean[idx]
                    ),
                    "importance_std": float(
                        xgb_perm.importances_std[idx]
                    ),
                }
            )

    # -------------------------------------------------------------
    # Save raw split-level results
    # -------------------------------------------------------------
    lr_raw = pd.DataFrame(lr_rows)
    rf_raw = pd.DataFrame(rf_rows)
    xgb_raw = pd.DataFrame(xgb_rows)
    performance_df = pd.DataFrame(performance_rows)

    lr_raw.to_csv(
        output_dir / "lr_coefficients_all_splits.csv",
        index=False,
    )
    rf_raw.to_csv(
        output_dir / "rf_permutation_all_splits.csv",
        index=False,
    )
    xgb_raw.to_csv(
        output_dir / "xgb_permutation_all_splits.csv",
        index=False,
    )
    performance_df.to_csv(
        output_dir / "model_test_performance.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------
    lr_summary = summarize_lr(lr_raw)
    rf_summary = summarize_permutation(rf_raw)
    xgb_summary = summarize_permutation(xgb_raw)

    lr_summary.to_csv(
        output_dir / "lr_feature_importance_summary.csv",
        index=False,
    )
    rf_summary.to_csv(
        output_dir / "rf_feature_importance_summary.csv",
        index=False,
    )
    xgb_summary.to_csv(
        output_dir / "xgb_feature_importance_summary.csv",
        index=False,
    )

    top_k = min(args.top_k, len(feature_cols))

    lr_top = lr_summary.head(top_k).copy()
    rf_top = rf_summary.head(top_k).copy()
    xgb_top = xgb_summary.head(top_k).copy()

    lr_top.to_csv(
        output_dir / "lr_top10.csv",
        index=False,
    )
    rf_top.to_csv(
        output_dir / "rf_top10.csv",
        index=False,
    )
    xgb_top.to_csv(
        output_dir / "xgb_top10.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Three figures
    # -------------------------------------------------------------
    plot_lr_top10(
        lr_top,
        output_dir / "lr_top10_coefficients.png",
    )
    plot_permutation_top10(
        rf_top,
        output_dir / "rf_top10_permutation_importance.png",
        "Random Forest: Top 10 Permutation Importances",
    )
    plot_permutation_top10(
        xgb_top,
        output_dir / "xgb_top10_permutation_importance.png",
        "XGBoost: Top 10 Permutation Importances",
    )

    # -------------------------------------------------------------
    # Performance summary
    # -------------------------------------------------------------
    performance_summary = (
        performance_df.groupby("model")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )

    performance_summary.to_csv(
        output_dir / "model_test_performance_summary.csv",
        index=False,
    )

    metadata = {
        "input": str(input_path),
        "feature_set": args.feature_set,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "label_column": label_col,
        "positive_class": str(positive_class),
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "permutation_repeats": args.permutation_repeats,
        "permutation_scoring": "roc_auc",
        "logistic_regression_importance": (
            "Standardized model coefficients from a pipeline using "
            "median imputation and StandardScaler. Ranking uses mean "
            "absolute coefficient across the repeated splits; plots show "
            "the mean signed coefficient."
        ),
        "tree_importance": (
            "Permutation importance computed on each held-out test split. "
            "Feature importance is the decrease in ROC AUC after shuffling "
            "the feature. Summary values aggregate importance across splits."
        ),
        "models": {
            "Logistic Regression": {
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 5000,
                "random_state": 0,
            },
            "Random Forest": {
                "n_estimators": 100,
                "class_weight": "balanced",
            },
            "XGBoost": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            },
        },
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 76)
    print("TOP FEATURES")
    print("=" * 76)

    print("\nLogistic Regression:")
    print(
        lr_top[
            [
                "feature",
                "stage",
                "signed_coefficient_mean",
                "absolute_coefficient_mean",
            ]
        ].to_string(index=False)
    )

    print("\nRandom Forest:")
    print(
        rf_top[
            [
                "feature",
                "stage",
                "permutation_importance_mean",
                "permutation_importance_std_across_splits",
            ]
        ].to_string(index=False)
    )

    print("\nXGBoost:")
    print(
        xgb_top[
            [
                "feature",
                "stage",
                "permutation_importance_mean",
                "permutation_importance_std_across_splits",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
