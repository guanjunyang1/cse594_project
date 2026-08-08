#!/usr/bin/env python3
import argparse, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("Please install xgboost first: pip install xgboost") from e

SCENARIO_STRENGTHS = {"weak": 0.30, "medium": 0.60, "strong": 1.00}
DEFAULT_THRESHOLDS = [(0.20, 0.80), (0.30, 0.70), (0.40, 0.60)]
POSITIVE_CANDIDATES = {"adhd", "case", "positive", "1", "true", "yes"}
META_NAMES = {
    "participant_id", "participant", "subject_id", "subject", "id",
    "group", "class", "label", "target", "diagnosis"
}


def parse_args():
    p = argparse.ArgumentParser(
        description="3 models x 3 thresholds x 30 repeated splits under weak/medium/strong Stage-1 scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--input",
        default="synthetic_stage1_stage2_dataset.csv",
        help="Input CSV file."
    )
    p.add_argument("--output_dir", default="signal_scenario_results")
    p.add_argument("--label_col", default=None)
    p.add_argument("--positive_label", default=None)
    p.add_argument("--stage1_cols", nargs="*", default=None)
    p.add_argument("--stage2_cols", nargs="*", default=None)
    p.add_argument("--thresholds", default="0.20,0.80;0.30,0.70;0.40,0.60")
    p.add_argument("--n_repeats", type=int, default=30)
    p.add_argument("--test_size", type=float, default=0.30)
    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()


def parse_thresholds(text):
    out = []
    for item in text.split(";"):
        lo, hi = [float(x.strip()) for x in item.split(",")]
        if not (0 <= lo < 0.5 < hi <= 1):
            raise ValueError(f"Invalid threshold pair: {(lo, hi)}")
        out.append((lo, hi))
    return out


def infer_label_col(df, explicit=None):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"label column not found: {explicit}")
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for name in ["label", "target", "diagnosis", "class", "group"]:
        if name in lower:
            return lower[name]
    binary = [c for c in df.columns if df[c].dropna().nunique() == 2]
    if len(binary) == 1:
        return binary[0]
    raise ValueError("Cannot infer label column; use --label_col.")


def detect_positive_label(s, explicit=None):
    vals = list(pd.Series(s.dropna().unique()).tolist())
    if len(vals) != 2:
        raise ValueError(f"Expected binary labels, got: {vals}")
    if explicit is not None:
        for v in vals:
            if str(v) == str(explicit):
                return v
        raise ValueError(f"positive label {explicit} not found in {vals}")
    try:
        if {float(v) for v in vals} == {0.0, 1.0}:
            return next(v for v in vals if float(v) == 1.0)
    except Exception:
        pass
    for v in vals:
        if str(v).strip().lower() in POSITIVE_CANDIDATES:
            return v
    adhd = [v for v in vals if "adhd" in str(v).lower()]
    if len(adhd) == 1:
        return adhd[0]
    raise ValueError(f"Cannot infer positive label from {vals}; use --positive_label.")


def infer_feature_columns(df, label_col, s1_explicit=None, s2_explicit=None):
    if s1_explicit and s2_explicit:
        return list(s1_explicit), list(s2_explicit)

    cols = list(df.columns)
    s1 = [c for c in cols if str(c).lower().startswith(("s1_", "stage1_"))]
    s2 = [c for c in cols if str(c).lower().startswith(("s2_", "stage2_"))]
    if len(s1) == 15 and len(s2) == 13:
        return s1, s2

    excluded = {label_col}
    for c in cols:
        if str(c).lower() in META_NAMES:
            excluded.add(c)
    numeric = [
        c for c in cols
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric) < 28:
        raise ValueError(
            "Could not infer 28 numerical features. Use --stage1_cols/--stage2_cols "
            "or rename features with s1_/s2_ prefixes."
        )
    if len(numeric) > 28:
        warnings.warn("More than 28 numeric candidates found; using first 28.")
    numeric = numeric[:28]
    return numeric[:15], numeric[15:28]


def encode_y(s, positive_label):
    return (s.to_numpy() == positive_label).astype(int)


def create_signal_scenario(df, y, stage1_cols, strength, seed=42):
    """
    Construct weak/medium/strong Stage-1 scenarios by controlling only
    class-mean separation.

    For each Stage-1 feature j:

        midpoint_j = (mu_TDC + mu_ADHD) / 2
        delta_j    = mu_ADHD - mu_TDC

        target_mu_TDC  = midpoint_j - strength * delta_j / 2
        target_mu_ADHD = midpoint_j + strength * delta_j / 2

    A single pooled residual distribution is shared by both classes. This
    removes class-specific variance, skewness, kurtosis, zero-mass, and other
    residual-shape differences that tree models could otherwise exploit when
    the intended Stage-1 mean signal is weak.

    Residuals are shuffled independently of the labels for each feature. The
    same feature scale is retained through the pooled standard deviation.
    Stage-2 features are left unchanged.

    For features that were originally non-negative, generated values are
    shifted upward as a whole if necessary instead of clipping at zero. A
    common shift preserves class separation and avoids creating an artificial
    point mass at zero.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)

    for col in stage1_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(x)
        x0 = x[(y == 0) & finite_mask]
        x1 = x[(y == 1) & finite_mask]

        if len(x0) < 2 or len(x1) < 2:
            warnings.warn(
                f"Skipping scenario transformation for {col}: "
                "not enough finite observations in both classes."
            )
            continue

        mu0, mu1 = np.mean(x0), np.mean(x1)
        sd0, sd1 = np.std(x0, ddof=1), np.std(x1, ddof=1)

        sd0s = sd0 if np.isfinite(sd0) and sd0 > 1e-12 else 1.0
        sd1s = sd1 if np.isfinite(sd1) and sd1 > 1e-12 else 1.0

        midpoint = (mu0 + mu1) / 2.0
        delta = mu1 - mu0
        target_mu0 = midpoint - strength * delta / 2.0
        target_mu1 = midpoint + strength * delta / 2.0

        pooled_sd = math.sqrt((sd0s**2 + sd1s**2) / 2.0)

        standardized_residuals = np.empty(finite_mask.sum(), dtype=float)
        finite_indices = np.flatnonzero(finite_mask)
        for k, idx in enumerate(finite_indices):
            if y[idx] == 0:
                standardized_residuals[k] = (x[idx] - mu0) / sd0s
            else:
                standardized_residuals[k] = (x[idx] - mu1) / sd1s

        # Use one common residual pool for both classes and break any
        # label-specific residual structure.
        shared_residuals = rng.permutation(standardized_residuals)

        new = np.full_like(x, np.nan, dtype=float)
        finite_positions = np.flatnonzero(finite_mask)
        for k, idx in enumerate(finite_positions):
            target_mean = target_mu1 if y[idx] == 1 else target_mu0
            new[idx] = target_mean + shared_residuals[k] * pooled_sd

        # Preserve the physical non-negative domain without clipping. A common
        # additive shift cannot create extra class-separation information.
        original_finite = x[finite_mask]
        generated_finite = new[finite_mask]
        if original_finite.size and np.min(original_finite) >= 0:
            min_generated = np.min(generated_finite)
            if min_generated < 0:
                new[finite_mask] = generated_finite - min_generated

        out[col] = new

    return out


def build_models(seed):
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=0
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1
            ))
        ])
    }


def calc_metrics(y_true, y_pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def run_split(df, y, train_idx, test_idx, stage1_cols, stage2_cols,
              thresholds, split_id, scenario, seed):
    X1 = df[stage1_cols]
    XF = df[stage1_cols + stage2_cols]

    X1tr, X1te = X1.iloc[train_idx], X1.iloc[test_idx]
    XFtr, XFte = XF.iloc[train_idx], XF.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    rows, uncertain_rows, prediction_rows = [], [], []

    for model_name, proto in build_models(seed).items():
        m1 = clone(proto)
        mf = clone(proto)
        m1.fit(X1tr, ytr)
        mf.fit(XFtr, ytr)

        p1 = m1.predict_proba(X1te)[:, 1]
        p2 = mf.predict_proba(XFte)[:, 1]
        pred1 = (p1 >= 0.5).astype(int)
        pred2 = (p2 >= 0.5).astype(int)
        for pos in range(len(yte)):
            prediction_rows.append({
                "scenario": scenario,
                "split": split_id,
                "model": model_name,
                "configuration": "Stage 1 only",
                "threshold_low": np.nan,
                "threshold_high": np.nan,
                "row_index": int(test_idx[pos]),
                "y_true": int(yte[pos]),
                "y_score": float(p1[pos]),
                "y_pred": int(pred1[pos])
            })
            prediction_rows.append({
                "scenario": scenario,
                "split": split_id,
                "model": model_name,
                "configuration": "Stage 1 + Stage 2",
                "threshold_low": np.nan,
                "threshold_high": np.nan,
                "row_index": int(test_idx[pos]),
                "y_true": int(yte[pos]),
                "y_score": float(p2[pos]),
                "y_pred": int(pred2[pos])
            })

        rows.append({
            "scenario": scenario, "split": split_id, "model": model_name,
            "configuration": "Stage 1 only",
            "threshold_low": np.nan, "threshold_high": np.nan,
            "stage2_usage": 0.0, "n_stage2": 0,
            **calc_metrics(yte, pred1, p1)
        })
        rows.append({
            "scenario": scenario, "split": split_id, "model": model_name,
            "configuration": "Stage 1 + Stage 2",
            "threshold_low": np.nan, "threshold_high": np.nan,
            "stage2_usage": 1.0, "n_stage2": len(yte),
            **calc_metrics(yte, pred2, p2)
        })

        for lo, hi in thresholds:
            uncertain = (p1 > lo) & (p1 < hi)
            cascade_pred = pred1.copy()
            cascade_pred[uncertain] = pred2[uncertain]

            cascade_prob = p1.copy()
            cascade_prob[uncertain] = p2[uncertain]

            n2 = int(uncertain.sum())
            rows.append({
                "scenario": scenario, "split": split_id, "model": model_name,
                "configuration": "Cascade",
                "threshold_low": lo, "threshold_high": hi,
                "stage2_usage": n2 / len(yte), "n_stage2": n2,
                **calc_metrics(yte, cascade_pred, cascade_prob)
            })

            for pos in np.flatnonzero(uncertain):
                uncertain_rows.append({
                    "scenario": scenario,
                    "split": split_id,
                    "model": model_name,
                    "threshold_low": lo,
                    "threshold_high": hi,
                    "row_index": int(test_idx[pos]),
                    "y_true": int(yte[pos]),
                    "p_stage1": float(p1[pos]),
                    "p_full": float(p2[pos]),
                    "stage1_pred": int(pred1[pos]),
                    "full_pred": int(pred2[pos]),
                    "changed_prediction": int(pred1[pos] != pred2[pos]),
                    "corrected_error": int(pred1[pos] != yte[pos] and pred2[pos] == yte[pos]),
                    "worsened_prediction": int(pred1[pos] == yte[pos] and pred2[pos] != yte[pos])
                })

            for pos in range(len(yte)):
                prediction_rows.append({
                    "scenario": scenario,
                    "split": split_id,
                    "model": model_name,
                    "configuration": "Cascade",
                    "threshold_low": lo,
                    "threshold_high": hi,
                    "row_index": int(test_idx[pos]),
                    "y_true": int(yte[pos]),
                    "y_score": float(cascade_prob[pos]),
                    "y_pred": int(cascade_pred[pos])
                })

    return rows, uncertain_rows, prediction_rows


def summarize(run_df):
    metrics = [
        "accuracy", "auc", "f1", "precision", "recall", "specificity",
        "stage2_usage", "n_stage2"
    ]
    keys = ["scenario", "model", "configuration", "threshold_low", "threshold_high"]
    g = run_df.groupby(keys, dropna=False)
    summary = g[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(x) for x in c if str(x) != ""]).rstrip("_")
        if isinstance(c, tuple) else c
        for c in summary.columns
    ]
    count = g.size().reset_index(name="n_splits")
    return count.merge(summary, on=keys, how="left")


def threshold_summary(run_df):
    metrics = [
        "accuracy", "auc", "f1", "precision", "recall", "specificity",
        "stage2_usage", "n_stage2"
    ]
    x = run_df[run_df["configuration"] == "Cascade"].copy()
    keys = ["scenario", "model", "threshold_low", "threshold_high"]
    out = x.groupby(keys)[metrics].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join([str(v) for v in c if str(v) != ""]).rstrip("_")
        if isinstance(c, tuple) else c
        for c in out.columns
    ]
    return out


def make_plots(summary_df, threshold_df, prediction_df, outdir):
    import matplotlib.pyplot as plt

    pdir = outdir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    labels = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGBoost"}
    markers = {"Logistic Regression": "o", "Random Forest": "s", "XGBoost": "^"}
    styles = {"Logistic Regression": "-", "Random Forest": "--", "XGBoost": "-."}
    scenarios = ["weak", "medium", "strong"]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8
    })
    # Mean ROC curves over 30 repeated splits for each signal scenario.
    # To keep the paper figure readable, compare Stage 1 only and
    # Stage 1 + Stage 2 for LR, RF, and XGBoost. Cascade ROC data remain
    # available in all_test_predictions.csv for separate analysis.
    mean_fpr = np.linspace(0.0, 1.0, 201)

    for s in scenarios:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ss = prediction_df[
            (prediction_df["scenario"] == s) &
            (prediction_df["configuration"].isin(["Stage 1 only", "Stage 1 + Stage 2"]))
        ]

        for model in models:
            for configuration, linestyle in [
                ("Stage 1 only", "--"),
                ("Stage 1 + Stage 2", "-")
            ]:
                tprs = []
                aucs = []
                subset = ss[
                    (ss["model"] == model) &
                    (ss["configuration"] == configuration)
                ]

                for split_id in sorted(subset["split"].unique()):
                    split_data = subset[subset["split"] == split_id]
                    y_true = split_data["y_true"].to_numpy()
                    y_score = split_data["y_score"].to_numpy()
                    if len(np.unique(y_true)) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(roc_auc_score(y_true, y_score))

                if not tprs:
                    continue

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = float(np.mean(aucs))
                label = f"{labels[model]} {'S1' if configuration == 'Stage 1 only' else 'S1+S2'} (AUC={mean_auc:.3f})"
                ax.plot(mean_fpr, mean_tpr, linestyle=linestyle, linewidth=1.2, label=label)

        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(pdir / f"roc_curve_{s}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for model in models:
        means, stds = [], []
        for s in scenarios:
            r = summary_df[
                (summary_df.scenario == s) &
                (summary_df.model == model) &
                (summary_df.configuration == "Stage 1 only")
            ].iloc[0]
            means.append(r["auc_mean"])
            stds.append(r["auc_std"])
        ax.errorbar(scenarios, means, yerr=stds,
                    marker=markers[model], linestyle=styles[model],
                    capsize=2, linewidth=1.3, label=labels[model])
    ax.set_xlabel("Stage 1 signal strength")
    ax.set_ylabel("AUC")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "stage1_auc_vs_signal_strength.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    common = threshold_df[
        np.isclose(threshold_df.threshold_low, 0.30) &
        np.isclose(threshold_df.threshold_high, 0.70)
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for model in models:
        means, stds = [], []
        for s in scenarios:
            r = common[(common.scenario == s) & (common.model == model)].iloc[0]
            means.append(100 * r["stage2_usage_mean"])
            stds.append(100 * r["stage2_usage_std"])
        ax.errorbar(scenarios, means, yerr=stds,
                    marker=markers[model], linestyle=styles[model],
                    capsize=2, linewidth=1.3, label=labels[model])
    ax.set_xlabel("Stage 1 signal strength")
    ax.set_ylabel("Stage 2 usage (%)")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "cascade_stage2_usage_vs_signal_strength.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    for s in scenarios:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ss = threshold_df[threshold_df.scenario == s]
        for model in models:
            m = ss[ss.model == model].sort_values("stage2_usage_mean")
            ax.errorbar(
                100 * m["stage2_usage_mean"], m["f1_mean"],
                xerr=100 * m["stage2_usage_std"], yerr=m["f1_std"],
                marker=markers[model], linestyle=styles[model],
                capsize=2, linewidth=1.3, label=labels[model]
            )
        ax.set_xlabel("Stage 2 usage (%)")
        ax.set_ylabel("F1 score")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / f"threshold_f1_tradeoff_{s}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\nUse --input <your_csv_file>."
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sdir = outdir / "scenario_datasets"
    sdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    label_col = infer_label_col(df, args.label_col)
    positive_label = detect_positive_label(df[label_col], args.positive_label)
    y = encode_y(df[label_col], positive_label)

    s1_cols, s2_cols = infer_feature_columns(
        df, label_col, args.stage1_cols, args.stage2_cols
    )

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_repeats,
        test_size=args.test_size,
        random_state=args.random_seed
    )
    splits = list(splitter.split(np.zeros(len(y)), y))

    rows, uncertain, predictions = [], [], []

    for scenario, strength in SCENARIO_STRENGTHS.items():
        sdf = create_signal_scenario(
            df, y, s1_cols, strength, seed=args.random_seed
        )
        sdf.to_csv(sdir / f"synthetic_{scenario}_stage1_signal.csv", index=False)

        for split_id, (train_idx, test_idx) in enumerate(splits, 1):
            r, u, p = run_split(
                sdf, y, train_idx, test_idx,
                s1_cols, s2_cols, thresholds,
                split_id, scenario, args.random_seed + split_id
            )
            rows.extend(r)
            uncertain.extend(u)
            predictions.extend(p)

    run_df = pd.DataFrame(rows)
    uncertain_df = pd.DataFrame(uncertain)
    prediction_df = pd.DataFrame(predictions)
    summary_df = summarize(run_df)
    threshold_df = threshold_summary(run_df)

    run_df.to_csv(outdir / "run_level_metrics.csv", index=False)
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    threshold_df.to_csv(outdir / "threshold_sensitivity.csv", index=False)
    uncertain_df.to_csv(outdir / "uncertain_case_predictions.csv", index=False)
    prediction_df.to_csv(outdir / "all_test_predictions.csv", index=False)

    make_plots(summary_df, threshold_df, prediction_df, outdir)

    metadata = {
        "input": str(input_path),
        "label_col": label_col,
        "positive_label": str(positive_label),
        "stage1_cols": s1_cols,
        "stage2_cols": s2_cols,
        "scenario_strengths": SCENARIO_STRENGTHS,
        "thresholds": thresholds,
        "n_repeats": args.n_repeats,
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "models": ["Logistic Regression", "Random Forest", "XGBoost"],
        "notes": [
            "Same stratified split indices reused across models/configurations/scenarios.",
            "Stage-1 scenarios use symmetric mean scaling with one label-independent shared residual distribution and pooled feature scale.",
            "Originally non-negative Stage-1 features are shifted, not clipped, if generated values cross zero.",
            "Stage-2 features are unchanged across signal scenarios.",
            "Cascade AUC uses p1 for confident cases and p2 for uncertain cases.",
            "all_test_predictions.csv stores y_true, y_score, and y_pred for ROC/PR/confusion-matrix analysis."
        ]
    }
    (outdir / "metadata.txt").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Done.")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Stage 1 features ({len(s1_cols)}): {s1_cols}")
    print(f"Stage 2 features ({len(s2_cols)}): {s2_cols}")


if __name__ == "__main__":
    main()
