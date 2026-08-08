#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier

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
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="synthetic_stage1_stage2_dataset.csv")
    p.add_argument("--output_dir", default="uniform_three_model_results")
    p.add_argument("--label_col", default=None)
    p.add_argument("--positive_label", default=None)
    p.add_argument("--n_splits", type=int, default=30)
    p.add_argument("--test_size", type=float, default=0.30)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--low", type=float, default=0.30)
    p.add_argument("--high", type=float, default=0.70)
    return p.parse_args()


def detect_label_col(df, explicit=None):
    if explicit:
        return explicit
    for c in ["label", "target", "adhd_label", "class_label", "diagnosis", "group"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot detect label column. Use --label_col.")


def encode_target(s, positive_label=None):
    vals = set(pd.Series(s.dropna().unique()).tolist())
    if vals == {0, 1}:
        return s.astype(int)

    sn = s.astype(str).str.strip().str.lower()
    if positive_label is not None:
        return sn.eq(str(positive_label).strip().lower()).astype(int)

    pos_names = {"adhd", "case", "positive", "1", "true", "yes"}
    y = sn.isin(pos_names).astype(int)
    if y.nunique() != 2:
        raise ValueError("Cannot detect positive class. Use --positive_label.")
    return y


def infer_features(df):
    if all(c in df.columns for c in STAGE1_FEATURES + STAGE2_FEATURES):
        return STAGE1_FEATURES, STAGE2_FEATURES
    s1 = [c for c in df.columns if c.lower().startswith("s1_")]
    s2 = [c for c in df.columns if c.lower().startswith("s2_")]
    if s1 and s2:
        return s1, s2
    raise ValueError("Cannot infer Stage 1/Stage 2 features.")


def make_models(seed):
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=0,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
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
                n_jobs=-1,
            )),
        ]),
    }


def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run_split(df, y, train_idx, test_idx, s1_cols, s2_cols, low, high, split_id, seed):
    rows = []
    full_cols = s1_cols + s2_cols

    X1_tr, X1_te = df.iloc[train_idx][s1_cols], df.iloc[test_idx][s1_cols]
    XF_tr, XF_te = df.iloc[train_idx][full_cols], df.iloc[test_idx][full_cols]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    for model_name, base_model in make_models(seed + split_id).items():
        # Stage 1 model
        m1 = clone(base_model)
        m1.fit(X1_tr, y_tr)
        p1 = m1.predict_proba(X1_te)[:, 1]
        pred1 = (p1 >= 0.5).astype(int)
        rows.append({
            "split": split_id,
            "model": model_name,
            "configuration": "Stage 1 only",
            "stage2_usage": 0.0,
            "n_stage2": 0,
            **metrics(y_te, pred1, p1),
        })

        # Full model
        mf = clone(base_model)
        mf.fit(XF_tr, y_tr)
        p2 = mf.predict_proba(XF_te)[:, 1]
        pred2 = (p2 >= 0.5).astype(int)
        rows.append({
            "split": split_id,
            "model": model_name,
            "configuration": "Stage 1 + Stage 2",
            "stage2_usage": 1.0,
            "n_stage2": len(test_idx),
            **metrics(y_te, pred2, p2),
        })

        # Cascade: same classifier family for both stages
        uncertain = (p1 > low) & (p1 < high)
        p_cas = p1.copy()
        p_cas[uncertain] = p2[uncertain]

        pred_cas = np.empty_like(pred1)
        pred_cas[p1 <= low] = 0
        pred_cas[p1 >= high] = 1
        pred_cas[uncertain] = (p2[uncertain] >= 0.5).astype(int)

        rows.append({
            "split": split_id,
            "model": model_name,
            "configuration": "Cascade",
            "stage2_usage": uncertain.mean(),
            "n_stage2": int(uncertain.sum()),
            **metrics(y_te, pred_cas, p_cas),
        })

    return rows


def summarize(run_df):
    metric_cols = [
        "accuracy", "auc", "f1", "precision", "recall",
        "specificity", "stage2_usage", "n_stage2"
    ]
    rows = []
    for (model, config), g in run_df.groupby(["model", "configuration"], sort=False):
        r = {"model": model, "configuration": config, "n_splits": len(g)}
        for c in metric_cols:
            r[f"{c}_mean"] = g[c].mean()
            r[f"{c}_std"] = g[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if not (0 <= args.low < 0.5 < args.high <= 1):
        raise ValueError("Need 0 <= low < 0.5 < high <= 1")

    df = pd.read_csv(args.input)
    label_col = detect_label_col(df, args.label_col)
    y = encode_target(df[label_col], args.positive_label)
    s1_cols, s2_cols = infer_features(df)

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    all_rows = []
    # SAME split indices are reused for LR, RF and XGBoost.
    for split_id, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        all_rows.extend(run_split(
            df, y, train_idx, test_idx,
            s1_cols, s2_cols,
            args.low, args.high,
            split_id, args.random_seed,
        ))
        print(f"Completed split {split_id:02d}/{args.n_splits}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_df = pd.DataFrame(all_rows)
    summary_df = summarize(run_df)

    run_path = out / "uniform_model_comparison_runs.csv"
    summary_path = out / "uniform_model_comparison_summary.csv"
    run_df.to_csv(run_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n=== SUMMARY ===")
    for _, r in summary_df.iterrows():
        print(
            f"{r['model']:20s} | {r['configuration']:18s} | "
            f"Acc {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} | "
            f"AUC {r['auc_mean']:.3f}±{r['auc_std']:.3f} | "
            f"F1 {r['f1_mean']:.3f}±{r['f1_std']:.3f} | "
            f"Recall {r['recall_mean']:.3f}±{r['recall_std']:.3f} | "
            f"S2 {100*r['stage2_usage_mean']:.1f}±{100*r['stage2_usage_std']:.1f}%"
        )

    print(f"\nSaved:\n  {run_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
