import numpy as np
import pandas as pd

np.random.seed(42)

n_control = 100
n_adhd = 100


stage1_reference = [
    # Sankesara et al., 2025 exact mean ± SD
    ("questionnaire_latency_mean", 11.69, 10.31, 19.14, 12.98),
    ("questionnaire_latency_sd", 8.97, 6.83, 15.88, 9.27),
    ("questionnaire_interval_mean", 0.18, 0.64, 1.12, 2.17),
    ("questionnaire_interval_sd", 0.35, 2.23, 2.01, 5.04),
    ("social_notification_latency_mean", 1516.45, 785.78, 2304.27, 1406.62),
    ("social_notification_latency_sd", 2125.15, 840.86, 2736.38, 1091.17),
    ("ambient_light_sd", 43.63, 73.57, 83.96, 155.57),
    ("fitbit_steps_active", 179.20, 177.26, 313.14, 263.88),
    ("active_session_duration", 1047.68, 1060.45, 1778.07, 1653.87),
    ("new_apps_added", 1.43, 0.81, 1.18, 0.40),

    # Kofler et al., 2013 effect-size simulation
    ("scroll_interval_sd", 0.00, 1.00, 0.76, 1.00),
    ("tap_interval_sd", 0.00, 1.00, 0.76, 1.00),
    ("dwell_time_sd", 0.00, 1.00, 0.76, 1.00),
    ("idle_interval_sd", 0.00, 1.00, 0.76, 1.00),
    ("session_fragmentation", 0.00, 1.00, 0.46, 1.00),
]


stage2_reference = [
    # Yoo et al., 2024 exact mean ± SD
    ("pst_fixation_duration", 71.79, 16.90, 64.57, 19.70),
    ("ast_saccade_time_max", 22.19, 7.41, 24.68, 6.58),
    ("mgst_saccade_degree_total", 10647.83, 4475.26, 14201.95, 6935.07),
    ("mgst_saccade_degree_mean", 37.48, 8.45, 40.88, 6.05),
    ("mgst_saccade_time_max", 14.96, 5.47, 17.91, 6.64),
    ("mgst_fixation_time_max", 35.54, 4.78, 32.41, 7.17),
    ("mgst_fixation_mean", 32.27, 6.36, 28.71, 8.09),
    ("mgst_saccade_latency_total", 55.75, 10.84, 51.14, 12.20),
    ("mgst_saccade_duration", 35.11, 14.18, 42.41, 17.58),
    ("cdt_saccade_degree_sd", 27.47, 6.61, 24.67, 4.76),
    ("stroop_saccade_degree_mean", 30.54, 6.35, 33.67, 6.43),
    ("stroop_screen_duration_mean", 0.33, 0.34, 0.42, 0.32),
    ("stroop_saccade_latency_total", 2.33, 0.65, 2.66, 0.75),
]


def generate_group(n, group_name, label):
    data = {
        "participant_id": [],
        "group": [],
        "label": []
    }

    all_features = stage1_reference + stage2_reference

    for feature_name, *_ in all_features:
        data[feature_name] = []

    for i in range(n):
        pid = f"{group_name}_{i:03d}"

        data["participant_id"].append(pid)
        data["group"].append(group_name)
        data["label"].append(label)

        for feature_name, control_mean, control_sd, adhd_mean, adhd_sd in all_features:
            if label == 0:
                value = np.random.normal(control_mean, control_sd)
            else:
                value = np.random.normal(adhd_mean, adhd_sd)


            standardized_features = [
                "scroll_interval_sd",
                "tap_interval_sd",
                "dwell_time_sd",
                "idle_interval_sd",
                "session_fragmentation"
            ]

            if feature_name not in standardized_features:
                value = max(value, 0)

            data[feature_name].append(value)

    return pd.DataFrame(data)


control_df = generate_group(n_control, "tdc", 0)
adhd_df = generate_group(n_adhd, "adhd", 1)

df = pd.concat([control_df, adhd_df], ignore_index=True)

df.to_csv("synthetic_stage1_stage2_dataset.csv", index=False)

print("Saved: synthetic_stage1_stage2_dataset.csv")
print(df.head())
print(df["group"].value_counts())
print(df.groupby("group").mean(numeric_only=True))