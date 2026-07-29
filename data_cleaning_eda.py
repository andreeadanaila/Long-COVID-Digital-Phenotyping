import pandas as pd
import numpy as np

# CONFIG

INPUT_PATH = "user_health_data_per_minute.csv"
OUTPUT_CLEAN_PATH = "wearable_data_cleaned.csv"
OUTPUT_REPORT_PATH = "data_quality_report.csv"

SIGNAL_COLS = [
    "heart_rate",
    "heart_rate_variability",
    "spo2",
    "steps",
    "respirations_per_minute",
    "distance",
    "body_battery",
]

# Threshold for a "small" gap (interpolatable) vs a "large" one (device removed)
SMALL_GAP_MAX_MINUTES = 5     

KEEP_ONLY_USER_IDS = None

# LOAD DATA

print("Loading data...")
df = pd.read_csv(INPUT_PATH)
df["received_date"] = pd.to_datetime(df["received_date"])

print(f"Total rows: {len(df):,}")
print(f"Patients (user_id) found: {sorted(df['user_id'].unique())}")
print(f"Number of patients: {df['user_id'].nunique()}")
print(f"Date range: {df['received_date'].min()} -> {df['received_date'].max()}")
print()

if KEEP_ONLY_USER_IDS is not None:
    before = df["user_id"].nunique()
    df = df[df["user_id"].isin(KEEP_ONLY_USER_IDS)]
    print(f"Filtered patients: {before} -> {df['user_id'].nunique()} "
          f"(kept {KEEP_ONLY_USER_IDS})")
    print()

# MISSING VALUES 

print("=== Missing values (NaN) per column ===")
print(df.isna().sum())
print()

# TIME GAPS

def find_time_gaps(user_df, freq="min"):
    """Returns a DataFrame with (gap_start, gap_end, duration_minutes) for a
    single patient, assuming there should be one row per minute between the
    first and last recorded timestamp."""
    user_df = user_df.sort_values("received_date")
    full_range = pd.date_range(
        start=user_df["received_date"].min(),
        end=user_df["received_date"].max(),
        freq=freq,
    )
    existing = set(user_df["received_date"])
    missing = sorted(t for t in full_range if t not in existing)

    if not missing:
        return pd.DataFrame(columns=["start", "end", "duration_minutes"])

    # group consecutive missing timestamps into a single "gap"
    gaps = []
    gap_start = missing[0]
    prev = missing[0]
    for t in missing[1:]:
        if (t - prev) > pd.Timedelta(minutes=1):
            gaps.append((gap_start, prev))
            gap_start = t
        prev = t
    gaps.append((gap_start, prev))

    gaps_df = pd.DataFrame(gaps, columns=["start", "end"])
    gaps_df["duration_minutes"] = (
        (gaps_df["end"] - gaps_df["start"]).dt.total_seconds() / 60 + 1
    )
    return gaps_df


print("=== Checking time gaps per patient ===")
all_gaps = []
for uid, g in df.groupby("user_id"):
    gaps_df = find_time_gaps(g)
    if len(gaps_df) > 0:
        gaps_df["user_id"] = uid
        all_gaps.append(gaps_df)
    print(f"  patient {uid}: {len(gaps_df)} gaps found")

if all_gaps:
    all_gaps_df = pd.concat(all_gaps, ignore_index=True)
    all_gaps_df["is_large_gap"] = all_gaps_df["duration_minutes"] > SMALL_GAP_MAX_MINUTES
    print()
    print(f"Total gaps: {len(all_gaps_df)}")
    print(f"  small gaps (<= {SMALL_GAP_MAX_MINUTES} min, interpolatable): "
          f"{(~all_gaps_df['is_large_gap']).sum()}")
    print(f"  large gaps (> {SMALL_GAP_MAX_MINUTES} min, device not worn): "
          f"{all_gaps_df['is_large_gap'].sum()}")
else:
    all_gaps_df = pd.DataFrame(
        columns=["user_id", "start", "end", "duration_minutes", "is_large_gap"]
    )
    print("No time gaps found -> complete per-minute series for all patients.")
print()

# REBUILD FULL SERIES + INTERPOLATE + FLAG DEVICE-NOT-WORN

print("Rebuilding full time series per patient and applying interpolation...")

cleaned_parts = []
for uid, g in df.groupby("user_id"):
    g = g.sort_values("received_date").set_index("received_date")

    full_index = pd.date_range(g.index.min(), g.index.max(), freq="min")
    g_full = g.reindex(full_index)
    g_full["user_id"] = uid

    # flag: this row was entirely missing from the original data
    g_full["was_missing"] = g_full[SIGNAL_COLS[0]].isna()

    # linearly interpolate all signals, but capped at SMALL_GAP_MAX_MINUTES
    for col in SIGNAL_COLS:
        g_full[col] = g_full[col].interpolate(
            method="linear", limit=SMALL_GAP_MAX_MINUTES, limit_area="inside"
        )

    # whatever is still NaN after interpolation = large gap = device not worn
    g_full["device_not_worn"] = g_full[SIGNAL_COLS[0]].isna()

    g_full = g_full.reset_index().rename(columns={"index": "received_date"})
    cleaned_parts.append(g_full)

df_clean = pd.concat(cleaned_parts, ignore_index=True)
df_clean = df_clean[["user_id", "received_date"] + SIGNAL_COLS +
                     ["was_missing", "device_not_worn"]]

print(f"Rows after rebuilding: {len(df_clean):,}")
print(f"Interpolated rows (small gaps): "
      f"{(df_clean['was_missing'] & ~df_clean['device_not_worn']).sum():,}")
print(f"Rows flagged 'device not worn' (large gaps): "
      f"{df_clean['device_not_worn'].sum():,}")
print()

# CHECK INTER-PATIENT BIAS (descriptive statistics)

print("=== Per-patient statistics (to check bias between patients) ===")
per_patient_stats = df_clean.groupby("user_id")[SIGNAL_COLS].agg(["mean", "std"])
print(per_patient_stats)
print()
print("Interpretation: if you see large differences between patients in")
print("mean/std for the same signal (e.g. average HR of 60 vs 100), that's")
print("a sign the model needs a PERSONALIZED baseline per patient, not one")
print("shared global threshold.")
print()

# SAVE

df_clean.to_csv(OUTPUT_CLEAN_PATH, index=False)
per_patient_stats.to_csv(OUTPUT_REPORT_PATH)

print(f"Saved: {OUTPUT_CLEAN_PATH}")
print(f"Saved: {OUTPUT_REPORT_PATH}")
print("Done. Next step: the 5 baseline models (simple threshold, logistic")
print("regression, Random Forest, Isolation Forest, one more advanced model).")