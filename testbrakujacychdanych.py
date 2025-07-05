import pandas as pd

df = pd.read_csv("features_output/train_per_frame.csv", sep=",")

# Sprawdzenie 1: brakujące klatki
missing_frames = []
for vid in df['video_index'].unique():
    frames = df[df['video_index'] == vid]['frame_index'].sort_values()
    expected = list(range(frames.min(), frames.max() + 1))
    actual = frames.tolist()
    if expected != actual:
        missing_frames.append(vid)

# Sprawdzenie 2: niepełne długości wierszy
expected_cols = df.shape[1]
bad_length_rows = df[df.apply(lambda row: row.count() != expected_cols, axis=1)]

# Sprawdzenie 3: NaN
rows_with_nans = df[df.isnull().any(axis=1)]

print(f"🔍 Brakujące klatki w: {missing_frames}")
print(f"⚠️  Wiersze z niepełną długością: {len(bad_length_rows)}")
print(f"⚠️  Wiersze z brakami (NaN): {len(rows_with_nans)}")
