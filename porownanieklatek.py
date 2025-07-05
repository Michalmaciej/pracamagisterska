import pandas as pd
import matplotlib.pyplot as plt

#Ścieżka do pliku CSV
csv_path = 'features_output/train_per_frame.csv'
video_index = 5
frame_shift = 20

#Wczytanie dane
df = pd.read_csv(csv_path)
video_data = df[df['video_index'] == video_index]

#Wybranie środkowej klatki i klatki +20
middle_idx = video_data['frame_index'].max() // 2
frame_a = video_data[video_data['frame_index'] == middle_idx]
frame_b = video_data[video_data['frame_index'] == middle_idx + frame_shift]

if frame_a.empty or frame_b.empty:
    print(f"Brakuje klatki: {middle_idx} lub {middle_idx + frame_shift}")
    exit()

#Liczba punktów (x, y)
num_features = (len(df.columns) - 3) // 2

#Punkty dla obu klatek
xa = [frame_a.iloc[0][f'f{i*2+1}'] for i in range(num_features)]
ya = [frame_a.iloc[0][f'f{i*2+2}'] for i in range(num_features)]
xb = [frame_b.iloc[0][f'f{i*2+1}'] for i in range(num_features)]
yb = [frame_b.iloc[0][f'f{i*2+2}'] for i in range(num_features)]

#Rysowanie
plt.figure(figsize=(6, 6))
plt.scatter(xa, ya, c='blue', s=10, label=f'frame {middle_idx}')
plt.scatter(xb, yb, c='orange', s=10, label=f'frame {middle_idx + frame_shift}')
plt.gca().invert_yaxis()
plt.title(f'Porównanie ruchu: klatka {middle_idx} vs {middle_idx + frame_shift} (video {video_index})')
plt.legend()
plt.tight_layout()
plt.show()
