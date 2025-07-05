import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

VIDEO_ID = '17722'
VIDEO_PATH = rf'C:\Users\michk\Desktop\WLASL2000\{VIDEO_ID}.mp4'
CSV_OUTPUT = f"landmarks_debug_{VIDEO_ID}.csv"
VIDEO_OUTPUT = f"landmarks_overlay_{VIDEO_ID}.mp4"
IMPORTANT_FACE_LANDMARKS = [33, 133, 263, 362, 1, 13, 14, 78, 308, 61, 291, 0]

NUM_POSE = 33
NUM_HAND = 21
NUM_FACE = len(IMPORTANT_FACE_LANDMARKS)

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
rows = []

with mp_pose.Pose() as pose, mp_hands.Hands(max_num_hands=2) as hands, mp_face.FaceMesh() as face_mesh:
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_pose = pose.process(rgb)
        res_hands = hands.process(rgb)
        res_face = face_mesh.process(rgb)

        frame_features = []

        #Pose
        if res_pose.pose_landmarks:
            for lm in res_pose.pose_landmarks.landmark:
                x, y = lm.x, lm.y
                frame_features += [x, y]
                cx, cy = int(x * width), int(y * height)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
        else:
            frame_features += [0.0, 0.0] * NUM_POSE

        #Hands
        hands_list = res_hands.multi_hand_landmarks if res_hands.multi_hand_landmarks else []
        for i in range(2):
            if i < len(hands_list):
                for lm in hands_list[i].landmark:
                    x, y = lm.x, lm.y
                    frame_features += [x, y]
                    cx, cy = int(x * width), int(y * height)
                    cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
            else:
                frame_features += [0.0, 0.0] * NUM_HAND

        #Face
        if res_face.multi_face_landmarks:
            for i in IMPORTANT_FACE_LANDMARKS:
                lm = res_face.multi_face_landmarks[0].landmark[i]
                x, y = lm.x, lm.y
                frame_features += [x, y]
                cx, cy = int(x * width), int(y * height)
                cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        else:
            frame_features += [0.0, 0.0] * NUM_FACE

        rows.append([frame_idx] + frame_features)
        out.write(frame)
        frame_idx += 1

cap.release()
out.release()

num_features = (NUM_POSE + 2 * NUM_HAND + NUM_FACE) * 2
header = ['frame_index'] + [f'f{i+1}' for i in range(num_features)]
df = pd.DataFrame(rows, columns=header)
df.to_csv(CSV_OUTPUT, index=False)
print(f"Zapisano: {CSV_OUTPUT}, {VIDEO_OUTPUT}")

data = pd.read_csv(CSV_OUTPUT)
frames = data.shape[0]
coords = data.iloc[:, 1:].values.reshape(frames, -1, 2)

fig, ax = plt.subplots()
sc = ax.scatter([], [], s=10)
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)
ax.set_title(f"Animacja punktów dla video_id: {VIDEO_ID}")

def init():
    sc.set_offsets(np.empty((1, 2)))
    return sc,

def update(frame):
    sc.set_offsets(coords[frame])
    return sc,

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=100)
ani.save(f"landmarks_debug_{VIDEO_ID}.gif", writer='pillow', fps=10)
plt.show()
