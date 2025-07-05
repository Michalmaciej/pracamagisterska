import cv2
import mediapipe as mp
import pandas as pd
import os

#MediaPipe setup
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468
NUM_POSE_LANDMARKS = 33

video_folder = r"C:\Users\michk\Desktop\WLASL2000\\"

def get_video_ids_and_gloss(txt_file_path):
    video_ids = []
    video_id_to_gloss = {}
    with open(txt_file_path, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                gloss = parts[0]
                video_id = parts[2]
                video_ids.append(video_id)
                video_id_to_gloss[video_id] = gloss
    return video_ids, video_id_to_gloss

def get_max_frames(video_ids):
    max_f = 0
    for video_id in video_ids:
        path = os.path.join(video_folder, f"{video_id}.mp4")
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        max_f = max(max_f, frames)
    return max_f

def process_dataset(txt_file_path, output_csv_path, max_frames):
    video_ids, video_id_to_gloss = get_video_ids_and_gloss(txt_file_path)
    all_videos_data = []

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        for video_idx, video_id in enumerate(video_ids):
            path = os.path.join(video_folder, f"{video_id}.mp4")
            cap = cv2.VideoCapture(path)
            frame_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_hands = hands.process(image_rgb)
                result_face_mesh = face_mesh.process(image_rgb)
                result_pose = pose.process(image_rgb)

                landmarks_frame = []

               #Hands
                hand_points = [[{'x': 0.0, 'y': 0.0}] * NUM_HAND_LANDMARKS for _ in range(2)]
                if result_hands.multi_hand_landmarks:
                    for i_h, hand_landmarks in enumerate(result_hands.multi_hand_landmarks[:2]):
                        for idx, lm in enumerate(hand_landmarks.landmark):
                            hand_points[i_h][idx] = {'x': lm.x, 'y': lm.y}
                for i_h in range(2):
                    for idx in range(NUM_HAND_LANDMARKS):
                        landmarks_frame.append({
                            'type': 'hand',
                            'point_index': i_h * NUM_HAND_LANDMARKS + idx,
                            'x': hand_points[i_h][idx]['x'],
                            'y': hand_points[i_h][idx]['y']
                        })

                #Face
                face_points = [{'x': 0.0, 'y': 0.0} for _ in range(NUM_FACE_LANDMARKS)]
                if result_face_mesh.multi_face_landmarks:
                    for idx, lm in enumerate(result_face_mesh.multi_face_landmarks[0].landmark[:NUM_FACE_LANDMARKS]):
                        face_points[idx] = {'x': lm.x, 'y': lm.y}
                for idx in range(NUM_FACE_LANDMARKS):
                    landmarks_frame.append({
                        'type': 'face',
                        'point_index': idx,
                        'x': face_points[idx]['x'],
                        'y': face_points[idx]['y']
                    })

                #Pose
                pose_points = [{'x': 0.0, 'y': 0.0} for _ in range(NUM_POSE_LANDMARKS)]
                if result_pose.pose_landmarks:
                    for idx, lm in enumerate(result_pose.pose_landmarks.landmark[:NUM_POSE_LANDMARKS]):
                        pose_points[idx] = {'x': lm.x, 'y': lm.y}
                for idx in range(NUM_POSE_LANDMARKS):
                    landmarks_frame.append({
                        'type': 'pose',
                        'point_index': idx,
                        'x': pose_points[idx]['x'],
                        'y': pose_points[idx]['y']
                    })


                frame_data.append(landmarks_frame)
            cap.release()

            #Padding do max_frames
            while len(frame_data) < max_frames:
                zero_frame = []
                for i in range(2 * NUM_HAND_LANDMARKS):
                    zero_frame.append({'type': 'hand', 'point_index': i, 'x': 0.0, 'y': 0.0})
                for i in range(NUM_FACE_LANDMARKS):
                    zero_frame.append({'type': 'face', 'point_index': i, 'x': 0.0, 'y': 0.0})
                for i in range(NUM_POSE_LANDMARKS):
                    zero_frame.append({'type': 'pose', 'point_index': i, 'x': 0.0, 'y': 0.0})
                frame_data.append(zero_frame)
            print(len(zero_frame))
            print(len(frame_data))

            all_videos_data.append((video_id, frame_data))
            print(len(all_videos_data))
            print(f"Przetworzono video_id={video_id} ({len(frame_data)} klatek)")

    #Zapis do CSV
    csv_rows = []
    for video_index, (video_id, video) in enumerate(all_videos_data):
        gloss = video_id_to_gloss.get(video_id, "UNKNOWN")
        for frame in video:
            row = {'video_index': video_index, 'gloss': gloss}
            #Zmiana indexu
            type_offset = {
                'hand': 0,
                'face': 1000,
                'pose': 2000
            }

            for point in frame:
                global_index = type_offset[point['type']] + point['point_index']
                row[f"x{global_index}"] = point['x']
                row[f"y{global_index}"] = point['y']
            csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(output_csv_path, index=False)
    print(f"\nDane zapisane do {output_csv_path}")

train_ids, _ = get_video_ids_and_gloss("train.txt")
test_ids, _ = get_video_ids_and_gloss("test.txt")
max_train = get_max_frames(train_ids)
max_test = get_max_frames(test_ids)
max_frames = max(max_train, max_test)

print(f"\nNajdłuższy film w obu zbiorach ma {max_frames} klatek (padding do tej długości)")

process_dataset("train.txt", "landmarks_flattened_train1.csv", max_frames)
process_dataset("test.txt", "landmarks_flattened_test1.csv", max_frames)
