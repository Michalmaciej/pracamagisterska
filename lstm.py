from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Multiply, Permute, Activation, Lambda
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU OK")
else:
    print("CPU only")

tf.keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

INPUT_CSV_TRAIN = '/content/drive/MyDrive/Colab Notebooks/landmarks_flattened_train.csv'
INPUT_CSV_TEST = '/content/drive/MyDrive/Colab Notebooks/landmarks_flattened_test.csv'

df_train = pd.read_csv(INPUT_CSV_TRAIN)
df_test = pd.read_csv(INPUT_CSV_TEST)
df_train.fillna(0.0, inplace=True)
df_test.fillna(0.0, inplace=True)

label_encoder = LabelEncoder()
df_train['label'] = label_encoder.fit_transform(df_train['gloss'])
df_test['label'] = label_encoder.transform(df_test['gloss'])
y_train_cat = to_categorical(df_train['label'])
y_test_cat = to_categorical(df_test['label'])
#y_train_cat = df_train['label']
#y_test_cat = df_test['label']
print(y_train_cat)

#df - DataFrame, każda klatka stanowi wiersz, kolumna vido_index identyfikuje nagranie
#kolumny zaczynające się od x/y zawierają współrzędne punktów charakterystycznych
#y_cat - macierz etykiet zakodowana one-hot tej samej długości co df
def prepare(df, y_cat):
    #inicjalizacja pustych list videos, labels
    videos, labels = [], []
    #iterowanie po wszystkich unikalnych wartościach video_index
    for video_idx in df['video_index'].unique():
        #video_data - podzbiór wierszy należący do tego nagrania
        video_data = df[df['video_index'] == video_idx].copy()
        #feat_cols - lista wszystkich kolumn zaczynających się od x lub y
        feat_cols = [c for c in video_data.columns if c.startswith(('x','y'))]
        #features - tablica NumPy o kształcie (liczba klatek, liczba cech)
        features = video_data[feat_cols].values
        #label - wektor one-hot pobrany z y_cat dla pierwszej klatki danego wideo
        label = y_cat[video_data.index[0]]
        #dodanie features do listy videos
        videos.append(features)
        #dodanie label do listy labels
        labels.append(label)
    return np.array(videos), np.array(labels)

X_train, y_train = prepare(df_train, y_train_cat)
X_test, y_test = prepare(df_test, y_test_cat)

""" 
#df - DataFrame, każda klatka stanowi wiersz, kolumna vido_index identyfikuje nagranie
#kolumny zaczynające się od x/y zawierają współrzędne punktów charakterystycznych
#y_cat - macierz etykiet zakodowana one-hot tej samej długości co df
def prepare(df, y_cat):
    #inicjalizacja pustych list videos, labels
    videos, labels = [], []
    #iterowanie po wszystkich unikalnych wartościach video_index
    for video_idx in df['video_index'].unique():
        #video_data - podzbiór wierszy należący do tego nagrania
        video_data = df[df['video_index'] == video_idx].copy()
        #feat_cols - lista wszystkich kolumn zaczynających się od x lub y
        feat_cols = [c for c in video_data.columns if c.startswith(('x','y','z'))]
        #features - tablica NumPy o kształcie (liczba klatek, liczba cech)
        features = video_data[feat_cols].values
        #label - wektor one-hot pobrany z y_cat dla pierwszej klatki danego wideo
        label = y_cat[video_data.index[0]]
        #dodanie features do listy videos
        videos.append(features)
        #dodanie label do listy labels
        labels.append(label)
    return np.array(videos), np.array(labels)

X_train, y_train = prepare(df_train, y_train_cat)
X_test, y_test = prepare(df_test, y_test_cat)
"""

"""
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
"""
scaler = MinMaxScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[2])
X_test_flat = X_test.reshape(-1, X_test.shape[2])
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

X_train_face_scaled = X_train_scaled[:,84:1020]
X_test_face_scaled = X_test_scaled[:,84:1020]

""" 
X_train_face_scaled = X_train_scaled[:,126:1530]
X_test_face_scaled = X_test_scaled[:,126:1530]
"""

encoding_dim = 16
input_dim = X_train_face_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(256, activation='relu')(encoded)
encoded_output = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(encoded_output)
decoded = Dense(512, activation='relu')(decoded)
decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder1 = Model(inputs=input_layer, outputs=decoded_output)
encoder1 = Model(inputs=input_layer, outputs=encoded_output)
autoencoder1.compile(optimizer='adam', loss='mse')

#RYSOWANIE STRUKTURY
#from tensorflow.keras.utils import model_to_dot
#import os

#dot = model_to_dot(autoencoder1, show_shapes=True, show_layer_activations=True, dpi=200)

#png_bytes = dot.create(prog="dot", format="png")

#out_path = "autoencoder_architecture.png"
#with open(out_path, "wb") as f:
#    f.write(png_bytes)

#print(f"Zapisano diagram do: {os.path.abspath(out_path)}")

print("\nTraining autoencoder...")
autoencoder1.fit(
    X_train_face_scaled, X_train_face_scaled,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

X_train_face_encoded = encoder1.predict(X_train_face_scaled)
X_test_face_encoded = encoder1.predict(X_test_face_scaled)

"""
#wybrane punkty
face_ids = [1, 33, 133, 362, 263, 61, 291, 13, 14, 168, 94, 289, 234, 46, 454, 278, 10, 152, 17, 356]

face_cols = []
for pid in face_ids:
    base = 84 + 2*pid  
    face_cols.extend([base, base+1]) 

X_train_face_sel = X_train_scaled[:, face_cols]
X_test_face_sel  = X_test_scaled[:, face_cols]

X_train_all_scaled = np.concatenate(
    [
        X_train_scaled[:, 0:84],      
        X_train_face_sel,        
        X_train_scaled[:, 1020:1087] 
    ],
    axis=1
)

X_test_all_scaled = np.concatenate(
    [
        X_test_scaled[:, 0:84],
        X_test_face_sel,
        X_test_scaled[:, 1020:1087]
    ],
    axis=1
)
"""
X_train_all_scaled = np.concatenate((
    #punkty rąk
    X_train_scaled[:, 0:84],       
    #zakodowane punkty twarzy
    X_train_face_encoded,    
    #szkielet
    X_train_scaled[:, 1020:1087])  
, axis=1)
X_test_all_scaled = np.concatenate((
    X_test_scaled[:, 0:84],
    X_test_face_encoded,
    X_test_scaled[:, 1020:1087]),
    axis=1)

"""
X_train_all_scaled = np.concatenate((
    #punkty rąk
    X_train_scaled[:, 0:126],       
    #zakodowane punkty twarzy
    X_train_face_encoded,    
    #szkielet
    X_train_scaled[:, 1530:1630])  
, axis=1)
X_test_all_scaled = np.concatenate((
    X_test_scaled[:, 0:126],
    X_test_face_encoded,
    X_test_scaled[:, 1530:1630]),
    axis=1)
"""

"""
X_train_all_scaled = np.concatenate((
    #punkty rąk
    X_train_scaled[:, 0:84],          
    #szkielet
    X_train_scaled[:, 1020:1087])  
, axis=1)
X_test_all_scaled = np.concatenate((
    X_test_scaled[:, 0:84],
    X_test_scaled[:, 1020:1087]),
    axis=1)
"""
encoding_dim = 32
input_dim = X_train_all_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded_output = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded_output)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder2 = Model(inputs=input_layer, outputs=decoded_output)
encoder2 = Model(inputs=input_layer, outputs=encoded_output)
autoencoder2.compile(optimizer='adam', loss='mse')

#RYSOWANIE STRUKTURY
#from tensorflow.keras.utils import model_to_dot
#import os

#dot = model_to_dot(autoencoder2, show_shapes=True, show_layer_activations=True, dpi=200)

#png_bytes = dot.create(prog="dot", format="png")

#out_path = "autoencoder2_architecture.png"
#with open(out_path, "wb") as f:
#    f.write(png_bytes)

#print(f"Zapisano diagram do: {os.path.abspath(out_path)}")

print("\nTraining autoencoder...")
autoencoder2.fit(
    X_train_all_scaled, X_train_all_scaled,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

X_train_encoded = encoder2.predict(X_train_all_scaled)
X_test_encoded = encoder2.predict(X_test_all_scaled)

X_train = X_train_encoded.reshape(X_train.shape[0], X_train.shape[1], encoding_dim)
X_test = X_test_encoded.reshape(X_test.shape[0], X_test.shape[1], encoding_dim)

"""
X_train = X_train_all_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train_all_scaled.shape[1])
X_test = X_test_all_scaled.reshape(X_test.shape[0], X_test.shape[1], X_train_all_scaled.shape[1])
"""

model = Sequential()
seq_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
model.add(seq_input)
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=100000,
    decay_rate=0.03)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
#model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')])
#early_stop = EarlyStopping(monitor='val_loss', patience=10)
#early_stop = EarlyStopping(monitor='loss', patience=10)

model.fit(X_train, y_train, epochs=50, validation_split=0.3, verbose=1)
#model.fit(X_train, y_train, epochs=200, validation_split=0.25, verbose=1)
#model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[early_stop], verbose=1)
#model.fit(X_train, y_train, epochs=200, callbacks=[early_stop], verbose=1)

#predykcje
#prawdopodobieństwa
y_prob = model.predict(X_test)
#klasa top-1     
y_pred = np.argmax(y_prob, axis=1)   
#klasa prawdziwa
y_true = np.argmax(y_test, axis=1)   

#metryki
acc  = accuracy_score(y_true, y_pred)

prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

top2_acc = np.mean([
    int(t in np.argsort(p)[-2:])          #czy klasa prawdziwa jest w 2 najlepszych
    for t, p in zip(y_true, y_prob)
])

cm = confusion_matrix(y_true, y_pred)

print(f"\nAccuracy      : {acc:.3f}")
print(f"Precision     : {prec:.3f}")
print(f"Recall        : {rec:.3f}")
print(f"F1-score      : {f1:.3f}")
print(f"Top-2 accuracy: {top2_acc:.3f}")

#confusion matrix i wizualizacja
plt.figure(figsize=(4.5,4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.colorbar(); plt.tight_layout(); plt.show()