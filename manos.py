import cv2
import mediapipe as mp
import csv
import time
import argparse
import sys
import tkinter as tk
from tkinter import filedialog
from pytube import YouTube
import youtube_dl
import string
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

def recognize_gesture(landmarks):

    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]


    thumb_up = thumb_tip.y < thumb_mcp.y
    thumb_down = thumb_tip.y > thumb_mcp.y


    index_up = index_tip.y < thumb_mcp.y
    middle_up = middle_tip.y < thumb_mcp.y


    if thumb_up and index_up and not middle_up:
        return 'Victoria'
    elif thumb_up and not index_up and not middle_up:
        return 'Pulgar arriba'
    elif thumb_down and not index_up and not middle_up:
        return 'Pulgar Abajo'
    elif thumb_up and index_up and middle_up:
        return 'Palma abierta'
    else:
        return 'Gesto desconocido'


parser = argparse.ArgumentParser(description='Detectar gestos de manos en un video y guardarlos en un archivo CSV.')
parser.add_argument('video_path', type=str, nargs='?', default=None, help='Ruta del video o URL del video')

args = parser.parse_args()

if args.video_path is None:
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename()
else:
    video_path = args.video_path

if 'youtube.com' in video_path or 'youtu.be' in video_path:
    try:
        yt = YouTube(video_path)
        stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
        video_url = stream.url
        # Configuración para youtube_dl
        ydl_opts = {
            'quiet': True,  # No imprimir salida
        }
        # Usar youtube_dl para extraer información del video
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get('title', None)
    except Exception as e:
        print(f"Error: No se pudo obtener el video de YouTube. Detalles: {e}")
        sys.exit(1)
else:
    video_url = video_path
    # Obtener el nombre del archivo de video
    video_title = os.path.splitext(os.path.basename(video_path))[0]

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en la ruta proporcionada: {video_url}")
    sys.exit(1)

last_gesture = None
gesture_start_time = None
min_gesture_duration = 1  

results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(image)

    if hand_results.multi_hand_landmarks is not None:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = recognize_gesture(hand_landmarks)

            if gesture != last_gesture:
                if last_gesture is not None:

                    gesture_duration = time.time() - gesture_start_time
                    if gesture_duration >= min_gesture_duration:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  
                        formatted_time = time.strftime('%H:%M:%S', time.gmtime(timestamp))  
                        result = {'time': formatted_time, 'gesture': last_gesture}
                        results.append(result)  

                gesture_start_time = time.time()

            last_gesture = gesture

        cv2.imshow('MediaPipe Hands', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

with open('results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time', 'gesture'])
    writer.writeheader()
    writer.writerows(results)

# Reemplazar caracteres no válidos en nombres de archivos
valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
video_title = ''.join(c for c in video_title if c in valid_chars)

# Cambiar el nombre del archivo
os.rename('results.csv', f"{video_title}.csv")
