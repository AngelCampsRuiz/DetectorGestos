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
import os

# Inicializa el detector de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializa el dibujante
mp_drawing = mp.solutions.drawing_utils

# Define la función de reconocimiento de gestos
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
# Reemplaza los argumentos con '/' por '--' para que argparse los pueda procesar
sys.argv = [arg if not arg.startswith('/') else '--' + arg[1:] for arg in sys.argv]
# Define el analizador de argumentos
parser = argparse.ArgumentParser(description='Detectar gestos de manos en un video y guardarlos en un archivo CSV.')
parser.add_argument('video_path', type=str, nargs='?', default=None, help='Ruta del video o URL del video')
parser.add_argument('--show', action='store_true', help='Si se incluye, muestra el video.')
parser.add_argument('--frames', type=int, default=1, help='Procesa cada "x" frames.')

# Parse los argumentos
args = parser.parse_args()

# Si no se proporciona una ruta de video, abre un cuadro de diálogo para seleccionar un archivo
if args.video_path is None:
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename()
else:
    video_path = args.video_path

# Verifica si la ruta del video es una URL de YouTube
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

# Abre el video directamente desde la URL
cap = cv2.VideoCapture(video_url)

# Verifica si el video se pudo abrir
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en la ruta proporcionada: {video_url}")
    sys.exit(1)

# Inicializa el último gesto reconocido y su contador
last_gesture = None
gesture_start_time = None
min_gesture_duration = 1  # Duración mínima del gesto en segundos

# Inicializa la lista de resultados
results = []

# Modifica el bucle principal para procesar cada "x" frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Incrementa el contador de frames
    frame_count += 1

    # Procesa solo cada "x" frames
    if frame_count % args.frames != 0:
        continue

    # Convierte el color BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa la imagen y detecta las manos
    hand_results = hands.process(image)

    # Dibuja los resultados
    if hand_results.multi_hand_landmarks is not None:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Reconoce el gesto
            gesture = recognize_gesture(hand_landmarks)

            # Si el gesto ha cambiado
            if gesture != last_gesture:
                if last_gesture is not None:
                    # Registra el gesto anterior si duró lo suficiente
                    gesture_duration = time.time() - gesture_start_time
                    if gesture_duration >= min_gesture_duration:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Obtiene el tiempo en segundos
                        formatted_time = time.strftime('%H:%M:%S', time.gmtime(timestamp))  # Formatea el tiempo en h:m:s
                        result = {
                            'time': formatted_time,
                            'title': video_title,  # Título del video
                            'description': 'Gestos',  # Descripción del gesto
                            'tags': last_gesture  # Etiqueta del gesto
                        }
                        results.append(result)  # Añade el resultado a la lista

                        # Guarda el fotograma en un archivo de imagen
                        frame_filename = f'gesture_{formatted_time.replace(":", "-")}.jpg'
                        cv2.imwrite(frame_filename, frame)

                # Reinicia el contador
                gesture_start_time = time.time()

            # Actualiza el último gesto reconocido
            last_gesture = gesture

    # Muestra la imagen solo si se proporciona el argumento /show y es el frame a procesar
    if args.show and frame_count % args.frames == 0:
        cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Cierra las ventanas y libera los recursos
cap.release()
cv2.destroyAllWindows()

# Guarda los resultados en un archivo CSV
with open('results.csv', 'w', newline='') as f:
    fieldnames = ['time', 'title', 'description', 'tags']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
