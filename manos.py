import cv2
import mediapipe as mp
import csv
import time
import argparse
import sys
from pytube import YouTube

# Inicializa el detector de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializa el dibujante
mp_drawing = mp.solutions.drawing_utils

# Define la función de reconocimiento de gestos
def recognize_gesture(landmarks):
    # Obtiene las coordenadas del pulgar
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    # Obtiene las coordenadas del índice y del dedo medio
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Comprueba si el pulgar está levantado o bajado
    thumb_up = thumb_tip.y < thumb_mcp.y
    thumb_down = thumb_tip.y > thumb_mcp.y

    # Comprueba si el índice y el dedo medio están levantados
    index_up = index_tip.y < thumb_mcp.y
    middle_up = middle_tip.y < thumb_mcp.y

    # Reconoce el gesto
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

# Define el analizador de argumentos
parser = argparse.ArgumentParser(description='Detectar gestos de manos en un video y guardarlos en un archivo CSV.')
parser.add_argument('video_path', type=str, help='Ruta del video o URL del video')

# Parse los argumentos
args = parser.parse_args()

# Verifica si la ruta del video es una URL de YouTube
if 'youtube.com' in args.video_path or 'youtu.be' in args.video_path:
    try:
        yt = YouTube(args.video_path)
        stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
        video_url = stream.url
    except Exception as e:
        print(f"Error: No se pudo obtener el video de YouTube. Detalles: {e}")
        sys.exit(1)
else:
    video_url = args.video_path

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
                        result = {'time': formatted_time, 'gesture': last_gesture}
                        results.append(result)  # Añade el resultado a la lista

                # Reinicia el contador
                gesture_start_time = time.time()

            # Actualiza el último gesto reconocido
            last_gesture = gesture

        # Muestra la imagen
        cv2.imshow('MediaPipe Hands', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Cierra las ventanas y libera los recursos aaa
cap.release()
cv2.destroyAllWindows()

# Guarda los resultados en un archivo CSV
with open('results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time', 'gesture'])
    writer.writeheader()
    writer.writerows(results)