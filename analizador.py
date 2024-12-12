import cv2
import mediapipe as mp
import csv
import time
import argparse
import sys
import tkinter as tk
from tkinter import filedialog
from pytube import YouTube
import os
from yt_dlp import YoutubeDL
import re

# Inicializa el detector de manos y el dibujante
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para reconocer gestos
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

# Función para extraer un frame
import os

def extract_frame(video_url, extract_time_ms, output_filename):
    """
    Extrae un frame específico de un video en un tiempo dado y lo guarda en la subcarpeta "Fotogramas".

    Parámetros:
    - video_url: Ruta o URL del video.
    - extract_time_ms: Tiempo en milisegundos donde se extraerá el frame.
    - output_filename: Nombre del archivo donde se guardará el frame.

    Retorna:
    - bool: True si se extrajo correctamente, False si hubo un error.
    """
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_url}")
        
        # Posiciona el video en el tiempo especificado
        cap.set(cv2.CAP_PROP_POS_MSEC, extract_time_ms)
        ret, frame = cap.read()
        if ret:
            # Crea la subcarpeta "Fotogramas" si no existe
            output_folder = "Fotogramas"
            os.makedirs(output_folder, exist_ok=True)

            # Define la ruta completa para guardar el archivo
            full_output_path = os.path.join(output_folder, output_filename)
            
            # Guarda el frame como imagen
            cv2.imwrite(full_output_path, frame)
            print(f"Frame extraído y guardado como {full_output_path}")
        else:
            print(f"No se pudo extraer el frame en el tiempo especificado.")
        cap.release()
    except Exception as e:
        print(f"Error al extraer el frame: {e}")
        return False
    return True

# Función para obtener información del video
def get_video_info(video_path):
    if 'youtube.com' in video_path or 'youtu.be' in video_path:
        try:
            ydl_opts = {'quiet': True, 'format': 'best'}
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_path, download=False)
                return info_dict['url'], info_dict.get('title', 'video')
        except Exception as e:
            print(f"Error al obtener el video de YouTube: {e}")
            sys.exit(1)
    else:
        return video_path, os.path.splitext(os.path.basename(video_path))[0]

# Procesa el video
def process_video(video_url, video_title, show_video, frame_step):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en la ruta proporcionada: {video_url}")
        sys.exit(1)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = []
    frame_count = 0
    last_gesture = None
    gesture_start_time = None
    min_gesture_duration = 1  # Duración mínima del gesto en segundos

    output_folder = "Fotogramas"
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_step != 0:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks)
                if gesture != last_gesture:
                    if last_gesture:
                        gesture_duration = time.time() - gesture_start_time
                        if gesture_duration >= min_gesture_duration:
                            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                            formatted_time = time.strftime('%H:%M:%S', time.gmtime(timestamp))
                            results.append({'time': formatted_time, 'title': video_title, 'description': 'Gestos', 'tags': last_gesture})
                            frame_filename = os.path.join(output_folder, f'gesture_{formatted_time.replace(":", "-")}.jpg')
                            cv2.imwrite(frame_filename, frame)
                    gesture_start_time = time.time()
                last_gesture = gesture
        if show_video and frame_count % frame_step == 0:
            cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if results:
        csv_filename = f'{video_title}_gestures.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['time', 'title', 'description', 'tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Resultados guardados en {csv_filename}")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

# Función principal
def main():
    """
    Punto de entrada principal del programa.
    Procesa argumentos de línea de comandos y ejecuta la funcionalidad seleccionada.
    """
    parser = argparse.ArgumentParser(description='Detectar gestos de manos en un video.')
    parser.add_argument('video_path', type=str, nargs='?', help='Ruta del video o URL del video.')
    parser.add_argument('--show', action='store_true', help='Muestra el video en tiempo real mientras se procesa.')
    parser.add_argument('--frames', type=int, default=1, help='Procesa cada "x" frames (por defecto, cada frame).')
    parser.add_argument('--extract', type=str, help='Extrae un frame específico (formato: mm:ss:fff).')
    args = parser.parse_args()

    # Establece el video por defecto solo si no se proporciona ninguno
    default_video = "https://youtu.be/PeeGp1S04Ys?si=Mo5gCLw8rpBSWbmd"
    video_path = args.video_path if args.video_path is not None else default_video

    # Obtiene la URL y título del video
    video_url, video_title = get_video_info(video_path)

    start_time = time.time()  # Marca el inicio del tiempo

    # Si se especifica --extract, procesa solo ese frame
    if args.extract:
        if not re.match(r'^\d{2}:\d{2}:\d{3}$', args.extract):
            print("Error: Formato de tiempo inválido. Usa el formato mm:ss:fff")
            sys.exit(1)

        # Convierte el tiempo a milisegundos
        try:
            minutes, seconds, milliseconds = map(int, args.extract.split(':'))
            extract_time_ms = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
        except ValueError:
            print("Error: No se pudo interpretar el tiempo proporcionado. Verifica el formato.")
            sys.exit(1)

        output_filename = f'extracted_frame_{args.extract.replace(":", "-")}.png'
        success = extract_frame(video_url, extract_time_ms, output_filename)
        if success:
            print(f"Frame extraído correctamente: {output_filename}")
        else:
            print("Hubo un problema al extraer el frame.")
    else:
        # Procesa el video completo con los argumentos proporcionados
        process_video(video_url, video_title, args.show, args.frames)

    end_time = time.time()  # Marca el fin del tiempo
    elapsed_time = end_time - start_time  # Calcula el tiempo total
    print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    main()