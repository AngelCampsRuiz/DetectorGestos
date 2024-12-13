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
import numpy as np

# Inicializa el detector de manos y el dibujante
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para reconocer gestos
def recognize_gesture(landmarks):
    # Obtiene las coordenadas de los puntos clave de los dedos
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Define los criterios para cada gesto
    thumb_up = thumb_tip.y < thumb_mcp.y
    thumb_down = thumb_tip.y > thumb_mcp.y
    index_up = index_tip.y < index_mcp.y
    middle_up = middle_tip.y < middle_mcp.y
    ring_up = ring_tip.y < middle_mcp.y
    pinky_up = pinky_tip.y < middle_mcp.y

    # Lógica para determinar el gesto
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return 'Mano abierta'
    elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return 'Pulgar arriba'
    elif not thumb_up and index_up and middle_up and ring_up and pinky_up:
        return 'Cuatro dedos arriba'
    elif thumb_down and not index_up and not middle_up and not ring_up and not pinky_up:
        return 'Pulgar abajo'
    elif not thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
        return 'Pinky arriba'
    elif thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return 'Señal de victoria'
    else:
        return 'Gesto desconocido'

# Función para extraer un frame
import os

def clean_filename(filename):
    """
    Limpia el nombre del archivo/carpeta eliminando caracteres no válidos
    y reemplazando espacios con guiones bajos.
    """
    # Solo permitir letras, números, guiones y guiones bajos
    filename = re.sub(r'[^a-zA-Z0-9\-_]', '', filename)
    # Reemplazar múltiples guiones o guiones bajos con uno solo
    filename = re.sub(r'[-_]+', '_', filename)
    # Limitar la longitud del nombre
    filename = filename[:50]
    # Eliminar guiones bajos al inicio o final
    filename = filename.strip('_')
    # Si el nombre quedó vacío, usar un nombre por defecto
    if not filename:
        filename = "video"
    return filename

def extract_frame(video_url, extract_time_ms, output_filename, video_title):
    """
    Extrae un frame específico de un video en un tiempo dado y lo guarda en una carpeta específica.
    """
    try:
        # Limpiar el título del video para el nombre de la carpeta
        clean_title = clean_filename(video_title)
        
        # Crear nombre de carpeta con título y fecha/hora actual
        current_time = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{clean_title}_frame_{current_time}"
        
        # Crear la estructura de carpetas
        base_folder = "Fotogramas"
        output_folder = os.path.join(base_folder, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        print(f"Carpeta creada: {output_folder}")

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_url}")
        
        print(f"Video abierto correctamente: {video_url}")
        print(f"Intentando extraer frame en tiempo: {extract_time_ms}ms")
        
        # Posiciona el video en el tiempo especificado
        cap.set(cv2.CAP_PROP_POS_MSEC, extract_time_ms)
        ret, frame = cap.read()
        
        if ret:
            # Define la ruta completa para guardar el archivo
            full_output_path = os.path.join(output_folder, output_filename)
            print(f"Intentando guardar frame en: {full_output_path}")
            
            # Verifica que el frame no esté vacío
            if frame is None:
                raise ValueError("El frame es None")
            if frame.size == 0:
                raise ValueError("El frame tiene tamaño 0")
                
            print(f"Dimensiones del frame: {frame.shape}")
            print(f"Tipo de datos del frame: {frame.dtype}")
            
            # Asegurarse de que el frame está en BGR (el formato que espera cv2.imwrite)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                print("Frame en formato correcto (3 canales)")
                frame_bgr = frame  # Ya está en BGR cuando viene de cv2.VideoCapture
            else:
                print(f"Frame en formato inesperado: {frame.shape}")
                frame_bgr = frame
                
            # Intenta guardar la imagen
            try:
                print("Intentando guardar como JPG...")
                success = cv2.imwrite(full_output_path, frame_bgr)
                if not success:
                    print("Falló JPG, intentando PNG...")
                    full_output_path = full_output_path.replace('.jpg', '.png')
                    success = cv2.imwrite(full_output_path, frame_bgr)
                    if not success:
                        print("También falló PNG")
                        # Intenta guardar una imagen de prueba para verificar permisos
                        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                        test_path = os.path.join(output_folder, 'test.png')
                        test_success = cv2.imwrite(test_path, test_image)
                        if not test_success:
                            raise ValueError(f"No se pueden escribir imágenes en {output_folder}")
                        else:
                            print("La prueba de escritura funcionó, el problema está en el frame")
                            raise ValueError("No se pudo guardar la imagen en ningún formato")
            except Exception as write_error:
                print(f"Error al escribir la imagen: {write_error}")
                raise
                
            print(f"Frame extraído y guardado como {full_output_path}")
            cap.release()
            
            # Verifica que el archivo se haya creado
            if not os.path.exists(full_output_path):
                raise ValueError(f"El archivo no se creó en: {full_output_path}")
                
            return {
                'folder_name': folder_name,
                'file_name': os.path.basename(full_output_path),
                'full_path': full_output_path
            }
        else:
            cap.release()
            raise ValueError("No se pudo extraer el frame en el tiempo especificado.")
    except Exception as e:
        print(f"Error al extraer el frame: {e}")
        if 'cap' in locals():
            cap.release()
        raise e

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
    # Limpiar el título del video para el nombre de la carpeta
    base_title = clean_filename(video_title)
    
    # Encontrar el siguiente número disponible para el título
    base_folder = "Fotogramas"
    folder_num = 1
    
    while True:
        if folder_num == 1:
            current_title = base_title
        else:
            current_title = f"{base_title}{folder_num}"
            
        output_folder = os.path.join(base_folder, current_title)
        if not os.path.exists(output_folder):
            break
        folder_num += 1
    
    # Crear la estructura de carpetas
    os.makedirs(output_folder, exist_ok=True)
    csv_folder = os.path.join(output_folder, "CSV")
    os.makedirs(csv_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video en la ruta proporcionada: {video_url}")

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = []
    frame_count = 0
    last_gesture = None
    gesture_start_time = None
    min_gesture_duration = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Solo procesa los frames según el frame_step
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
                            results.append({
                                'time': formatted_time, 
                                'title': current_title, 
                                'description': 'Gestos', 
                                'tags': last_gesture
                            })
                            # Guardar frame en la carpeta específica del video
                            frame_filename = os.path.join(output_folder, f'gesture_{formatted_time.replace(":", "-")}.jpg')
                            cv2.imwrite(frame_filename, frame)
                    gesture_start_time = time.time()
                last_gesture = gesture
        if show_video and frame_count % frame_step == 0:
            cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    results_data = []
    csv_filename = None
    if results:
        csv_filename = os.path.join(csv_folder, f'{current_title}_gestures.csv')
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['time', 'title', 'description', 'tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            results_data = results

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    
    return {
        'csv_filename': csv_filename if results else None,
        'gestures_detected': len(results),
        'results': results_data,
        'folder_name': current_title
    }

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
    default_video = "https://www.youtube.com/watch?v=SRXWrMbE1jw&t=19s"
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
        success = extract_frame(video_url, extract_time_ms, output_filename, video_title)
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