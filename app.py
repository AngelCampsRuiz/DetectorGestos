from flask import Flask, render_template, request, jsonify
import os
import time
from analizador import extract_frame, process_video, get_video_info, detect_objects
import cv2
import numpy as np
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configurar carpeta temporal para subidas
UPLOAD_FOLDER = 'temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('analizador.html')

@app.route('/process', methods=['POST'])
def process_request():
    try:
        video_file = request.files.get('videoInput')
        video_url = request.form.get('videoUrl')
        frame_time = request.form.get('frameTime')
        action = request.form.get('gestureAction')
        frame_step = int(request.form.get('frameStep', 1))
        show_video = request.form.get('showVideo') == 'true'

        if not (video_file or video_url):
            return jsonify({'status': 'error', 'message': 'No se proporcionó un video ni una URL.'})

        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)
            video_title = os.path.splitext(video_file.filename)[0]
        else:
            video_path, video_title = get_video_info(video_url)

        # Crear directorio Fotogramas si no existe
        os.makedirs("Fotogramas", exist_ok=True)

        if action == 'extract':
            minutes, seconds, milliseconds = map(int, frame_time.split(':'))
            extract_time_ms = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
            
            output_filename = f"frame_{minutes}-{seconds}-{milliseconds}.jpg"
            result = extract_frame(video_path, extract_time_ms, output_filename, video_title)
            return jsonify({
                'status': 'success',
                'message': 'Frame extraído correctamente',
                'data': {
                    'folder_name': result['folder_name'],
                    'file_name': result['file_name'],
                    'full_path': result['full_path']
                }
            })

        elif action == 'process':
            result = process_video(video_path, video_title, show_video, frame_step)
            return jsonify({
                'status': 'success', 
                'message': f'Gestos procesados correctamente. Guardados en la carpeta {result["folder_name"]}',
                'data': result
            })

        elif action == 'detect_objects':
            result = detect_objects(video_path, video_title, show_video, frame_step)
            return jsonify({
                'status': 'success', 
                'message': f'Objetos detectados correctamente. Guardados en la carpeta {result["folder_name"]}',
                'data': result
            })

        return jsonify({'status': 'error', 'message': 'Acción no válida.'})

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Esto imprimirá el error completo en la consola
        return jsonify({
            'status': 'error',
            'message': f'Error al procesar el video: {str(e)}'
        })

def clean_filename(filename):
    # Reemplazar caracteres no válidos con guiones
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '-')
    return filename.strip()

if __name__ == '__main__':
    app.run(debug=True)