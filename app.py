from flask import Flask, render_template, request, jsonify
import os
import time
from analizador import extract_frame, process_video, get_video_info

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('analizador.html')

@app.route('/process', methods=['POST'])
def process_request():
    video_file = request.files.get('videoInput')
    video_url = request.form.get('videoUrl')
    frame_time = request.form.get('frameTime')
    action = request.form.get('gestureAction')
    frame_step = int(request.form.get('frameStep', 1))
    show_video = request.form.get('showVideo') == 'true'

    if not (video_file or video_url):
        return jsonify({'status': 'error', 'message': 'No se proporcionó un video ni una URL.'})

    try:
        if video_file:
            video_path = os.path.join("temp", video_file.filename)
            os.makedirs("temp", exist_ok=True)
            video_file.save(video_path)
            video_title = os.path.splitext(video_file.filename)[0]
        else:
            video_path, video_title = get_video_info(video_url)

        # Crear directorio Fotogramas si no existe
        os.makedirs("Fotogramas", exist_ok=True)

        if action == 'extract':
            # Convertir tiempo de mm:ss:fff a milisegundos
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

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

    return jsonify({'status': 'error', 'message': 'Acción no válida.'})

if __name__ == '__main__':
    app.run(debug=True)