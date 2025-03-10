import soundfile as sf
import numpy as np
import scipy.signal as signal
import csv

# Función para detectar disparos en el audio
def detectar_disparo(audio_path, umbral=0.1, min_duracion=0.02):
    # Cargar el archivo de audio con soundfile
    y, sr = sf.read(audio_path)
    print("Archivo cargado correctamente")

    # Verificar la duración del audio
    duracion_audio = len(y) / sr
    print(f"Duración del audio: {duracion_audio} segundos")
    
    # Ajustar el tamaño de la ventana y la cantidad de muestras de salto en función de la duración del audio
    nperseg = min(128, len(y) // 10)  # Ajustamos nperseg para que no sea mayor que la longitud de los datos
    hop_size = int(0.1 * sr)  # Tamaño del paso de 0.1 segundos
    window_size = int(0.4 * sr)  # Tamaño de la ventana de 0.4 segundos

    marcas = []

    # Recorrer el audio en segmentos
    for i in range(0, len(y) - window_size, hop_size):
        segment = y[i:i + window_size]
        
        # Calcular el espectro de la señal usando la FFT
        f, t, Zxx = signal.stft(segment, fs=sr, nperseg=nperseg)  # Ajustar nperseg
        # Calcular la energía en el dominio de frecuencia
        energia = np.abs(Zxx) ** 2
        energia_promedio = np.mean(energia)

        # Detectar picos de energía que superen el umbral
        if energia_promedio > umbral:
            marcas.append(i / sr)

    # Filtrar marcas cercanas entre sí (min_duracion) para evitar duplicados
    marcas_filtradas = []
    for i, marca in enumerate(marcas):
        if i == 0 or marca - marcas_filtradas[-1] > min_duracion:
            marcas_filtradas.append(marca)

    return marcas_filtradas

# Guardar las marcas en un archivo CSV
def guardar_marcadores_csv(marcadores, archivo_csv):
    # Comprobar si hay marcas para guardar
    if not marcadores:
        print("No se detectaron marcas de disparos.")
        return
    
    with open(archivo_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Marca de Disparo (segundos)"])  # Cabecera
        
        # Escribir las marcas detectadas
        for marca in marcadores:
            writer.writerow([marca])  # Escribir cada marca en una fila nueva

# Ruta del archivo de audio
audio_path = "arma.wav"

# Detectar los disparos
marcas = detectar_disparo(audio_path)

# Mostrar marcas para ver si fueron detectadas correctamente
print("Marcas detectadas:", marcas)

# Guardar las marcas en un archivo CSV
guardar_marcadores_csv(marcas, "marcas_disparo.csv")
print("Las marcas de disparos se han guardado en 'marcas_disparo.csv'.")
