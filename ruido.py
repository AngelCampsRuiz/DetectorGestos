import soundfile as sf
import numpy as np
import scipy.signal as signal
import csv
import matplotlib.pyplot as plt

# Función para detectar disparos en el audio
def detectar_disparo(audio_path, umbral=0.1, min_duracion=0.02):
    # Cargar el archivo de audio con soundfile
    y, sr = sf.read(audio_path)
    print("Archivo cargado correctamente")

    # Verificar la duración del audio
    duracion_audio = len(y) / sr
    print(f"Duración del audio: {duracion_audio} segundos")
    
    hop_size = int(0.05 * sr)  # Tamaño del paso de 0.05 segundos (más pequeño)
    window_size = int(0.1 * sr)  # Tamaño de la ventana de 0.1 segundos (más pequeño)
    marcas = []

    # Recorrer el audio en segmentos
    for i in range(0, len(y) - window_size, hop_size):
        segment = y[i:i + window_size]
        
        # Ajuste dinámico de nperseg en función del tamaño del segmento
        nperseg = min(len(segment), 64)  # Asegurarse de que nperseg no sea mayor que el segmento
        f, t, Zxx = signal.stft(segment, fs=sr, nperseg=nperseg)
        
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

    return marcas_filtradas, y, sr

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
marcas, audio_data, sr = detectar_disparo(audio_path)

# Mostrar marcas para ver si fueron detectadas correctamente
print("Marcas detectadas:", marcas)

# Guardar las marcas en un archivo CSV
guardar_marcadores_csv(marcas, "marcas_disparo.csv")
print("Las marcas de disparos se han guardado en 'marcas_disparo.csv'.")

# Graficar el audio y las marcas
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, len(audio_data) / sr, len(audio_data)), audio_data)
plt.title('Señal de audio')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Amplitud')
for marca in marcas:
    plt.axvline(x=marca, color='r', linestyle='--')
plt.show()
