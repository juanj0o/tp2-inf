import numpy as np

from data import *

import soundfile as sf


from scipy.io import wavfile

import matplotlib.pyplot as plt


vocales = ["a.wav", "e.wav", "i.wav", "o.wav", "u.wav"]
consonantes = ["f.wav", "j.wav", "s.wav", "sh.wav"]
audios = []
recortados = []

frec = 14700
N = int(0.2 * frec)  # 200 ms

def graficar_señal(señal):
    audio, _  = sf.read(señal)
    audios.append(audio)
    audio_recortado = audio[:N]
    recortados.append(audio_recortado)


    tiempo = np.arange(0, N) / frec

    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, datos_recortados)
    plt.title(f"señal de {señal}")
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()





# def graficar_señal_robusta(señal):
#     try:
#         # Cargar con librosa (más tolerante)
#         datos, frecuencia_muestreo = librosa.load(señal, sr=None)
#         duracion = 200  # milisegundos
#         num_muestras = int(frecuencia_muestreo * duracion / 1000)
#         datos_recortados = datos[:num_muestras]
        
#         # Guardar el archivo recortado
#         sf.write("recortado.wav", datos_recortados, frecuencia_muestreo)
        
#         # Crear vector de tiempo
#         tiempo = np.arange(0, num_muestras) / frecuencia_muestreo
        
#         # Graficar
#         plt.figure(figsize=(10, 4))
#         plt.plot(tiempo, datos_recortados)
#         plt.title(f"Señal de {os.path.basename(señal)}")
#         plt.xlabel('Tiempo (segundos)')
#         plt.ylabel('Amplitud')
#         plt.grid(True)
#         plt.show()
        
#         print(f"Procesado exitoso: {señal}")
#         return True
#     except Exception as e:
#         print(f"Error al procesar {señal}: {str(e)}")
#         return False
    
for vocal in vocales:
    graficar_señal(vocal)
for consonante in consonantes:
    graficar_señal(consonante)

