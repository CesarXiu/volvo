import os
import json
import threading
from fuzzywuzzy import fuzz, process
import unidecode
import queue
import vosk
import sounddevice as sd
from whisper import load_model
import numpy as np
import threading
import time
import torch

# Configuración de modelos
WAKE_WORD = "alfred"  # Define tu palabra clave
#whisper_model = load_model("tiny")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = load_model("tiny").to(device)
vosk_model_path = "vosk-model-small-es-0.42"#"vosk-model-es-0.42"
# Ruta de la carpeta donde están los archivos JSON
json_folder = "json/"
# Verificar modelos
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"No se encuentra el modelo en {vosk_model_path}")
vosk_model = vosk.Model(vosk_model_path)

# Cola de audio
audio_queue = queue.Queue()
#--------------------------------------------#
def execute_with_timeout(func, args=(), kwargs=None, timeout=5):
    """
    Ejecuta una función con un tiempo límite.
    
    :param func: Función a ejecutar.
    :param args: Argumentos posicionales para la función.
    :param kwargs: Argumentos de palabra clave para la función.
    :param timeout: Tiempo límite en segundos.
    :return: Resultado de la función o None si se excede el tiempo.
    """
    kwargs = kwargs or {}
    result = [None]
    exception = [None]
    
    def wrapper():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"Proceso interrumpido después de {timeout} segundos.")
        return None  # Tiempo excedido
    if exception[0]:
        raise exception[0]
    return result[0]
#--------------------------------------------#
def limpiar_texto(texto):
    """ Función para limpiar y normalizar texto (minúsculas y eliminar acentos) """
    texto = texto.lower()  # Convertir a minúsculas
    texto = unidecode.unidecode(texto)  # Eliminar acentos y caracteres especiales
    return texto

def buscar_coincidencias(input_text):
    resultados = []

    # Limpiar el texto de entrada
    input_text = limpiar_texto(input_text)

    # Recorremos todos los archivos JSON en la carpeta
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder, filename)

            # Abrimos y cargamos el archivo JSON
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Revisamos las claves del JSON para encontrar las listas
            for key, items in data.items():
                for item in items:
                    # Limpiar el texto de las descripciones
                    desc_limpia = limpiar_texto(item['desc'])

                    # Usamos fuzzywuzzy para encontrar coincidencias aproximadas
                    match = process.extractOne(input_text, [desc_limpia], scorer=fuzz.token_sort_ratio)

                    # Si la similitud es mayor que un umbral (por ejemplo, 90%)
                    if match and match[1] >= 90:
                        resultados.append({
                            'archivo': filename,
                            'objeto_principal': key,
                            'descripcion': item['desc'],
                            'estado': item['est'],
                            'similitud': match[1]
                        })
    return resultados

def transcribe_audio(audio_data):
    """ Transcribe el audio utilizando Whisper. """
    start_time = time.time()  # Marca inicial
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalizar a rango [-1.0, 1.0]
    transcription = whisper_model.transcribe(audio_data.flatten(), language="es")["text"]
    elapsed_time = time.time() - start_time  # Tiempo transcurrido
    print(f"Tiempo de transcripción: {elapsed_time:.2f} segundos")
    return transcription

def listen_for_commands():
    """ Graba audio después de detectar la palabra clave. """
    fs = 16000  # Frecuencia de muestreo
    duration = 3  # Duración de la grabación en segundos
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    return audio_data

def process_audio():
    """ Procesa los datos de audio y detecta la palabra clave. """
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=lambda indata, *_: audio_queue.put(bytes(indata))):
        rec = vosk.KaldiRecognizer(vosk_model, 16000)

        while True:
            data = audio_queue.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                detected_text = result.get("text", "").lower()
                print(f"Detectado: {detected_text}")

                if WAKE_WORD in detected_text:
                    start_time = time.time()
                    command = detected_text.split(WAKE_WORD, 1)[1].strip()
                    
                    if command:
                        print(f"Comando detectado en la misma frase: '{command}'")
                        try:
                            coincidencias = execute_with_timeout(buscar_coincidencias, args=(command,), timeout=5)
                            if coincidencias:
                                for resultado in coincidencias:
                                    print(f"Archivo: {resultado['archivo']}")
                                    print(f"Objeto principal: {resultado['objeto_principal']}")
                                    print(f"Descripción: {resultado['descripcion']}")
                                    print(f"Estado: {resultado['estado']}")
                                    print(f"Similitud: {resultado['similitud']}%")
                                    print()
                            else:
                                print("No se encontraron coincidencias o el tiempo se agotó.")
                        except Exception as e:
                            print(f"Error durante la búsqueda de coincidencias: {e}")
                    else:
                        print(f"Palabra clave '{WAKE_WORD}' detectada. Grabando comando...")
                        audio_data = listen_for_commands()
                        try:
                            transcription = execute_with_timeout(transcribe_audio, args=(audio_data,), timeout=5)
                            if transcription:
                                print(f"Comando detectado: '{transcription}'")
                                coincidencias = execute_with_timeout(buscar_coincidencias, args=(transcription,), timeout=5)
                                if coincidencias:
                                    for resultado in coincidencias:
                                        print(f"Archivo: {resultado['archivo']}")
                                        print(f"Objeto principal: {resultado['objeto_principal']}")
                                        print(f"Descripción: {resultado['descripcion']}")
                                        print(f"Estado: {resultado['estado']}")
                                        print(f"Similitud: {resultado['similitud']}%")
                                        print()
                                else:
                                    print("No se encontraron coincidencias o el tiempo se agotó.")
                        except Exception as e:
                            print(f"Error al procesar el comando: {e}")

def main():
    """ Función principal que ejecuta el procesamiento de audio en un hilo. """
    print(f"Esperando la palabra clave '{WAKE_WORD}'...")
    print(f"Usando: {device}")
    # Crear un hilo para la detección de comandos
    audio_thread = threading.Thread(target=process_audio)
    audio_thread.daemon = True  # Hacer que el hilo se cierre al finalizar el programa
    audio_thread.start()

    # Mantener el programa activo
    try:
        while True:
            time.sleep(1)  # Mantener el hilo activo
    except KeyboardInterrupt:
        print("Detenido por el usuario.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detenido por el usuario.")

