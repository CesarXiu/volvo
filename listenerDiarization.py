import os
import json
import queue
import vosk
import sounddevice as sd
from whisper import load_model
import numpy as np
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav

# Cargar el modelo de Whisper
encoder = VoiceEncoder()  # Modelo de Resemblyzer
whisper_model = load_model("tiny")  # Cambiar a "tiny", "small", etc., si necesitas optimización

# Perfil del conductor autorizado (embeddings pregrabados)
authorized_voice_paths = ["VolvoMaster.wav", "Volvo.wav", "Volvo2.wav", "Volvo3.wav", "Volvo4.wav", "Volvo5.wav", "Volvo6.wav", "Volvo7.wav", "Globo1.wav", "Globo2.wav", "Globo3.wav"]  # Archivos de voz del conductor
authorized_embeddings = []

for path in authorized_voice_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra el archivo de voz en {path}")
    wav = preprocess_wav(path)
    embedding = encoder.embed_utterance(wav)
    authorized_embeddings.append(embedding)

# Calcular el promedio de los embeddings
authorized_embedding = np.mean(authorized_embeddings, axis=0)

# Configuración de Vosk
vosk_model_path = "vosk-model-small-es-0.42"  # Ruta al modelo descargado
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"No se encuentra el modelo en {vosk_model_path}")

vosk_model = vosk.Model(vosk_model_path)

# Palabra clave
WAKE_WORD = "globo"  # Define tu palabra clave aquí

# Cola para manejar el audio grabado
audio_queue = queue.Queue()

def verify_audio(audio_data, threshold=0.60):
    """
    Verifica si el audio pertenece al conductor autorizado.
    
    Parámetros:
        audio_data (numpy array): Datos de audio para verificar.
        threshold (float): Umbral de similitud para considerar que es la misma persona.

    Retorna:
        (bool, str): Un booleano indicando si la voz es del conductor y la transcripción.
    """
    # Preprocesar el audio
    # Convertir los datos de audio a float32
    audio_data = audio_data.astype(np.float32) / 32768.0

    # Preprocesar el audio
    wav = preprocess_wav(audio_data.flatten())
    # Generar el embedding de la voz
    test_embedding = encoder.embed_utterance(wav)
    
    # Normalizar los embeddings
    authorized_embedding_norm = authorized_embedding / np.linalg.norm(authorized_embedding)
    test_embedding_norm = test_embedding / np.linalg.norm(test_embedding)
    
    # Calcular la similitud coseno entre las dos voces
    similarity = np.dot(authorized_embedding_norm, test_embedding_norm)
    print(f"Similitud calculada: {similarity:.2f}")

    # Verificar si la similitud supera el umbral
    if similarity >= threshold:
        print("Voz autorizada reconocida. Procesando transcripción...")
        return True, whisper_model.transcribe(audio_data, language="es")["text"]
    else:
        print("Voz no autorizada.")
        return False, None #whisper_model.transcribe(audio_data, language="es")["text"]

# Función para escuchar comandos
def listen_for_commands():
    print("¡Palabra clave detectada! Escuchando comando...")

    # Grabar audio durante un período definido
    duration = 5  # Duración de la grabación en segundos
    fs = 16000  # Frecuencia de muestreo
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    # Transcribir el audio con Whisper
    print("Procesando comando...")
    return audio_data

# Función principal
def main():
    print("Esperando la palabra clave 'Volvo'...")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=lambda indata, frames, time, status: audio_queue.put(bytes(indata))):
        rec = vosk.KaldiRecognizer(vosk_model, 16000)

        while True:
            data = audio_queue.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "text" in result:
                    detected_text = result["text"].lower()
                    print(f"Detectado: {detected_text}")

                    # Verificar si la palabra clave está presente
                    if WAKE_WORD in detected_text:
                        audio_data = listen_for_commands()

                        # Verificar si el audio corresponde a la voz autorizada
                        is_authorized, transcription = verify_audio(audio_data)
                        if is_authorized:
                            print(f"Comando reconocido: {transcription}")
                        else:
                            print("Voz no autorizada.")
                        print(f"Comando completo: {detected_text}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detenido por el usuario.")