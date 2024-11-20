import os
import wave
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

# Perfil del conductor autorizado (embedding pregrabado)
# En la vida real, este embedding debería ser almacenado en una base de datos o archivo.
authorized_voice_path = "Volvo.m4a"  # Archivo de voz del conductor
authorized_embedding = encoder.embed_utterance(preprocess_wav(authorized_voice_path))

# Configuración de Vosk
vosk_model_path = "vosk-model-small-es-0.42"  # Ruta al modelo descargado
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"No se encuentra el modelo en {vosk_model_path}")

vosk_model = vosk.Model(vosk_model_path)

# Palabra clave
WAKE_WORD = "volvo"  # Define tu palabra clave aquí

# Cola para manejar el audio grabado
audio_queue = queue.Queue()

def verify_audio(audio_path, threshold=0.75):
    """
    Verifica si el audio pertenece al conductor autorizado.
    
    Parámetros:
        audio_path (str): Ruta del archivo de audio a verificar.
        threshold (float): Umbral de similitud para considerar que es la misma persona.

    Retorna:
        (bool, str): Un booleano indicando si la voz es del conductor y la transcripción.
    """
    # Preprocesar el audio
    wav = preprocess_wav(audio_path)
    
    # Generar el embedding de la voz
    test_embedding = encoder.embed_utterance(wav)
    
    # Calcular la similitud coseno entre las dos voces
    similarity = np.dot(authorized_embedding, test_embedding)
    print(f"Similitud calculada: {similarity:.2f}")

    # Verificar si la similitud supera el umbral
    if similarity >= threshold:
        print("Voz autorizada reconocida. Procesando transcripción...")
        # Transcribir el audio con Whisper
        result = whisper_model.transcribe(audio_path, language="es")
        return True, result["text"]
    else:
        
        result = whisper_model.transcribe(audio_path, language="es")
        print("Voz no autorizada.", result["text"])
        return False, None


# Función para escuchar comandos
def listen_for_commands():
    print("¡Palabra clave detectada! Escuchando comando...")
    
    # Grabar audio durante un período definido
    duration = 5  # Duración de la grabación en segundos
    fs = 16000  # Frecuencia de muestreo
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    # Guardar el audio en un archivo temporal
    temp_audio_path = "command.wav"
    with wave.open(temp_audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

    # Transcribir con Whisper
    result = whisper_model.transcribe(temp_audio_path, language="es")
    print(f"Transcripción: {result['text']}")

    # Opcional: eliminar el archivo temporal
    #os.remove(temp_audio_path)

# Función principal
def main():
    print("Esperando la palabra clave 'Volvo'...")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=lambda indata, *args: audio_queue.put(bytes(indata))):
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
                        listen_for_commands()
                        if os.path.exists("command.wav"):
                            is_authorized, transcription = verify_audio("command.wav")
                            if is_authorized:
                                print(f"Transcripción del audio: {transcription}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detenido por el usuario.")
