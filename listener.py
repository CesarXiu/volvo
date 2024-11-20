import os
import wave
import json
import queue
import vosk
import sounddevice as sd
from whisper import load_model

# Cargar el modelo de Whisper
whisper_model = load_model("base")  # Cambiar a "tiny", "small", etc., si necesitas optimización

# Configuración de Vosk
vosk_model_path = "vosk-model-small-es-0.42"  # Ruta al modelo descargado
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"No se encuentra el modelo en {vosk_model_path}")

vosk_model = vosk.Model(vosk_model_path)

# Palabra clave
WAKE_WORD = "volvo"  # Define tu palabra clave aquí

# Cola para manejar el audio grabado
audio_queue = queue.Queue()

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
    os.remove(temp_audio_path)

# Función principal
def main():
    print("Esperando la palabra clave 'Volvo'...")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=lambda indata, _, __, ___: audio_queue.put(bytes(indata))):
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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detenido por el usuario.")
