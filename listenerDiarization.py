import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import json

# Configuración
TRIGGER_WORD = "Voro"  # Palabra clave para activar el asistente
AUTHORIZED_VOICE_PATH = "VolvoMaster.m4a"  # Archivo de referencia de voz
SAMPLE_RATE = 16000  # Tasa de muestreo
LISTEN_DURATION = 5  # Segundos de escucha para validación

# Cargar el modelo de Vosk (asegúrate de descargar el modelo en español)
MODEL_PATH = "vosk-model-small-es-0.42"  # Cambia esto si usas otro directorio
model = Model(MODEL_PATH)
voice_encoder = VoiceEncoder()

# Cargar voz autorizada
authorized_voice_wav = preprocess_wav(Path(AUTHORIZED_VOICE_PATH))
authorized_voice_embedding = voice_encoder.embed_utterance(authorized_voice_wav)


def record_audio(duration, samplerate):
    """Graba audio por un tiempo determinado."""
    print("Escuchando...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()  # Esperar hasta que termine la grabación
    return audio.flatten()


def detect_trigger_word(audio, samplerate):
    """Detecta si la palabra clave está en el audio."""
    print("Procesando para detectar palabra clave...")
    recognizer = KaldiRecognizer(model, samplerate)
    audio_data = (audio * 32768).astype(np.int16).tobytes()  # Convertir audio al formato correcto
    if recognizer.AcceptWaveform(audio_data):
        result = json.loads(recognizer.Result())
        print(f"Texto detectado: {result.get('text', '')}")
        return TRIGGER_WORD in result.get("text", "").lower()
    return False


def validate_voice(audio, samplerate):
    """Valida si la voz grabada coincide con el usuario autorizado."""
    print("Validando voz...")
    wav = preprocess_wav(audio, source_sr=samplerate)
    embedded_voice = voice_encoder.embed_utterance(wav)
    similarity = np.dot(authorized_voice_embedding, embedded_voice)
    print(f"Similitud de voz: {similarity:.2f}")
    return similarity > 0.60  # Umbral ajustable


def main():
    """Ciclo principal de escucha."""
    while True:
        # Escucha continuo
        audio = record_audio(LISTEN_DURATION, SAMPLE_RATE)

        # Detectar palabra clave
        if detect_trigger_word(audio, SAMPLE_RATE):
            print("Palabra clave detectada: Volvot")
            print("Por favor, hable para la validación de voz...")

            # Validar voz
            validation_audio = record_audio(LISTEN_DURATION, SAMPLE_RATE)
            if validate_voice(validation_audio, SAMPLE_RATE):
                print("Acceso permitido. Usuario autenticado.")
            else:
                print("Acceso denegado. Usuario no autorizado.")


if __name__ == "__main__":
    main()
