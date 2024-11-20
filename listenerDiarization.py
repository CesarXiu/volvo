import os
import json
import queue
import vosk
import sounddevice as sd
from whisper import load_model, log_mel_spectrogram
import numpy as np
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav

# Configuración de modelos
WAKE_WORD = "globo"  # Define tu palabra clave
encoder = VoiceEncoder()
whisper_model = load_model("tiny")
vosk_model_path = "vosk-model-small-es-0.42"

# Verificar modelos
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"No se encuentra el modelo en {vosk_model_path}")
vosk_model = vosk.Model(vosk_model_path)

# Cargar voces autorizadas
authorized_voice_paths = ["VolvoMaster.wav", "Volvo.wav", "Volvo2.wav", "Volvo3.wav", "Volvo4.wav", "Volvo5.wav", "Volvo6.wav", "Volvo7.wav", "Globo1.wav", "Globo2.wav", "Globo3.wav", "Globo4.wav"] 
authorized_embeddings = [
    encoder.embed_utterance(preprocess_wav(path)) for path in authorized_voice_paths
]
authorized_embedding = np.mean(authorized_embeddings, axis=0)

# Cola de audio
audio_queue = queue.Queue()

def verify_audio(audio_data, threshold=0.60):
    # Normaliza el audio a float32
    audio_data = audio_data.astype(np.float32) / 32768.0  
    wav = preprocess_wav(audio_data.flatten())
    test_embedding = encoder.embed_utterance(wav)
    
    # Normalizar los embeddings
    authorized_norm = authorized_embedding / np.linalg.norm(authorized_embedding)
    test_norm = test_embedding / np.linalg.norm(test_embedding)
    similarity = np.dot(authorized_norm, test_norm)
    print(f"Similitud calculada: {similarity:.2f}")

    if similarity >= threshold:
        # Transcribe directamente el audio normalizado
        transcription = whisper_model.transcribe(audio_data.flatten(), language="es")["text"]
        return True, transcription

    return False, None

def listen_for_commands():
    print("Escuchando comando...")
    fs = 16000  # Frecuencia de muestreo
    duration = 5  # Duración de la grabación en segundos
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    return audio_data

def main():
    print(f"Esperando la palabra clave '{WAKE_WORD}'...")
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
                    audio_data = listen_for_commands()
                    is_authorized, transcription = verify_audio(audio_data)
                    if is_authorized:
                        print(f"Comando autorizado: {transcription}")
                    else:
                        print("Voz no autorizada.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detenido por el usuario.")
