import vosk
import sounddevice as sd
import queue
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import whisper
import json

# Configuración
KEYWORD = "volvo"
AUTHORIZED_EMBEDDING = np.load("authorized_user.npy")  # Carga un embedding autorizado
SAMPLE_RATE = 16000
MODEL_PATH = "vosk-model-small-es-0.42"  # Modelo de Vosk
vosk_model = vosk.Model(MODEL_PATH)

# Inicializar modelos
encoder = VoiceEncoder()
whisper_model = whisper.load_model("base")

# Cola de audio
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))
    print("Audio data added to queue")

def recognize_keyword():
    rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
    while True:
        data = audio_queue.get()
        if rec.AcceptWaveform(data):
            result = rec.Result()
            print(f"Resultado de Vosk: {result}")  # Imprimir resultado de Vosk
            result_dict = json.loads(result)
            if 'text' in result_dict and KEYWORD in result_dict['text']:
                print("Keyword detected!")
                return

def capture_command():
    print("Listening for command...")
    rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
    command_audio = []
    while True:
        data = audio_queue.get()
        command_audio.append(data)
        if rec.AcceptWaveform(data):
            result = rec.Result()
            return b''.join(command_audio), result

def process_command(command_audio):
    wav = np.frombuffer(command_audio, dtype=np.int16)
    wav = preprocess_wav(wav, SAMPLE_RATE)
    embedding = encoder.embed_utterance(wav)
    similarity = np.dot(embedding, AUTHORIZED_EMBEDDING) / (np.linalg.norm(embedding) * np.linalg.norm(AUTHORIZED_EMBEDDING))
    return similarity

def main():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
        print("Listening for keyword...")
        recognize_keyword()
        command_audio, result = capture_command()
        similarity = process_command(command_audio)
        if similarity > 0.6:
            print(f"Command executed with similarity: {similarity:.2f}")
        else:
            print(f"Command failed with similarity: {similarity:.2f}")
            print(f"Attempted command: {result}")

if __name__ == "__main__":
    main()