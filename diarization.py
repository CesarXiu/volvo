import numpy as np
from resemblyzer import VoiceEncoder
from whisper import load_model
from resemblyzer.audio import preprocess_wav
import os

# Configurar modelos
encoder = VoiceEncoder()  # Modelo de Resemblyzer
model = load_model("base")  # Modelo de Whisper

# Perfil del conductor autorizado (embedding pregrabado)
# En la vida real, este embedding debería ser almacenado en una base de datos o archivo.
authorized_voice_path = "Volvo.m4a"  # Archivo de voz del conductor
authorized_embedding = None

# Generar el embedding del conductor autorizado
if os.path.exists(authorized_voice_path):
    wav = preprocess_wav(authorized_voice_path)
    authorized_embedding = encoder.embed_utterance(wav)
else:
    raise FileNotFoundError(f"El archivo de voz del conductor {authorized_voice_path} no existe.")

# Función para verificar si un audio pertenece al conductor autorizado
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
        result = model.transcribe(audio_path, language="es")
        return True, result["text"]
    else:
        print("Voz no autorizada.")
        return False, None

# Prueba con un archivo de audio
audio_to_test = "Volvo.m4a"  # Cambia a la ruta del archivo a probar

if __name__ == "__main__":
    if os.path.exists(audio_to_test):
        is_authorized, transcription = verify_audio(audio_to_test)
        if is_authorized:
            print(f"Transcripción del audio: {transcription}")
        else:
            print("Acceso denegado. Voz no coincide con el conductor autorizado.")
    else:
        print(f"El archivo de prueba {audio_to_test} no existe.")
