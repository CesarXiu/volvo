from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path

# Inicializa el encoder de Resemblyzer
encoder = VoiceEncoder()

def register_user_profile(audio_path):
    # Cargar y preprocesar el archivo de audio del usuario
    wav = preprocess_wav(Path(audio_path))
    
    # Generar el embedding de la voz
    embedding = encoder.embed_utterance(wav)
    
    # Guardar el embedding en un archivo .npy
    np.save("authorized_user.npy", embedding)
    print("Perfil de usuario autorizado registrado con éxito.")

# Ruta al archivo de audio del usuario autorizado
audio_path = "VolvoMaster.m4a"  # Reemplaza con tu archivo .wav
register_user_profile(audio_path)
