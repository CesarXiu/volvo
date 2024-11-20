import whisper

model = whisper.load_model("base")

result = model.transcribe("Volvo.m4a", language="es")
print(result['text'])