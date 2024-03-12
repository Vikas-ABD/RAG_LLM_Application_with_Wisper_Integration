#resource 
##https://wandb.ai/wandb_fc/gentle-intros/reports/OpenAI-Whisper-How-to-Transcribe-Your-Audio-to-Text-for-Free-with-SRTs-VTTs---VmlldzozNDczNTI0
##before running we have to install the ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor,pipeline
import torch

# Import langdetect
import langdetect

model = AutoModelForSpeechSeq2Seq.from_pretrained("saved_model")
processor = AutoProcessor.from_pretrained("saved_model")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    device=device

)



# Transcription
result = pipe("video_audio.mp3")
print(result["text"])

# Language Detection
try:
  detected_language = langdetect.detect(result["text"])
  print(f"Detected Language: {detected_language}")
except langdetect.LangDetectException as e:
  print(f"Language Detection Error: {e}")


# Translation (corrected: set "language" in generate_kwargs)
result = pipe(
    "video_audio.mp3", generate_kwargs={"task": "translate"}  # Set target language default it translate into english
)
print(result["text"])



##store translated text into a text file translated_text.txt

with open("translated_text.txt", "w", encoding="utf-8") as txt:
    txt.write(result["text"]+ "\n")
    txt.write(detected_language)

