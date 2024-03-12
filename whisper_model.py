##pip install --upgrade pip
##pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]
#source link https://huggingface.co/openai/whisper-large-v3
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)


model.save_pretrained("saved_model")  # Save model to the "saved_model" directory
processor.save_pretrained("saved_model")  # Save processor to the same directory

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )


# result = pipe("video_audio.mp3")
# print(result["text"])


# ##translate to english default

# result = pipe("video_audio.mp3", generate_kwargs={"task": "translate"})
# print(result['test'])
