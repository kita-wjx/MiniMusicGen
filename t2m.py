from audiocraft.models import musicgen
# from audiocraft.data.audio import audio_write
import torchaudio
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = musicgen.MusicGen.get_pretrained('./musicgen-small', device=device)

# 这里调整生成时长
model.set_generation_params(duration=12)

# 这里调整描述词
discriptions = [
    'Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach',
    'A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.'
]
output_dir = "./example/t2m"

# 测试用 只用最后一个提示词
res = model.generate(discriptions, progress=True)

for idx, one_wav in enumerate(res):
    torchaudio.save(f'{output_dir}/{idx}.wav', one_wav.cpu(), model.sample_rate)
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    # audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
