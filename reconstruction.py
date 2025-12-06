import torch
import os
import torchaudio
from audiocraft.models.loaders import load_compression_model

name = "./musicgen-small"
sample_rate = 32000
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_compression_model(name, device=device)

music_dir = "./example/background"
output_dir = "./example/reconstruction"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(music_dir):
    if file.endswith(".mp3") or file.endswith(".wav"):
        music_file = os.path.join(music_dir, file)
        music, sr = torchaudio.load(music_file)
        if sr != sample_rate:
            music = torchaudio.functional.resample(music, orig_freq=sr, new_freq=sample_rate)
        music = music.reshape(1, music.shape[0], -1)
        music = torch.mean(music, 1, keepdim=True).to(device)

        with torch.no_grad():
            codes, _ = model.encode(music)
            gen_audio = model.decode(codes, None)
            gen_audio = gen_audio.cpu()
            torchaudio.save(os.path.join(output_dir, file), gen_audio[0], model.sample_rate)
