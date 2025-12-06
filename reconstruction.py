import torch
import os
import torchaudio
from audiocraft.models.loaders import load_compression_model

# 模型路径
name = "./musicgen-small"
# 音乐采样率，与模型有关
sample_rate = 32000
# # 设备选择，没有GPU会用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模型
model = load_compression_model(name, device=device)

# 原音乐目录
music_dir = "./example/background"
# 输出目录
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
