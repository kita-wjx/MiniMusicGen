from audiocraft.models import musicgen
# from audiocraft.data.audio import audio_write
import torchaudio
import torch
import os

# cuda节省显存配置，可以不管
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 设备选择，没有GPU会用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 导入模型，这里“./musicgen-small”可替换为其他预训练模型路径，比如“./musicgen-medium”或“./musicgen-melody”或“./musicgen-large”
model = musicgen.MusicGen.get_pretrained('./musicgen-small', device=device)

# 这里调整生成时长
model.set_generation_params(duration=12)

# 这里调整提示词，列表每个元素会生成一段音乐
discriptions = [
    'Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach',
    'A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.'
]

# 输出路径，默认是./example/t2m，可以修改
output_dir = "./example/t2m"

res = model.generate(discriptions, progress=True)

for idx, one_wav in enumerate(res):
    torchaudio.save(f'{output_dir}/{idx}.wav', one_wav.cpu(), model.sample_rate)
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    # audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
