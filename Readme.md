这个仓库是[MusicGen](https://github.com/facebookresearch/audiocraft)代码的简单实现，目前只提供推理生成代码。

## 🛠️ 环境安装

```bash
conda create --name musicgen python=3.10.19
conda activate musicgen
pip install -r requirements.txt
```

## 🎯 代码运行
- 文生音乐
    ```bash
    # 可以在文件里面调整采用的音乐生成模型（4个版本），生成时长，多个输入提示词
    python t2m.py
    ```
- 音乐编解码
    ```bash
    python reconstruction.py
    ```