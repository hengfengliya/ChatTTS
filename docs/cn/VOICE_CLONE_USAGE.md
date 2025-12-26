# 声音克隆使用说明（零样本音色模拟）

> 说明：ChatTTS 的“声音克隆”属于**零样本音色模拟**，更像是“用一段音频提取说话人特征，再合成相似音色”，不是 1:1 完全复刻。

## 1. 准备一段清晰的参考音频

- 建议 **5~15 秒**，单人讲话，背景噪声越少越好  
- 支持常见格式（如 `mp3` / `wav`）  
- 如果采样率不是 24000，代码会自动重采样  

## 2. 使用 Python 代码生成“音色签名”

```python
import ChatTTS  # 作用：导入 ChatTTS 主入口（概念：API=对外调用接口）
from tools.audio import load_audio  # 作用：导入音频加载工具（概念：重采样=统一采样率）
import torch  # 作用：导入 torch（概念：张量=深度学习数据结构）
import torchaudio  # 作用：导入 torchaudio（概念：音频读写库）
chat = ChatTTS.Chat()  # 作用：创建模型实例（原理：初始化推理入口）
chat.load(compile=False)  # 作用：加载模型权重（原理：读取预训练参数到内存）
ref_wav = load_audio("sample.mp3", 24000)  # 作用：读取参考音频并重采样（概念：采样率=每秒采样次数）
spk_smp = chat.sample_audio_speaker(ref_wav)  # 作用：提取音色签名（概念：嵌入向量=说话人特征）
texts = ["你好，这是我的声音克隆测试。"]  # 作用：准备要合成的文本（概念：输入文本=模型驱动内容）
params_infer_code = ChatTTS.Chat.InferCodeParams(  # 作用：创建推理配置对象（概念：参数对象=配置集合）
    spk_smp=spk_smp,  # 作用：指定音色签名（原理：用说话人特征控制音色）
)  # 作用：结束参数定义（概念：对象构造=生成配置实例）
wavs = chat.infer(texts, params_infer_code=params_infer_code)  # 作用：执行推理生成音频（原理：文本转语音）
torchaudio.save("voice_clone_output.wav", torch.from_numpy(wavs[0]), 24000)  # 作用：保存音频文件（概念：采样率=保存时的时间分辨率）
```

## 3. 如果要多次复用音色

> 你可以把 `spk_smp` 保存下来，后续直接复用，避免每次都加载参考音频。

```python
with open("spk_smp.txt", "w", encoding="utf-8") as f:  # 作用：打开文件用于写入（概念：编码=utf-8 中文兼容）
    f.write(spk_smp)  # 作用：保存音色签名（原理：音色签名是可编码的字符串）
with open("spk_smp.txt", "r", encoding="utf-8") as f:  # 作用：打开文件用于读取（概念：持久化=长期保存）
    spk_smp = f.read()  # 作用：读取音色签名（原理：读取文本恢复音色信息）
```

## 4. 通过 WebUI 使用参考音频（可选）

```bash
python examples\web\webui.py # 作用：启动 WebUI（原理：启动本地网页服务）
```

在界面中上传 **Sample Audio**，系统会自动生成对应的音色签名并用于合成。
