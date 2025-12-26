# 本地部署指南（Windows 友好）

## 1. 环境准备

- **Python 3.11**（推荐版本，与项目依赖更兼容）
- **Git**（用于拉取代码）
- **足够的磁盘空间**（首次运行会下载模型文件）

## 2. 创建并激活虚拟环境

```bash
cd C:\Users\18805\Desktop\word\Vibe Coding\voice\ChatTTS # 作用：进入项目根目录（原理：定位工作路径）
python -m venv .venv # 作用：创建虚拟环境（概念：虚拟环境=隔离依赖的独立空间）
.\.venv\Scripts\Activate.ps1 # 作用：激活虚拟环境（原理：修改 PATH 指向虚拟环境）
```

## 3. 安装依赖

```bash
python -m pip install --upgrade pip # 作用：升级 pip（概念：pip=Python 包管理器）
pip install -r requirements.txt # 作用：安装依赖（原理：读取 requirements.txt 列表）
```

## 4. 启动并验证

```bash
python examples\cmd\run.py "你好，我是 ChatTTS。" # 作用：生成示例语音（原理：文本转语音推理）
```

成功后会在项目根目录生成 `output_audio_0.mp3`。

## 5. 常见问题

- **首次运行时间较长**：需要下载模型文件（概念：预训练模型=训练好的权重参数）。
- **显存不足**：可先使用 CPU 运行（速度较慢，但可验证流程）。
