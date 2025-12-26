import os  # 作用：操作系统接口（概念：环境变量与路径管理）
import sys  # 作用：系统参数与输入输出（概念：标准输出编码控制）
import argparse  # 作用：命令行参数解析（概念：参数配置入口）
from typing import Optional, List  # 作用：类型注解（概念：提高可读性与静态检查）
import numpy as np  # 作用：数值计算库（概念：多维数组=矩阵数据结构）
import torch  # 作用：深度学习计算库（概念：张量=深度学习数据结构）
import torchaudio  # 作用：音频读写库（概念：保存音频文件）

now_dir = os.getcwd()  # 作用：获取当前工作目录（概念：运行路径）
sys.path.append(now_dir)  # 作用：加入模块搜索路径（概念：保证能导入本地包）

import ChatTTS  # 作用：导入 ChatTTS 主模块（概念：模型推理入口）
from tools.logger import get_logger  # 作用：获取日志器（概念：日志=运行过程记录）
from tools.audio import load_audio, pcm_arr_to_mp3_view  # 作用：音频加载与转码（概念：重采样/编码）
from tools.normalizer.en import normalizer_en_nemo_text  # 作用：英文文本规范化（概念：文本归一化）
from tools.normalizer.zh import normalizer_zh_tn  # 作用：中文文本规范化（概念：文本清洗）

def load_normalizer(chat: ChatTTS.Chat) -> None:  # 作用：加载文本规范化器（概念：中英文文本归一化）
    try:  # 作用：尝试加载英文规范化器（原理：捕获缺失依赖）
        chat.normalizer.register("en", normalizer_en_nemo_text())  # 作用：注册英文规范化器（概念：注册=绑定功能）
    except ValueError as e:  # 作用：处理配置错误（概念：异常=运行错误）
        get_logger("Command").error(e)  # 作用：输出错误日志（概念：日志分级）
    except BaseException:  # 作用：兜底处理缺失依赖（概念：异常兜底）
        get_logger("Command").warning("Package nemo_text_processing not found!")  # 作用：提示缺依赖（概念：依赖=外部库）
    try:  # 作用：尝试加载中文规范化器（原理：捕获缺失依赖）
        chat.normalizer.register("zh", normalizer_zh_tn())  # 作用：注册中文规范化器（概念：注册=绑定功能）
    except ValueError as e:  # 作用：处理配置错误（概念：异常=运行错误）
        get_logger("Command").error(e)  # 作用：输出错误日志（概念：日志分级）
    except BaseException:  # 作用：兜底处理缺失依赖（概念：异常兜底）
        get_logger("Command").warning("Package WeTextProcessing not found!")  # 作用：提示缺依赖（概念：依赖=外部库）

def trim_and_normalize_ref(wav: np.ndarray, threshold: float) -> np.ndarray:  # 作用：裁剪静音并归一化参考音频（概念：信号清洗）
    abs_wav = np.abs(wav)  # 作用：取绝对值幅度（原理：忽略正负号）  # 概念：幅度=信号强度
    indices = np.where(abs_wav > threshold)[0]  # 作用：找到非静音样本（原理：阈值筛选）  # 概念：阈值=静音判断标准
    if indices.size == 0:  # 作用：没有有效样本时直接返回（原理：边界处理）  # 概念：空样本=无语音
        return wav  # 作用：返回原始音频（原理：避免空裁剪）  # 概念：兜底=失败保护
    start = int(indices[0])  # 作用：定位裁剪起点（原理：取首个索引）  # 概念：起点=开始位置
    end = int(indices[-1]) + 1  # 作用：定位裁剪终点（原理：取末尾索引加一）  # 概念：终点=结束位置
    trimmed = wav[start:end]  # 作用：裁剪有效片段（原理：数组切片）  # 概念：裁剪=去除静音
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0  # 作用：计算峰值（原理：绝对值最大）  # 概念：峰值=最大振幅
    if peak > 0:  # 作用：避免除零（原理：条件保护）  # 概念：安全检查=防止异常
        trimmed = trimmed / peak * 0.9  # 作用：归一化音量（原理：比例缩放）  # 概念：归一化=统一振幅
    return trimmed  # 作用：返回处理后音频（原理：函数输出）  # 概念：返回值=处理结果

def save_audio(wav: np.ndarray, path: str, fmt: str) -> None:  # 作用：保存音频文件（概念：输出持久化）
    if fmt == "mp3":  # 作用：判断输出格式（概念：条件分支）
        data = pcm_arr_to_mp3_view(wav)  # 作用：PCM 转 MP3（概念：编码=压缩音频）
        with open(path, "wb") as f:  # 作用：写入二进制文件（概念：二进制=原始字节）
            f.write(data)  # 作用：保存 MP3 数据（原理：写入文件）
    else:  # 作用：处理 WAV 输出（概念：无损音频格式）
        torchaudio.save(path, torch.from_numpy(wav).unsqueeze(0), 24000)  # 作用：保存 WAV（概念：采样率=每秒采样次数）

def parse_args() -> argparse.Namespace:  # 作用：解析命令行参数（概念：配置入口）
    parser = argparse.ArgumentParser(  # 作用：创建参数解析器（概念：解析器=命令行工具）
        description="ChatTTS Config Runner",  # 作用：命令说明（概念：帮助信息）
    )  # 作用：结束解析器创建（概念：对象构造）
    parser.add_argument(  # 作用：添加文本参数（概念：可选输入）
        "text",  # 作用：参数名称（概念：位置参数）
        nargs="*",  # 作用：允许不输入文本（概念：可选参数）
        default=["马斯克的SpaceX马上就要上市IPO了，是不是可以布局一下商业航天啊！"],  # 作用：默认文本（概念：默认值）
        help="输入文本（可空，空则使用默认文案）",  # 作用：参数说明（概念：帮助信息）
    )  # 作用：结束参数定义（概念：解析器配置）
    parser.add_argument("--source", choices=["local", "huggingface", "custom"], default="local", help="模型来源")  # 作用：模型来源选择（概念：数据源）
    parser.add_argument("--custom_path", default="", help="自定义模型路径")  # 作用：自定义路径（概念：本地模型目录）
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3", help="输出格式")  # 作用：输出格式（概念：音频编码）
    parser.add_argument("--output", default="output_audio_0.mp3", help="输出文件名")  # 作用：输出文件名（概念：文件路径）
    parser.add_argument("--spk_emb", default=None, help="音色嵌入字符串")  # 作用：指定音色嵌入（概念：说话人特征）
    parser.add_argument("--spk_file", default=None, help="音色嵌入文件路径")  # 作用：读取音色嵌入文件（概念：持久化特征）
    parser.add_argument("--spk_audio", default=None, help="参考音频路径，用于提取音色")  # 作用：参考音频（概念：零样本音色）
    parser.add_argument("--txt_smp", default=None, help="参考音频对应转写文本")  # 作用：参考转写文本（概念：音频文本对齐）
    parser.add_argument("--min_new_token", type=int, default=None, help="最小生成步数（用于避免首步结束）")  # 作用：最小步数（概念：EOS=结束标记）
    parser.add_argument("--trim_ref", action="store_true", help="裁剪静音并归一化参考音频")  # 作用：清洗参考音频（概念：静音裁剪）
    parser.add_argument("--trim_threshold", type=float, default=0.01, help="静音阈值（裁剪参考音频用）")  # 作用：静音阈值（概念：能量阈值）
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")  # 作用：采样温度（概念：随机性控制）
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-P 采样")  # 作用：Top-P（概念：概率截断采样）
    parser.add_argument("--top_k", type=int, default=20, help="Top-K 采样")  # 作用：Top-K（概念：候选数量限制）
    parser.add_argument("--compile", action="store_true", help="开启编译优化")  # 作用：编译加速（概念：图优化）
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="计算设备")  # 作用：设备选择（概念：CPU/GPU）
    return parser.parse_args()  # 作用：返回解析结果（概念：参数对象）

def main() -> None:  # 作用：程序主入口（概念：执行流程起点）
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # 作用：设置输出编码（概念：避免中文乱码）
    if hasattr(sys.stdout, "reconfigure"):  # 作用：检查输出是否可配置（概念：兼容性判断）
        sys.stdout.reconfigure(encoding="utf-8")  # 作用：设置标准输出编码（概念：UTF-8 编码）
    args = parse_args()  # 作用：读取命令行参数（概念：配置获取）
    logger = get_logger("Command")  # 作用：创建日志器（概念：日志通道）
    chat = ChatTTS.Chat(get_logger("ChatTTS"))  # 作用：创建模型实例（概念：推理对象）
    load_normalizer(chat)  # 作用：加载文本规范化器（概念：文本清洗）
    custom_path = args.custom_path if args.custom_path else None  # 作用：处理自定义路径（概念：空值转换）
    device = None  # 作用：默认设备选择（概念：自动选择）
    if args.device == "cpu":  # 作用：手动指定 CPU（概念：设备强制）
        device = torch.device("cpu")  # 作用：创建 CPU 设备对象（概念：计算设备）
    elif args.device == "cuda":  # 作用：手动指定 GPU（概念：设备强制）
        device = torch.device("cuda")  # 作用：创建 GPU 设备对象（概念：CUDA=英伟达 GPU 计算）
    is_load = chat.load(source=args.source, custom_path=custom_path, compile=args.compile, device=device)  # 作用：加载模型权重（概念：预训练参数）
    if not is_load:  # 作用：判断加载是否成功（概念：流程控制）
        logger.error("模型加载失败")  # 作用：输出错误信息（概念：错误日志）
        sys.exit(1)  # 作用：退出程序（概念：非零退出码）
    spk_emb: Optional[str] = None  # 作用：初始化音色嵌入（概念：说话人特征）
    spk_smp: Optional[str] = None  # 作用：初始化音色签名（概念：音频提取特征）
    if args.spk_file:  # 作用：从文件读取音色嵌入（概念：持久化特征）
        with open(args.spk_file, "r", encoding="utf-8") as f:  # 作用：打开音色文件（概念：文本读取）
            spk_emb = f.read().strip()  # 作用：读取并去除空白（概念：文本清理）
    elif args.spk_emb:  # 作用：直接使用命令行音色嵌入（概念：参数优先）
        spk_emb = args.spk_emb  # 作用：赋值音色嵌入（概念：变量绑定）
    if args.spk_audio:  # 作用：参考音频提取音色（概念：零样本音色模拟）
        ref_wav = load_audio(args.spk_audio, 24000)  # 作用：加载并重采样音频（概念：采样率统一）
        if args.trim_ref:  # 作用：按需裁剪与归一化（概念：音频清洗）
            ref_wav = trim_and_normalize_ref(ref_wav, args.trim_threshold)  # 作用：清洗参考音频（原理：阈值裁剪+归一化）  # 概念：静音裁剪=去除低能量
        spk_smp = chat.sample_audio_speaker(ref_wav)  # 作用：提取音色签名（概念：嵌入向量）
        spk_emb = None  # 作用：使用音频签名时禁用嵌入（概念：避免冲突）
        if args.txt_smp is None:  # 作用：缺少参考转写时提示（概念：对齐检查）
            logger.warning("参考音频未提供对应转写（txt_smp），克隆效果可能不稳定")  # 作用：输出警告（概念：日志提示）
    if spk_emb is None and spk_smp is None:  # 作用：无音色输入时随机生成（概念：兜底策略）
        spk_emb = chat.sample_random_speaker()  # 作用：随机音色（概念：说话人采样）
    min_new_token = args.min_new_token  # 作用：读取最小步数（概念：生成下限）
    if min_new_token is None:  # 作用：未显式设置时自动决定（概念：默认策略）
        min_new_token = 64 if spk_smp is not None else 0  # 作用：克隆场景提高下限（原理：避免首步结束）  # 概念：EOS=结束标记
    params = ChatTTS.Chat.InferCodeParams(  # 作用：构建推理参数对象（概念：参数集合）
        spk_emb=spk_emb,  # 作用：设置音色嵌入（概念：说话人特征）
        spk_smp=spk_smp,  # 作用：设置音色签名（概念：参考音频特征）
        txt_smp=args.txt_smp if spk_smp is not None and args.txt_smp else None,  # 作用：设置参考转写（原理：音频文本对齐）  # 概念：txt_smp=参考文本
        temperature=args.temperature,  # 作用：设置采样温度（概念：随机性强度）
        top_P=args.top_p,  # 作用：设置 Top-P（概念：概率截断）
        top_K=args.top_k,  # 作用：设置 Top-K（概念：候选数量）
        min_new_token=min_new_token,  # 作用：设置最小生成步数（原理：避免首步 EOS）  # 概念：生成下限=最短长度
    )  # 作用：结束参数构建（概念：对象完成）
    wavs = chat.infer(args.text, params_infer_code=params)  # 作用：执行推理生成音频（概念：文本转语音）
    output_path = args.output  # 作用：读取输出路径（概念：文件名）
    if args.format == "wav" and not output_path.endswith(".wav"):  # 作用：自动修正后缀（概念：文件格式匹配）
        output_path = "output_audio_0.wav"  # 作用：设置默认 WAV 文件名（概念：输出兜底）
    if args.format == "mp3" and not output_path.endswith(".mp3"):  # 作用：自动修正后缀（概念：文件格式匹配）
        output_path = "output_audio_0.mp3"  # 作用：设置默认 MP3 文件名（概念：输出兜底）
    save_audio(wavs[0], output_path, args.format)  # 作用：保存生成音频（概念：结果落盘）
    logger.info("音频已保存到 %s", output_path)  # 作用：输出成功信息（概念：日志提示）

if __name__ == "__main__":  # 作用：脚本入口判断（概念：模块直接执行）
    main()  # 作用：调用主函数（概念：启动流程）
