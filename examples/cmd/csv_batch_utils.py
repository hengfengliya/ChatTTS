import os  # 作用：操作系统接口（概念：路径与文件处理）
import sys  # 作用：系统参数与输入输出（概念：标准输出控制）
import csv  # 作用：读取 CSV 文件（概念：表格数据）
import hashlib  # 作用：哈希计算（概念：稳定种子生成）
import numpy as np  # 作用：数值计算库（概念：数组与信号处理）
from typing import Optional, List, Any  # 作用：类型注解（概念：可读性与检查）

from tools.logger import get_logger  # 作用：获取日志器（概念：日志记录）
from tools.audio import load_audio, pcm_arr_to_mp3_view  # 作用：音频加载与编码（概念：重采样/压缩）
from tools.normalizer.en import normalizer_en_nemo_text  # 作用：英文文本规范化（概念：文本归一化）
from tools.normalizer.zh import normalizer_zh_tn  # 作用：中文文本规范化（概念：文本清洗）


def load_normalizer(chat: Any) -> None:  # 作用：加载文本规范化器（概念：语言清洗）
    try:  # 作用：尝试加载英文规范化器（原理：捕获缺失依赖）
        chat.normalizer.register("en", normalizer_en_nemo_text())  # 作用：注册英文规范化器（概念：绑定功能）
    except ValueError as e:  # 作用：处理配置错误（概念：异常）
        get_logger("CSVBatch").error(e)  # 作用：输出错误日志（概念：日志分级）
    except BaseException:  # 作用：兜底处理缺失依赖（概念：异常兜底）
        get_logger("CSVBatch").warning("Package nemo_text_processing not found!")  # 作用：提示缺依赖（概念：依赖库）
    try:  # 作用：尝试加载中文规范化器（原理：捕获缺失依赖）
        chat.normalizer.register("zh", normalizer_zh_tn())  # 作用：注册中文规范化器（概念：绑定功能）
    except ValueError as e:  # 作用：处理配置错误（概念：异常）
        get_logger("CSVBatch").error(e)  # 作用：输出错误日志（概念：日志分级）
    except BaseException:  # 作用：兜底处理缺失依赖（概念：异常兜底）
        get_logger("CSVBatch").warning("Package WeTextProcessing not found!")  # 作用：提示缺依赖（概念：依赖库）


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


def ensure_dir(path: str) -> None:  # 作用：确保目录存在（概念：目录初始化）
    os.makedirs(path, exist_ok=True)  # 作用：创建目录（原理：若不存在则创建）


def save_audio(wav: Any, path: str, fmt: str) -> None:  # 作用：保存音频文件（概念：结果落盘）
    if fmt == "mp3":  # 作用：判断输出格式（概念：条件分支）
        data = pcm_arr_to_mp3_view(wav)  # 作用：PCM 转 MP3（概念：音频编码）
        with open(path, "wb") as f:  # 作用：写入二进制文件（概念：字节流）
            f.write(data)  # 作用：保存 MP3 数据（原理：写入文件）
    else:  # 作用：处理 WAV 输出（概念：无损格式）
        import torch  # 作用：延迟导入 torch（概念：减少启动耗时）
        import torchaudio  # 作用：延迟导入 torchaudio（概念：减少启动耗时）
        torchaudio.save(path, torch.from_numpy(wav).unsqueeze(0), 24000)  # 作用：保存 WAV（概念：采样率）


def read_csv_rows(csv_path: str) -> List[dict]:  # 作用：读取 CSV 行（概念：表格数据）
    with open(csv_path, "r", encoding="utf-8", newline="") as f:  # 作用：打开 CSV（概念：UTF-8）
        reader = csv.DictReader(f)  # 作用：按列名读取（概念：字典行）
        return [row for row in reader]  # 作用：转成列表（概念：批量处理）


def resolve_speaker_from_key(speaker_key: str, logger: Any) -> Optional[str]:  # 作用：解析说话人字段（概念：固定音色来源）
    if not speaker_key:  # 作用：空值直接返回（概念：边界处理）
        return None  # 作用：无音色嵌入（概念：空结果）
    if speaker_key.startswith("file:"):  # 作用：文件模式（概念：外部音色文件）
        file_path = speaker_key.replace("file:", "", 1).strip()  # 作用：提取文件路径（概念：去前缀）
        if not os.path.exists(file_path):  # 作用：校验文件存在（概念：路径检查）
            logger.warning("说话人音色文件不存在：%s", file_path)  # 作用：提示缺失（概念：日志提示）
            return None  # 作用：返回空（概念：回退）
        with open(file_path, "r", encoding="utf-8") as f:  # 作用：读取文件（概念：文本读取）
            return f.read().strip()  # 作用：返回音色嵌入（概念：固定音色）
    if speaker_key.startswith("emb:"):  # 作用：直接嵌入模式（概念：内联音色）
        return speaker_key.replace("emb:", "", 1).strip()  # 作用：返回嵌入内容（概念：固定音色）
    return None  # 作用：未知模式返回空（概念：回退）


def resolve_ref_audio_path(ref_dir: str, ref_prefix: str) -> Optional[str]:  # 作用：根据前缀查找参考音频（概念：文件匹配）
    if not ref_prefix:  # 作用：空值直接返回（概念：边界处理）
        return None  # 作用：无参考音频（概念：空结果）
    direct_path = os.path.join(ref_dir, ref_prefix)  # 作用：支持带扩展名的直连路径（概念：完整文件名）
    if os.path.exists(direct_path):  # 作用：优先检查直连文件（概念：路径校验）
        return direct_path  # 作用：返回直连路径（概念：匹配成功）
    ref_prefix = os.path.splitext(ref_prefix)[0]  # 作用：去掉扩展名再尝试（概念：前缀归一化）
    exts = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]  # 作用：支持的音频扩展名（概念：格式列表）
    for ext in exts:  # 作用：遍历扩展名（概念：循环匹配）
        candidate = os.path.join(ref_dir, ref_prefix + ext)  # 作用：拼接候选路径（概念：路径组合）
        if os.path.exists(candidate):  # 作用：检查文件存在（概念：路径校验）
            return candidate  # 作用：返回匹配路径（概念：找到结果）
    return None  # 作用：未找到匹配文件（概念：回退）


def stable_seed_from_key(key: str) -> int:  # 作用：生成稳定随机种子（概念：可复现）
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()  # 作用：计算哈希（概念：固定映射）
    return int(digest[:8], 16)  # 作用：取前 8 位转整数（概念：种子值）


def sanitize_text(text: str) -> str:  # 作用：清理文本（概念：减少非法字符）
    if not text:  # 作用：空文本直接返回（概念：边界处理）
        return text  # 作用：返回原值（概念：无修改）
    digit_map = {  # 作用：数字映射（概念：数字中文化）
        "0": "零",  # 作用：映射 0（概念：中文数字）
        "1": "一",  # 作用：映射 1（概念：中文数字）
        "2": "二",  # 作用：映射 2（概念：中文数字）
        "3": "三",  # 作用：映射 3（概念：中文数字）
        "4": "四",  # 作用：映射 4（概念：中文数字）
        "5": "五",  # 作用：映射 5（概念：中文数字）
        "6": "六",  # 作用：映射 6（概念：中文数字）
        "7": "七",  # 作用：映射 7（概念：中文数字）
        "8": "八",  # 作用：映射 8（概念：中文数字）
        "9": "九",  # 作用：映射 9（概念：中文数字）
    }  # 作用：结束映射定义（概念：字典）
    text = "".join(digit_map.get(ch, ch) for ch in text)  # 作用：替换数字（概念：逐字符处理）
    text = text.replace("？", "。").replace("?", "。")  # 作用：替换问号（概念：标点规范）
    text = text.replace("！", "。").replace("!", "。")  # 作用：替换感叹号（概念：标点规范）
    return text  # 作用：返回清理结果（概念：输出文本）


def safe_infer(chat: Any, text: str, params_refine: Any, params_infer: Any) -> List[Any]:  # 作用：安全推理封装（概念：异常兜底）
    try:  # 作用：优先使用标准推理（概念：正常流程）
        return chat.infer(  # 作用：执行推理（概念：文本转语音）
            [text],  # 作用：输入文本列表（概念：批处理输入）
            params_refine_text=params_refine,  # 作用：传入细化参数（概念：稳定前处理）
            params_infer_code=params_infer,  # 作用：传入推理参数（概念：稳定生成）
        )  # 作用：返回推理结果（概念：音频数组）
    except ValueError:  # 作用：捕获空拼接异常（概念：兜底处理）
        pass  # 作用：继续尝试兜底方案（概念：流程控制）
    try:  # 作用：第二次尝试（概念：禁用细化与规范）
        return chat.infer(  # 作用：执行推理（概念：文本转语音）
            [text],  # 作用：输入文本列表（概念：批处理输入）
            skip_refine_text=True,  # 作用：跳过细化（概念：减少不稳定）
            do_text_normalization=False,  # 作用：关闭文本规范化（概念：避免清空）
            do_homophone_replacement=False,  # 作用：关闭同音替换（概念：减少变换）
            split_text=False,  # 作用：禁用分段（概念：减少切分）
            params_refine_text=params_refine,  # 作用：传入细化参数（概念：占位）
            params_infer_code=params_infer,  # 作用：传入推理参数（概念：稳定生成）
        )  # 作用：返回推理结果（概念：音频数组）
    except BaseException:  # 作用：捕获其他异常（概念：兜底）
        pass  # 作用：继续低层推理（概念：流程控制）
    wavs: List[Any] = []  # 作用：准备输出列表（概念：容器）
    for batch in chat._infer(  # 作用：使用内部推理避免拼接（概念：低层输出）
        [text],  # 作用：输入文本列表（概念：批处理输入）
        False,  # 作用：关闭流式（概念：完整输出）
        None,  # 作用：不指定语言（概念：自动判断）
        True,  # 作用：跳过细化（概念：兜底策略）
        False,  # 作用：不只细化（概念：生成音频）
        True,  # 作用：使用解码器（概念：解码生成）
        False,  # 作用：关闭文本规范化（概念：避免清空）
        False,  # 作用：关闭同音替换（概念：减少变换）
        False,  # 作用：不分段（概念：减少切分）
        4,  # 作用：最大分段批次（概念：批处理大小）
        params_refine,  # 作用：细化参数（概念：前处理控制）
        params_infer,  # 作用：推理参数（概念：生成控制）
    ):  # 作用：迭代批次输出（概念：生成批）
        for wav in batch:  # 作用：展开批次（概念：逐条处理）
            wavs.append(wav)  # 作用：收集音频（概念：结果聚合）
    return wavs  # 作用：返回兜底结果（概念：音频列表）
