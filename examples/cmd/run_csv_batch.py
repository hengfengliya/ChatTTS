import os  # 作用：操作系统接口（概念：路径与文件处理）
import sys  # 作用：系统参数与输入输出（概念：标准输出控制）
import argparse  # 作用：命令行参数解析（概念：配置入口）
from typing import Optional, List  # 作用：类型注解（概念：可读性与检查）
now_dir = os.getcwd()  # 作用：获取当前工作目录（概念：运行路径）
sys.path.append(now_dir)  # 作用：加入模块搜索路径（概念：导入本地包）

from tools.logger import get_logger  # 作用：获取日志器（概念：日志记录）


def parse_args() -> argparse.Namespace:  # 作用：解析命令行参数（概念：配置入口）
    parser = argparse.ArgumentParser(description="ChatTTS CSV Batch Runner")  # 作用：创建解析器（概念：命令行工具）
    parser.add_argument("--csv", default="examples/cmd/batch_input.csv", help="CSV 文件路径")  # 作用：CSV 路径（概念：输入数据）
    parser.add_argument("--output_dir", default="voice_output", help="输出目录")  # 作用：输出目录（概念：结果路径）
    parser.add_argument("--ref_dir", default="voice_input", help="参考音频目录")  # 作用：参考音频目录（概念：输入音频）
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3", help="输出格式")  # 作用：输出格式（概念：编码方式）
    parser.add_argument("--merge_name", default="merged.mp3", help="合并文件名")  # 作用：合并输出名（概念：最终结果）
    parser.add_argument("--silence_ms", type=int, default=200, help="片段间静音毫秒")  # 作用：静音间隔（概念：停顿）
    parser.add_argument("--ref_max_sec", type=float, default=10.0, help="参考音频最大时长（秒）")  # 作用：参考音频截断（概念：防止过长卡顿）
    parser.add_argument("--trim_ref", action="store_true", help="裁剪静音并归一化参考音频")  # 作用：清洗参考音频（概念：静音裁剪）
    parser.add_argument("--trim_threshold", type=float, default=0.01, help="静音阈值（裁剪参考音频用）")  # 作用：静音阈值（概念：能量阈值）
    parser.add_argument("--source", choices=["local", "huggingface", "custom"], default="local", help="模型来源")  # 作用：模型来源（概念：权重来源）
    parser.add_argument("--custom_path", default="", help="自定义模型路径")  # 作用：自定义路径（概念：本地模型目录）
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="计算设备")  # 作用：设备选择（概念：CPU/GPU）
    parser.add_argument("--compile", action="store_true", help="开启编译优化")  # 作用：编译加速（概念：图优化）
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")  # 作用：采样温度（概念：随机性强度）
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-P 采样")  # 作用：Top-P（概念：概率截断）
    parser.add_argument("--top_k", type=int, default=20, help="Top-K 采样")  # 作用：Top-K（概念：候选限制）
    parser.add_argument("--min_new_token", type=int, default=64, help="最小生成步数，避免首步即结束")  # 作用：最小生成长度（概念：防止空输出）
    return parser.parse_args()  # 作用：返回解析结果（概念：参数对象）


def main() -> None:  # 作用：程序主入口（概念：执行起点）
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # 作用：设置输出编码（概念：避免中文乱码）
    if hasattr(sys.stdout, "reconfigure"):  # 作用：检查输出是否可配置（概念：兼容性）
        sys.stdout.reconfigure(encoding="utf-8")  # 作用：设置标准输出编码（概念：UTF-8）
    print("启动批量生成：开始加载依赖...", flush=True)  # 作用：输出启动提示（概念：进度可见）
    import numpy as np  # 作用：延迟导入 numpy（概念：减少启动卡顿）
    import torch  # 作用：延迟导入 torch（概念：减少启动卡顿）
    import ChatTTS  # 作用：延迟导入 ChatTTS（概念：模型入口）
    from tools.audio import load_audio  # 作用：延迟导入音频加载（概念：重采样读取）
    from examples.cmd.csv_batch_utils import (  # 作用：延迟导入批量工具（概念：复用逻辑）
        load_normalizer,  # 作用：加载规范化器（概念：文本清洗）
        trim_and_normalize_ref,  # 作用：裁剪参考音频（概念：静音清洗）
        ensure_dir,  # 作用：确保目录存在（概念：路径准备）
        save_audio,  # 作用：保存音频（概念：结果落盘）
        read_csv_rows,  # 作用：读取 CSV（概念：表格数据）
        resolve_speaker_from_key,  # 作用：解析说话人（概念：固定音色）
        resolve_ref_audio_path,  # 作用：解析参考音频（概念：前缀匹配）
        stable_seed_from_key,  # 作用：生成稳定种子（概念：可复现）
        sanitize_text,  # 作用：清理文本（概念：减少非法字符）
        safe_infer,  # 作用：安全推理（概念：异常兜底）
    )  # 作用：结束工具导入（概念：模块引用）
    print("依赖加载完成：开始解析参数...", flush=True)  # 作用：输出进度提示（概念：可见反馈）
    args = parse_args()  # 作用：读取参数（概念：配置获取）
    logger = get_logger("CSVBatch")  # 作用：创建日志器（概念：日志通道）
    def is_valid_wav(wav: np.ndarray) -> bool:  # 作用：判断音频有效性（概念：空音频检测）
        if wav is None:  # 作用：空对象直接判无效（概念：空值保护）
            return False  # 作用：返回无效（概念：布尔结果）
        if not hasattr(wav, "size"):  # 作用：确保有数组大小（概念：属性检查）
            return False  # 作用：返回无效（概念：布尔结果）
        if wav.size == 0:  # 作用：检测空数组（概念：长度为零）
            return False  # 作用：返回无效（概念：布尔结果）
        if not np.isfinite(wav).all():  # 作用：检测数值异常（概念：非有限值）
            return False  # 作用：返回无效（概念：布尔结果）
        if np.max(np.abs(wav)) < 1e-5:  # 作用：检测全静音（概念：幅度阈值）
            return False  # 作用：返回无效（概念：布尔结果）
        return True  # 作用：通过校验（概念：有效音频）
    if not os.path.exists(args.csv):  # 作用：检查 CSV 是否存在（概念：路径校验）
        logger.error("CSV 文件不存在：%s", args.csv)  # 作用：输出错误（概念：错误日志）
        sys.exit(1)  # 作用：退出程序（概念：非零退出码）
    ensure_dir(args.output_dir)  # 作用：创建输出目录（概念：路径准备）
    ensure_dir(args.ref_dir)  # 作用：创建输入目录（概念：参考音频）
    chat = ChatTTS.Chat(get_logger("ChatTTS"))  # 作用：创建模型实例（概念：推理对象）
    load_normalizer(chat)  # 作用：加载规范化器（概念：文本清洗）
    custom_path = args.custom_path if args.custom_path else None  # 作用：处理自定义路径（概念：空值转换）
    device = None  # 作用：默认设备（概念：自动选择）
    if args.device == "cpu":  # 作用：指定 CPU（概念：设备强制）
        device = torch.device("cpu")  # 作用：创建 CPU 设备对象（概念：计算设备）
    elif args.device == "cuda":  # 作用：指定 GPU（概念：设备强制）
        device = torch.device("cuda")  # 作用：创建 GPU 设备对象（概念：CUDA）
    is_load = chat.load(source=args.source, custom_path=custom_path, compile=args.compile, device=device)  # 作用：加载模型（概念：权重加载）
    if not is_load:  # 作用：判断加载是否成功（概念：流程控制）
        logger.error("模型加载失败")  # 作用：输出错误信息（概念：错误日志）
        sys.exit(1)  # 作用：退出程序（概念：非零退出码）
    rows = read_csv_rows(args.csv)  # 作用：读取 CSV 行（概念：批量数据）
    if len(rows) == 0:  # 作用：检查空数据（概念：边界处理）
        logger.error("CSV 内容为空")  # 作用：输出错误（概念：错误日志）
        sys.exit(1)  # 作用：退出程序（概念：非零退出码）
    speaker_cache: dict = {}  # 作用：缓存说话人音色（概念：同名复用）
    silence = np.zeros(int(24000 * args.silence_ms / 1000), dtype=np.float32)  # 作用：生成静音段（概念：零数组）
    merged_wavs: List[np.ndarray] = []  # 作用：合并音频列表（概念：序列拼接）
    for idx, row in enumerate(rows, start=1):  # 作用：逐行处理（概念：循环）
        text = sanitize_text((row.get("文本") or "").strip())  # 作用：读取并清理文本（概念：字符规范）
        speaker_key = (row.get("说话人") or "").strip()  # 作用：读取说话人列（概念：角色标识）
        ref_audio = (row.get("参考音频") or "").strip()  # 作用：读取参考音频列（概念：文件名前缀）
        ref_text = (row.get("参考文本") or row.get("参考转写") or "").strip()  # 作用：读取参考转写（概念：音频文本对齐）
        if not text:  # 作用：跳过空文本（概念：边界处理）
            logger.warning("第 %d 行文本为空，已跳过", idx)  # 作用：提示跳过（概念：日志提示）
            continue  # 作用：跳过本行（概念：流程控制）
        spk_emb: Optional[str] = None  # 作用：初始化音色嵌入（概念：说话人特征）
        spk_smp: Optional[str] = None  # 作用：初始化音色签名（概念：音频特征）
        manual_seed: Optional[int] = None  # 作用：初始化随机种子（概念：可复现）
        if ref_audio:  # 作用：存在参考音频时克隆（概念：零样本音色）
            ref_path = resolve_ref_audio_path(args.ref_dir, ref_audio)  # 作用：按前缀查找文件（概念：路径解析）
            if not ref_path:  # 作用：找不到参考音频（概念：路径校验）
                logger.warning("第 %d 行参考音频不存在，改用随机音色：%s", idx, ref_audio)  # 作用：提示降级（概念：日志提示）
            else:  # 作用：参考音频有效（概念：条件分支）
                logger.info("第 %d 行使用参考音频：%s", idx, ref_path)  # 作用：输出参考音频路径（概念：可追踪性）
                if not ref_text:  # 作用：缺少参考文本时提示（概念：条件检查）
                    logger.warning("第 %d 行缺少参考文本，音色可能不稳定", idx)  # 作用：提示补全转写（概念：日志提示）
                ref_wav = load_audio(ref_path, 24000, max_seconds=args.ref_max_sec)  # 作用：加载参考音频并截断（概念：限制时长）
                if args.trim_ref:  # 作用：按需裁剪与归一化（概念：音频清洗）
                    ref_wav = trim_and_normalize_ref(ref_wav, args.trim_threshold)  # 作用：清洗参考音频（原理：阈值裁剪+归一化）  # 概念：静音裁剪=去除低能量
                spk_smp = chat.sample_audio_speaker(ref_wav)  # 作用：提取音色签名（概念：嵌入向量）
                spk_emb = None  # 作用：克隆优先（概念：避免混用）
                manual_seed = None  # 作用：参考音频不固定种子（概念：避免过早结束）
        if spk_smp is None and spk_emb is None:  # 作用：无参考音频时走说话人逻辑（概念：固定音色）
            if speaker_key and speaker_key in speaker_cache:  # 作用：命中缓存（概念：同名复用）
                cached = speaker_cache[speaker_key]  # 作用：读取缓存（概念：缓存命中）
                spk_emb = cached.get("spk_emb")  # 作用：读取嵌入（概念：固定音色）
                spk_smp = cached.get("spk_smp")  # 作用：读取签名（概念：克隆音色）
                manual_seed = cached.get("manual_seed")  # 作用：读取种子（概念：稳定生成）
            if spk_smp is None and spk_emb is None and speaker_key:  # 作用：解析固定音色（概念：说话人绑定）
                spk_emb = resolve_speaker_from_key(speaker_key, logger)  # 作用：解析固定音色（概念：嵌入来源）
            if spk_smp is None and spk_emb is None:  # 作用：无参考音频时随机音色（概念：兜底策略）
                spk_emb = chat.sample_random_speaker()  # 作用：随机音色（概念：说话人采样）
            if speaker_key and manual_seed is None:  # 作用：为说话人生成稳定种子（概念：可复现）
                manual_seed = stable_seed_from_key(speaker_key)  # 作用：生成种子（概念：稳定哈希）
            if speaker_key:  # 作用：写入缓存（概念：固定音色）
                speaker_cache[speaker_key] = {  # 作用：保存音色与种子（概念：同名一致）
                    "spk_emb": spk_emb,  # 作用：保存嵌入（概念：固定音色）
                    "spk_smp": spk_smp,  # 作用：保存签名（概念：克隆音色）
                    "manual_seed": manual_seed,  # 作用：保存种子（概念：可复现）
                }  # 作用：结束缓存写入（概念：字典结构）
        params_refine = ChatTTS.Chat.RefineTextParams(  # 作用：构建文本细化参数（概念：前处理控制）
            manual_seed=manual_seed,  # 作用：设置细化随机种子（概念：稳定输出）
        )  # 作用：结束细化参数构建（概念：对象完成）
        params_infer = ChatTTS.Chat.InferCodeParams(  # 作用：构建推理参数（概念：参数集合）
            spk_emb=spk_emb,  # 作用：设置音色嵌入（概念：说话人特征）
            spk_smp=spk_smp,  # 作用：设置音色签名（概念：参考音频特征）
            temperature=args.temperature,  # 作用：设置采样温度（概念：随机性）
            top_P=args.top_p,  # 作用：设置 Top-P（概念：概率截断）
            top_K=args.top_k,  # 作用：设置 Top-K（概念：候选限制）
            manual_seed=manual_seed,  # 作用：设置随机种子（概念：稳定输出）
            min_new_token=max(1, args.min_new_token),  # 作用：限制最小步数（原理：避免首步 EOS）
            txt_smp=ref_text if spk_smp is not None and ref_text else None,  # 作用：设置参考转写（概念：音频文本对齐）
        )  # 作用：结束参数构建（概念：对象完成）
        wavs = safe_infer(chat, text, params_refine, params_infer)  # 作用：执行安全推理（概念：异常兜底）
        if len(wavs) == 0 or not is_valid_wav(wavs[0]):  # 作用：检测空音频或全静音（概念：质量检查）
            logger.warning("第 %d 行生成音频异常，尝试提高采样并重试", idx)  # 作用：提示重试（概念：日志提示）
            retry_params = ChatTTS.Chat.InferCodeParams(  # 作用：构建重试参数（概念：更激进采样）
                spk_emb=spk_emb,  # 作用：设置音色嵌入（概念：说话人特征）
                spk_smp=spk_smp,  # 作用：设置音色签名（概念：参考音频特征）
                temperature=max(args.temperature, 0.7),  # 作用：提高温度（概念：增加随机性）
                top_P=max(args.top_p, 0.9),  # 作用：提高 Top-P（概念：扩大候选）
                top_K=max(args.top_k, 40),  # 作用：提高 Top-K（概念：扩大候选）
                manual_seed=None,  # 作用：清空种子（概念：允许重新采样）
                min_new_token=max(128, args.min_new_token),  # 作用：提高最小步数（概念：避免过短）
                txt_smp=ref_text if spk_smp is not None and ref_text else None,  # 作用：设置参考转写（概念：音频文本对齐）
            )  # 作用：结束重试参数构建（概念：对象完成）
            wavs = safe_infer(chat, text, params_refine, retry_params)  # 作用：再次推理（概念：二次尝试）
        if len(wavs) == 0:  # 作用：处理空输出（概念：异常保护）
            logger.warning("第 %d 行未生成音频", idx)  # 作用：提示异常（概念：日志）
            continue  # 作用：跳过（概念：流程控制）
        wav = wavs[0]  # 作用：取第一个音频（概念：结果选择）
        if not is_valid_wav(wav):  # 作用：最终兜底校验（概念：防止空音频）
            logger.warning("第 %d 行音频全静音或为空，已跳过", idx)  # 作用：提示跳过（概念：日志提示）
            continue  # 作用：跳过本行（概念：流程控制）
        file_name = f"{idx:03d}.{args.format}"  # 作用：生成文件名（概念：序号命名）
        out_path = os.path.join(args.output_dir, file_name)  # 作用：拼接输出路径（概念：路径拼接）
        save_audio(wav, out_path, args.format)  # 作用：保存短音频（概念：文件输出）
        merged_wavs.append(wav)  # 作用：加入合并列表（概念：序列拼接）
        merged_wavs.append(silence)  # 作用：加入静音段（概念：停顿）
        logger.info("已生成第 %d 行音频：%s", idx, out_path)  # 作用：输出进度（概念：日志提示）
    if len(merged_wavs) == 0:  # 作用：检查合并列表为空（概念：边界处理）
        logger.error("没有可合并的音频")  # 作用：输出错误（概念：错误日志）
        sys.exit(1)  # 作用：退出程序（概念：非零退出码）
    merged = np.concatenate(merged_wavs)  # 作用：拼接音频（概念：数组合并）
    merge_path = os.path.join(args.output_dir, args.merge_name)  # 作用：合并文件路径（概念：输出路径）
    if args.format == "wav" and not merge_path.endswith(".wav"):  # 作用：修正合并文件后缀（概念：格式匹配）
        merge_path = os.path.join(args.output_dir, "merged.wav")  # 作用：兜底文件名（概念：默认值）
    if args.format == "mp3" and not merge_path.endswith(".mp3"):  # 作用：修正合并文件后缀（概念：格式匹配）
        merge_path = os.path.join(args.output_dir, "merged.mp3")  # 作用：兜底文件名（概念：默认值）
    save_audio(merged, merge_path, args.format)  # 作用：保存合并音频（概念：最终结果）
    logger.info("合并音频已保存到 %s", merge_path)  # 作用：输出成功提示（概念：日志提示）


if __name__ == "__main__":  # 作用：脚本入口判断（概念：直接执行）
    main()  # 作用：调用主函数（概念：启动流程）
