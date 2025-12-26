from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

import av
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
import numpy as np


video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}


def wav2(i: BytesIO, o: BufferedWriter, format: str):
    """
    https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/412a9950a1e371a018c381d1bfb8579c4b0de329/infer/lib/audio.py#L20
    """
    inp = av.open(i, "r")
    format = video_format_dict.get(format, format)
    out = av.open(o, "w", format=format)
    format = audio_format_dict.get(format, format)

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(
    file: Union[str, BytesIO, Path],
    sr: Optional[int] = None,
    format: Optional[str] = None,
    mono=True,
    max_seconds: Optional[float] = None,  # 作用：限制最大时长（概念：防止超长卡顿）
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """
    https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/412a9950a1e371a018c381d1bfb8579c4b0de329/infer/lib/audio.py#L39
    """
    if (isinstance(file, str) and not Path(file).exists()) or (
        isinstance(file, Path) and not file.exists()
    ):
        raise FileNotFoundError(f"File not found: {file}")
    rate = 0

    container = av.open(file, format=format)
    audio_stream = next(s for s in container.streams if s.type == "audio")
    channels = 1 if audio_stream.layout == "mono" else 2
    container.seek(0)
    resampler = (
        AudioResampler(format="fltp", layout=audio_stream.layout, rate=sr)
        if sr is not None
        else None
    )

    # Estimated maximum total number of samples to pre-allocate the array
    # AV stores length in microseconds by default
    duration_us = container.duration or 0  # 作用：获取容器时长（概念：微秒时间戳）
    estimated_total_samples = (  # 作用：估算样本数（概念：预分配容量）
        int(duration_us * sr // 1_000_000) if sr is not None and duration_us else 48000  # 作用：按时长估算（概念：微秒转采样点）
    )
    max_samples: Optional[int] = None  # 作用：初始化最大样本数（概念：截断上限）
    if sr is not None and max_seconds is not None:  # 作用：启用时长上限（概念：防止超大音频）
        max_samples = max(int(sr * max_seconds), 1)  # 作用：计算上限样本数（概念：秒转采样点）
        if estimated_total_samples > max_samples:  # 作用：限制预估容量（概念：防止超大分配）
            estimated_total_samples = max_samples  # 作用：应用上限（概念：内存保护）
    decoded_audio = np.zeros(
        (
            estimated_total_samples + 1
            if channels == 1
            else (channels, estimated_total_samples + 1)
        ),
        dtype=np.float32,
    )

    offset = 0

    def process_packet(packet: List[AudioFrame]):
        frames_data = []
        rate = 0
        for frame in packet:
            # frame.pts = None  # 清除时间戳，避免重新采样问题
            resampled_frames = (
                resampler.resample(frame) if resampler is not None else [frame]
            )
            for resampled_frame in resampled_frames:
                frame_data = resampled_frame.to_ndarray()
                rate = resampled_frame.rate
                frames_data.append(frame_data)
        return (rate, frames_data)

    def frame_iter(container):
        for p in container.demux(container.streams.audio[0]):
            yield p.decode()

    for r, frames_data in map(process_packet, frame_iter(container)):  # 作用：逐包解码音频（概念：流式解码）
        if not rate:
            rate = r
        for frame_data in frames_data:
            end_index = offset + len(frame_data[0])  # 作用：计算写入终点（概念：样本累加）
            if max_samples is not None and end_index > max_samples:  # 作用：超过上限时截断（概念：时长限制）
                end_index = max_samples  # 作用：应用上限终点（概念：截断边界）

            # 检查 decoded_audio 是否有足够的空间，并在必要时调整大小
            if end_index > decoded_audio.shape[1]:
                decoded_audio = np.resize(
                    decoded_audio, (decoded_audio.shape[0], end_index * 4)
                )

            np.copyto(  # 作用：拷贝音频数据（概念：内存复制）
                decoded_audio[..., offset:end_index],  # 作用：目标切片（概念：缓冲区片段）
                frame_data[..., : end_index - offset],  # 作用：源切片（概念：截断后的帧数据）
            )
            offset = end_index  # 作用：更新写入位置（概念：写指针）
            if max_samples is not None and offset >= max_samples:  # 作用：达到上限时停止（概念：提前退出）
                break  # 作用：跳出当前帧循环（概念：停止解码）
        if max_samples is not None and offset >= max_samples:  # 作用：达到上限时停止（概念：提前退出）
            break  # 作用：跳出包循环（概念：停止解码）

    container.close()

    # Truncate the array to the actual size
    decoded_audio = decoded_audio[..., :offset]

    if mono and decoded_audio.shape[0] > 1:
        decoded_audio = decoded_audio.mean(0)

    if sr is not None:
        return decoded_audio
    return decoded_audio, rate
