"""
Stage 2: 从视频中提取高保真音频
=================================
FFmpeg 提取 48kHz/16bit/Mono WAV。
支持断点续传，已提取的视频自动跳过。
"""

import os
import subprocess

from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    get_logger,
    get_video_files,
    load_config,
)


def extract_audio(
    video_path: str,
    output_dir: str,
    sample_rate: int = 48000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    logger=None,
) -> str | None:
    """
    从单个视频提取音频。
    返回输出 WAV 路径，失败返回 None。
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{basename}.wav")

    if os.path.exists(output_path):
        if logger:
            logger.debug(f"已存在，跳过: {output_path}")
        return output_path

    cmd = [
        "ffmpeg",
        "-y",                        # 覆盖输出
        "-i", video_path,
        "-vn",                       # 丢弃视频流
        "-acodec", codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-map_metadata", "-1",       # 移除元数据
        "-loglevel", "error",        # 不打印无关信息
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            if logger:
                logger.error(f"FFmpeg 失败 [{video_path}]: {result.stderr.strip()}")
            return None

        return output_path
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"超时 [{video_path}]")
        return None
    except Exception as e:
        if logger:
            logger.error(f"异常 [{video_path}]: {e}")
        return None


def extract_all(config: dict, logger=None):
    """批量提取所有视频的音频"""
    paths = config["paths"]

    if logger is None:
        logger = get_logger("02_extract_audio")

    audio_dir = ensure_dir(paths["extracted_audio"])

    videos = get_video_files(paths["raw_videos"])

    if not videos:
        logger.warning("没有找到视频文件，请先运行 01_download.py")
        return []

    logger.info(f"=" * 60)
    logger.info(f"Stage 2: 音频提取")
    logger.info(f"共 {len(videos)} 个视频 → 输出至 {audio_dir}")
    logger.info(f"格式: {config['audio_extract']['sample_rate']}Hz / "
                f"{config['audio_extract']['channels']}ch / "
                f"{config['audio_extract']['codec']}")
    logger.info(f"=" * 60)

    results = []
    for video_path in tqdm(videos, desc="提取音频"):
        out_path = extract_audio(
            video_path,
            audio_dir,
            sample_rate=config["audio_extract"]["sample_rate"],
            channels=config["audio_extract"]["channels"],
            codec=config["audio_extract"]["codec"],
            logger=logger,
        )

        if out_path and os.path.exists(out_path):
            results.append(out_path)
            logger.info(f"✓ {video_path}")
        else:
            logger.warning(f"✗ {video_path} 提取失败")

    logger.info(f"完成: {len(results)}/{len(videos)}")
    return results


def main():
    config = load_config()
    from pipeline.utils import setup_logger
    logger = setup_logger("02_extract_audio", config["paths"]["logs"])
    extract_all(config, logger)


if __name__ == "__main__":
    main()
