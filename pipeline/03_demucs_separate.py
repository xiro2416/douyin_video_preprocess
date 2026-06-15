"""
Stage 3: Demucs 人声分离 (Python API — no ffmpeg needed)
==========================================================
使用 htdemucs + --two-stems vocals 分离出人声茎。
改用 Demucs Python API 直接加载，绕过 torchaudio / ffmpeg 依赖。
使用 soundfile 读取音频，numpy → tensor 传给 Demucs 推理。
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    get_audio_files,
    get_logger,
    load_config,
)


def run_demucs_separation(
    audio_path: str,
    output_dir: str,
    model_name: str = "htdemucs",
    shifts: int = 3,
    overlap: float = 0.25,
    segment: int = 5,
    device: str = "cuda",
    logger=None,
) -> dict | None:
    """
    对单个音频运行 Demucs 分离（Python API，回避 torchaudio 加载）。
    返回 {"vocals": path, "no_vocals": path}，失败返回 None。
    """
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model_name, basename)
    os.makedirs(stem_dir, exist_ok=True)
    vocals_path = os.path.join(stem_dir, "vocals.wav")
    no_vocals_path = os.path.join(stem_dir, "no_vocals.wav")

    if os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
        if logger:
            logger.debug(f"Demucs 已处理，跳过: {basename}")
        return {"vocals": vocals_path, "no_vocals": no_vocals_path}

    if logger:
        logger.info(f"正在分离: {basename}")

    try:
        from demucs import pretrained
        from demucs.apply import apply_model
        from demucs.audio import convert_audio_channels

        # ── 1. 用 soundfile 读取音频（不需要 ffmpeg/torchaudio）──
        wav_np, sr = sf.read(audio_path)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)  # 降混到单声道
        wav_np = wav_np.astype(np.float32)

        # ── 2. 转为 torch tensor [channels, samples] ──
        wav = torch.from_numpy(wav_np).unsqueeze(0)  # [1, N]

        # ── 3. 重采样到 44.1kHz (Demucs 期望) ──
        if sr != 44100:
            from julius import resample_frac
            wav = resample_frac(wav, sr, 44100)
            sr = 44100

        # ── 4. 确保立体声 (htdemucs 是 4 源模型，期望 2 声道) ──
        wav = convert_audio_channels(wav, 2)

        # ── 5. 加载模型 ──
        model = pretrained.get_model(model_name)
        model.to(device)
        model.eval()

        # ── 6. 归一化 & 分离 ──
        ref = wav.mean(0, keepdim=True)
        wav_centered = wav - ref.mean()
        std_val = wav_centered.std()
        if std_val > 1e-8:
            wav_centered = wav_centered / std_val

        with torch.no_grad():
            sources = apply_model(
                model,
                wav_centered.unsqueeze(0),  # [1, ch, N]
                device=device,
                shifts=shifts,
                split=True,
                overlap=overlap,
                progress=True,
                segment=segment,
            )[0]

        # 反归一化
        if std_val > 1e-8:
            sources = sources * std_val
        sources = sources + ref

        # ── 7. 提取 vocals / no_vocals ──
        # htdemucs 顺序: drums, bass, other, vocals
        idx_vocals = model.sources.index("vocals")
        vocals = sources[idx_vocals].cpu()
        # no_vocals = drums + bass + other
        no_vocals = torch.zeros_like(vocals)
        for i, name in enumerate(model.sources):
            if i != idx_vocals:
                no_vocals += sources[i].cpu()

        # ── 8. 保存（使用 soundfile，避免 torchaudio 的 FFmpeg 依赖）──
        def _save_stem(tensor, path, sample_rate, clip="rescale"):
            """保存音频茎，带 clamp / rescale 保护"""
            import numpy as np
            arr = tensor.numpy()
            # 将多声道 [C, N] 转 [N, C] (soundfile 格式)
            if arr.ndim == 2:
                arr = arr.T
            # clip mode: rescale
            peak = np.max(np.abs(arr))
            if clip == "rescale" and peak > 1.0:
                arr = arr / peak
            elif clip == "clamp":
                arr = np.clip(arr, -1.0, 1.0)
            # 目标采样率 44100
            sf.write(path, arr, sample_rate, subtype="PCM_16")

        _save_stem(vocals, vocals_path, sr, clip="rescale")
        _save_stem(no_vocals, no_vocals_path, sr, clip="rescale")

        if logger:
            logger.info(f"✓ {basename}: vocals 已分离")

        return {"vocals": vocals_path, "no_vocals": no_vocals_path}

    except Exception as e:
        if logger:
            logger.error(f"Demucs 失败 [{basename}]: {e}", exc_info=True)
        return None


def separate_all(config: dict, logger=None):
    """批量运行 Demucs 分离"""
    paths = config["paths"]
    demucs_cfg = config["demucs"]

    if logger is None:
        logger = get_logger("03_demucs_separate")

    demucs_dir = ensure_dir(paths["demucs_output"])

    audio_files = get_audio_files(paths["extracted_audio"])
    if not audio_files:
        logger.warning("没有找到音频文件，请先运行 02_extract_audio.py")
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"=" * 60)
    logger.info(f"Stage 3: Demucs 人声分离 (Python API)")
    logger.info(f"模型: {demucs_cfg['model']} (--two-stems vocals)")
    logger.info(f"shifts={demucs_cfg['shifts']}, overlap={demucs_cfg['overlap']}, segment={demucs_cfg['segment']}")
    logger.info(f"设备: {device}")
    logger.info(f"共 {len(audio_files)} 个音频")
    logger.info(f"=" * 60)

    results = []

    for audio_path in tqdm(audio_files, desc="Demucs 分离"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]

        stems = run_demucs_separation(
            audio_path, demucs_dir,
            model_name=demucs_cfg["model"],
            shifts=demucs_cfg["shifts"],
            overlap=demucs_cfg["overlap"],
            segment=demucs_cfg["segment"],
            device=device,
            logger=logger,
        )

        if stems is None:
            logger.warning(f"✗ {basename}: 分离失败")
            continue

        results.append(stems["vocals"])
        logger.info(f"✓ {basename}")

    logger.info(f"完成: {len(results)}/{len(audio_files)} 个音频分离成功")
    return results


def main():
    config = load_config()
    from pipeline.utils import setup_logger
    logger = setup_logger("03_demucs_separate", config["paths"]["logs"])
    separate_all(config, logger)


if __name__ == "__main__":
    main()
