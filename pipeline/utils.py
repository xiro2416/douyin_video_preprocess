"""
共享工具模块：日志、文件 I/O、音频工具、断点续传
"""

import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa  # noqa: E402, loaded early to avoid speechbrain lazy-import conflict
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


# ── 日志 ──────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    """配置带文件和终端输出的 logger"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Windows 终端（Git Bash）通常用 UTF-8，但 Python 误检测为 GBK
    # 强制 stdout/stderr 使用 UTF-8，避免日志乱码和 UnicodeEncodeError
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, 'reconfigure'):
            try:
                _stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass  # 某些环境可能不支持 reconfigure

    if logger.handlers:
        return logger  # 避免重复添加

    # 文件 handler（每日轮转用文件名区分）
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d}.log"),
        encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%m-%d %H:%M:%S"
    ))

    # 终端 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ── 配置 ──────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """加载 YAML 配置"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 音频工具 ──────────────────────────────────────────────────────

def read_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    读取音频并重采样到 target_sr。
    返回 (wav, sr)，wav 形状为 (samples,) 的 float32 数组，值域 [-1, 1]。
    注意: 使用 scipy 而非 librosa 重采样，避免 speechbrain lazy-import 冲突。
    """
    from scipy import signal as _signal
    wav, orig_sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # 混音到 mono
    if orig_sr != target_sr:
        # 用 resample_poly 做带限重采样（等价于 librosa 的 quality）
        gcd = np.gcd(orig_sr, target_sr)
        wav = _signal.resample_poly(wav, target_sr // gcd, orig_sr // gcd)
    return wav.astype(np.float32), target_sr


def write_audio(path: str, wav: np.ndarray, sr: int):
    """写入 WAV 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sr)


def compute_snr(wav: np.ndarray, noise_floor: Optional[np.ndarray] = None) -> float:
    """估算信噪比 (dB)"""
    if noise_floor is None:
        # 用最低 5% 能量帧估算噪声
        frame_len = 512
        frames = librosa.util.frame(wav, frame_length=frame_len, hop_length=frame_len).T
        energies = np.sum(frames ** 2, axis=0)
        threshold = np.percentile(energies, 5)
        noise_mask = energies <= threshold
        signal_mask = energies > threshold
        noise_power = np.mean(energies[noise_mask]) if noise_mask.any() else 1e-12
        signal_power = np.mean(energies[signal_mask]) if signal_mask.any() else 1e-12
    else:
        signal_power = np.mean(wav ** 2)
        noise_power = np.mean(noise_floor ** 2)

    snr = 10 * np.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12))
    return snr


def compute_loudness(wav: np.ndarray, sr: int) -> float:
    """使用 pyloudnorm 计算响度 (LKFS / LUFS)"""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        return meter.integrated_loudness(wav)
    except ImportError:
        # fallback: RMS 近似
        rms = np.sqrt(np.mean(wav ** 2))
        return 20 * np.log10(max(rms, 1e-12))


def normalize_loudness(wav: np.ndarray, sr: int, target_db: float = -26.0) -> np.ndarray:
    """将音频响度标准化到 target_db (LUFS)"""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        current = meter.integrated_loudness(wav)
        return pyln.normalize.loudness(wav, current, target_db)
    except ImportError:
        # fallback: peak normalization to -3dB then scale
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak * 0.707  # -3dB
        rms = np.sqrt(np.mean(wav ** 2))
        if rms > 1e-12:
            gain = 10 ** ((target_db - 20 * np.log10(rms)) / 20)
            wav = wav * gain
        return wav


def detect_clipping(wav: np.ndarray, threshold: float = 0.99) -> float:
    """检测削波比例"""
    clipped = np.sum(np.abs(wav) > threshold)
    return clipped / len(wav)


def detect_dc_offset(wav: np.ndarray) -> float:
    """检测直流偏移"""
    return float(np.mean(wav))


def trim_silence(
    wav: np.ndarray,
    sr: int,
    top_db: float = 40,
    frame_length: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """修剪首尾静音"""
    trimmed, _ = librosa.effects.trim(
        wav, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return trimmed


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样"""
    if orig_sr == target_sr:
        return wav
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)

# ── 文件工具 ──────────────────────────────────────────────────────

def get_video_files(directory: str, extensions=(".mp4", ".mov", ".mkv", ".webm")) -> List[str]:
    """递归获取目录下所有视频文件"""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(str(f) for f in files)


def get_audio_files(directory: str, extensions=(".wav", ".mp3", ".flac", ".m4a")) -> List[str]:
    """递归获取目录下所有音频文件"""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(str(f) for f in files)


def ensure_dir(path: str) -> str:
    """确保目录存在并返回路径"""
    os.makedirs(path, exist_ok=True)
    return path


def safe_filename(s: str) -> str:
    """移除文件名中的非法字符"""
    return "".join(c for c in s if c.isalnum() or c in "._- ").strip()
