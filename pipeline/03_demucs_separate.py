"""
Stage 3: Demucs 人声分离 (Python API — no ffmpeg needed)
==========================================================
使用 Demucs Python API 直接加载，绕过 torchaudio / ffmpeg 依赖。
使用 soundfile 读取音频，numpy → tensor 传给 Demucs 推理。

首次运行需要模型文件。本地仓库位于 pretrained_models/，
优先使用本地模型（无需联网），fallback 到远程下载。
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# ── PyTorch 2.6+ 兼容：注册 demucs 自定义类到 safe globals ──
# PyTorch 2.6 开始 torch.load 默认 weights_only=True，
# 而 demucs .th 文件包含序列化的模型类，需要注册才能加载。
import demucs.hdemucs as _hd
import demucs.htdemucs as _ht

_torch_classes = [
    _hd.HDemucs, _hd.HEncLayer, _hd.HDecLayer, _hd.ScaledEmbedding, _hd.MultiWrap,
    _ht.HTDemucs,
]
try:
    from demucs.demucs import DConv
    _torch_classes.append(DConv)
except ImportError:
    pass
try:
    from demucs.transformer import CrossTransformerEncoder
    _torch_classes.append(CrossTransformerEncoder)
except ImportError:
    pass
torch.serialization.add_safe_globals(_torch_classes)

from pipeline.utils import (
    ensure_dir,
    get_audio_files,
    get_logger,
    load_config,
)

# 本地模型仓库（项目根目录下的 pretrained_models/）
_LOCAL_MODEL_REPO = Path(__file__).resolve().parent.parent / "pretrained_models"


def _load_model(demucs_cfg: dict, device: str):
    """加载 Demucs 模型，优先本地仓库，fallback 远程下载。"""
    # PyTorch 2.6+ 默认 weights_only=True，而 demucs 库内部
    # (states.py) 调用 torch.load 未传此参数，需要临时覆盖。
    _orig_torch_load = torch.load
    def _demucs_compat_load(path, map_location='cpu'):
        return _orig_torch_load(path, map_location=map_location, weights_only=False)

    from demucs import pretrained

    model_name = demucs_cfg.get("model", "htdemucs")
    repo_path = demucs_cfg.get("model_repo", "").strip()

    if repo_path:
        repo = Path(repo_path)
    elif _LOCAL_MODEL_REPO.is_dir() and any(_LOCAL_MODEL_REPO.glob("*.th")):
        repo = _LOCAL_MODEL_REPO
    else:
        repo = None

    # ── 临时 monkey-patch torch.load 以兼容 PyTorch 2.6+ ──
    torch.load = _demucs_compat_load

    if repo:
        try:
            model = pretrained.get_model(model_name, repo=repo)
        except Exception as e:
            get_logger("03_demucs_separate").warning(
                f"本地模型加载失败: {e}，回退到远程下载..."
            )
            model = pretrained.get_model(model_name)
    else:
        model = pretrained.get_model(model_name)

    # ── 恢复原始 torch.load ──
    torch.load = _orig_torch_load

    model.to(device)
    model.eval()
    return model


def run_demucs_separation(
    audio_path: str,
    output_dir: str,
    model,
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
        from demucs.apply import apply_model
        from demucs.audio import convert_audio_channels

        # ── 1. 用 soundfile 读取音频（不需要 ffmpeg/torchaudio）──
        wav_np, sr = sf.read(audio_path)
        wav_np = wav_np.astype(np.float32)

        # ── 2. 转为 torch tensor [channels, samples] ──
        wav = torch.from_numpy(wav_np).unsqueeze(0)  # [1, N]

        # ── 3. 重采样到 44.1kHz (Demucs 期望) ──
        if sr != 44100:
            from julius import resample_frac
            wav = resample_frac(wav, sr, 44100)
            sr = 44100

        # ── 4. 确保立体声 ──
        wav = convert_audio_channels(wav, 2)

        # ── 5. 归一化 & 分离 ──
        mean = wav.mean().item()
        std = wav.std().item()
        if std > 1e-8:
            wav = (wav - mean) / std

        with torch.no_grad():
            sources = apply_model(
                model,
                wav.unsqueeze(0),  # [1, ch, N]
                device=device,
                shifts=shifts,
                split=True,
                overlap=overlap,
                progress=True,
                segment=segment,
            )[0]

        # 反归一化（加标量，匹配官方实现）
        if std > 1e-8:
            sources = sources * std + mean

        # ── 6. 提取 vocals / no_vocals ──
        sources_list = getattr(model, 'sources', ['drums', 'bass', 'other', 'vocals'])
        idx_vocals = sources_list.index("vocals")
        vocals = sources[idx_vocals].cpu()
        # no_vocals = 其他茎之和
        no_vocals = torch.zeros_like(vocals)
        for i, name in enumerate(sources_list):
            if i != idx_vocals:
                no_vocals += sources[i].cpu()

        # ── 7. 保存（使用 soundfile，避免 torchaudio 的 FFmpeg 依赖）──
        def _save_stem(tensor, path, sample_rate, clip="rescale"):
            """保存音频茎，带 clamp / rescale 保护"""
            arr = tensor.numpy()
            if arr.ndim == 2:
                arr = arr.T
            peak = np.max(np.abs(arr))
            if clip == "rescale" and peak > 1.0:
                arr = arr / peak
            elif clip == "clamp":
                arr = np.clip(arr, -1.0, 1.0)
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
    """批量运行 Demucs 分离（零输入或全部失败时抛异常终止流水线）"""
    paths = config["paths"]
    demucs_cfg = config["demucs"]

    if logger is None:
        logger = get_logger("03_demucs_separate")

    demucs_dir = ensure_dir(paths["demucs_output"])

    audio_files = get_audio_files(paths["extracted_audio"])
    if not audio_files:
        raise RuntimeError(
            "Stage 03 错误：没有找到音频文件。请先运行 Stage 02 (音频提取)。"
        )

    # ── GPU 设备选择（优先全局配置）──
    gpu_cfg = config.get("gpu", {})
    if not gpu_cfg.get("enabled", True):
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        cvd = gpu_cfg.get("cuda_visible", "").strip()
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        device_id = gpu_cfg.get("device_id", 0)
        if torch.cuda.is_available():
            device = f"cuda:{device_id}" if torch.cuda.device_count() > device_id else "cuda"
        else:
            device = "cpu"

    logger.info(f"=" * 60)
    logger.info(f"Stage 3: Demucs 人声分离 (Python API)")
    model_name = demucs_cfg.get("model", "htdemucs")
    logger.info(f"模型: {model_name}")
    logger.info(f"本地仓库: {_LOCAL_MODEL_REPO if _LOCAL_MODEL_REPO.is_dir() else '远程下载'}")
    logger.info(f"shifts={demucs_cfg.get('shifts', 6)}, overlap={demucs_cfg.get('overlap', 0.25)}, segment={demucs_cfg.get('segment', 7)}")
    logger.info(f"设备: {device}")
    logger.info(f"共 {len(audio_files)} 个音频")
    logger.info(f"=" * 60)

    # ── 加载模型（单例，所有文件复用）──
    logger.info("加载 Demucs 模型...")
    model = _load_model(demucs_cfg, device)
    logger.info(f"模型已加载: {type(model).__name__}")

    results = []

    for audio_path in tqdm(audio_files, desc="Demucs 分离"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]

        stems = run_demucs_separation(
            audio_path, demucs_dir,
            model=model,
            model_name=model_name,
            shifts=demucs_cfg.get("shifts", 6),
            overlap=demucs_cfg.get("overlap", 0.25),
            segment=demucs_cfg.get("segment", 7),
            device=device,
            logger=logger,
        )

        if stems is None:
            logger.warning(f"✗ {basename}: 分离失败")
            continue

        results.append(stems["vocals"])
        logger.info(f"✓ {basename}")

    logger.info(f"完成: {len(results)}/{len(audio_files)} 个音频分离成功")

    if len(results) == 0 and len(audio_files) > 0:
        raise RuntimeError(
            f"Stage 03 错误：所有 {len(audio_files)} 个音频分离均失败。"
            f" 请检查模型文件是否完整（pretrained_models/ 目录）或 GPU 显存是否充足。"
        )

    return results


def main():
    config = load_config()
    from pipeline.utils import setup_logger
    logger = setup_logger("03_demucs_separate", config["paths"]["logs"])
    separate_all(config, logger)


if __name__ == "__main__":
    main()
