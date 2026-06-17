"""
Stage 3: Ensemble vocal separation — Demucs + MDX23C (STFT soft-mask averaging)
===============================================================================

将 Demucs (htdemucs_ft_vocals, 时域 U-Net) 和 MDX23C (频域 MDX, ONNX) 两个
架构互补的模型的分离结果在 STFT 域用软掩码平均融合，产出更干净、更自然的 vocals。

Ensemble 策略 (双分辨率 STFT soft-mask averaging):
  M_d = |V_d|² / (|V_d|² + |N_d|² + ε)      ← Demucs 的 Wiener 掩码 @ n_fft=4096
  M_m = |V_m|² / (|V_m|² + |N_m|² + ε)      ← MDX23C 的 Wiener 掩码 @ n_fft=8192
  M_d↑ = upsample(M_d)                       ← 上采样 Demucs 掩码到 8192 频域 bins
  M_e = (w_d·M_d↑ + w_m·M_m) / (w_d + w_m)  ← 在 8192-bin 分辨率下加权平均
  V_e = ISTFT(M_e · STFT{mixture})           ← 用原始混合相位重建
  N_e = mixture - V_e                         ← 完美重建

双分辨率设计使每个模型在其原生 STFT 分辨率下计算掩码，避免分辨率不匹配
导致的频率选择性损失，然后在高分辨率下融合。

输出格式与现有 stage 03 完全一致:
  separate_all(config, logger)
  data/03_ensemble_output/{model_short}/{basename}/{vocals.wav, no_vocals.wav}
"""

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    get_audio_files,
    get_logger,
    load_config,
)

# ── 内部辅助：加载数字前缀模块 ──


def _load_stage_module(module_filename: str):
    """
    用 importlib 加载 pipeline 下带数字前缀的模块
    (e.g. "03_demucs_separate.py" → module object)
    """
    pipeline_dir = Path(__file__).resolve().parent
    module_path = pipeline_dir / module_filename
    module_name = module_filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 模块级缓存：加载后的 stage 03 模块 ──
# 用 importlib 加载 (数字前缀不能直接 import)
_DEMUCS_MODULE = None
_MDX_MODULE = None


def _get_demucs_module():
    global _DEMUCS_MODULE
    if _DEMUCS_MODULE is None:
        _DEMUCS_MODULE = _load_stage_module("03_demucs_separate.py")
    return _DEMUCS_MODULE


def _get_mdx_module():
    global _MDX_MODULE
    if _MDX_MODULE is None:
        _MDX_MODULE = _load_stage_module("03_mdx23c_separate.py")
    return _MDX_MODULE


# ── 用于加权平均可配置检查 ──
_ENSEMBLE_EPSILON = 1e-8


def ensemble_stems(
    vocals_a: np.ndarray,
    no_vocals_a: np.ndarray,
    vocals_b: np.ndarray,
    no_vocals_b: np.ndarray,
    mixture: np.ndarray,
    sr: int,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    n_fft_a: int = 4096,
    n_fft_b: int = 8192,
    n_fft_fusion: int = 8192,
    hop_length: int = 1024,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    STFT 域软掩码平均融合两个模型的分离结果（双分辨率）。

    每个模型在自己的原生 STFT 分辨率下计算 Wiener 掩码：
      - 模型 A (Demucs) 在 n_fft_a 分辨率下分析
      - 模型 B (MDX23C) 在 n_fft_b 分辨率下分析
    然后将掩码上采样到 n_fft_fusion 分辨率进行融合，保留原始混合相位。

    Args:
        vocals_a: 模型 A 的 vocals 波形 [samples] 或 [channels, samples]
        no_vocals_a: 模型 A 的 no_vocals 波形
        vocals_b: 模型 B 的 vocals 波形
        no_vocals_b: 模型 B 的 no_vocals 波形
        mixture: 原始混合波形
        sr: 采样率
        weight_a: 模型 A 的融合权重
        weight_b: 模型 B 的融合权重
        n_fft_a: 模型 A 的 STFT 窗口大小
        n_fft_b: 模型 B 的 STFT 窗口大小
        n_fft_fusion: 融合掩码的 STFT 窗口大小（取高分辨率）
        hop_length: STFT 步长（各模型共享）
        device: 计算设备

    Returns:
        (vocals_ensembled, no_vocals_ensembled) 波形
    """
    # ── 统一 shape 为 [channels, samples] ──
    def _to_chan_first(wav: np.ndarray) -> np.ndarray:
        if wav.ndim == 1:
            return wav[np.newaxis, :]  # [1, N]
        return wav  # already [C, N]

    vocals_a = _to_chan_first(vocals_a)
    no_vocals_a = _to_chan_first(no_vocals_a)
    vocals_b = _to_chan_first(vocals_b)
    no_vocals_b = _to_chan_first(no_vocals_b)
    mixture = _to_chan_first(mixture)

    # ── 统一长度（取所有信号的最短长度，截断到短半帧以避免 ISTFT 边界问题）──
    min_len = min(
        vocals_a.shape[-1],
        no_vocals_a.shape[-1],
        vocals_b.shape[-1],
        no_vocals_b.shape[-1],
        mixture.shape[-1],
    )
    # 对齐到 hop_length 的整数倍（避免 ISTFT 长度不匹配）
    aligned_len = (min_len // hop_length) * hop_length
    if aligned_len < hop_length:
        # 太短，直接返回其中一种模型的结果
        return (vocals_a[:, :min_len], no_vocals_a[:, :min_len])

    vocals_a = vocals_a[:, :aligned_len]
    no_vocals_a = no_vocals_a[:, :aligned_len]
    vocals_b = vocals_b[:, :aligned_len]
    no_vocals_b = no_vocals_b[:, :aligned_len]
    mixture = mixture[:, :aligned_len]

    # ── 统一通道数（以 mixture 为准，广播）──
    n_chan = mixture.shape[0]
    if vocals_a.shape[0] < n_chan:
        vocals_a = np.broadcast_to(vocals_a, (n_chan, vocals_a.shape[1]))
        no_vocals_a = np.broadcast_to(no_vocals_a, (n_chan, no_vocals_a.shape[1]))
    if vocals_b.shape[0] < n_chan:
        vocals_b = np.broadcast_to(vocals_b, (n_chan, vocals_b.shape[1]))
        no_vocals_b = np.broadcast_to(no_vocals_b, (n_chan, no_vocals_b.shape[1]))

    # ── 转为 torch tensor ──
    def _to_torch(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).to(device)

    v_a = _to_torch(vocals_a)
    n_a = _to_torch(no_vocals_a)
    v_b = _to_torch(vocals_b)
    n_b = _to_torch(no_vocals_b)
    mix = _to_torch(mixture)

    # ── 各模型在各自原生分辨率下计算 Wiener 掩码 ──
    def _soft_mask(vocals_stft, no_vocals_stft):
        v_mag2 = torch.abs(vocals_stft) ** 2
        n_mag2 = torch.abs(no_vocals_stft) ** 2
        return v_mag2 / (v_mag2 + n_mag2 + _ENSEMBLE_EPSILON)

    def _stft_at(x, n_fft):
        w = torch.hann_window(n_fft, device=device)
        return torch.stft(x, n_fft, hop_length, window=w, return_complex=True)

    # 模型 B (MDX23C)：在 n_fft_b 分辨率下分析
    M_b = _soft_mask(_stft_at(v_b, n_fft_b), _stft_at(n_b, n_fft_b))

    # 模型 A (Demucs)：在 n_fft_a 分辨率下分析
    M_a = _soft_mask(_stft_at(v_a, n_fft_a), _stft_at(n_a, n_fft_a))

    # ── 上采样 Demucs 掩码到融合分辨率 ──
    target_bins = n_fft_fusion // 2 + 1
    if M_a.shape[1] != target_bins:
        # M_a: [C, F_a, T] → bilinear 上采样频域 → [C, target_bins, T]
        M_a = F.interpolate(
            M_a.unsqueeze(0),
            size=(target_bins, M_a.shape[2]),
            mode='bilinear',
            align_corners=False,
        )[0]

    # ── 在融合分辨率下加权平均 ──
    total_weight = weight_a + weight_b
    M_ens = (weight_a * M_a + weight_b * M_b) / total_weight

    # ── 混合 STFT 用融合分辨率，保留原始相位 ──
    window_fusion = torch.hann_window(n_fft_fusion, device=device)
    X = torch.stft(mix, n_fft_fusion, hop_length, window=window_fusion, return_complex=True)
    V_ens = M_ens * X

    # ── ISTFT ──
    v_ens = torch.istft(V_ens, n_fft_fusion, hop_length, window=window_fusion, length=aligned_len)

    # ── 减法得到 no_vocals（完美重建）──
    n_ens = mix[:, :aligned_len] - v_ens

    # ── 转回 numpy [C, N] ──
    v_out = v_ens.cpu().numpy()
    n_out = n_ens.cpu().numpy()

    # ── 如果输入是 mono，降回 [N] ──
    if vocals_a.shape[0] == 1 and mixture.shape[0] == 1:
        v_out = v_out[0]
        n_out = n_out[0]

    return (v_out, n_out)


def _save_stem(tensor: np.ndarray, path: str, sample_rate: int):
    """保存音频，带峰值 clip 保护"""
    if tensor.ndim == 2:
        tensor = tensor.T  # [channels, N] → [N, channels] for soundfile
    peak = np.max(np.abs(tensor))
    if peak > 1.0:
        tensor = tensor / peak
    sf.write(path, tensor, sample_rate, subtype="PCM_16")


def run_ensemble_separation(
    audio_path: str,
    demucs_dir: str,
    mdx_dir: str,
    ensemble_dir: str,
    demucs_model,
    mdx_separator,
    config: dict,
    logger,
) -> dict | None:
    """
    对单个音频运行 ensemble 分离。

    Flow:
      1. 运行 Demucs（跳过已处理）→ 返回路径
      2. 运行 MDX23C（跳过已处理）→ 返回路径
      3. 如果 ensemble 输出已存在 → 跳过
      4. 如果任一模型失败 → fallback 到另一模型
      5. 两个模型都成功 → STFT 软掩码 ensemble
      6. 写入 ensemble 输出目录

    Returns:
        {"vocals": str, "no_vocals": str} | None
    """
    demucs_cfg = config.get("demucs", {})
    mdx23c_cfg = config.get("mdx23c", {})
    ensemble_cfg = config.get("ensemble", {})

    model_name = ensemble_cfg.get("model_name", "demucs+mdx23c_ens")
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(ensemble_dir, model_name, basename)
    os.makedirs(stem_dir, exist_ok=True)
    vocals_path = os.path.join(stem_dir, "vocals.wav")
    no_vocals_path = os.path.join(stem_dir, "no_vocals.wav")

    # ── 检查 ensemble 是否已处理 ──
    if os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
        if logger:
            logger.debug(f"Ensemble 已处理，跳过: {basename}")
        return {"vocals": vocals_path, "no_vocals": no_vocals_path}

    if logger:
        logger.info(f"正在 ensemble 分离: {basename}")

    try:
        # ── 1. 运行 Demucs ──
        _demucs_mod = _get_demucs_module()
        demucs_result = _demucs_mod.run_demucs_separation(
            audio_path,
            demucs_dir,
            model=demucs_model,
            model_name=demucs_cfg.get("model", "htdemucs"),
            shifts=demucs_cfg.get("shifts", 10),
            overlap=demucs_cfg.get("overlap", 0.25),
            segment=demucs_cfg.get("segment", 7),
            device=next(demucs_model.parameters()).device,
            logger=logger,
        )

        # ── 2. 运行 MDX23C ──
        mdx_result = None
        if mdx_separator is not None:
            _mdx_mod = _get_mdx_module()
            mdx_result = _mdx_mod.run_mdx23c_separation(
                audio_path,
                mdx_dir,
                separator=mdx_separator,
                model_name=mdx23c_cfg.get("model", "MDX23C-8KFFT-InstVoc_HQ_2.ckpt"),
                logger=logger,
            )
        else:
            if logger:
                logger.warning("MDX23C separator 为 None，跳过 MDX23C")

        # ── 3. Fallback 处理 ──
        if demucs_result is None and mdx_result is None:
            if logger:
                logger.error(f"Demucs 和 MDX23C 都失败: {basename}")
            return None

        if demucs_result is None:
            if logger:
                logger.warning(f"Demucs 失败，使用 MDX23C 结果: {basename}")
            shutil.copy(mdx_result["vocals"], vocals_path)
            shutil.copy(mdx_result["no_vocals"], no_vocals_path)
            return {"vocals": vocals_path, "no_vocals": no_vocals_path}

        if mdx_result is None:
            if logger:
                logger.warning(f"MDX23C 失败，使用 Demucs 结果: {basename}")
            shutil.copy(demucs_result["vocals"], vocals_path)
            shutil.copy(demucs_result["no_vocals"], no_vocals_path)
            return {"vocals": vocals_path, "no_vocals": no_vocals_path}

        # ── 4. 两个模型都成功 → 做 ensemble ──

        # 读取两个模型的输出
        v_d, sr_d = sf.read(demucs_result["vocals"])
        n_d, _ = sf.read(demucs_result["no_vocals"])
        v_m, sr_m = sf.read(mdx_result["vocals"])
        n_m, _ = sf.read(mdx_result["no_vocals"])
        mixture, sr_x = sf.read(audio_path)

        # ── 采样率统一到目标 SR ──
        target_sr = ensemble_cfg.get("target_sr", 44100)

        def _resample_if_needed(wav, orig_sr, tgt_sr):
            if orig_sr == tgt_sr:
                return wav
            from scipy import signal
            # sf.read → shape 为 [samples] 或 [samples, channels]
            new_len = int(wav.shape[0] * tgt_sr / orig_sr)
            return signal.resample(wav, new_len, axis=0)

        v_d = _resample_if_needed(v_d, sr_d, target_sr)
        n_d = _resample_if_needed(n_d, sr_d, target_sr)
        v_m = _resample_if_needed(v_m, sr_m, target_sr)
        n_m = _resample_if_needed(n_m, sr_m, target_sr)
        mixture = _resample_if_needed(mixture, sr_x, target_sr)

        # ── 转为 [channels, samples] 或 [samples] ──
        def _transpose(*args):
            result = []
            for arr in args:
                # [samples, channels] → [channels, samples]
                if arr.ndim == 2 and arr.shape[1] <= 8:
                    result.append(arr.T)
                else:
                    result.append(arr)
            return result

        v_d, n_d, v_m, n_m, mixture = _transpose(v_d, n_d, v_m, n_m, mixture)

        # ── 做 ensemble ──
        gpu_cfg = config.get("gpu", {})
        device_id = gpu_cfg.get("device_id", 0)
        use_cuda = gpu_cfg.get("enabled", True) and torch.cuda.is_available()
        device = f"cuda:{device_id}" if use_cuda and torch.cuda.device_count() > device_id else "cuda" if use_cuda else "cpu"

        w_a = ensemble_cfg.get("demucs_weight", 0.5)
        w_b = ensemble_cfg.get("mdx23c_weight", 0.5)
        n_fft_a = ensemble_cfg.get("n_fft_demucs", 4096)
        n_fft_b = ensemble_cfg.get("n_fft_mdx23c", 8192)
        n_fft_fusion = ensemble_cfg.get("n_fft_fusion", 8192)
        hop_len = ensemble_cfg.get("hop_length", 1024)

        vocals_ens, no_vocals_ens = ensemble_stems(
            v_d, n_d,
            v_m, n_m,
            mixture,
            target_sr,
            weight_a=w_a,
            weight_b=w_b,
            n_fft_a=n_fft_a,
            n_fft_b=n_fft_b,
            n_fft_fusion=n_fft_fusion,
            hop_length=hop_len,
            device=device,
        )

        # ── 保存 ──
        _save_stem(vocals_ens, vocals_path, target_sr)
        _save_stem(no_vocals_ens, no_vocals_path, target_sr)

        if logger:
            logger.info(f"✓ {basename}: ensemble vocals 已保存")

        return {"vocals": vocals_path, "no_vocals": no_vocals_path}

    except Exception as e:
        if logger:
            logger.error(f"Ensemble 失败 [{basename}]: {e}", exc_info=True)
        return None


def separate_all(config: dict, logger=None):
    """
    批量运行 Demucs + MDX23C ensemble 分离。

    入口签名与旧版 03 模块完全一致：separate_all(config, logger)
    """
    paths = config["paths"]
    demucs_cfg = config.get("demucs", {})
    mdx23c_cfg = config.get("mdx23c", {})
    ensemble_cfg = config.get("ensemble", {})

    if not ensemble_cfg.get("enabled", False):
        if logger:
            logger.warning("Ensemble 模式未启用 (config.ensemble.enabled=false)，跳过")
        return []

    if logger is None:
        logger = get_logger("03_ensemble_separate")

    # ── 输出目录 ──
    demucs_dir = ensure_dir(paths.get("demucs_output", "./data/03_demucs_output"))
    mdx_dir = ensure_dir(paths.get("voice_sep_output", "./data/03_mdx23c_output"))
    ensemble_dir = ensure_dir(
        ensemble_cfg.get("output_dir", paths.get("ensemble_output", "./data/03_ensemble_output"))
    )

    # ── 输入发现 ──
    audio_files = get_audio_files(paths["extracted_audio"])
    if not audio_files:
        raise RuntimeError(
            "Stage 03 Ensemble 错误：没有找到音频文件。请先运行 Stage 02 (音频提取)。"
        )

    # ── GPU 设备选择 ──
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

    model_name = ensemble_cfg.get("model_name", "demucs+mdx23c_ens")

    logger.info("=" * 60)
    logger.info("Stage 3: Ensemble 人声分离 (Demucs + MDX23C)")
    logger.info(f"模型融合: {model_name}")
    logger.info(f"权重: Demucs={ensemble_cfg.get('demucs_weight', 0.5)}, "
                f"MDX23C={ensemble_cfg.get('mdx23c_weight', 0.5)}")
    logger.info(f"STFT: Demucs@{ensemble_cfg.get('n_fft_demucs', 4096)}, "
                f"MDX23C@{ensemble_cfg.get('n_fft_mdx23c', 8192)}, "
                f"Fusion@{ensemble_cfg.get('n_fft_fusion', 8192)}, "
                f"hop={ensemble_cfg.get('hop_length', 1024)}")
    logger.info(f"设备: {device}")
    logger.info(f"共 {len(audio_files)} 个音频")
    logger.info("=" * 60)

    # ── 加载 Demucs 模型 ──
    logger.info("加载 Demucs 模型...")
    _demucs_mod = _get_demucs_module()
    demucs_model = _demucs_mod._load_model(demucs_cfg, device)
    logger.info(f"Demucs 模型已加载: {type(demucs_model).__name__}")

    # ── 加载 MDX23C 模型（audio_separator Separator）──
    import logging as _logging
    from audio_separator.separator import Separator

    logger.info("加载 MDX23C 模型 (ONNX)...")
    _mdx_mod = _get_mdx_module()
    local_model_dir = _mdx_mod._ensure_model_local(mdx23c_cfg, logger)
    os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = str(local_model_dir)
    _logging.getLogger("audio_separator").setLevel(_logging.WARNING)

    mdx_model_name = mdx23c_cfg.get("model", "MDX23C-8KFFT-InstVoc_HQ_2.ckpt")
    mdx_separator = Separator(
        output_dir=mdx_dir,
        model_file_dir=str(local_model_dir),
        log_level=_logging.WARNING,
        output_format="WAV",
        normalization_threshold=0.9,
        mdxc_params={
            "segment_size": mdx23c_cfg.get("segment", 256),
            "overlap": mdx23c_cfg.get("overlap", 8),
            "batch_size": 1,
        },
    )
    try:
        mdx_separator.load_model(model_filename=mdx_model_name)
    except Exception as e:
        logger.error(f"MDX23C 模型加载失败: {e}")
        logger.warning("使用 pure-Demucs fallback (无 MDX23C)")
        mdx_separator = None

    logger.info("MDX23C 模型已加载" if mdx_separator else "MDX23C 模型不可用，使用 Demucs-only fallback")

    # ── 主循环 ──
    results = []

    for audio_path in tqdm(audio_files, desc="Ensemble 分离"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]

        stems = run_ensemble_separation(
            audio_path,
            demucs_dir,
            mdx_dir,
            ensemble_dir,
            demucs_model=demucs_model,
            mdx_separator=mdx_separator,
            config=config,
            logger=logger,
        )

        if stems is None:
            logger.warning(f"✗ {basename}: ensemble 分离失败")
            continue

        results.append(stems["vocals"])
        logger.info(f"✓ {basename}")

    logger.info(f"完成: {len(results)}/{len(audio_files)} 个音频 ensemble 分离成功")

    if len(results) == 0 and len(audio_files) > 0:
        raise RuntimeError(
            f"Stage 03 Ensemble 错误：所有 {len(audio_files)} 个音频分离均失败。"
            f" 请检查模型文件或 GPU 显存。"
        )

    return results


def main():
    config = load_config()
    from pipeline.utils import setup_logger

    logger = setup_logger("03_ensemble_separate", config["paths"]["logs"])
    separate_all(config, logger)


if __name__ == "__main__":
    main()
