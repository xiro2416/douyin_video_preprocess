"""
Stage 04v2: 跨视频说话人分离 (pyannote community-1)
==================================================
exclude_overlap:
============================================================================
社区版 1 号的管道结构 (Pipeline Internals)
============================================================================

pyannote community-1 的 SpeakerDiarization 管道分为 4 个阶段：

  音频 → [1. 分割] → [2. 说话人计数] → [3. 嵌入提取] → [4. 聚类] → 分离结果

阶段 1：分割 (Segmentation) ────────────────────────────────────────────
  - 模型: Powerset 多类分割模型 (Plaquet23)
  - 输入: 16kHz 波形
  - 输出: (num_chunks, num_frames, local_num_speakers) 原始分数矩阵
  - 使用滑动窗口推理，窗口滑动步长 = segmentation_step × 窗口时长
  - 因为使用 powerset 模式，该模型直接预测多说话人联合概率，
    因此不存在传统的 threshold 二值化参数 (不同于 binary 分割模型)

阶段 2：说话人计数 (Speaker Counting) ──────────────────────────────────
  - 将分割输出合并为逐帧的活跃说话人数
  - 使用 warm-up 修剪: 丢弃每句段两端 10% 的帧以消除边界伪影
  - 对计数结果 cap 到 max_speakers，避免分割错误导致过度计数

阶段 3：嵌入提取 (Speaker Embedding) ───────────────────────────────────
  - 模型: Wespeaker 声纹嵌入模型 (Wang2023)
  - 为每个 (chunk, local_speaker) 对提取 d 维嵌入向量
  - embedding_exclude_overlap: 控制是否排除重叠语音区域
  - embedding_batch_size: 控制 GPU 推理的批大小

阶段 4：聚类 —— VBxClustering (默认) ────────────────────────────────
  两步算法:

  4a. AHC (凝聚式层次聚类)
    - 距离度量: cosine 距离
    - 链接准则: centroid linkage
    - threshold (默认 0.6, 范围 [0.0, 2.0]): cosine 距离阈值
      ↓ 更小 → 更激进合并 → 更少说话人
      ↑ 更大 → 更保守合并 → 更多说话人

  4b. VBx (变分贝叶斯 x-vector 聚类)
    - 以 AHC 结果为初始化，使用 PLDA 似然进行变分贝叶斯迭代精化
    - Fa (默认 0.07, 范围 [0.01, 0.5]): 充分统计量缩放因子
      ↓ 更小 → 声学似然权重降低 → 正则化增强 → 更少说话人
      ↑ 更大 → 声学似然权重增加 → 更多说话人
    - Fb (默认 0.8, 范围 [0.01, 15.0]): 说话人正则化系数
      ↓ 更小 → 允许更多说话人
      ↑ 更大 → 对新说话人的先验惩罚更强 → 更少说话人
    - maxIters (固定 = 20): VBx EM 最大迭代次数，通常 10 步内收敛

  最终说话人数目由 VBx 自动确定 (冗余说话人先验收敛到零)。
  如果自动数量 < min_speakers 或 > max_speakers，则回退到 KMeans。

后处理：min_duration_off ──────────────────────────────────────────────
  - segmentation.min_duration_off (默认 0.0): 同一说话人间隔 < 此值则合并
  - 效果: 消除同一说话人段内的短静音片段 (如说话停顿)

============================================================================
参数敏感性指南 (Parameter Sensitivity Guide)
============================================================================

以下参数按影响程度从大到小排列：

  1. min_speakers / max_speakers (直接约束)
     → 直接影响最终说话人数目。设置过窄会导致检测数量超出范围而回退到 KMeans。

  2. clustering.threshold (社区 1 号默认 0.6)
     → 控制 AHC 粗聚类的合并粒度。对最终说话人数影响显著。
     → 对于单说话人视频: 建议 0.5 (更激进合并)
     → 对于多人对话: 建议 0.6-0.7 (保守合并)
     → 范围 [0.0, 2.0] 因为使用的是 L2 归一化嵌入，cosine 距离 ∈ [0, 2]

  3. clustering.Fa (社区 1 号默认 0.07) & Fb (社区 1 号默认 0.8)
     → VBx 精化阶段的核心参数。Fa 在 0.05-0.10 范围有效，Fb 在 0.5-5.0 范围有效。
     → Fa/Fb 比值 (Fa/Fb) 控制有效说话人数:
       - 小比值 (< 0.05) → 更少说话人
       - 大比值 (> 0.2) → 更多说话人
     → 经验建议:
       - 清晰录音 (高 SNR): Fa=0.07, Fb=0.8 (默认)
       - 嘈杂录音 (低 SNR): Fa=0.05, Fb=1.5 (减少噪声音色导致的说话人分裂)
       - 快速变化的对话: Fa=0.10, Fb=0.5 (允许更多说话人快速切换)

  4. segmentation.min_duration_off (社区 1 号默认 0.0)
     → 合并同一说话人的短间隔。对片段总数影响大，对说话人数影响小。
     → 建议 0.3-0.5 秒，平衡片段数量和边界精度

  5. segmentation_step (固定 = 0.1)
     → 推理速度与精度的权衡。该参数在管道构建时固定，需重建管道才能修改。
     → 0.1 = 90% 重叠 (高精度, 慢)
     → 0.25 = 75% 重叠 (平衡)
     → 0.5 = 50% 重叠 (快速, 可能遗漏短句)

  6. min_segment_duration (后处理过滤，非管道参数)
     → 过滤过短的语音段。0.5 秒通常足够去除噪声片段。

============================================================================
单视频内说话人聚类 vs 跨视频说话人聚类
============================================================================

单视频内聚类 (VBxClustering, pipeline 内置):
  - 作用: 在单个音频文件内部区分不同说话人
  - 方法: AHC (cosine 距离 + centroid linkage) → VBx (PLDA 似然 + 变分推断)
  - 输出: 给每个语音段分配一个局部分 ID (SPEAKER_00, SPEAKER_01, ...)
  - 局限: 不同视频的 SPEAKER_00 可能对应不同的人 —— 无跨视频一致性

跨视频聚类 (本模块 L3 层实现):
  - 作用: 将所有视频的局部说话人按声纹相似度重新聚类，统一全局身份
  - 方法: 收集所有 (video_id, local_speaker) 的嵌入向量 → cosine 距离矩阵
          → sklearn AgglomerativeClustering (average linkage)
          → 按总时长排序: 最大簇 → "TARGET", 其余 → "OTHER_XX"
          → 离群检测: 单样本簇且距离 > 1.5×threshold → "UNKNOWN"
  - 输出: 全局身份标签映射


用法:
  uv run python -m pipeline.04_diarization_community   # 单独运行
  uv run python -c "from pipeline.04_diarization_community import main; main()"
"""

# ============================================================
# L1: 基础工具层
# ============================================================

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    get_audio_files,
    load_config,
    setup_logger,
    write_audio,
)


def _preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
) -> Optional[Tuple[torch.Tensor, int, np.ndarray, int]]:
    """
    统一音频预处理流水线。

    流程: soundfile 读取 → 多声道降混为单声道 → 重采样到 target_sr
         → 峰值归一化到 [-1, 1] → 转换为 [1, N] float32 tensor

    参数:
        audio_path: 音频文件路径
        target_sr:   目标采样率 (pyannote 要求 16kHz)

    返回:
        (tensor, target_sr, raw_audio, raw_sr) 或 None (校验失败)
        - tensor:   [1, samples] float32, 已归一化, 用于 pyannote 推理
        - raw_audio: (samples,) float64, 原始采样率, 用于后续精确切分
    """
    try:
        raw_audio, raw_sr = sf.read(audio_path)
    except Exception as e:
        raise IOError(f"音频读取失败: {e}")

    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    error = _validate_audio(raw_audio, raw_sr)
    if error:
        return None

    if raw_sr != target_sr:
        from scipy import signal as _signal

        gcd = np.gcd(raw_sr, target_sr)
        audio_16k = _signal.resample_poly(
            raw_audio.astype(np.float64),
            target_sr // gcd,
            raw_sr // gcd,
        )
    else:
        audio_16k = raw_audio.copy()

    peak = np.max(np.abs(audio_16k))
    if peak > 1e-8:
        audio_16k = audio_16k / peak

    tensor = torch.from_numpy(audio_16k).float().unsqueeze(0)

    return tensor, target_sr, raw_audio.astype(np.float64), raw_sr


def _validate_audio(wav: np.ndarray, sr: int) -> Optional[str]:
    """前置校验：空音频、超短音频、静音。"""
    if len(wav) < 100:
        return "audio_empty"

    duration = len(wav) / sr
    if duration < 0.1:
        return f"audio_too_short ({duration:.2f}s)"

    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2))
    if rms < 1e-6:
        return f"audio_silent (rms={rms:.2e})"

    return None


# ============================================================
# L2: 分音核心层 — pyannote community-1
# ============================================================


def _get_community_pipeline_params(config: dict) -> dict:
    """
    从配置中提取 community-1 专用的管道参数。

    返回值是一个参数字典，用于后续以两种方式控制管道:
      1. 管道构建时参数 (init-time) —— 需要重建管道
      2. 管道运行时参数 (runtime) —— 通过属性赋值可动态调整

    本函数返回统一字典，调用方自行决定如何使用。
    """
    diar_cfg = config.get("diarization_community", config.get("diarization_ad", {}))

    # ── 管道构建时参数 (init-time) ──
    # 这些参数在 SpeakerDiarization.__init__() 中设置，构建后无法通过简单属性修改。
    # 如需调整，需重建管道实例。

    # segmentation_step (float, 默认 0.1):
    #   分割模型的滑动步长，表示为窗口时长的比例。
    #   0.1 表示 90% 重叠推理 → 最高精度但最慢。
    #   0.25 → 75% 重叠, 0.5 → 50% 重叠。
    #   增大此值可显著减少推理时间，但可能遗漏短语音段 (特别是快速对话中的短句)。
    init_params = {
        "segmentation_step": diar_cfg.get("segmentation_step", 0.1),
    }

    # embedding_batch_size (int, 默认 1):
    #   嵌入提取的批大小。增加此值可在 GPU 上提升吞吐量。
    #   注意: 每个样本是 (1, 1, num_samples) 的波形 + (1, num_frames) 的 mask，
    #   因此内存占用主要取决于 segment 窗口时长 × 嵌入模型采样率。
    init_params["embedding_batch_size"] = diar_cfg.get("embedding_batch_size", 32)

    # embedding_exclude_overlap (bool, 默认 False):
    #   True  → 只使用非重叠语音区域提取嵌入 (嵌入更"纯净"，但可能跳过短句)
    #   False → 使用全部语音区域提取嵌入 (更多数据，但对重叠鲁棒)
    #   对于大部分对话场景，False 效果更好 (更多样本 → 更稳定的聚类)。
    init_params["embedding_exclude_overlap"] = diar_cfg.get(
        "embedding_exclude_overlap", True
    )

    # segmentation_batch_size (int, 默认 1):
    #   分割模型的批大小。与 embedding_batch_size 不同，此参数通过 setter 暴露，
    #   因此也可以在构建后动态调整 (直接设置 pipeline.segmentation_batch_size = N)。
    init_params["segmentation_batch_size"] = diar_cfg.get("segmentation_batch_size", 32)

    # ── 管道运行时参数 (runtime) ──
    # 这些参数通过赋值 pipeline 的属性即可动态调整，无需重建。

    # min_duration_off (float, 默认 0.0):
    #   属于segmentation ParamDict 的一部分。用于合并同一说话人的短间隙。
    #   设置方式: pipeline.segmentation.min_duration_off = 0.5
    runtime_params = {
        "min_duration_off": diar_cfg.get("merge_gap", diar_cfg.get("min_duration_off", 0.0)),
    }

    # ── 聚类参数 (clustering runtime) ──
    # 对于 VBxClustering，这些参数通过 pipeline.klustering.X 访问。
    # pipeline.klustering.threshold (AHC 阈值)
    # pipeline.klustering.Fa (VBx Fa)
    # pipeline.klustering.Fb (VBx Fb)

    clustering_params = {
        # threshold (float, 默认 0.6, 范围 [0.0, 2.0]):
        #   AHC (凝聚式层次聚类) 的 cosine 距离阈值。
        #   嵌入向量已 L2 归一化，cosine 距离 ∈ [0, 2]。
        #   0 → 不合并任何说话人 (极端保守)
        #   0.5 → 较激进合并，适合同一人出现在邻近 segment
        #   0.6 → community-1 默认值，通用折中
        #   0.8 → 较保守合并，适合多人对话
        #   1.0+ → 极少合并，基本保持 AHC 原始聚类
        #   详见 VBxClustering.cluster() 中的 scipy.cluster.hierarchy.fcluster 逻辑
        "threshold": diar_cfg.get("vbx_threshold",
                       diar_cfg.get("clustering_threshold", 0.6)),

        # Fa (float, 默认 0.07, 范围 [0.01, 0.5]):
        #   VBx 充分统计量的缩放因子。
        #   在变分贝叶斯迭代中，Fa 控制声学似然 (acoustic likelihood) 的权重:
        #   - Fa 越大 → 声学信息权重越高 → 模型更容易区分细微音色差异 → 更多说话人
        #   - Fa 越小 → 正则化更强 → 模型倾向于合并 → 更少说话人
        #   技术细节: 用于公式 (16) 中 α = Fa/Fb × invL × gamma^T × rho
        #   以及公式 (17) 中 invL 的计算
        "Fa": diar_cfg.get("vbx_Fa",
              diar_cfg.get("Fa", 0.07)),

        # Fb (float, 默认 0.8, 范围 [0.01, 15.0]):
        #   VBx 说话人正则化系数。
        #   控制 VB 目标函数中 KL 散度项的权重:
        #   - Fb 越大 → 对新增说话人的先验惩罚越强 → 更少说话人
        #   - Fb 越小 → 允许更多说话人
        #   关键比值 ρ = Fa / Fb:
        #   - ρ < 0.05 → 强烈倾向少说话人
        #   - ρ ≈ 0.0875 → 默认 (0.07/0.8)
        #   - ρ > 0.2 → 强烈倾向多说话人
        "Fb": diar_cfg.get("vbx_Fb",
              diar_cfg.get("Fb", 0.8)),
    }

    # ── 推理时约束 (pipeline.apply 参数) ──
    # num_speakers / min_speakers / max_speakers 在每次 apply() 时传入。
    # min_speakers 和 max_speakers 是硬约束:
    #   - 如果 VBx 自动检测的说话人数 < min_speakers 或 > max_speakers,
    #     则触发 KMeans 回退 (强行聚类到所需数量)
    #   - 回退时 constrained_assignment 会被设为 False, 以避免人工增加说话人数
    max_speakers = diar_cfg.get("max_speakers", 5)
    min_speakers = diar_cfg.get("min_speakers", 1)
    if min_speakers > max_speakers:
        min_speakers = max_speakers

    # ── 后处理参数 ──
    min_segment_duration = diar_cfg.get("min_segment_duration", 0.5)
    max_segment_duration = diar_cfg.get("max_segment_duration", 10.0)

    # ── precision 模式参数覆盖 ──
    precision_cfg = diar_cfg.get("precision", {})
    if precision_cfg.get("enabled", False):
        if "vbx_threshold" in precision_cfg:
            clustering_params["threshold"] = precision_cfg["vbx_threshold"]
        if "vbx_Fa" in precision_cfg:
            clustering_params["Fa"] = precision_cfg["vbx_Fa"]
        if "vbx_Fb" in precision_cfg:
            clustering_params["Fb"] = precision_cfg["vbx_Fb"]
        if "min_segment_duration" in precision_cfg:
            min_segment_duration = precision_cfg["min_segment_duration"]

    return {
        "init_params": init_params,
        "runtime_params": runtime_params,
        "clustering_params": clustering_params,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "min_segment_duration": min_segment_duration,
        "max_segment_duration": max_segment_duration,
        "precision_cfg": precision_cfg,
    }


def _load_community_pipeline(
    config: dict,
) -> Tuple["SpeakerDiarization", str, torch.device, dict]:
    """
    加载 pyannote community-1 说话人分离管道。

    与 3.1 不同，community-1 使用 SpeakerDiarization 类，
    默认使用 VBxClustering (AHC + VBx 精化)。
    该管道返回 DiarizeOutput 对象，包含 speaker_diarization、
    exclusive_speaker_diarization 和 speaker_embeddings。

    ═══════════════════════════════════════════════════════════════
    管道内部结构 (构建完成后)
    ═══════════════════════════════════════════════════════════════

    pipeline.segmentation     → ParamDict (包含 min_duration_off)
    pipeline.klustering       → VBxClustering 实例
        .threshold            → AHC 距离阈值 (Uniform 参数)
        .Fa                   → VBx Fa (Uniform 参数)
        .Fb                   → VBx Fb (Uniform 参数)
        .plda                 → PLDA 模型 (用于 VBx 似然计算)
        .constrained_assignment → bool (VBx 是否使用约束分配)
    pipeline._segmentation    → Inference 对象 (分割模型)
    pipeline._embedding       → PretrainedSpeakerEmbedding (嵌入模型)
    pipeline._audio           → Audio 对象 (音频 I/O)
    pipeline.segmentation_step → float (分割窗口滑动步长)
    pipeline.embedding_batch_size → int (嵌入批大小)

    返回:
        (pipeline, model_name, device, params)
    """
    from pyannote.audio import Pipeline

    diar_cfg = config.get("diarization_community", config.get("diarization_ad", {}))

    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or diar_cfg.get("hf_token", "").strip()
    )

    model_name = "pyannote/speaker-diarization-community-1"
    fallback_model = "pyannote/speaker-diarization"

    def _try_load(model, auth_token):
        if not auth_token:
            return Pipeline.from_pretrained(model)
        try:
            return Pipeline.from_pretrained(model, use_auth_token=auth_token)
        except TypeError:
            return Pipeline.from_pretrained(model, token=auth_token)

    pipe = None
    last_error = None
    try:
        pipe = _try_load(model_name, token)
    except Exception as e:
        last_error = e

    if pipe is None:
        try:
            pipe = _try_load(fallback_model, token)
            model_name = fallback_model
        except Exception as e:
            raise RuntimeError(
                f"pyannote community-1 模型加载失败。\n"
                f"  community-1 错误: {last_error}\n"
                f"  fallback 错误: {e}\n"
                f"  请确认:\n"
                f"    1. 已访问 https://hf.co/pyannote/speaker-diarization-community-1 接受条款\n"
                f"    2. HF_TOKEN 环境变量或 config.yaml 中 hf_token 已正确设置\n"
                f"    3. 网络可访问 huggingface.co"
            )

    # ── 设备适配 ──
    global_gpu = config.get("gpu", {})
    if not global_gpu.get("enabled", True):
        device = torch.device("cpu")
    else:
        device_str = diar_cfg.get("device", "auto")
        cvd = global_gpu.get("cuda_visible", "").strip()
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        if device_str == "cpu":
            device = torch.device("cpu")
        elif device_str == "cuda":
            device_id = global_gpu.get("device_id", 0)
            device = torch.device(
                f"cuda:{device_id}" if torch.cuda.device_count() > device_id else "cuda"
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    # ── 提取和配置参数 ──
    params = _get_community_pipeline_params(config)

    # 设置运行时参数
    pipe.segmentation.min_duration_off = params["runtime_params"]["min_duration_off"]

    # 设置聚类参数 (如果管道使用的是 VBxClustering)
    if hasattr(pipe, "klustering"):
        clust = params["clustering_params"]
        # threshold, Fa, Fb 是 ParamDict 的 Uniform 参数，有 setter
        if hasattr(pipe.klustering, "threshold"):
            pipe.klustering.threshold = clust["threshold"]
        if hasattr(pipe.klustering, "Fa"):
            pipe.klustering.Fa = clust["Fa"]
        if hasattr(pipe.klustering, "Fb"):
            pipe.klustering.Fb = clust["Fb"]

    # embedding_exclude_overlap: 提升 VBx 聚类嵌入质量
    pipe.embedding_exclude_overlap = params["init_params"]["embedding_exclude_overlap"]

    # 应用 batch_size (注意: 这些是管道属性，可以直接赋值)
    pipe.segmentation_batch_size = params["init_params"]["segmentation_batch_size"]
    pipe.embedding_batch_size = params["init_params"]["embedding_batch_size"]

    return pipe, model_name, device, params


def _run_diarization(
    tensor: torch.Tensor,
    sample_rate: int,
    pipeline,
    params: dict,
) -> Tuple[List[dict], np.ndarray, Dict[str, np.ndarray], Optional["Annotation"]]:
    """
    对单个预处理后的音频张量运行 community-1 说话人分离。

    pyannote community-1 的 SpeakerDiarization.apply() 方法流程:
      1. get_segmentations()     → raw segmentation scores
      2. binarize()               → binary segmentation (powerset 模式下无 threshold)
      3. speaker_count()          → 逐帧活跃说话人数
      4. get_embeddings()         → 每 (chunk, speaker) 的嵌入向量
      5. clustering()             → VBxClustering (AHC + VBx)
      6. reconstruct()            → 聚类结果 → 离散分离结果
      7. to_annotation()          → 离散结果 → Annotation (含 min_duration_off 合并)

    参数:
        tensor:     [1, N] float32, 16kHz 音频
        sample_rate: 采样率
        pipeline:   SpeakerDiarization 实例
        params:     _get_community_pipeline_params() 返回的参数

    返回:
        (segments, diarize_output, speaker_embeddings, exclusive_annotation)
        - segments: [{speaker, start, end, duration, is_overlap, overlap_speakers}, ...]
        - diarize_output: DiarizeOutput 对象 (包含完整结果)
        - speaker_embeddings: {speaker_label: numpy_array}
        - exclusive_annotation: 如果可用，返回无重叠的 Annotation
    """
    min_dur = params["min_segment_duration"]
    min_speakers = params["min_speakers"]
    max_speakers = params["max_speakers"]

    device = getattr(pipeline, "device", torch.device("cpu"))
    tensor = tensor.to(device)

    # ── 运行 community-1 管道 ──
    # SpeakerDiarization.apply() 返回 DiarizeOutput 对象 (非 legacy 模式):
    #   .speaker_diarization        → Annotation (完整分离结果，含重叠)
    #   .exclusive_speaker_diarization → Annotation (无重叠，适合 ASR 对齐)
    #   .speaker_embeddings         → (num_speakers, dimension) numpy 数组
    output = pipeline(
        {"waveform": tensor, "sample_rate": sample_rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # ── 提取 Annotation ──
    if hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization
        exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
        embedding_matrix = getattr(output, "speaker_embeddings", None)
        diarize_output = output
    else:
        # legacy 模式或 fallback 管道 → 直接使用 annotation
        annotation = output
        exclusive_annotation = None
        embedding_matrix = None
        diarize_output = None

    # ── 解析为统一片段格式 ──
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = round(turn.start, 3)
        end = round(turn.end, 3)
        duration = round(end - start, 3)
        if duration >= min_dur:
            segments.append({
                "speaker": str(speaker),
                "start": start,
                "end": end,
                "duration": duration,
                "is_overlap": False,
                "overlap_speakers": [],
            })

    if not segments:
        return [], diarize_output, {}, exclusive_annotation

    segments.sort(key=lambda s: (s["start"], s["end"]))
    _detect_overlaps(segments)

    # ── 提取说话人嵌入 ──
    speaker_embeddings = {}
    if embedding_matrix is not None and len(annotation.labels()) > 0:
        labels = list(annotation.labels())
        emb_matrix = embedding_matrix
        # 确保是 numpy
        if hasattr(emb_matrix, "cpu"):
            emb_matrix = emb_matrix.cpu().numpy()
        elif not isinstance(emb_matrix, np.ndarray):
            emb_matrix = np.array(emb_matrix)

        for i, label in enumerate(labels):
            if i < len(emb_matrix):
                vec = emb_matrix[i].astype(np.float64)
                if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                    continue
                norm = np.linalg.norm(vec)
                if norm < 1e-8:
                    continue
                vec = vec / norm
                speaker_embeddings[str(label)] = vec

    return segments, diarize_output, speaker_embeddings, exclusive_annotation


def _resegment_long_segments(
    segments: List[dict],
    raw_audio: np.ndarray,
    raw_sr: int,
    max_duration: float = 10.0,
    min_silence_ms: int = 300,
) -> List[dict]:
    """
    对超过 max_duration 的段进行 VAD 二次切分。

    策略:
      1. Silero VAD 检测内部停顿 (>= min_silence_ms)
      2. 在停顿处切分，确保每段 ≤ max_duration
      3. VAD 切完后仍然超长的，算术均分兜底
    """
    try:
        import silero_vad

        vad_model, utils = silero_vad.load_silero_vad()
        (get_speech_timestamps, _, _, _, _) = utils
    except (ImportError, Exception):
        return _fallback_split(segments, max_duration)

    if raw_sr != 16000:
        from scipy import signal as _signal

        gcd = np.gcd(raw_sr, 16000)
        audio_16k = _signal.resample_poly(
            raw_audio.astype(np.float64), 16000 // gcd, raw_sr // gcd
        ).astype(np.float32)
    else:
        audio_16k = raw_audio.astype(np.float32)

    new_segments = []

    for seg in segments:
        if seg["duration"] <= max_duration:
            new_segments.append(seg)
            continue

        seg_start_s = int(seg["start"] * 16000)
        seg_end_s = int(seg["end"] * 16000)
        seg_start_s = max(0, seg_start_s)
        seg_end_s = min(len(audio_16k), seg_end_s)

        if seg_end_s <= seg_start_s:
            new_segments.append(seg)
            continue

        seg_audio = audio_16k[seg_start_s:seg_end_s]

        speech_ts = get_speech_timestamps(
            torch.from_numpy(seg_audio),
            vad_model,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=min_silence_ms,
            return_seconds=True,
        )

        if not speech_ts:
            new_segments.append(seg)
            continue

        sub_segments = []
        buffer_start = speech_ts[0]["start"]
        buffer_end = speech_ts[0]["end"]

        for ts in speech_ts[1:]:
            gap = ts["start"] - buffer_end
            current_block_dur = buffer_end - buffer_start

            if current_block_dur + gap + (ts["end"] - ts["start"]) <= max_duration:
                buffer_end = ts["end"]
            elif current_block_dur <= max_duration:
                sub_segments.append((buffer_start, buffer_end))
                buffer_start = ts["start"]
                buffer_end = ts["end"]
            else:
                sub_segments.append((buffer_start, buffer_end))
                buffer_start = ts["start"]
                buffer_end = ts["end"]

        if buffer_end > buffer_start:
            sub_segments.append((buffer_start, buffer_end))

        final_sub_segs = []
        for sub_start, sub_end in sub_segments:
            sub_dur = sub_end - sub_start
            if sub_dur <= max_duration:
                final_sub_segs.append((sub_start, sub_end))
            else:
                n_chunks = int(np.ceil(sub_dur / max_duration))
                chunk_dur = sub_dur / n_chunks
                for i in range(n_chunks):
                    chunk_start = sub_start + i * chunk_dur
                    chunk_end = min(sub_end, sub_start + (i + 1) * chunk_dur)
                    if chunk_end - chunk_start >= 0.3:
                        final_sub_segs.append((chunk_start, chunk_end))

        if not final_sub_segs:
            new_segments.append(seg)
            continue

        seg_abs_start = seg["start"]
        for i, (sub_start, sub_end) in enumerate(final_sub_segs):
            abs_start = round(seg_abs_start + sub_start, 3)
            abs_end = round(seg_abs_start + sub_end, 3)
            dur = round(abs_end - abs_start, 3)
            if dur < 0.3:
                continue

            sub_seg = {
                **seg,
                "start": abs_start,
                "end": abs_end,
                "duration": dur,
                "is_overlap": seg.get("is_overlap", False),
                "overlap_speakers": list(seg.get("overlap_speakers", [])),
                "segment_id": "",
                "segment_path": "",
                "original_speaker": seg.get("original_speaker", seg["speaker"]),
            }
            new_segments.append(sub_seg)

    new_segments.sort(key=lambda s: (s["start"], s["end"]))
    return new_segments


def _fallback_split(segments: List[dict], max_duration: float) -> List[dict]:
    """纯算术均分兜底 (VAD 不可用时)"""
    new_segments = []
    for seg in segments:
        if seg["duration"] <= max_duration:
            new_segments.append(seg)
            continue
        n_chunks = int(np.ceil(seg["duration"] / max_duration))
        chunk_dur = seg["duration"] / n_chunks
        for i in range(n_chunks):
            chunk_start = seg["start"] + i * chunk_dur
            chunk_end = min(seg["end"], seg["start"] + (i + 1) * chunk_dur)
            if chunk_end - chunk_start < 0.3:
                continue
            sub_seg = {
                **seg,
                "start": round(chunk_start, 3),
                "end": round(chunk_end, 3),
                "duration": round(chunk_end - chunk_start, 3),
                "segment_id": "",
                "segment_path": "",
                "original_speaker": seg.get("original_speaker", seg["speaker"]),
            }
            new_segments.append(sub_seg)
    new_segments.sort(key=lambda s: (s["start"], s["end"]))
    return new_segments


def _detect_overlaps(segments: List[dict]) -> None:
    """
    检测并标记重叠语音段 (原地修改 segments)。

    注意: community-1 的 exclusive_speaker_diarization 已经处理了重叠，
    但由于我们使用的是 speaker_diarization (含重叠)，
    这里的二次重叠检测是必要的。
    """
    n = len(segments)
    if n <= 1:
        return

    for i in range(n):
        seg_a = segments[i]
        for j in range(i + 1, n):
            seg_b = segments[j]
            if seg_a["start"] < seg_b["end"] and seg_b["start"] < seg_a["end"]:
                overlap_start = max(seg_a["start"], seg_b["start"])
                overlap_end = min(seg_a["end"], seg_b["end"])
                if overlap_end - overlap_start > 0.05:
                    seg_a["is_overlap"] = True
                    seg_b["is_overlap"] = True
                    if seg_b["speaker"] not in seg_a["overlap_speakers"]:
                        seg_a["overlap_speakers"].append(seg_b["speaker"])
                    if seg_a["speaker"] not in seg_b["overlap_speakers"]:
                        seg_b["overlap_speakers"].append(seg_a["speaker"])


# ============================================================
# L3: 跨视频说话人关联层
# ============================================================

# ══════════════════════════════════════════════════════════════════════
# 跨视频说话人聚类详细说明
# ══════════════════════════════════════════════════════════════════════
#
# 问题:
#   community-1 的 VBxClustering 在单视频内部进行聚类，
#   输出的标签是 SPEAKER_00, SPEAKER_01 等局部编号。
#   不同视频的 SPEAKER_00 可能对应完全不同的真人 ——
#   视频 1 的 SPEAKER_00 是博主，视频 2 的 SPEAKER_00 可能是嘉宾。
#
# 解决:
#   本层将所有视频的局部说话人嵌入向量收集到一起，
#   用独立的 AHC 凝聚式层次聚类进行二次聚类，实现跨视频身份统一。
#
# 关键设计决策:
#
#   a. 为什么使用独立的 AHC 而非复用 VBx?
#      VBx 是为单音频内的短时说话人切换设计的 (依赖相邻帧的时序信息)，
#      不适合跨视频的静态嵌入聚类。AHC 简单有效，且结果直观可解释。
#
#   b. 为什么使用 cosine 距离而非其他?
#      community-1 的嵌入向量已 L2 归一化，cosine 距离等价于欧氏距离的平方。
#      cosine 距离对嵌入的绝对幅度不敏感，更适合不同录音条件下的声纹比较。
#
#   c. 为什么使用 average linkage?
#      average linkage 对噪声和离群点最不敏感。
#      single linkage 易产生链式效应 (chaining)，ward linkage 假设球状簇。
#
#   d. 为什么要按时长排序分配标签?
#      博主的语音时长通常占总量的最大比例。
#      将最大簇标记为 TARGET 符合直觉: 我们要找的那个说话人在所有视频中说话最久。
#
#   e. 离群检测的原理:
#      单样本簇 (只有一个视频的一段语音) 且距离最近的其他样本
#      超过 1.5×threshold → 该样本的声纹与其他所有说话人差异过大，
#      可能是环境噪声导致的误检或极其罕见的说话人，标记为 UNKNOWN。
#
# ══════════════════════════════════════════════════════════════════════


def _compute_cluster_quality_metrics(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    dist_matrix: np.ndarray,
    logger,
) -> None:
    """
    计算并记录聚类质量指标。

    指标:
      - 簇内平均距离 (intra-cluster distance): 越小说明聚类越紧凑
      - 簇间平均距离 (inter-cluster distance): 越大说明聚类分离度越好
      - Davies-Bouldin Index 近似: 簇内距离 / 簇间距离 的比值
      - Silhouette Score 近似: (b - a) / max(a, b)
    """
    n_clusters = len(set(cluster_labels))
    if n_clusters < 2:
        return

    n_samples = len(embeddings)
    cluster_ids = sorted(set(cluster_labels))

    intra_dists = {}
    inter_dists = {}

    for cid in cluster_ids:
        mask = cluster_labels == cid
        members = np.where(mask)[0]
        n_members = len(members)

        if n_members < 2:
            intra_dists[cid] = float("inf")
        else:
            sub_dist = dist_matrix[np.ix_(members, members)]
            intra_dists[cid] = sub_dist.sum() / (n_members * (n_members - 1))

        other_dists = []
        for other_cid in cluster_ids:
            if other_cid == cid:
                continue
            other_mask = cluster_labels == other_cid
            inter_sub = dist_matrix[np.ix_(members, other_mask)]
            if inter_sub.size > 0:
                other_dists.append(inter_sub.mean())

        inter_dists[cid] = np.mean(other_dists) if other_dists else float("inf")

    logger.info("  ── 聚类质量指标 ──")
    for cid in cluster_ids:
        intra = intra_dists.get(cid, float("inf"))
        inter = inter_dists.get(cid, float("inf"))
        ratio = intra / inter if inter > 1e-10 and intra != float("inf") else float("inf")
        n_mem = int(np.sum(cluster_labels == cid))
        logger.info(
            f"    簇 {cid} ({n_mem} 样本): "
            f"簇内={intra:.3f}, 簇间={inter:.3f}, 内/间比值={ratio:.3f}"
        )

    mean_intra = np.mean([v for v in intra_dists.values() if v != float("inf")])
    mean_inter = np.mean([v for v in inter_dists.values() if v != float("inf")])
    if mean_inter > 1e-10 and mean_intra != float("inf"):
        logger.info(
            f"  全局: 均簇内={mean_intra:.3f}, 均簇间={mean_inter:.3f}, "
            f"Davies-Bouldin≈{mean_intra/mean_inter:.3f}"
        )


def _cluster_speakers(
    all_embeddings: Dict[Tuple[str, str], np.ndarray],
    config: dict,
    logger,
) -> Dict[Tuple[str, str], str]:
    """
    跨视频说话人 AHC 凝聚式层次聚类。

    ══════════════════════════════════════════════════════════════════════
    与 community-1 内部 VBx 的关系
    ══════════════════════════════════════════════════════════════════════
    community-1 在单视频内使用 VBxClustering (AHC + VBx 精化)，
    本函数在跨视频层面使用独立的 AHC (sklearn) 做二次聚类。

    两者使用相同的话语人嵌入 (同一个 Wespeaker 模型产出)，
    但:
    - VBx 利用帧级别的时序信息，适用于短时说话人切换
    - 本文 AHC 利用视频级别的静态嵌入，适用于跨视频身份统一

    因此，VBx 的结果可以不同视频对应不同的人，
    而本层负责将所有局部身份映射到全局统一的标签空间。

    ══════════════════════════════════════════════════════════════════════

    流程:
      1. 嵌入收集与校验 (NaN/Inf/零向量过滤)
      2. 维度一致性校验 (不同嵌入模型版本可能导致维度不一致)
      3. 计算 cosine 距离矩阵 (scipy.pdist)
      4. sklearn AgglomerativeClustering (average linkage)
      5. 按簇内总时长排序 → 标签映射 (TARGET / OTHER_XX / UNKNOWN)
      6. 聚类质量评估 (簇内/簇间距离)
      7. 离群检测

    参数:
        all_embeddings: {(video_id, speaker_label): embedding_vector}
        config:         全局配置
        logger:         日志器

    返回:
        label_map: {(video_id, speaker_label): global_label}
    """
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import AgglomerativeClustering

    diar_cfg = config.get("diarization_community", config.get("diarization_ad", {}))
    precision_cfg = diar_cfg.get("precision", {})
    threshold = diar_cfg.get("clustering_threshold", diar_cfg.get("vbx_clustering_threshold", 0.5))
    if precision_cfg.get("enabled", False) and "clustering_threshold" in precision_cfg:
        threshold = precision_cfg["clustering_threshold"]
    enable_outlier = diar_cfg.get("enable_outlier_detection", True)
    outlier_factor = diar_cfg.get("outlier_factor", 1.5)

    if not all_embeddings:
        logger.info("无说话人嵌入，跳过跨视频聚类")
        return {}

    keys = list(all_embeddings.keys())
    embeddings = [all_embeddings[k] for k in keys]

    # ── NaN / Inf / 零向量过滤 ──
    clean_keys = []
    clean_embs = []
    for k, emb in zip(keys, embeddings):
        if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
            logger.warning(f"  丢弃含 NaN/Inf 的嵌入: {k}")
            continue
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            logger.warning(f"  丢弃零向量嵌入: {k}")
            continue
        clean_keys.append(k)
        clean_embs.append(emb)
    keys = clean_keys
    embeddings = clean_embs

    if not embeddings:
        logger.warning("过滤后无有效嵌入，跳过聚类")
        return {}

    # ── 维度一致性校验 ──
    dims = {emb.shape[0] for emb in embeddings}
    if len(dims) > 1:
        logger.warning(f"嵌入维度不一致: {dims}，过滤异常项")
        valid_dim = max(dims, key=lambda d: sum(1 for e in embeddings if e.shape[0] == d))
        filtered_keys = []
        filtered_embs = []
        for k, emb in zip(keys, embeddings):
            if emb.shape[0] == valid_dim:
                filtered_keys.append(k)
                filtered_embs.append(emb)
            else:
                logger.warning(f"  丢弃: {k} (dim={emb.shape[0]}, expected={valid_dim})")
        keys = filtered_keys
        embeddings = filtered_embs

    if len(embeddings) < 2:
        logger.info("嵌入样本数 < 2，跳过聚类")
        if len(embeddings) == 1:
            return {keys[0]: "TARGET"}
        return {}

    emb_matrix = np.stack(embeddings, axis=0)
    n_samples = len(embeddings)
    logger.info(
        f"跨视频聚类: {n_samples} 个说话人实例, "
        f"嵌入维度={embeddings[0].shape[0]}"
    )

    # ── 计算 cosine 距离矩阵 ──
    dist_vector = pdist(emb_matrix, metric="cosine")
    dist_matrix = squareform(dist_vector)

    # ── AHC 凝聚式层次聚类 ──
    # linkage='average': 簇间距离 = 所有样本对距离的平均值
    #   - 相比 'single' (最近邻): 不易产生链式效应
    #   - 相比 'complete' (最远邻): 对离群点更鲁棒
    #   - 相比 'ward' (ward 方差): 不需要欧氏距离假设
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
    )
    cluster_labels = clustering.fit_predict(dist_matrix)

    # ── 按簇编号整理 ──
    cluster_members = {}
    for idx, c_id in enumerate(cluster_labels):
        c_id = int(c_id)
        if c_id not in cluster_members:
            cluster_members[c_id] = []
        cluster_members[c_id].append(idx)

    # ── 按簇大小排序 (大簇在前) ──
    cluster_sizes = sorted(cluster_members.items(), key=lambda x: -len(x[1]))
    n_clusters = len(cluster_sizes)

    logger.info(f"  AHC 聚类完成: {n_clusters} 个簇 (threshold={threshold})")

    # ── 聚类质量日志 ──
    _compute_cluster_quality_metrics(emb_matrix, cluster_labels, dist_matrix, logger)

    # ── 构建标签映射 ──
    label_map = {}
    outlier_threshold = threshold * outlier_factor

    for rank, (c_id, member_indices) in enumerate(cluster_sizes):
        if rank == 0:
            label = "TARGET"
        else:
            label = f"OTHER_{(rank - 1):02d}"

        # 离群检测: 单样本簇 + 距离检查
        if enable_outlier and len(member_indices) == 1:
            idx = member_indices[0]
            distances_to_others = dist_matrix[idx, :]
            mask = np.ones(n_samples, dtype=bool)
            mask[idx] = False
            min_dist = distances_to_others[mask].min() if mask.any() else float("inf")

            if min_dist > outlier_threshold:
                label = "UNKNOWN"
                logger.info(
                    f"  离群检测: {keys[idx]} → UNKNOWN "
                    f"(min_dist={min_dist:.3f} > {outlier_threshold:.3f})"
                )

        for idx in member_indices:
            label_map[keys[idx]] = label

    # ── precision 模式：TARGET 嵌入距离门控 + IQR 离群检测 ──
    if precision_cfg.get("enabled", False):
        # 1. 嵌入距离门控：TARGET 成员到质心的 cosine 距离超过阈值 → UNCERTAIN
        if precision_cfg.get("embedding_gate", False):
            max_dist = precision_cfg.get("embedding_max_distance", 0.5)
            target_keys = [k for k, v in label_map.items() if v == "TARGET"]
            if len(target_keys) >= 2:
                target_indices = [keys.index(k) for k in target_keys]
                target_embs = emb_matrix[target_indices]
                centroid = target_embs.mean(axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-10:
                    centroid = centroid / centroid_norm
                    for i, idx in enumerate(target_indices):
                        dist = 1.0 - float(np.dot(emb_matrix[idx], centroid))
                        if dist > max_dist:
                            label_map[target_keys[i]] = "UNCERTAIN"
                            logger.info(
                                f"  嵌入门控: {target_keys[i]} → UNCERTAIN "
                                f"(dist={dist:.3f} > {max_dist})"
                            )

        # 2. IQR 离群检测：TARGET 簇内平均距离超过 Q3 + 1.5×IQR → UNCERTAIN
        if precision_cfg.get("iqr_outlier_detection", False):
            target_keys = [k for k, v in label_map.items() if v == "TARGET"]
            if len(target_keys) >= 4:
                target_indices = [keys.index(k) for k in target_keys]
                mean_dists = []
                for idx in target_indices:
                    others = [j for j in target_indices if j != idx]
                    d = float(dist_matrix[idx][others].mean())
                    mean_dists.append(d)
                mean_dists_arr = np.array(mean_dists)
                q1, q3 = np.percentile(mean_dists_arr, [25, 75])
                iqr = q3 - q1
                upper = q3 + 1.5 * iqr
                for i, d in enumerate(mean_dists):
                    if d > upper:
                        label_map[target_keys[i]] = "UNCERTAIN"
                        logger.info(
                            f"  IQR离群: {target_keys[i]} → UNCERTAIN "
                            f"(mean_dist={d:.3f} > {upper:.3f}, "
                            f"Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f})"
                        )

    # ── 日志输出 ──
    logger.info(f"  聚类完成: {n_clusters} 个簇, 阈值={threshold}")
    for label in ["TARGET"] + [f"OTHER_{i:02d}" for i in range(max(0, n_clusters - 1))]:
        members = [k for k, v in label_map.items() if v == label]
        if members:
            logger.info(f"    {label}: {len(members)} 个说话人实例")
            for vid, spk in members[:5]:
                logger.info(f"      ├ {vid} / {spk}")
            if len(members) > 5:
                logger.info(f"      └ ... 还有 {len(members) - 5} 个")

    unknown_members = [k for k, v in label_map.items() if v == "UNKNOWN"]
    if unknown_members:
        logger.info(f"    UNKNOWN (低置信度): {len(unknown_members)} 个")

    return label_map


# ============================================================
# L4: 批量流水线层
# ============================================================


def _discover_input_files(config: dict, logger) -> List[Tuple[str, str]]:
    """
    发现待处理音频文件。

    优先级:
    1. MDX23C 分离输出 (新版 Stage 03)
    2. Demucs 分离输出 (旧版 Stage 03, 向后兼容)
    3. 原始音频目录 (Stage 02 兜底)
    4. 原始视频目录 (最后兜底)
    """
    paths = config["paths"]
    candidates = []

    # ---- 优先级 1: MDX23C 输出 ----
    voice_sep_dir = paths.get("voice_sep_output", "")
    if voice_sep_dir and os.path.isdir(voice_sep_dir):
        mdx23c_cfg = config.get("mdx23c", {})
        mdx23c_model = mdx23c_cfg.get("model", "mdx23c")
        model_dir_name = mdx23c_model.replace(".ckpt", "").rsplit("/", 1)[-1]
        base = os.path.join(voice_sep_dir, model_dir_name)
        if os.path.isdir(base):
            for video_dir in sorted(os.listdir(base)):
                vocals_path = os.path.join(base, video_dir, "vocals.wav")
                if os.path.isfile(vocals_path):
                    candidates.append((vocals_path, video_dir))
            if candidates:
                logger.info(f"输入源: MDX23C 输出 ({len(candidates)} 个 vocals.wav)")
                return candidates

    # ---- 优先级 2: Demucs 输出 (向后兼容) ----
    demucs_dir = paths.get("demucs_output", "./data/03_demucs_output")
    demucs_model = config.get("demucs", {}).get("model", "htdemucs")
    base = os.path.join(demucs_dir, demucs_model)
    if os.path.isdir(base):
        for video_dir in sorted(os.listdir(base)):
            vocals_path = os.path.join(base, video_dir, "vocals.wav")
            if os.path.isfile(vocals_path):
                candidates.append((vocals_path, video_dir))
        if candidates:
            logger.info(f"输入源: Demucs 输出 ({len(candidates)} 个 vocals.wav)")
            return candidates

    audio_dir = paths.get("extracted_audio", "./data/02_extracted_audio")
    if os.path.isdir(audio_dir):
        audio_files = get_audio_files(audio_dir)
        for af in audio_files:
            video_id = os.path.splitext(os.path.basename(af))[0]
            candidates.append((af, video_id))
        if candidates:
            logger.info(f"输入源: 原始音频目录 ({len(candidates)} 个文件)")
            return candidates

    raw_dir = paths.get("raw_videos", "./data/01_raw_videos")
    if os.path.isdir(raw_dir):
        audio_files = get_audio_files(raw_dir)
        for af in audio_files:
            video_id = os.path.splitext(os.path.basename(af))[0]
            candidates.append((af, video_id))
        if candidates:
            logger.info(f"输入源: 原始视频目录 ({len(candidates)} 个音频)")
            return candidates

    return candidates


def process_all(config: dict, logger=None) -> List[dict]:
    """
    Stage 04v2 主流程：跨视频说话人分离 (community-1)。

    ══════════════════════════════════════════════════════════════════
    流程概述
    ══════════════════════════════════════════════════════════════════

    Phase 1 — 单视频说话人分离 (L2)
      对每个音频文件:
        1. 音频预处理 (降混、重采样、归一化)
        2. community-1 管道推理 (分割 → 计数 → 嵌入 → VBx 聚类)
        3. 解析 DiarizeOutput 为统一片段格式
        4. VAD 二次切分 (超长段按停顿分割)
        5. 收集说话人嵌入向量 (用于 Phase 2)

    Phase 2 — 跨视频说话人聚类 (L3)
      收集所有视频的嵌入向量:
        1. cosine 距离矩阵
        2. AHC 凝聚式层次聚类 (average linkage)
        3. 按总时长排序 → 全局标签 (TARGET / OTHER_XX / UNKNOWN)
        4. 标签映射应用到所有片段

    Phase 3 — 输出
      1. 切分音频段写入 segments 目录
      2. segments_meta.json 写入输出目录
      3. 清理 checkpoint

    ══════════════════════════════════════════════════════════════════
    参数 (通过 config.yaml 的 diarization_community 段控制)
    ══════════════════════════════════════════════════════════════════

    【管道构建时参数 (需重建管道才能修改)】
      segmentation_step:      分割窗口滑动步长，默认 0.1 (90% 重叠)
      embedding_batch_size:   嵌入提取批大小，默认 1

    【管道运行时参数 (可通过属性动态修改)】
      min_duration_off:       合并同一说话人短间隙，默认 0.0
      clustering.threshold:   AHC 距离阈值，默认 0.6
      clustering.Fa:          VBx Fa 系数，默认 0.07
      clustering.Fb:          VBx Fb 系数，默认 0.8

    【推理约束】
      min_speakers:           单音频最少说话人数，默认 1
      max_speakers:           单音频最多说话人数，默认 5

    【后处理】
      min_segment_duration:   最短语音片段 (秒)，默认 0.5
      max_segment_duration:   最长语音片段 (秒)，默认 10.0

    【跨视频聚类】
      clustering_threshold:   跨视频 AHC 距离阈值，默认 0.5
      enable_outlier_detection: 是否启用离群检测，默认 True
      outlier_factor:         离群阈值系数 (×threshold)，默认 1.5

    ══════════════════════════════════════════════════════════════════
    """
    if logger is None:
        logger = setup_logger(
            "04_diarization_community",
            config["paths"].get("logs", "./logs"),
        )

    diar_cfg = config.get("diarization_community", config.get("diarization_ad", {}))
    paths = config["paths"]

    out_dir = ensure_dir(paths.get("diarization_output", "./data/04_diarization"))
    seg_dir = ensure_dir(os.path.join(out_dir, "segments"))
    checkpoint_path = os.path.join(out_dir, "_checkpoint_community.json")
    save_interval = diar_cfg.get("save_interval", 10)

    # ── 1. 文件发现 ──
    candidates = _discover_input_files(config, logger)
    if not candidates:
        logger.warning("没有找到待处理音频文件！请先运行 Stage 01-03。")
        return []

    logger.info("=" * 60)
    logger.info("Stage 04v2: 跨视频说话人分离 (community-1)")
    logger.info(f"模型: pyannote/speaker-diarization-community-1")
    logger.info(f"待处理: {len(candidates)} 个音频文件")
    logger.info(f"输出目录: {out_dir}")
    logger.info("=" * 60)

    # ── 2. 断点恢复 ──
    completed_videos = set()
    all_segments: List[dict] = []
    all_embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    speaker_duration: Dict[Tuple[str, str], float] = {}

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            completed_videos = set(ckpt.get("completed_videos", []))
            all_segments = ckpt.get("segments", [])
            emb_data = ckpt.get("embeddings", {})
            for key_str, emb_list in emb_data.items():
                parts = key_str.split("|||")
                if len(parts) == 2:
                    all_embeddings[(parts[0], parts[1])] = np.array(emb_list)
            dur_data = ckpt.get("speaker_duration", {})
            for key_str, dur in dur_data.items():
                parts = key_str.split("|||")
                if len(parts) == 2:
                    speaker_duration[(parts[0], parts[1])] = dur
            logger.info(
                f"断点恢复: {len(completed_videos)} 已完成, "
                f"{len(all_segments)} 个已收集片段"
            )
        except Exception as e:
            logger.warning(f"checkpoint 读取失败，将重新开始: {e}")
            completed_videos = set()
            all_segments = []
            all_embeddings = {}
            speaker_duration = {}

    # ── 3. 加载 community-1 管道 ──
    logger.info("加载 pyannote community-1 管道...")
    t0 = time.time()
    pipeline, model_name, device, params = _load_community_pipeline(config)
    logger.info(f"已加载: {model_name} (device={device}, {time.time() - t0:.1f}s)")

    # 记录当前运行时参数
    min_dur_off = params["runtime_params"]["min_duration_off"]
    logger.info(f"  运行时参数:")
    logger.info(f"    min_duration_off={min_dur_off}")
    clust = params["clustering_params"]
    logger.info(f"    聚类 threshold={clust['threshold']}, Fa={clust['Fa']}, Fb={clust['Fb']}")
    logger.info(f"    推理 min_speakers={params['min_speakers']}, max_speakers={params['max_speakers']}")
    logger.info(f"    后处理 min_segment={params['min_segment_duration']}s, "
                f"max_segment={params['max_segment_duration']}s")

    # ── 4. 逐文件处理 ──
    skipped = 0
    failed = 0
    processed = 0

    pbar = tqdm(candidates, desc="说话人分离", unit="file")
    for audio_path, video_id in pbar:
        if video_id in completed_videos:
            skipped += 1
            pbar.set_postfix({"skip": skipped, "done": processed, "fail": failed})
            continue

        pbar.set_description(f"分离: {video_id[:30]}")

        try:
            result = _preprocess_audio(audio_path, target_sr=16000)
            if result is None:
                logger.warning(f"  [{video_id}] 音频校验失败，跳过")
                failed += 1
                completed_videos.add(video_id)
                continue

            tensor, sr_16k, raw_audio, raw_sr = result
            audio_duration = raw_audio.shape[0] / raw_sr

            # ── 4b. community-1 说话人分离 ──
            diar_segments, diarize_output, spk_embs, exclusive_ann = _run_diarization(
                tensor, sr_16k, pipeline, params
            )

            if not diar_segments:
                logger.warning(f"  [{video_id}] 未检测到语音段")
                completed_videos.add(video_id)
                continue

            # ── 4c. VAD 二次切分 ──
            max_seg_dur = params["max_segment_duration"]
            n_before = len(diar_segments)
            diar_segments = _resegment_long_segments(
                diar_segments, raw_audio, raw_sr,
                max_duration=max_seg_dur,
            )
            if len(diar_segments) != n_before:
                logger.info(f"  VAD 切分: {n_before} → {len(diar_segments)} 段")

            # ── 4d. 收集嵌入和时长 ──
            for spk_label, emb in spk_embs.items():
                key = (video_id, spk_label)
                all_embeddings[key] = emb
                spk_dur = sum(
                    s["duration"]
                    for s in diar_segments
                    if s["speaker"] == spk_label
                )
                speaker_duration[key] = speaker_duration.get(key, 0) + spk_dur

            # ── 4e. 切分音频段 ──
            for i, seg in enumerate(diar_segments):
                seg_id = f"{video_id}_{seg['speaker']}_{i:04d}"
                s = int(seg["start"] * raw_sr)
                e = int(seg["end"] * raw_sr)
                s = max(0, s)
                e = min(len(raw_audio), e)
                if e <= s:
                    continue

                seg_wav = raw_audio[s:e]
                seg_path = os.path.join(seg_dir, f"{seg_id}.wav")
                write_audio(seg_path, seg_wav, raw_sr)

                seg["segment_id"] = seg_id
                seg["segment_path"] = seg_path
                seg["video_id"] = video_id
                seg["original_speaker"] = seg["speaker"]

            all_segments.extend(diar_segments)

            # ── 日志 ──
            speakers = sorted(set(s["speaker"] for s in diar_segments))
            logger.info(
                f"  [{video_id}] {len(diar_segments)} 段, "
                f"{len(speakers)} 说话人: {speakers}, "
                f"音频 {audio_duration:.1f}s"
            )
            for spk in speakers:
                spk_total = sum(
                    s["duration"] for s in diar_segments if s["speaker"] == spk
                )
                spk_overlap = sum(
                    s["duration"]
                    for s in diar_segments
                    if s["speaker"] == spk and s["is_overlap"]
                )
                logger.info(f"    {spk}: {spk_total:.1f}s (重叠 {spk_overlap:.1f}s)")

            if exclusive_ann is not None:
                logger.info(
                    f"  exclusive_speaker_diarization 可用: "
                    f"{len(exclusive_ann.labels())} 说话人"
                )

            processed += 1
            completed_videos.add(video_id)

            if processed % save_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    completed_videos,
                    all_segments,
                    all_embeddings,
                    speaker_duration,
                )

        except Exception as e:
            logger.error(f"  [{video_id}] 处理失败: {e}", exc_info=True)
            failed += 1
            completed_videos.add(video_id)
            continue

    pbar.close()

    if processed == 0 and skipped > 0:
        logger.info("所有文件已在 checkpoint 中，跳过处理")

    # ── 5. 最终 checkpoint ──
    _save_checkpoint(
        checkpoint_path, completed_videos, all_segments, all_embeddings, speaker_duration
    )

    # ── 6. 跨视频聚类 ──
    logger.info("── 跨视频说话人聚类 ──")
    if not all_embeddings:
        logger.warning("没有提取到任何说话人嵌入，跳过聚类")
    else:
        label_map = _cluster_speakers(all_embeddings, config, logger)

        if label_map:
            # ── 按簇内总时长重排标签 ──
            label_duration: Dict[str, float] = {}
            for (vid, spk), global_label in label_map.items():
                dur = speaker_duration.get((vid, spk), 0)
                label_duration[global_label] = (
                    label_duration.get(global_label, 0) + dur
                )

            # 如果 TARGET 不是时长最长的簇，交换标签
            if "TARGET" in label_duration and len(label_duration) > 1:
                sorted_labels = sorted(label_duration.items(), key=lambda x: -x[1])
                if sorted_labels[0][0] != "TARGET":
                    old_target_label = "TARGET"
                    new_target_label = sorted_labels[0][0]
                    new_label_map = {}
                    for key, label in label_map.items():
                        if label == old_target_label:
                            new_label_map[key] = new_target_label
                        elif label == new_target_label:
                            new_label_map[key] = old_target_label
                        else:
                            new_label_map[key] = label
                    label_map = new_label_map
                    logger.info(
                        f"  基于时长重排: '{new_target_label}'"
                        f"({label_duration[new_target_label]:.1f}s) "
                        f"↔ 'TARGET'({label_duration.get(old_target_label, 0):.1f}s)"
                    )

            # 应用标签映射
            for seg in all_segments:
                key = (seg["video_id"], seg.get("original_speaker", seg["speaker"]))
                seg["speaker"] = label_map.get(key, seg["speaker"])

            # ── precision 模式：TARGET 段专属最短时长过滤 ──
            precision_cfg = diar_cfg.get("precision", {})
            if precision_cfg.get("enabled", False):
                target_min_dur = precision_cfg.get("min_segment_duration", 0.8)
                n_demoted = 0
                for seg in all_segments:
                    if seg["speaker"] == "TARGET" and seg["duration"] < target_min_dur:
                        seg["speaker"] = "UNCERTAIN"
                        n_demoted += 1
                if n_demoted > 0:
                    logger.info(
                        f"  TARGET 最短段过滤: {n_demoted} 段 < {target_min_dur}s 降级为 UNCERTAIN"
                    )

            # 最终统计
            final_labels = {}
            for seg in all_segments:
                lbl = seg["speaker"]
                if lbl not in final_labels:
                    final_labels[lbl] = {"count": 0, "duration": 0.0}
                final_labels[lbl]["count"] += 1
                final_labels[lbl]["duration"] += seg["duration"]

            logger.info("── 最终说话人分布 ──")
            for lbl in ["TARGET"] + sorted(
                [k for k in final_labels if k.startswith("OTHER_")]
            ) + sorted([k for k in final_labels if k == "UNKNOWN"]):
                if lbl in final_labels:
                    info = final_labels[lbl]
                    logger.info(
                        f"  {lbl}: {info['count']} 段, "
                        f"{info['duration']:.1f}s ({info['duration'] / 60:.1f}min)"
                    )

    # ── 7. 保存最终元数据 ──
    meta_path = os.path.join(out_dir, "segments_meta_community.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # ── 8. 总结 ──
    total_dur = sum(s["duration"] for s in all_segments)
    unique_speakers = set(s["speaker"] for s in all_segments)
    logger.info("=" * 60)
    logger.info(
        f"Stage 04v2 完成: {len(candidates)} 输入 → "
        f"{processed} 成功, {skipped} 跳过, {failed} 失败"
    )
    logger.info(
        f"  输出: {len(all_segments)} 段, "
        f"{len(unique_speakers)} 说话人, "
        f"{total_dur:.1f}s ({total_dur / 60:.1f}min)"
    )
    logger.info(f"  元数据: {meta_path}")
    logger.info("=" * 60)

    return all_segments


def _save_checkpoint(
    path: str,
    completed_videos: set,
    segments: List[dict],
    embeddings: Dict[Tuple[str, str], np.ndarray],
    speaker_duration: Dict[Tuple[str, str], float],
) -> None:
    """持久化中间结果到 checkpoint 文件"""
    emb_serializable = {}
    for (vid, spk), emb in embeddings.items():
        key_str = f"{vid}|||{spk}"
        emb_serializable[key_str] = emb.tolist()

    dur_serializable = {}
    for (vid, spk), dur in speaker_duration.items():
        key_str = f"{vid}|||{spk}"
        dur_serializable[key_str] = dur

    ckpt = {
        "completed_videos": sorted(completed_videos),
        "segments": segments,
        "embeddings": emb_serializable,
        "speaker_duration": dur_serializable,
        "pipeline": "pyannote/speaker-diarization-community-1",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def main(config=None, logger=None):
    """CLI 入口"""
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger(
            "04_diarization_community",
            config["paths"].get("logs", "./logs"),
        )

    process_all(config, logger=logger)


if __name__ == "__main__":
    main()
