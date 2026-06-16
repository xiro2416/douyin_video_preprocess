"""
Stage 04: 跨视频说话人分离
===========================
基于 pyannote/speaker-diarization-3.1 完成单音频说话人时间边界切分，
通过声纹嵌入 + AHC 凝聚式层次聚类实现跨视频同一说话人身份统一，
最终输出带身份标签的语音片段与结构化元数据。

4 层架构:
  L1 基础工具层 ─ 音频预处理、speechbrain 兼容补丁、前置校验
  L2 分音核心层 ─ pyannote 流水线加载、单音频 diarization、重叠检测、嵌入提取
  L3 跨视频关联层 ─ 嵌入收集、AHC 聚类、身份标签映射、离群识别
  L4 批量流水线层 ─ 文件发现、批量调度、异常隔离、断点续传、片段导出

用法:
  uv run python -m pipeline.run_pipeline --from 04 --to 04   # 单独运行
  uv run python -m pipeline.run_pipeline --all                # 完整流水线
"""

# ============================================================
# L0: 兼容补丁 (必须在其他 import 之前执行)
# ============================================================
# speechbrain lazy import 在 Windows 下会触发崩溃：
# inspect.getframeinfo 遍历 sys.modules 时触发 LazyModule.__getattr__
# → ensure_module → 尝试 import 不存在的子模块 → ModuleNotFoundError
# 方案：拦截 importlib.import_module，对 speechbrain.* 子模块失败时注入占位模块

import importlib as _il
import sys as _sys
import types as _types

_orig_import_module = _il.import_module


def _patched_import_module(name, package=None):
    try:
        return _orig_import_module(name, package=package)
    except (ImportError, ModuleNotFoundError):
        if name.startswith("speechbrain."):
            dummy = _types.ModuleType(name)
            dummy.__file__ = "<sb_patch>"
            _sys.modules[name] = dummy
            return dummy
        raise


_il.import_module = _patched_import_module

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
    # 1. 读取原始音频
    try:
        raw_audio, raw_sr = sf.read(audio_path)
    except Exception as e:
        raise IOError(f"音频读取失败: {e}")

    # 2. 多声道 → 单声道
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    # 3. 前置校验
    error = _validate_audio(raw_audio, raw_sr)
    if error:
        return None

    # 4. 重采样到目标采样率
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

    # 5. 峰值归一化
    peak = np.max(np.abs(audio_16k))
    if peak > 1e-8:
        audio_16k = audio_16k / peak

    # 6. 转 tensor [1, N]
    tensor = torch.from_numpy(audio_16k).float().unsqueeze(0)

    return tensor, target_sr, raw_audio.astype(np.float64), raw_sr


def _validate_audio(wav: np.ndarray, sr: int) -> Optional[str]:
    """
    前置校验：空音频、超短音频、静音。
    返回错误描述字符串，通过则返回 None。
    """
    # 空音频
    if len(wav) < 100:
        return "audio_empty"

    # 超短 (< 0.1s)
    duration = len(wav) / sr
    if duration < 0.1:
        return f"audio_too_short ({duration:.2f}s)"

    # 静音 (RMS < 1e-6)
    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2))
    if rms < 1e-6:
        return f"audio_silent (rms={rms:.2e})"

    return None


# ============================================================
# L2: 分音核心层
# ============================================================


def _load_diarization_pipeline(config: dict):
    """
    加载 pyannote 说话人分离流水线，支持二级降级。

    优先加载 3.1 正式版，失败自动回退 community-1。
    自动设备适配 (CUDA/CPU)。
    """
    from pyannote.audio import Pipeline

    diar_cfg = config.get("diarization_ad", {})

    # Token: 优先环境变量
    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or diar_cfg.get("hf_token", "").strip()
    )

    model_name = diar_cfg.get("model", "pyannote/speaker-diarization-3.1")

    def _try_load(model, auth_token):
        """尝试加载模型，兼容新旧版 pyannote auth API"""
        if not auth_token:
            return Pipeline.from_pretrained(model)
        # 优先 use_auth_token (旧版 API)
        try:
            return Pipeline.from_pretrained(model, use_auth_token=auth_token)
        except TypeError:
            # 回退到 token (新版 API)
            return Pipeline.from_pretrained(model, token=auth_token)

    # 尝试加载 3.1
    pipe = None
    last_error = None
    try:
        pipe = _try_load(model_name, token)
    except Exception as e:
        last_error = e

    # 降级到 community-1 (speaker-diarization)
    if pipe is None:
        fallback_model = "pyannote/speaker-diarization"
        try:
            pipe = _try_load(fallback_model, token)
            model_name = fallback_model
        except Exception as e:
            raise RuntimeError(
                f"pyannote 模型加载失败。\n"
                f"  3.1 错误: {last_error}\n"
                f"  community 错误: {e}\n"
                f"  请确认:\n"
                f"    1. 已访问 https://hf.co/pyannote/speaker-diarization-3.1 接受条款\n"
                f"    2. HF_TOKEN 环境变量或 config.yaml 中 hf_token 已正确设置\n"
                f"    3. 网络可访问 huggingface.co"
            )

    # --- 设备适配: 优先全局 gpu 配置，次之 diarization 专用配置 ---
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
            device = torch.device(f"cuda:{device_id}" if torch.cuda.device_count() > device_id else "cuda")
        else:  # auto
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    return pipe, model_name, device


def _set_merge_gap(pipeline, config: dict):
    """通过 pyannote 原生的 min_duration_off 合并同说话人近段"""
    diar_cfg = config.get("diarization_ad", {})
    merge_gap = diar_cfg.get("merge_gap", 0.0)
    if merge_gap > 0:
        pipeline.segmentation.min_duration_off = merge_gap


def _run_diarization(
    tensor: torch.Tensor,
    sample_rate: int,
    pipeline,
    config: dict,
) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    """
    对单个预处理后的音频张量运行说话人分离。

    参数:
        tensor:     [1, N] float32, 16kHz 音频
        sample_rate: 采样率
        pipeline:   pyannote Pipeline 实例
        config:     全局配置

    返回:
        (segments, speaker_embeddings)
        segments: [{speaker, start, end, duration, is_overlap, overlap_speakers}, ...]
        speaker_embeddings: {speaker_label: numpy_array}
    """
    diar_cfg = config.get("diarization_ad", {})
    min_dur = diar_cfg.get("min_segment_duration", 0.5)
    min_speakers = diar_cfg.get("min_speakers", 1)
    max_speakers = diar_cfg.get("max_speakers", 5)

    # 将 tensor 移到 pipeline 所在设备
    device = getattr(pipeline, "device", torch.device("cpu"))
    tensor = tensor.to(device)

    # 运行 diarization
    diar_result = pipeline(
        {"waveform": tensor, "sample_rate": sample_rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # ── 解析分音结果 ──
    # pyannote 3.x/4.x 兼容: 从 speaker_diarization 属性取 Annotation
    if hasattr(diar_result, "speaker_diarization"):
        annotation = diar_result.speaker_diarization
    else:
        annotation = diar_result

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
        return [], {}

    # 按起始时间排序
    segments.sort(key=lambda s: (s["start"], s["end"]))

    # ── 重叠检测 ──
    _detect_overlaps(segments)

    # ── 提取说话人嵌入 ──
    speaker_embeddings = {}
    if (
        hasattr(diar_result, "speaker_embeddings")
        and diar_result.speaker_embeddings is not None
    ):
        labels = list(annotation.labels())
        emb_matrix = diar_result.speaker_embeddings
        # emb_matrix 可能是 tensor 或 numpy
        if hasattr(emb_matrix, "cpu"):
            emb_matrix = emb_matrix.cpu().numpy()
        elif not isinstance(emb_matrix, np.ndarray):
            emb_matrix = np.array(emb_matrix)

        for i, label in enumerate(labels):
            if i < len(emb_matrix):
                vec = emb_matrix[i].astype(np.float64)
                # 过滤 NaN / Inf / 零向量
                if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                    continue
                norm = np.linalg.norm(vec)
                if norm < 1e-8:
                    continue
                # L2 归一化
                vec = vec / norm
                speaker_embeddings[str(label)] = vec

    return segments, speaker_embeddings


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

    参数:
        segments:  pyannote 输出段列表
        raw_audio: 原始音频 (用于 VAD)
        raw_sr:    原始采样率
        max_duration: 最大段时长 (秒)
        min_silence_ms: 最小停顿时长 (ms)

    返回:
        重新切分后的段列表
    """
    try:
        import silero_vad
        vad_model, utils = silero_vad.load_silero_vad()
        (get_speech_timestamps, _, _, _, _) = utils
    except (ImportError, Exception):
        # Silero VAD 不可用，直接用算术均分
        return _fallback_split(segments, max_duration)

    # 将音频降到 16kHz (Silero VAD 要求)
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

        # 提取该段的 16kHz 音频
        seg_start_s = int(seg["start"] * 16000)
        seg_end_s = int(seg["end"] * 16000)
        seg_start_s = max(0, seg_start_s)
        seg_end_s = min(len(audio_16k), seg_end_s)

        if seg_end_s <= seg_start_s:
            new_segments.append(seg)
            continue

        seg_audio = audio_16k[seg_start_s:seg_end_s]

        # Silero VAD
        speech_ts = get_speech_timestamps(
            torch.from_numpy(seg_audio),
            vad_model,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=min_silence_ms,
            return_seconds=True,
        )

        if not speech_ts:
            # VAD 没检测到语音，保留原段
            new_segments.append(seg)
            continue

        # 在语音区间之间切分（停顿 ≥ min_silence_ms 处）
        sub_segments = []
        buffer_start = speech_ts[0]["start"]
        buffer_end = speech_ts[0]["end"]

        for ts in speech_ts[1:]:
            # 从上一个语音结束到当前语音开始的间隔
            gap = ts["start"] - buffer_end
            current_block_dur = buffer_end - buffer_start

            if current_block_dur + gap + (ts["end"] - ts["start"]) <= max_duration:
                # 合并到当前块
                buffer_end = ts["end"]
            elif current_block_dur <= max_duration:
                # 当前块可以独立成段，开始新块
                sub_segments.append((buffer_start, buffer_end))
                buffer_start = ts["start"]
                buffer_end = ts["end"]
            else:
                # 当前块自身已超长，需要拆分
                sub_segments.append((buffer_start, buffer_end))
                buffer_start = ts["start"]
                buffer_end = ts["end"]

        # 最后一个块
        if buffer_end > buffer_start:
            sub_segments.append((buffer_start, buffer_end))

        # 如果 VAD 切分后仍然有超长块，算术均分兜底
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
                    if chunk_end - chunk_start >= 0.3:  # 最小 0.3s
                        final_sub_segs.append((chunk_start, chunk_end))

        if not final_sub_segs:
            new_segments.append(seg)
            continue

        # 将子段的时间偏移回全局时间轴
        seg_abs_start = seg["start"]
        for i, (sub_start, sub_end) in enumerate(final_sub_segs):
            abs_start = round(seg_abs_start + sub_start, 3)
            abs_end = round(seg_abs_start + sub_end, 3)
            dur = round(abs_end - abs_start, 3)
            if dur < 0.3:
                continue

            sub_seg = {
                **seg,  # 继承原段的 speaker, video_id 等
                "start": abs_start,
                "end": abs_end,
                "duration": dur,
                "is_overlap": seg.get("is_overlap", False),
                "overlap_speakers": list(seg.get("overlap_speakers", [])),
                # 重置 segment_id / segment_path (后续写入时重新生成)
                "segment_id": "",
                "segment_path": "",
                "original_speaker": seg.get("original_speaker", seg["speaker"]),
            }
            new_segments.append(sub_seg)

    # 按时间排序
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
    检测并标记重叠语音段（原地修改 segments）。

    对所有时间段逐一检查：若某时段内活跃说话人数 > 1，
    则所有涉事段标记 is_overlap=True 并记录 overlap_speakers。
    """
    n = len(segments)
    if n <= 1:
        return

    for i in range(n):
        seg_a = segments[i]
        for j in range(i + 1, n):
            seg_b = segments[j]
            # 检查时间重叠: a.start < b.end AND b.start < a.end
            if seg_a["start"] < seg_b["end"] and seg_b["start"] < seg_a["end"]:
                # 确认有实际重叠（不只是边界接触）
                overlap_start = max(seg_a["start"], seg_b["start"])
                overlap_end = min(seg_a["end"], seg_b["end"])
                if overlap_end - overlap_start > 0.05:  # > 50ms 才算有效重叠
                    seg_a["is_overlap"] = True
                    seg_b["is_overlap"] = True
                    if seg_b["speaker"] not in seg_a["overlap_speakers"]:
                        seg_a["overlap_speakers"].append(seg_b["speaker"])
                    if seg_a["speaker"] not in seg_b["overlap_speakers"]:
                        seg_b["overlap_speakers"].append(seg_a["speaker"])


# ============================================================
# L3: 跨视频说话人关联层
# ============================================================


def _cluster_speakers(
    all_embeddings: Dict[Tuple[str, str], np.ndarray],
    config: dict,
    logger,
) -> Dict[Tuple[str, str], str]:
    """
    跨视频说话人 AHC 凝聚式层次聚类。

    流程:
      1. 维度一致性校验
      2. 计算余弦距离矩阵 (scipy.pdist)
      3. sklearn AgglomerativeClustering (average linkage)
      4. 身份标签映射: 最大簇 → TARGET, 其他 → OTHER_XX
      5. 离群识别: 单样本簇 + 距所有簇中心 > 1.5×阈值 → UNKNOWN

    参数:
        all_embeddings: {(video_id, speaker_label): embedding_vector}
        config:         全局配置
        logger:         日志器

    返回:
        label_map: {(video_id, speaker_label): global_label}
    """
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import AgglomerativeClustering

    diar_cfg = config.get("diarization_ad", {})
    threshold = diar_cfg.get("clustering_threshold", 0.5)
    enable_outlier = diar_cfg.get("enable_outlier_detection", True)

    if not all_embeddings:
        logger.info("无说话人嵌入，跳过跨视频聚类")
        return {}

    keys = list(all_embeddings.keys())
    embeddings = [all_embeddings[k] for k in keys]

    # 0. NaN / Inf / 零向量过滤
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

    # 1. 维度一致性校验
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
        # 单样本直接标记为 TARGET
        if len(embeddings) == 1:
            return {keys[0]: "TARGET"}
        return {}

    emb_matrix = np.stack(embeddings, axis=0)
    n_samples = len(embeddings)
    logger.info(f"跨视频聚类: {n_samples} 个说话人实例, 嵌入维度={embeddings[0].shape[0]}")

    # 2. 计算余弦距离矩阵
    # pdist 返回 compact 格式，squareform 转为方阵
    dist_vector = pdist(emb_matrix, metric="cosine")
    dist_matrix = squareform(dist_vector)

    # 3. 层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
    )
    cluster_labels = clustering.fit_predict(dist_matrix)

    # 4. 按簇内说话人总时长排序 → 标签映射
    # 计算每个簇的总时长
    cluster_duration = {}
    cluster_members = {}
    for idx, c_id in enumerate(cluster_labels):
        c_id = int(c_id)
        if c_id not in cluster_members:
            cluster_members[c_id] = []
        cluster_members[c_id].append(idx)

    # 每个 key 的时长 (从 all_embeddings 之外的元数据补充 -- 此处仅用样本数作为 proxy)
    # 实际聚类基于嵌入，时长信息需从 segments 获取（见 process_all 中的实现）
    # 这里先用样本数排序作为 fallback；process_all 会传入时长权重

    # 按簇大小排序
    cluster_sizes = sorted(cluster_members.items(), key=lambda x: -len(x[1]))
    n_clusters = len(cluster_sizes)

    # 5. 构建标签映射
    label_map = {}
    outlier_threshold = threshold * 1.5

    for rank, (c_id, member_indices) in enumerate(cluster_sizes):
        if rank == 0:
            label = "TARGET"
        else:
            label = f"OTHER_{(rank - 1):02d}"

        # 离群检测：单样本簇 + 距离检查
        if enable_outlier and len(member_indices) == 1:
            idx = member_indices[0]
            # 计算该样本到所有其他样本的最小距离
            distances_to_others = dist_matrix[idx, :]
            # 排除自己
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

    # 6. 日志输出
    logger.info(f"  聚类完成: {n_clusters} 个簇, 阈值={threshold}")
    for label in ["TARGET"] + [f"OTHER_{i:02d}" for i in range(max(0, n_clusters - 1))]:
        members = [k for k, v in label_map.items() if v == label]
        if members:
            logger.info(f"    {label}: {len(members)} 个说话人实例")
            for vid, spk in members[:5]:  # 最多显示 5 个
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

    优先扫描 Demucs 输出目录，按 视频ID/vocals.wav 规则匹配。
    Demucs 目录不存在时兜底扫描原始音频目录。

    返回: [(audio_path, video_id), ...]
    """
    paths = config["paths"]
    demucs_dir = paths.get("demucs_output", "./data/03_demucs_output")
    demucs_model = config.get("demucs", {}).get("model", "htdemucs")

    candidates = []

    # 优先: Demucs 输出
    base = os.path.join(demucs_dir, demucs_model)
    if os.path.isdir(base):
        for video_dir in sorted(os.listdir(base)):
            vocals_path = os.path.join(base, video_dir, "vocals.wav")
            if os.path.isfile(vocals_path):
                candidates.append((vocals_path, video_dir))
        if candidates:
            logger.info(f"输入源: Demucs 输出 ({len(candidates)} 个 vocals.wav)")
            return candidates

    # 兜底: 原始音频目录
    audio_dir = paths.get("extracted_audio", "./data/02_extracted_audio")
    if os.path.isdir(audio_dir):
        audio_files = get_audio_files(audio_dir)
        for af in audio_files:
            video_id = os.path.splitext(os.path.basename(af))[0]
            candidates.append((af, video_id))
        if candidates:
            logger.info(f"输入源: 原始音频目录 ({len(candidates)} 个文件)")
            return candidates

    # 再次兜底: 扫描 raw_videos 对应音频
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
    Stage 04 主流程：跨视频说话人分离。

    1. 发现输入文件 (优先 Demucs vocals.wav)
    2. 加载 pyannote 流水线 (单例，复用)
    3. 逐文件: 预处理 → diarization → 收集片段和嵌入
    4. AHC 跨视频聚类 + 标签映射
    5. 导出切分音频段 + segments_meta.json

    异常隔离: 单文件失败不中断整体流程
    断点续传: 每 N 文件持久化 checkpoint
    """
    if logger is None:
        logger = setup_logger("04_diarization", config["paths"].get("logs", "./logs"))

    diar_cfg = config.get("diarization_ad", {})
    paths = config["paths"]

    out_dir = ensure_dir(paths.get("diarization_output", "./data/04_diarization"))
    seg_dir = ensure_dir(os.path.join(out_dir, "segments"))
    checkpoint_path = os.path.join(out_dir, "_checkpoint.json")
    save_interval = diar_cfg.get("save_interval", 10)

    # ── 1. 文件发现 ──
    candidates = _discover_input_files(config, logger)
    if not candidates:
        logger.warning("没有找到待处理音频文件！请先运行 Stage 01-03。")
        return []

    logger.info("=" * 60)
    logger.info("Stage 04: 跨视频说话人分离")
    logger.info(f"模型: {diar_cfg.get('model', 'pyannote/speaker-diarization-3.1')}")
    logger.info(f"聚类阈值: {diar_cfg.get('clustering_threshold', 0.5)}")
    logger.info(f"待处理: {len(candidates)} 个音频文件")
    logger.info(f"输出目录: {out_dir}")
    logger.info("=" * 60)

    # ── 2. 断点恢复 ──
    completed_videos = set()
    all_segments: List[dict] = []
    all_embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    # 记录每个 (video_id, speaker) 的总时长，用于聚类排序
    speaker_duration: Dict[Tuple[str, str], float] = {}

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            completed_videos = set(ckpt.get("completed_videos", []))
            all_segments = ckpt.get("segments", [])
            # 恢复嵌入 (从 list 转回 dict)
            emb_data = ckpt.get("embeddings", {})
            for key_str, emb_list in emb_data.items():
                # key_str 格式: "video_id|||speaker_label"
                parts = key_str.split("|||")
                if len(parts) == 2:
                    all_embeddings[(parts[0], parts[1])] = np.array(emb_list)
            # 恢复 speaker_duration
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

    # ── 3. 加载 pyannote 流水线 (单例) ──
    logger.info("加载 pyannote 流水线...")
    t0 = time.time()
    pipeline, model_name, device = _load_diarization_pipeline(config)
    logger.info(f"已加载: {model_name} (device={device}, {time.time() - t0:.1f}s)")

    # 启用 pyannote 原生合并：同一说话人间隔 < merge_gap 的段自动合并
    _set_merge_gap(pipeline, config)
    merge_gap = config.get("diarization_ad", {}).get("merge_gap", 0.0)
    if merge_gap > 0:
        logger.info(f"  合并间隙: min_duration_off={merge_gap}s (pyannote 原生)")

    # ── 4. 逐文件处理 ──
    skipped = 0
    failed = 0
    processed = 0

    pbar = tqdm(candidates, desc="说话人分离", unit="file")
    for audio_path, video_id in pbar:
        # 跳过已完成
        if video_id in completed_videos:
            skipped += 1
            pbar.set_postfix({"skip": skipped, "done": processed, "fail": failed})
            continue

        pbar.set_description(f"分离: {video_id[:30]}")

        try:
            # 4a. 音频预处理
            result = _preprocess_audio(audio_path, target_sr=16000)
            if result is None:
                logger.warning(f"  [{video_id}] 音频校验失败，跳过")
                failed += 1
                # 标记为已处理（避免反复重试）
                completed_videos.add(video_id)
                continue

            tensor, sr_16k, raw_audio, raw_sr = result
            audio_duration = raw_audio.shape[0] / raw_sr

            # 4b. 说话人分离
            diar_segments, spk_embs = _run_diarization(
                tensor, sr_16k, pipeline, config
            )

            if not diar_segments:
                logger.warning(f"  [{video_id}] 未检测到语音段")
                completed_videos.add(video_id)
                continue

            max_seg_dur = diar_cfg.get("max_segment_duration", 10.0)

            # 4c. VAD 二次切分: 对 >max_seg_dur 的段按停顿边界切开
            n_before = len(diar_segments)
            diar_segments = _resegment_long_segments(
                diar_segments, raw_audio, raw_sr,
                max_duration=max_seg_dur,
            )
            if len(diar_segments) != n_before:
                logger.info(f"  VAD 切分: {n_before} → {len(diar_segments)} 段")

            # 4d. 收集嵌入和时长
            for spk_label, emb in spk_embs.items():
                key = (video_id, spk_label)
                all_embeddings[key] = emb
                # 累计时长
                spk_dur = sum(
                    s["duration"]
                    for s in diar_segments
                    if s["speaker"] == spk_label
                )
                speaker_duration[key] = speaker_duration.get(key, 0) + spk_dur

            # 4e. 切分音频段 (使用原始采样率音频以保证质量)
            for i, seg in enumerate(diar_segments):
                seg_id = f"{video_id}_{seg['speaker']}_{i:04d}"
                s = int(seg["start"] * raw_sr)
                e = int(seg["end"] * raw_sr)
                # 边界保护
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

            # 日志
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

            processed += 1
            completed_videos.add(video_id)

            # 4e. 断点持久化
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
            completed_videos.add(video_id)  # 标记为已处理，避免反复重试
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
        # 将时长信息融入聚类排序：先聚类，再按簇内总时长排序
        label_map = _cluster_speakers(all_embeddings, config, logger)

        if label_map:
            # 按簇内总时长重排标签 (TARGET 应对应总时长最大的簇)
            # 收集每个全局标签的总时长
            label_duration: Dict[str, float] = {}
            for (vid, spk), global_label in label_map.items():
                dur = speaker_duration.get((vid, spk), 0)
                label_duration[global_label] = label_duration.get(global_label, 0) + dur

            # 如果 TARGET 不是时长最长的簇，交换标签
            if "TARGET" in label_duration and len(label_duration) > 1:
                sorted_labels = sorted(label_duration.items(), key=lambda x: -x[1])
                if sorted_labels[0][0] != "TARGET":
                    # 重新映射：最长时长 → TARGET，原 TARGET → OTHER
                    old_target_label = "TARGET"
                    new_target_label = sorted_labels[0][0]
                    # 重新构建 label_map
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
                        f"  基于时长重排: '{new_target_label}'({label_duration[new_target_label]:.1f}s) "
                        f"↔ 'TARGET'({label_duration.get(old_target_label, 0):.1f}s)"
                    )

            # 应用标签映射到所有段
            for seg in all_segments:
                key = (seg["video_id"], seg.get("original_speaker", seg["speaker"]))
                seg["speaker"] = label_map.get(key, seg["speaker"])

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
            ) + sorted(
                [k for k in final_labels if k == "UNKNOWN"]
            ):
                if lbl in final_labels:
                    info = final_labels[lbl]
                    logger.info(
                        f"  {lbl}: {info['count']} 段, "
                        f"{info['duration']:.1f}s ({info['duration'] / 60:.1f}min)"
                    )

    # ── 7. 保存最终元数据 ──
    meta_path = os.path.join(out_dir, "segments_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    # 删除 checkpoint (流程完整结束)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # ── 8. 总结 ──
    total_dur = sum(s["duration"] for s in all_segments)
    unique_speakers = set(s["speaker"] for s in all_segments)
    logger.info("=" * 60)
    logger.info(
        f"Stage 04 完成: {len(candidates)} 输入 → "
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
    # 将 tuple keys 转为字符串 (JSON 不支持 tuple key)
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
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def main(config=None, logger=None):
    """CLI 入口 (兼容 run_pipeline.py 调用)"""
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("04_diarization", config["paths"].get("logs", "./logs"))

    process_all(config, logger=logger)


if __name__ == "__main__":
    main()
