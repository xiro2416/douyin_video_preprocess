"""
Stage 04b: SNR + SFM + HNR 质量过滤
======================================
读取 Stage 04 输出的 segments_meta_community.json 和 TARGET 片段 WAV，
计算每段的三维质量指标，按可配置阈值丢弃低质量片段：

  1. SNR  — 信噪比 (P10/P90 帧能量法)
  2. SFM  — 频谱平坦度 (谐波 vs 噪声)
  3. HNR  — 谐波噪声比 (声带周期性)

输出: 更新后的 segments_meta_community.json，
      rejected/ 目录下保留丢弃的 WAV 及其 rejected_segments.json 元数据。
"""

import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy import signal as _signal
from scipy.fft import rfft, rfftfreq

from pipeline.utils import (
    ensure_dir,
    get_logger,
    load_config,
    setup_logger,
)


# ═══════════════════════════════════════════════════════════════
# Silero VAD (lazy load)
# ═══════════════════════════════════════════════════════════════

_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
    return _vad_model, _vad_utils


# ═══════════════════════════════════════════════════════════════
# Computation
# ═══════════════════════════════════════════════════════════════

def _resample(wav: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """重采样到 target_sr 并峰值归一化。"""
    g = np.gcd(orig_sr, target_sr)
    wav = _signal.resample_poly(
        wav.astype(np.float64), target_sr // g, orig_sr // g
    ).astype(np.float32)
    peak = np.max(np.abs(wav))
    if peak > 1e-8:
        wav = wav / peak
    return wav


def _get_speech_mask(
    wav_16k: np.ndarray,
    sr: int = 16000,
    vad_threshold: float = 0.5,
) -> np.ndarray:
    """Silero VAD → speech validity mask (bool array)."""
    model, (get_speech_timestamps, _, _, _, _) = _load_vad()
    tensor = torch.from_numpy(wav_16k).float()
    ts = get_speech_timestamps(
        tensor, model,
        sampling_rate=sr, threshold=vad_threshold,
        return_seconds=True,
    )
    mask = np.zeros(len(wav_16k), dtype=bool)
    for t in ts:
        s = max(0, int(t["start"] * sr))
        e = min(len(wav_16k), int(t["end"] * sr))
        mask[s:e] = True
    return mask


def compute_snr(wav_16k: np.ndarray) -> float:
    """
    SNR via energy percentile method.

    噪声底 = P10 帧能量 (最安静的10%)
    信号   = P90 帧能量
    SNR = 10 * log10(P90 / P10)
    """
    frame_len, hop = 512, 256
    total = len(wav_16k)
    n_frames = max(1, (total - frame_len) // hop + 1)
    energies = np.empty(n_frames, dtype=np.float64)

    for i in range(n_frames):
        seg = wav_16k[i * hop : i * hop + frame_len]
        energies[i] = np.sum(seg ** 2)

    p10 = np.percentile(energies, 10)
    p90 = np.percentile(energies, 90)
    p10 = max(p10, 1e-12)
    p90 = max(p90, 1e-12)

    snr = 10.0 * np.log10(p90 / p10)
    return float(np.clip(snr, 0, 60))


def compute_sfm(wav_16k: np.ndarray, speech_mask: np.ndarray,
                n_fft: int = 1024, hop: int = 512,
                eps: float = 1e-12) -> float:
    """
    Spectral Flatness Measure. 0 = pure harmonic, 1 = white noise.

    仅在 VAD 标记的语音帧上计算。
    取所有帧的中位数 SFM。
    """
    speech_signal = wav_16k[speech_mask]
    if len(speech_signal) < n_fft:
        return 1.0

    n_frames = max(1, (len(speech_signal) - n_fft) // hop + 1)
    sfm_vals = []

    for i in range(n_frames):
        frame = speech_signal[i * hop : i * hop + n_fft]
        frame *= np.hanning(n_fft)
        mag = np.abs(rfft(frame))[1:]         # skip DC
        mag = np.maximum(mag, eps)
        power = mag ** 2

        gm = np.exp(np.mean(np.log(power)))
        am = np.mean(power)
        sfm = gm / max(am, eps)
        sfm_vals.append(sfm)

    if not sfm_vals:
        return 1.0
    return float(np.median(sfm_vals))


def compute_hnr(wav_16k: np.ndarray, speech_mask: np.ndarray,
                sr: int = 16000,
                f0_min: int = 80, f0_max: int = 400,
                frame_len: int = 1024, hop: int = 512) -> float:
    """
    HNR via autocorrelation (Boersma 1993).

    在原始音频上逐帧滑动，仅处理 speech_mask 占比 > 50% 的帧，
    取所有帧的中位数 HNR。
    HNR = 10 * log10(r_max / (1 - r_max))
    r_max 是自相关函数在基频滞后范围内的最大值。
    """
    total = len(wav_16k)
    if total < frame_len:
        return 0.0

    n_frames = max(1, (total - frame_len) // hop + 1)
    lag_min = sr // f0_max
    lag_max = sr // f0_min

    hnr_vals = []
    for i in range(n_frames):
        frame = wav_16k[i * hop : i * hop + frame_len]
        frame_mask = speech_mask[i * hop : i * hop + frame_len]

        # 只处理语音帧占比 > 50% 的帧
        if frame_mask.sum() < frame_len * 0.5:
            continue

        frame = frame * np.hanning(frame_len)

        # Autocorrelation
        ac = np.correlate(frame, frame, mode="full")
        ac = ac[len(ac) // 2:]                    # positive lags
        ac /= max(ac[0], 1e-12)                    # normalize r[0]=1

        r_max = np.max(ac[lag_min:lag_max])

        if 0 < r_max < 1:
            hnr = 10.0 * np.log10(r_max / (1.0 - r_max + 1e-12))
            hnr_vals.append(float(np.clip(hnr, 0, 40)))

    if not hnr_vals:
        return 0.0
    return float(np.median(hnr_vals))


# ═══════════════════════════════════════════════════════════════
# Filter logic
# ═══════════════════════════════════════════════════════════════

def filter_target_segments(config: dict, logger=None) -> Dict:
    """
    主入口：读取 Stage 04 输出的 segments_meta，过滤 TARGET 段。

    返回统计 dict:
      {total, keep, discard, by_reason: {...}}
    """
    if logger is None:
        logger = get_logger("04b_snr_filter")

    cfg = config.get("snr_filter", {})
    enabled = cfg.get("enabled", True)
    if not enabled:
        logger.info("snr_filter.enabled = false, 跳过过滤")
        return {"total": 0, "keep": 0, "discard": 0, "skipped": True}

    snr_thresh = cfg.get("snr_threshold", 15.0)
    sfm_thresh = cfg.get("sfm_threshold", 0.01)
    hnr_thresh = cfg.get("hnr_threshold", 6.0)
    vad_thresh = cfg.get("vad_threshold", 0.5)
    min_speech_ratio = cfg.get("min_speech_ratio", 0.1)
    diar_out = config["paths"]["diarization_output"]

    meta_path = os.path.join(diar_out, "segments_meta_community.json")
    seg_dir = os.path.join(diar_out, "segments")

    if not os.path.exists(meta_path):
        logger.warning(f"元数据不存在: {meta_path}")
        return {"total": 0, "keep": 0, "discard": 0, "error": "meta_not_found"}

    # ── 读取元数据 ──
    with open(meta_path, "r", encoding="utf-8") as f:
        all_segments: List[dict] = json.load(f)

    total = len(all_segments)
    target_indices = [
        i for i, s in enumerate(all_segments)
        if s["speaker"] == "TARGET"
    ]
    logger.info(
        f"总 {total} 段，其中 TARGET {len(target_indices)} 段"
    )

    if not target_indices:
        logger.info("无 TARGET 段，跳过")
        return {"total": total, "keep": 0, "discard": 0}

    # ── 备份 ──
    backup_path = meta_path.replace(".json", "_pre_snr_filter.json")
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(all_segments, f, ensure_ascii=False, indent=2)
        logger.info(f"备份元数据 → {backup_path}")

    # ── 逐段分析 ──
    stats = {
        "total": total,
        "target": len(target_indices),
        "keep": 0,
        "discard": 0,
        "by_reason": {
            "snr": 0,          # 低 SNR
            "sfm": 0,          # 高 SFM (噪声谱)
            "hnr": 0,          # 低 HNR (气息/沙哑)
            "speech_ratio": 0, # 非语音段
            "multi": 0,        # 多个维度不达标
        },
        "snr_vals": [],
        "sfm_vals": [],
        "hnr_vals": [],
    }

    # 预加载 VAD
    logger.info("加载 Silero VAD...")
    _load_vad()
    logger.info("Silero VAD 就绪")

    for idx in target_indices:
        seg = all_segments[idx]
        wav_path = seg["segment_path"]
        seg_id = seg["segment_id"]

        if not os.path.exists(wav_path):
            logger.warning(f"  [缺失] {wav_path}")
            continue

        # ── 加载音频 ──
        wav, orig_sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        wav_16k = _resample(wav, orig_sr, 16000)
        speech_mask = _get_speech_mask(wav_16k, 16000, vad_thresh)

        # 非语音段检测
        speech_ratio = speech_mask.sum() / max(len(speech_mask), 1)
        if speech_ratio < min_speech_ratio:
            stats["by_reason"]["speech_ratio"] += 1
            stats["discard"] += 1
            seg["filter_decision"] = f"speech_ratio={speech_ratio:.2f}"
            continue

        # ── 计算 ──
        snr = compute_snr(wav_16k)
        sfm = compute_sfm(wav_16k, speech_mask)
        hnr = compute_hnr(wav_16k, speech_mask)

        seg["snr_db"] = round(snr, 1)
        seg["sfm"] = round(sfm, 4)
        seg["hnr_db"] = round(hnr, 1)

        stats["snr_vals"].append(snr)
        stats["sfm_vals"].append(sfm)
        stats["hnr_vals"].append(hnr)

        # ── 判定 ──
        fail_snr = snr < snr_thresh
        fail_sfm = sfm > sfm_thresh
        fail_hnr = hnr < hnr_thresh
        failures = sum([fail_snr, fail_sfm, fail_hnr])

        if fail_snr or fail_sfm or fail_hnr:
            stats["discard"] += 1

            if failures >= 2:
                stats["by_reason"]["multi"] += 1
                seg["filter_decision"] = (
                    f"multi(snr={seg['snr_db']:.1f} sfm={seg['sfm']:.4f} hnr={seg['hnr_db']:.1f})"
                )
            elif fail_snr:
                stats["by_reason"]["snr"] += 1
                seg["filter_decision"] = f"snr={seg['snr_db']:.1f}<{snr_thresh}"
            elif fail_sfm:
                stats["by_reason"]["sfm"] += 1
                seg["filter_decision"] = f"sfm={seg['sfm']:.4f}>{sfm_thresh}"
            elif fail_hnr:
                stats["by_reason"]["hnr"] += 1
                seg["filter_decision"] = f"hnr={seg['hnr_db']:.1f}<{hnr_thresh}"

            # 移动到 rejected 目录保留
            rejected_seg_dir = ensure_dir(
                os.path.join(diar_out, "rejected", "segments")
            )
            rejected_wav_path = os.path.join(
                rejected_seg_dir, os.path.basename(wav_path)
            )
            try:
                shutil.move(wav_path, rejected_wav_path)
                seg["rejected_wav_path"] = rejected_wav_path
            except OSError as e:
                logger.warning(f"  移动失败 {wav_path}: {e}")
        else:
            stats["keep"] += 1
            seg["filter_decision"] = "keep"

        if (stats["keep"] + stats["discard"]) % 30 == 0:
            logger.info(
                f"  [{stats['keep'] + stats['discard']}/{len(target_indices)}] "
                f"keep={stats['keep']} discard={stats['discard']}"
            )

    # ── 分离保留段与 rejected 段 ──
    filtered = []
    rejected_segments = []
    for s in all_segments:
        if s["speaker"] != "TARGET":
            # 非 TARGET 段无条件保留
            filtered.append(s)
            continue
        if s.get("filter_decision") == "keep":
            # TARGET 通过段：保留 SNR 字段，清理 filter_decision
            s.pop("filter_decision", None)
            filtered.append(s)
        else:
            # TARGET 丢弃段：收集到 rejected 列表
            rejected_segments.append(s)

    # ── 写入 rejected 元数据 ──
    if rejected_segments:
        rejected_dir = ensure_dir(os.path.join(diar_out, "rejected"))
        rejected_meta_path = os.path.join(rejected_dir, "rejected_segments.json")
        for s in rejected_segments:
            s["original_segment_path"] = s.get("original_segment_path", s["segment_path"])
            if "rejected_wav_path" in s:
                s["segment_path"] = s.pop("rejected_wav_path")
        with open(rejected_meta_path, "w", encoding="utf-8") as f:
            json.dump(rejected_segments, f, ensure_ascii=False, indent=2)
        logger.info(f"  已拒绝段元数据: {rejected_meta_path} ({len(rejected_segments)} 段)")

    # ── 写回 ──
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    # ── 日志 ──
    dur_before = sum(s["duration"] for s in all_segments if s["speaker"] == "TARGET")
    dur_after = sum(s["duration"] for s in filtered if s["speaker"] == "TARGET")

    snr_arr = np.array(stats["snr_vals"])
    sfm_arr = np.array(stats["sfm_vals"])
    hnr_arr = np.array(stats["hnr_vals"])

    logger.info(f"\n{'=' * 60}")
    logger.info("SNR + SFM + HNR 过滤完成")
    logger.info(f"{'=' * 60}")
    logger.info(f"  TARGET 段: {stats['target']} → keep {stats['keep']}, discard {stats['discard']}")
    logger.info(f"  总时长:    {dur_before:.1f}s → {dur_after:.1f}s ({dur_after/60:.1f}min)")
    logger.info(f"  ├─ 丢弃原因 ──────────────────────────")
    for reason, count in stats["by_reason"].items():
        if count > 0:
            pct = count / max(stats["discard"], 1) * 100
            logger.info(f"  │  {reason}: {count} ({pct:.1f}%)")
    logger.info(f"  ├─ SNR 分布 ──────────────────────────")
    if len(snr_arr) > 0:
        logger.info(f"  │  均值={snr_arr.mean():.1f}  中位数={np.median(snr_arr):.1f}  "
                     f"范围=[{snr_arr.min():.1f}, {snr_arr.max():.1f}]")
    logger.info(f"  ├─ SFM 分布 ──────────────────────────")
    if len(sfm_arr) > 0:
        logger.info(f"  │  均值={sfm_arr.mean():.4f} 中位数={np.median(sfm_arr):.4f}  "
                     f"范围=[{sfm_arr.min():.4f}, {sfm_arr.max():.4f}]")
    logger.info(f"  ├─ HNR 分布 ──────────────────────────")
    if len(hnr_arr) > 0:
        logger.info(f"  │  均值={hnr_arr.mean():.1f}  中位数={np.median(hnr_arr):.1f}  "
                     f"范围=[{hnr_arr.min():.1f}, {hnr_arr.max():.1f}]")
    logger.info(f"  └─ 阈值: SNR≥{snr_thresh}  SFM≤{sfm_thresh}  HNR≥{hnr_thresh}")
    logger.info(f"  元数据更新: {meta_path} ({len(filtered)} 段)")
    logger.info(f"{'=' * 60}")

    return stats


def main(config=None, logger=None):
    """CLI 入口"""
    if config is None:
        config = load_config()
    if logger is None:
        log_dir = config["paths"].get("logs", "./logs")
        logger = setup_logger("04b_snr_filter", log_dir)
    filter_target_segments(config, logger=logger)


if __name__ == "__main__":
    main()
