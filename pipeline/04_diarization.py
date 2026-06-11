"""
Stage 4: 目标说话人日志 + 声纹验证
====================================
核心流水线：
  1. VAD — 语音活动检测，切割出语音段
  2. ECAPA-TDNN — 提取声纹嵌入，与目标模板比较（本地预下载模型）
  3. 仅保留目标说话人的语音片段
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# 预加载 librosa 子模块（必须在 speechbrain 之前），
# 否则 librosa 的 lazy_loader 调用 inspect.stack() 时会触发 speechbrain 的损坏 lazy import
import librosa.core.audio       # noqa: E402 提前解析 samplerate
import librosa.feature          # noqa: E402
import librosa.effects          # noqa: E402

from pipeline.utils import (
    ensure_dir,
    load_config,
    read_audio,
    write_audio,
)

# module-level logger (unused, kept for backward compatibility)
_logger = None


# ── VAD ───────────────────────────────────────────────────────────

def vad_segments(
    wav: np.ndarray,
    sr: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 300,
) -> List[dict]:
    """
    Silero VAD 语音活动检测。
    返回 [{start, end, duration}, ...] (单位: 秒)
    """
    try:
        import silero_vad
        vad_model, utils = silero_vad.load_silero_vad()
        (get_speech_timestamps, _, _, _, _) = utils

        if sr != 16000:
            wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        else:
            wav_16k = wav

        speech_ts = get_speech_timestamps(
            torch.from_numpy(wav_16k),
            vad_model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )

        segments = []
        for ts in speech_ts:
            segments.append({
                "start": ts["start"],
                "end": ts["end"],
                "duration": ts["end"] - ts["start"],
            })
        return segments
    except (ImportError, NotImplementedError):
        return _energy_vad_fallback(wav, sr)


def _energy_vad_fallback(wav: np.ndarray, sr: int,
                          min_speech_duration_ms: int = 1500,
                          merge_gap_ms: int = 700) -> List[dict]:
    """基于能量的 VAD fallback"""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]

    threshold = np.percentile(energy, 85) * 0.6  # 取第85百分位的60%作为阈值
    threshold = max(threshold, np.mean(energy) * 0.15)
    is_speech = energy > threshold
    min_frames = int(min_speech_duration_ms / 1000 * sr / hop_length)
    merge_frames = int(merge_gap_ms / 1000 * sr / hop_length)

    raw_segments = []
    i = 0
    while i < len(is_speech):
        if is_speech[i]:
            start_frame = i
            while i < len(is_speech) and is_speech[i]:
                i += 1
            end_frame = i
            raw_segments.append((start_frame, end_frame))
        else:
            i += 1

    if raw_segments:
        merged = [raw_segments[0]]
        for seg in raw_segments[1:]:
            if seg[0] - merged[-1][1] < merge_frames:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)
        raw_segments = merged

    segments = []
    for start_frame, end_frame in raw_segments:
        if end_frame - start_frame >= min_frames:
            start_sec = start_frame * hop_length / sr
            end_sec = end_frame * hop_length / sr
            segments.append({"start": start_sec, "end": end_sec, "duration": end_sec - start_sec})
    return segments


# ── 声纹嵌入 ─────────────────────────────────────────────────────

class SpeakerEmbeddingExtractor:
    """ECAPA-TDNN 声纹嵌入提取器（使用本地预下载模型）"""

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            local_model_dir = os.path.abspath("pretrained_models/ecapa_model")
            if not os.path.exists(os.path.join(local_model_dir, "embedding_model.ckpt")):
                raise FileNotFoundError(f"模型文件未找到: {local_model_dir}")

            # monkey-patch: speechbrain Pretrainer 默认用 SYMLINK，Windows 非管理员会失败
            import speechbrain.utils.parameter_transfer as _pt
            import speechbrain.utils.fetching as _fetching
            _orig = _pt.Pretrainer.collect_files
            def _patched(self, default_source=None, local_strategy=None, **kw):
                return _orig(self, default_source=default_source,
                             local_strategy=_fetching.LocalStrategy.COPY, **kw)
            _pt.Pretrainer.collect_files = _patched

            from speechbrain.inference.speaker import SpeakerRecognition
            self.model = SpeakerRecognition.from_hparams(
                source=local_model_dir,
                savedir=local_model_dir,
                run_opts={"device": self.device},
            )
            self.available = True
        except Exception as e:
            print(f"ECAPA-TDNN 加载失败: {e}")
            self.available = False

    def extract(self, wav: np.ndarray, sr: int) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            if sr != 16000:
                wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            else:
                wav_16k = wav
            wav_16k = wav_16k / (np.max(np.abs(wav_16k)) + 1e-8)
            tensor = torch.from_numpy(wav_16k).float().to(self.device)
            embedding = self.model.encode_batch(tensor)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"嵌入提取失败: {e}")
            return None

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        from scipy.spatial.distance import cosine
        return 1 - float(cosine(emb1, emb2))


# ── 说话人注册 ──────────────────────────────────────────────────

def auto_register_speaker(audio_path: str, extractor: SpeakerEmbeddingExtractor) -> Optional[np.ndarray]:
    wav, sr = read_audio(audio_path)
    segs = vad_segments(wav, sr)
    if not segs:
        return None
    segs.sort(key=lambda x: x["duration"], reverse=True)
    embeddings = []
    for seg in segs[:3]:
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        emb = extractor.extract(wav[s:e], sr)
        if emb is not None:
            embeddings.append(emb)
    if not embeddings:
        return None
    template = np.mean(embeddings, axis=0)
    return template / (np.linalg.norm(template) + 1e-8)


def discover_high_quality_segments(
    video_audio_pairs: List[Tuple[str, str]],
    extractor: SpeakerEmbeddingExtractor,
) -> np.ndarray:
    all_embeddings = []
    for video_id, audio_path in tqdm(video_audio_pairs, desc="注册说话人"):
        wav, sr = read_audio(audio_path)
        segs = vad_segments(wav, sr)
        if segs:
            segs.sort(key=lambda x: x["duration"], reverse=True)
            best = segs[0]
            s, e = int(best["start"] * sr), int(best["end"] * sr)
            emb = extractor.extract(wav[s:e], sr)
            if emb is not None:
                all_embeddings.append(emb)
    if not all_embeddings:
        raise RuntimeError("未能从任何视频中提取到声纹嵌入！")
    emb_matrix = np.stack(all_embeddings)
    centroid = np.mean(emb_matrix, axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-8)
    from scipy.spatial.distance import cosine
    dist = np.array([cosine(centroid, emb) for emb in emb_matrix])
    mean_d, std_d = np.mean(dist), np.std(dist)
    filtered = [emb_matrix[i] for i in range(len(emb_matrix)) if dist[i] <= mean_d + 2 * std_d]
    if not filtered:
        filtered = all_embeddings
    final = np.mean(filtered, axis=0)
    return final / (np.linalg.norm(final) + 1e-8)


def match_segments(
    segments: List[dict], wav: np.ndarray, sr: int,
    target_embedding: np.ndarray,
    extractor: SpeakerEmbeddingExtractor,
    similarity_threshold: float = 0.65,
) -> List[dict]:
    matched = []
    for seg in tqdm(segments, desc="声纹匹配", leave=False):
        s, e = int(seg["start"] * sr), int(seg["end"] * sr)
        if e - s < sr:  # < 1s
            continue
        emb = extractor.extract(wav[s:e], sr)
        if emb is None:
            continue
        sim = extractor.cosine_similarity(target_embedding, emb)
        seg["similarity"] = sim
        if sim >= similarity_threshold:
            matched.append(seg)
    return matched


def split_long_segments(
    segments: List[dict],
    wav: np.ndarray,
    sr: int,
    max_duration: float = 10.0,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 300,
    threshold: float = 0.5,
) -> List[dict]:
    """
    将超过 max_duration 秒的段用 VAD（300ms 停顿阈值）重新切分。
    """
    result = []
    for seg in segments:
        dur = seg["duration"]
        if dur <= max_duration:
            result.append(seg)
            continue

        # 提取这段音频
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        sub_wav = wav[s:e]

        # 用更严格的停顿阈值（300ms）重新做 VAD
        sub_segs = vad_segments(
            sub_wav, sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        if not sub_segs:
            # VAD 没切出来 → 算术均分兜底
            n = int(dur / max_duration) + 1
            chunk_len = dur / n
            for i in range(n):
                s = seg["start"] + i * chunk_len
                e = s + chunk_len
                result.append({
                    "start": round(s, 2),
                    "end": round(e, 2),
                    "duration": round(chunk_len, 2),
                })
            continue

        # VAD 切出来的子段，偏移回原始时间轴
        for sub in sub_segs:
            sub["start"] = round(seg["start"] + sub["start"], 2)
            sub["end"] = round(seg["start"] + sub["end"], 2)
            sub["duration"] = round(sub["end"] - sub["start"], 2)
            # 如果子段仍 > 10s → 算术均分兜底
            if sub["duration"] > max_duration:
                n = int(sub["duration"] / max_duration) + 1
                chunk_len = sub["duration"] / n
                for i in range(n):
                    s = sub["start"] + i * chunk_len
                    e = s + chunk_len
                    result.append({
                        "start": round(s, 2),
                        "end": round(e, 2),
                        "duration": round(chunk_len, 2),
                    })
            else:
                result.append(sub)

    return result


# ── 主流程 ───────────────────────────────────────────────────────

def process_all(config: dict, logger=None):
    from pipeline.utils import setup_logger
    log = logger or setup_logger("04_diarization", config["paths"]["logs"])
    paths = config["paths"]
    diar_cfg = config["diarization"]
    vad_cfg = config["vad"]
    reg_cfg = config["speaker_registration"]

    demucs_dir = paths["demucs_output"]
    demucs_model = config["demucs"]["model"]
    diar_dir = ensure_dir(paths["diarization_output"])
    seg_dir = ensure_dir(paths["segments"])

    vocals_files = []
    if os.path.exists(os.path.join(demucs_dir, demucs_model)):
        for vd in os.listdir(os.path.join(demucs_dir, demucs_model)):
            vp = os.path.join(demucs_dir, demucs_model, vd, "vocals.wav")
            if os.path.exists(vp):
                vocals_files.append(vp)
    if not vocals_files:
        from pipeline.utils import get_audio_files
        vocals_files = get_audio_files(paths["extracted_audio"])
    if not vocals_files:
        log.warning("没有找到音频文件！")
        return [], None

    log.info(f"=" * 60)
    log.info(f"Stage 4: 说话人日志 + 声纹验证")
    log.info(f"共 {len(vocals_files)} 个音频")
    log.info(f"相似度阈值: {diar_cfg['similarity_threshold']}")
    log.info(f"VAD 最短语音: {vad_cfg['min_speech_duration_ms']}ms")
    log.info(f"=" * 60)

    # 加载 ECAPA-TDNN
    log.info("正在加载 ECAPA-TDNN 声纹模型...")
    extractor = SpeakerEmbeddingExtractor()
    use_speaker_matching = extractor.available
    if not use_speaker_matching:
        log.warning("声纹模型加载失败，跳过声纹匹配，保留所有 VAD 段")

    # 目标说话人注册
    target_embedding = None
    if use_speaker_matching:
        log.info("正在自动注册目标说话人声纹...")
        if reg_cfg["method"] == "auto":
            pairs = [(os.path.splitext(os.path.basename(os.path.dirname(ap)))[0], ap)
                     for ap in vocals_files[:20]]
            try:
                target_embedding = discover_high_quality_segments(pairs, extractor)
                tmpl_path = os.path.join(paths["dataset"], "speaker_embedding.npy")
                ensure_dir(paths["dataset"])
                np.save(tmpl_path, target_embedding)
                log.info(f"说话人模板已保存: {tmpl_path}")
            except RuntimeError as e:
                log.warning(f"说话人注册失败: {e}")
                use_speaker_matching = False
        else:
            tmpl_path = os.path.join(paths["dataset"], "speaker_embedding.npy")
            if os.path.exists(tmpl_path):
                target_embedding = np.load(tmpl_path)
                log.info(f"已加载现有说话人模板: {tmpl_path}")
            else:
                log.warning("手动模式但未找到模板文件")
                use_speaker_matching = False

    # 逐音频处理
    all_segments = []
    for audio_path in tqdm(vocals_files, desc="VAD & 声纹匹配"):
        video_id = os.path.splitext(os.path.basename(os.path.dirname(audio_path)))[0]
        log.info(f"处理: {video_id}")
        wav, sr = read_audio(audio_path)
        segments = vad_segments(wav, sr, threshold=vad_cfg["threshold"],
                                 min_speech_duration_ms=vad_cfg["min_speech_duration_ms"],
                                 min_silence_duration_ms=vad_cfg["min_silence_duration_ms"])
        if not segments:
            log.warning(f"  未检测到语音: {video_id}")
            continue

        # 切分超过 10s 的长段
        before_split = len(segments)
        segments = split_long_segments(segments, wav, sr, max_duration=10.0)
        if len(segments) != before_split:
            log.info(f"  切分: {before_split}段 → {len(segments)}段")

        if use_speaker_matching and target_embedding is not None:
            matched = match_segments(segments, wav, sr, target_embedding, extractor,
                                      similarity_threshold=diar_cfg["similarity_threshold"])
            log.info(f"  VAD: {len(segments)}段 → 匹配: {len(matched)}段")
            source_segments = matched
        else:
            log.info(f"  VAD: {len(segments)} 段（跳过声纹匹配）")
            source_segments = segments
        for i, seg in enumerate(source_segments):
            seg_id = f"{video_id}_seg{i:04d}"
            s, e = int(seg["start"] * sr), int(seg["end"] * sr)
            seg_path = os.path.join(seg_dir, f"{seg_id}.wav")
            write_audio(seg_path, wav[s:e], sr)
            seg["segment_path"] = seg_path
            seg["segment_id"] = seg_id
            seg["video_id"] = video_id
            all_segments.append(seg)

    meta_path = os.path.join(diar_dir, "segments_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    log.info(f"说话人日志完成: 共 {len(all_segments)} 个语音段 | {meta_path}")
    return all_segments, target_embedding


def main():
    config = load_config()
    process_all(config)


if __name__ == "__main__":
    main()
