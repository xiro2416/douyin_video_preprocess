"""
Stage 04-ad: pyannote 多说话人分离
=====================================
使用 pyannote speaker-diarization 模型，对每个段精准切分出各
说话人的时间边界。
uv run python -m 04_diarization_ad.py


核心流水线：
  1. pyannote diarization — 分析整段音频，"谁在什么时候说话"
  2. 输出带 speaker 标签的语音片段

与 Stage 04（声纹匹配）不同：本阶段做的是"分离不同说话人"，
而非"只保留目标说话人"。

输出格式 (segments_meta_ad.json):
  [
    {
      "segment_id": "xxx_spkA_0000",
      "speaker": "SPEAKER_00",
      "start": 1.2,
      "end": 5.0,
      "duration": 3.8,
      "segment_path": "...",
      "video_id": "xxx"
    },
    ...
  ]
"""

import json
import os
import sys
import types as _types
from typing import Dict, List

# ── speechbrain 1.1.0 lazy import 兼容修补 ──
# liborsa/torch 的 inspect.getframeinfo 遍历 sys.modules 时触发 speechbrain
# LazyModule.__getattr__ → ensure_module，后者尝试 import 不存在的模块崩溃。
# 方案：替换 importlib.import_module，对 speechbrain 子模块失败时自动创建 dummy。
import speechbrain as _sb
import importlib as _il
_orig_import_module = _il.import_module
def _patched_import_module(name, package=None):
    try:
        return _orig_import_module(name, package=package)
    except (ImportError, ModuleNotFoundError):
        if name.startswith('speechbrain.'):
            dummy = _types.ModuleType(name)
            dummy.__file__ = '<sb_patch>'
            sys.modules[name] = dummy
            return dummy
        raise
_il.import_module = _patched_import_module

import librosa
import numpy as np
import torch
from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    load_config,
    read_audio,
    write_audio,
)


# ── pyannote diarization ──────────────────────────────────

def load_diarization_pipeline(config: dict):
    """加载 pyannote 说话人分离流水线"""
    from pyannote.audio import Pipeline

    model = config.get("diarization_ad", {}).get("model", "pyannote/speaker-diarization-3.1")
    # token 来源：config > 环境变量
    token = (
        config.get("diarization_ad", {}).get("hf_token", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    )
    kwargs = {}
    if token:
        kwargs["token"] = token

    print(f"[04-ad] 加载 pyannote: {model} ...")
    pipe = None
    # 先尝试本地缓存
    try:
        pipe = Pipeline.from_pretrained(model, **kwargs)
    except Exception:
        pass

    if pipe is None:
        # 回退到 community-1
        model = "pyannote/speaker-diarization-community-1"
        print(f"[04-ad] 回退到: {model}")
        try:
            pipe = Pipeline.from_pretrained(model, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"pyannote 模型加载失败: {e}\n"
                "请访问 https://hf.co/pyannote/speaker-diarization-3.1 接受条款，"
                "然后在 config.yaml 中设置 hf_token。"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(torch.device(device))
    print(f"[04-ad] pyannote 已加载 (device={device})")
    return pipe


def run_diarization(
    audio_path: str,
    pipeline,
    config: dict,
) -> List[dict]:
    """
    对单个音频运行说话人分离。

    返回: [{speaker, start, end, duration}, ...]
    """
    diar_cfg = config.get("diarization_ad", {})
    min_dur = diar_cfg.get("min_segment_duration", 0.5)

    # 读取音频 → 内存（绕过 torchcodec 问题）
    import soundfile as sf
    audio, orig_sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # pyannote 期望 16kHz
    target_sr = 16000
    if orig_sr != target_sr:
        audio = librosa.resample(
            y=audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
        )

    tensor = torch.from_numpy(audio).float().unsqueeze(0)

    # 运行 diarization
    diar_result = pipeline(
        {"waveform": tensor, "sample_rate": target_sr}
    )

    # pyannote 4.x 返回 DiarizeOutput 对象
    # 格式: [(segment, _, speaker_label), ...]
    annotation = diar_result.speaker_diarization
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = round(turn.start, 2)
        end = round(turn.end, 2)
        duration = round(end - start, 2)
        if duration >= min_dur:
            segments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": duration,
            })

    # 按时间排序
    segments.sort(key=lambda s: s["start"])

    # 提取 pyannote 内置的说话人嵌入（每个 SPK 一个向量）
    spk_embeddings = {}
    if hasattr(diar_result, "speaker_embeddings") and diar_result.speaker_embeddings is not None:
        labels = list(annotation.labels())
        emb_matrix = diar_result.speaker_embeddings  # (num_speakers, dim)
        for i, label in enumerate(labels):
            if i < len(emb_matrix):
                spk_embeddings[label] = emb_matrix[i]

    return segments, spk_embeddings


# ── 主流程 ───────────────────────────────────────────────

def process_all(config: dict, logger=None):
    """对 Demucs 输出的所有 vocals.wav 做多说话人分离"""
    from pipeline.utils import setup_logger
    log = logger or setup_logger("04_diarization_ad", config["paths"]["logs"])

    paths = config["paths"]
    demucs_dir = paths["demucs_output"]
    demucs_model = config["demucs"]["model"]
    out_dir = ensure_dir(paths.get("diarization_ad_output", "./data/04_diarization_ad"))
    seg_dir = ensure_dir(os.path.join(out_dir, "segments"))

    # 收集所有 vocals.wav
    vocals_files = []
    base = os.path.join(demucs_dir, demucs_model)
    if os.path.exists(base):
        for vd in os.listdir(base):
            vp = os.path.join(base, vd, "vocals.wav")
            if os.path.exists(vp):
                vocals_files.append(vp)
    if not vocals_files:
        from pipeline.utils import get_audio_files
        vocals_files = get_audio_files(paths["extracted_audio"])
    if not vocals_files:
        log.warning("没有找到音频文件！")
        return []

    log.info(f"=" * 60)
    log.info(f"Stage 04-ad: pyannote 多说话人分离")
    log.info(f"共 {len(vocals_files)} 个音频")
    log.info(f"输出: {out_dir}")
    log.info(f"=" * 60)

    # 加载模型
    pipe = load_diarization_pipeline(config)

    all_segments = []
    all_embeddings = {}  # {(video_id, speaker): np.array}

    for audio_path in tqdm(vocals_files, desc="说话人分离"):
        video_id = os.path.splitext(os.path.basename(os.path.dirname(audio_path)))[0]
        log.info(f"分离: {video_id}")

        diar_segs, spk_embs = run_diarization(audio_path, pipe, config)

        if not diar_segs:
            log.warning(f"  未检测到任何说话人: {video_id}")
            continue

        # 统计各说话人
        speakers = set(s["speaker"] for s in diar_segs)
        log.info(f"  检测到 {len(speakers)} 个说话人: {sorted(speakers)}")

        # 保存说话人嵌入
        for label, emb in spk_embs.items():
            all_embeddings[(video_id, label)] = emb

        # 按 speaker 统计总时长
        for spk in sorted(speakers):
            spk_total = sum(s["duration"] for s in diar_segs if s["speaker"] == spk)
            log.info(f"    {spk}: {spk_total:.1f}s")

        # 读取原始音频（48kHz, 原始时间轴对应）
        import soundfile as sf
        raw_wav, raw_sr = sf.read(audio_path)
        if raw_wav.ndim > 1:
            raw_wav = raw_wav.mean(axis=1)

        # 写出切分段
        for i, seg in enumerate(diar_segs):
            seg_id = f"{video_id}_{seg['speaker']}_{i:04d}"
            s = int(seg["start"] * raw_sr)
            e = int(seg["end"] * raw_sr)
            seg_path = os.path.join(seg_dir, f"{seg_id}.wav")
            write_audio(seg_path, raw_wav[s:e], raw_sr)

            seg["segment_id"] = seg_id
            seg["segment_path"] = seg_path
            seg["video_id"] = video_id

        all_segments.extend(diar_segs)

    # ── 跨视频说话人关联 ──────────────────────────────────
    # pyannote 的 speaker_embeddings 是 192 维，与旧模板 (256维) 不兼容
    # 改为直接用 pyannote 嵌入做跨视频聚类
    if all_embeddings:
        from scipy.spatial.distance import cosine

        # 计算所有 (video, speaker) 两两之间的余弦相似度
        all_keys = list(all_embeddings.keys())
        n = len(all_keys)
        sim_matrix = np.eye(n)
        for i in range(n):
            emb_i = all_embeddings[all_keys[i]]
            emb_i = emb_i / (np.linalg.norm(emb_i) + 1e-8)
            for j in range(i + 1, n):
                emb_j = all_embeddings[all_keys[j]]
                emb_j = emb_j / (np.linalg.norm(emb_j) + 1e-8)
                sim = 1 - float(cosine(emb_i, emb_j))
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # 阈值 0.5 做简单聚类
        threshold = 0.5
        groups = []
        assigned = set()
        for i in range(n):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, n):
                if j not in assigned and sim_matrix[i, j] >= threshold:
                    group.append(j)
                    assigned.add(j)
            groups.append(group)

        # 最大组为 TARGET
        groups.sort(key=lambda g: -len(g))
        label_map = {}
        for g_idx, group in enumerate(groups):
            label = "TARGET" if g_idx == 0 else f"OTHER_{g_idx - 1:02d}"
            for i in group:
                label_map[all_keys[i]] = label

        for seg in all_segments:
            key = (seg["video_id"], seg["speaker"])
            seg["original_speaker"] = seg["speaker"]
            seg["speaker"] = label_map.get(key, seg["speaker"])

        log.info(f"── 跨视频说话人关联 ──")
        log.info(f"  共 {n} 个说话人实例, 聚合为 {len(groups)} 组")
        for g_idx, group in enumerate(groups):
            label = "TARGET" if g_idx == 0 else f"OTHER_{g_idx - 1:02d}"
            keys_in_group = [all_keys[i] for i in group]
            # 过滤掉没有段的空说话人（pyannote 注册了但没有 ≥ min_dur 的段）
            valid_keys = []
            for vid, spk in keys_in_group:
                if any(s["video_id"] == vid and s.get("original_speaker", s["speaker"]) == spk for s in all_segments):
                    valid_keys.append((vid, spk))
            if not valid_keys:
                continue
            dur = sum(s["duration"] for s in all_segments
                      if (s["video_id"], s.get("original_speaker", s["speaker"])) in valid_keys)
            log.info(f"  {label}: {len(valid_keys):d} 个, {dur:.1f}s")
            for vid, spk in valid_keys:
                cnt = sum(1 for s in all_segments
                          if s["video_id"] == vid and s.get("original_speaker", s["speaker"]) == spk)
                sdur = sum(s["duration"] for s in all_segments
                           if s["video_id"] == vid and s.get("original_speaker", s["speaker"]) == spk)
                log.info(f"    ├ {vid:<20s} {spk:<12s} {cnt:3d}段 {sdur:.1f}s")

        target_segs = [s for s in all_segments if s["speaker"] == "TARGET"]
        target_dur = sum(s["duration"] for s in target_segs)
        log.info(f"  TARGET 合计: {len(target_segs)}段, {target_dur:.1f}s ({target_dur/60:.1f}min)")

    # 保存元数据
    meta_path = os.path.join(out_dir, "segments_meta_ad.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    # 最终统计
    if all_segments:
        speakers = set(s["speaker"] for s in all_segments)
        total_dur = sum(s["duration"] for s in all_segments)
        log.info(f"完成: 共 {len(all_segments)} 段, {len(speakers)} 个说话人, {total_dur:.1f}s")
        for spk in sorted(speakers):
            cnt = sum(1 for s in all_segments if s["speaker"] == spk)
            dur = sum(s["duration"] for s in all_segments if s["speaker"] == spk)
            log.info(f"  {spk}: {cnt}段, {dur:.1f}s")
    log.info(f"元数据: {meta_path}")

    return all_segments


def main():
    config = load_config()
    from pipeline.utils import setup_logger
    logger = setup_logger("04_diarization_ad", config["paths"]["logs"])
    process_all(config, logger)


if __name__ == "__main__":
    main()
