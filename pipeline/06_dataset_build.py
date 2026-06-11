"""
Stage 6: 数据集打包 — JSON + 音频目录
=======================================
将 ASR 验证通过的音频段整理为：

  data/06_dataset/
    metadata.json    ← JSON 数组，每项含 text / audio 路径 / duration 等
    audio/           ← 所有 WAV 文件（按 segment_id.wav 命名，扁平存放）

metadata.json 格式:
  [
    {
      "segment_id": "xxx_seg0000",
      "text": "转写文本",
      "audio": "audio/xxx_seg0000.wav",
      "duration": 3.2,
      "video_id": "xxx",
      "language": "ZH",
      "speaker": "target_blogger"
    },
    ...
  ]
"""

import json
import os
from typing import Dict, List

import numpy as np

from pipeline.utils import (
    ensure_dir,
    get_logger,
    load_config,
    read_audio,
    write_audio,
    normalize_loudness,
)


def build_dataset(
    segments_meta: List[dict],
    config: dict,
    logger=None,
) -> list:
    """
    打包为 JSON + audio/ 目录结构。

    参数:
        segments_meta: ASR 验证通过的段列表
        config: 全局配置

    返回:
        metadata list
    """
    if logger is None:
        logger = get_logger("06_dataset_build")

    ds_cfg = config["dataset"]
    paths = config["paths"]
    dataset_dir = ensure_dir(paths["dataset"])
    audio_dir = ensure_dir(os.path.join(dataset_dir, "audio"))

    logger.info(f"=" * 60)
    logger.info(f"Stage 6: 数据集打包")
    logger.info(f"输入: {len(segments_meta)} 段")
    logger.info(f"输出: {dataset_dir}")
    logger.info(f"=" * 60)

    metadata = []

    for seg in segments_meta:
        seg_id = seg.get("segment_id", "")
        text = (seg.get("asr") or {}).get("text", "").strip()
        duration = (seg.get("asr") or {}).get("duration", 0) or seg.get("duration", 0)
        similarity = seg.get("similarity", 0)

        if not text:
            continue

        # 源音频（来自 Stage 05 ASR 输出）
        src = seg.get("asr_path") or seg.get("segment_path", "")
        if not src or not os.path.exists(src):
            logger.warning(f"音频文件不存在，跳过: {src}")
            continue

        # 读取并归一化（峰值 → -26 LUFS 响度）
        dst_name = f"{seg_id}.wav"
        dst_path = os.path.join(audio_dir, dst_name)
        wav, sr = read_audio(src)
        wav = wav / (np.max(np.abs(wav)) + 1e-8)  # 峰值归一化，防削波
        wav = normalize_loudness(wav, sr, target_db=-26.0)  # 响度归一化
        write_audio(dst_path, wav, sr)

        metadata.append({
            "segment_id": seg_id,
            "text": text,
            "audio": os.path.join("audio", dst_name),
            "duration": round(duration, 2),
            "similarity": round(similarity, 4),
            "video_id": seg.get("video_id", ""),
            "language": ds_cfg.get("language", "ZH"),
            "speaker": ds_cfg.get("speaker_name", "target_blogger"),
        })

    # 写 metadata.json
    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 统计
    total_dur = sum(m["duration"] for m in metadata)
    logger.info(f"打包完成")
    logger.info(f"  段数: {len(metadata)}")
    logger.info(f"  总时长: {total_dur:.1f}s ({total_dur/60:.1f}min)")
    logger.info(f"  音频目录: {audio_dir}")
    logger.info(f"  metadata: {meta_path}")

    if total_dur < ds_cfg.get("min_total_minutes", 30) * 60:
        logger.warning(f"⚠ 仅 {total_dur/60:.1f}min，TTS 建议至少 {ds_cfg['min_total_minutes']}min")

    logger.info("前 5 条:")
    for m in metadata[:5]:
        logger.info(f"  [{m['segment_id']}] {m['text'][:60]}")

    return metadata


def main(config=None, logger=None):
    if config is None:
        config = load_config()
    if logger is None:
        from pipeline.utils import setup_logger
        logger = setup_logger("06_dataset_build", config["paths"]["logs"])

    meta_path = os.path.join(config["paths"]["asr_output"], "asr_passed_meta.json")
    if not os.path.exists(meta_path):
        logger.error(f"未找到 ASR 结果: {meta_path}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    build_dataset(segments, config, logger=logger)


if __name__ == "__main__":
    main()
