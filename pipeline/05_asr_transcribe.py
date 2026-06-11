"""
Stage 5: Whisper ASR 转写与文本置信度验证
===========================================
使用 OpenAI Whisper large-v3 对每个语音段进行转写。
验证策略：
  1. 文本不为空且包含中文字符（中文博主）
  2. 文本长度合理（太短可能是噪声）
"""

import json
import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from pipeline.utils import (
    get_logger,
    load_config,
    read_audio,
)


def contains_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[一-鿿㐀-䶿]', text))


def transcribe_segment(
    audio_path: str,
    model,
    config: dict,
    logger=None,
) -> Optional[Dict]:
    """
    对单个音频段进行 Whisper 转写。

    返回:
        {
            "text": "转写文本",
            "language": "zh",
            "confidence": float,
            "no_speech_prob": float,
            "duration": float,
            "valid": bool,
            "reason": str
        }
        若无效则 valid=False，reason 说明原因。
    """
    asr_cfg = config["asr"]

    # 读取音频
    wav, sr = read_audio(audio_path)

    # 如果太短，Whisper 可能不准
    if len(wav) / sr < 0.5:
        return {
            "text": "",
            "confidence": 0.0,
            "no_speech_prob": 1.0,
            "valid": False,
            "reason": "too_short",
        }

    # Whisper 转写
    result = model.transcribe(
        wav,
        language=asr_cfg["language"],
        temperature=asr_cfg["temperature"],
        no_speech_threshold=asr_cfg["no_speech_threshold"],
        condition_on_previous_text=asr_cfg["condition_on_previous_text"],
        word_timestamps=asr_cfg["word_timestamps"],
        verbose=False,
    )

    # 提取结果
    text = result.get("text", "").strip()
    segments_info = result.get("segments", [])

    # 置信度评估
    no_speech_prob = 0.0
    avg_confidence = 0.0

    if segments_info:
        no_speech_prob = max(
            s.get("no_speech_prob", 0) for s in segments_info
        )
        # 平均 token 置信度
        all_probs = []
        for s in segments_info:
            for token in s.get("tokens", []):
                if isinstance(token, dict) and "probability" in token:
                    all_probs.append(token["probability"])
        if all_probs:
            avg_confidence = float(np.mean(all_probs))

    # 验证逻辑
    valid = True
    reason = "ok"

    # 1. 文本为空
    if not text:
        valid = False
        reason = "empty_text"

    # 2. 文本过短（少于 2 个字符）
    elif len(text) < 2:
        valid = False
        reason = f"text_too_short: {text}"

    # 3. 中文博主，检查是否包含中文
    elif not contains_chinese(text):
        valid = False
        reason = f"no_chinese: {text[:50]}"

    # 4. 纯标点或数字
    elif re.match(r'^[\d\s\W]+$', text):
        valid = False
        reason = f"non_linguistic: {text[:50]}"

    return {
        "text": text,
        "language": result.get("language", "zh"),
        "confidence": avg_confidence,
        "no_speech_prob": no_speech_prob,
        "duration": len(wav) / sr,
        "valid": valid,
        "reason": reason,
        "segments_count": len(segments_info),
    }


def transcribe_all(
    segments_meta: List[dict],
    config: dict,
    logger=None,
) -> List[dict]:
    """
    对所有段运行 ASR 转写并验证。

    参数:
        segments_meta: Stage 04 输出的段列表

    返回:
        验证通过的段列表，每项附加 ASR 结果
    """
    if logger is None:
        logger = get_logger("05_asr_transcribe")

    asr_cfg = config["asr"]

    logger.info(f"=" * 60)
    logger.info(f"Stage 5: ASR 转写与验证")
    logger.info(f"模型: Whisper {asr_cfg['model']}, 语言: {asr_cfg['language']}")
    logger.info(f"待转写: {len(segments_meta)} 段")
    logger.info(f"=" * 60)

    # 加载 Whisper
    logger.info("正在加载 Whisper 模型（首次运行会下载）...")
    import whisper

    model = whisper.load_model(asr_cfg["model"])
    device = "cuda" if hasattr(model, "device") and model.device else "cpu"
    logger.info(f"Whisper 已加载 (device={device})")

    paths = config["paths"]
    asr_audio_dir = os.path.join(paths["asr_output"], "audio")
    os.makedirs(asr_audio_dir, exist_ok=True)

    passed = []
    failed_stats = {
        "too_short": 0,
        "empty_text": 0,
        "text_too_short": 0,
        "no_chinese": 0,
        "non_linguistic": 0,
    }

    for seg in tqdm(segments_meta, desc="ASR 转写"):
        audio_path = seg.get("segment_path", "")
        if not audio_path or not os.path.exists(audio_path):
            failed_stats["no_file"] = failed_stats.get("no_file", 0) + 1
            continue

        result = transcribe_segment(audio_path, model, config, logger=logger)

        if result is None:
            failed_stats["error"] = failed_stats.get("error", 0) + 1
            continue

        seg["asr"] = result

        if result["valid"]:
            # 复制音频到 ASR 输出目录
            asr_path = os.path.join(asr_audio_dir, f"{seg['segment_id']}.wav")
            shutil.copy2(audio_path, asr_path)
            seg["asr_path"] = asr_path
            passed.append(seg)
        else:
            reason = result.get("reason", "unknown")
            # 分类统计
            for key in failed_stats:
                if reason.startswith(key):
                    failed_stats[key] += 1
                    break
            else:
                failed_stats["other"] = failed_stats.get("other", 0) + 1

    # 报告
    logger.info(f"有效段: {len(passed)} / 总段: {len(segments_meta)}")
    for reason, count in sorted(failed_stats.items()):
        if count > 0:
            logger.info(f"  - {reason}: {count}")

    # 示例输出
    if passed:
        logger.info(f"转写示例（前 3 条）:")
        for p in passed[:3]:
            logger.info(f"  [{p['segment_id']}] {p['asr']['text'][:80]}")

    return passed


def main(config=None, logger=None):
    if config is None:
        config = load_config()
    if logger is None:
        from pipeline.utils import setup_logger
        logger = setup_logger("05_asr_transcribe", config["paths"]["logs"])

    # 读取 Stage 04 的说话人日志结果
    meta_path = os.path.join(
        config["paths"]["diarization_output"], "segments_meta.json"
    )
    if not os.path.exists(meta_path):
        logger.error(f"未找到段元数据: {meta_path}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    passed = transcribe_all(segments, config, logger=logger)

    # 保存
    out_path = os.path.join(
        config["paths"]["asr_output"], "asr_passed_meta.json"
    )
    os.makedirs(config["paths"]["asr_output"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)
    logger.info(f"ASR 结果保存至: {out_path}")


if __name__ == "__main__":
    main()
