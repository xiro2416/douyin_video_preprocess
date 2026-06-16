"""
Stage 5: Paraformer-large-zh ASR 转写与文本置信度验证
====================================================
使用 FunASR Paraformer-large-zh 对每个语音段进行转写。
模型从 HuggingFace 下载（通过 hf-mirror.com 加速）。

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
import torch
from tqdm import tqdm

from pipeline.utils import (
    get_logger,
    load_config,
    read_audio,
)


def _resolve_device(config: dict) -> torch.device:
    """
    根据全局 GPU 配置解析推理设备。
    优先级: config.gpu.enabled → config.gpu.cuda_visible → config.gpu.device_id
    """
    gpu_cfg = config.get("gpu", {})
    if not gpu_cfg.get("enabled", True):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")
    cvd = gpu_cfg.get("cuda_visible", "").strip()
    if cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    device_id = gpu_cfg.get("device_id", 0)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > device_id:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")
    return torch.device("cpu")


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
    对单个音频段进行 Paraformer 转写。

    返回:
        {
            "text": "转写文本",
            "language": "zh",
            "duration": float,
            "valid": bool,
            "reason": str
        }
        若无效则 valid=False，reason 说明原因。
    """
    # 读取音频（Paraformer 使用 16kHz mono）
    wav, sr = read_audio(audio_path)
    duration = len(wav) / sr

    # 如果太短，跳过
    if duration < 0.5:
        return {
            "text": "",
            "valid": False,
            "reason": "too_short",
        }

    try:
        # Paraformer 转写（返回 list[dict]）
        result_list = model.generate(input=wav, hotword='', use_itn=True)

        if not result_list:
            return {
                "text": "",
                "valid": False,
                "reason": "empty_result",
            }

        result = result_list[0]
        text = result.get("text", "").strip()

        # 获取时间戳信息（Paraformer 支持 sentence-level timestamp）
        sentence_info = result.get("sentence_info", [])

    except Exception as e:
        if logger:
            logger.warning(f"转写失败 {audio_path}: {e}")
        return {
            "text": "",
            "valid": False,
            "reason": f"transcribe_error: {e}",
        }

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
        "language": "zh",
        "duration": duration,
        "valid": valid,
        "reason": reason,
        "sentence_count": len(sentence_info) if sentence_info else 0,
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
    logger.info(f"模型: Paraformer-large-zh (FunASR, hub={asr_cfg.get('hub', 'hf')})")
    logger.info(f"待转写: {len(segments_meta)} 段")
    logger.info(f"=" * 60)

    # 设置 hf-mirror 加速下载（如果使用 HuggingFace hub）
    if asr_cfg.get("hub", "hf") == "hf":
        hf_endpoint = asr_cfg.get("hf_endpoint", "https://hf-mirror.com")
        os.environ["HF_ENDPOINT"] = hf_endpoint
        logger.info(f"HuggingFace 镜像: {hf_endpoint}")

    # 加载 Paraformer 模型
    logger.info("正在加载 Paraformer-large-zh 模型（首次运行会下载）...")
    from funasr import AutoModel

    device = _resolve_device(config)
    device_str = f"cuda:{device.index}" if device.type == "cuda" else "cpu"
    logger.info(f"推理设备: {device_str}")

    model_kwargs = {
        "model": asr_cfg["model"],
        "hub": asr_cfg.get("hub", "hf"),
        "device": device_str,
    }
    logger.info(f"模型参数: {model_kwargs}")
    model = AutoModel(**model_kwargs)
    logger.info("Paraformer 模型已加载")

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
        "empty_result": 0,
        "transcribe_error": 0,
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
    # 优先 segments_meta.json，fallback 到 segments_meta_community.json
    meta_path = os.path.join(
        config["paths"]["diarization_output"], "segments_meta.json"
    )
    if not os.path.exists(meta_path):
        meta_path = os.path.join(
            config["paths"]["diarization_output"], "segments_meta_community.json"
        )
    if not os.path.exists(meta_path):
        logger.error(f"未找到段元数据 (尝试过 segments_meta.json 和 segments_meta_community.json)")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # 过滤：仅保留 TARGET 且无重叠的片段
    before = len(segments)
    segments = [
        s for s in segments
        if s.get("speaker") == "TARGET" and not s.get("is_overlap", False)
    ]
    logger.info(f"过滤后保留 {len(segments)}/{before} 段 (仅 TARGET + 无重叠)")

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
