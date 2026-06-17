"""
Stage 5: Paraformer-large-zh ASR 转写与文本+置信度验证
=====================================================
使用 FunASR Paraformer-large-zh 对每个语音段进行转写。
模型从 HuggingFace 下载（通过 hf-mirror.com 加速）。

验证策略：
  1. 文本不为空且包含中文字符（中文博主）
  2. 文本长度合理（太短可能是噪声）
  3. 置信度 threshold（token-level softmax 均值，低于阈值放入 fail 目录）
"""

import inspect
import json
import os
import re
import shutil
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from pipeline.utils import (
    get_logger,
    load_config,
    read_audio,
)


# ═══════════════════════════════════════════════════════════════════
# Paraformer 置信度 patch（class 级）
# ═══════════════════════════════════════════════════════════════════

def patch_paraformer_confidence(model, logger=None) -> bool:
    """
    Class 级替换 Paraformer.inference，插入 token-level 置信度计算。

    在 result_i 字典中添加：
      - confidence    : 所有 token softmax 最大概率的均值
      - confidence_min: 所有 token 中最小的 softmax 概率

    通过 setattr 替换 class 方法（影响所有实例，比 instance patch 可靠）。
    """
    cls = type(model.model)
    source = textwrap.dedent(inspect.getsource(cls.inference))
    lines = source.split("\n")

    # 找到所有 result_i 行（需替换）
    replacements = []
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("result_i = {"):
            continue
        if '"confidence"' in stripped:
            continue          # 已打过补丁
        indent = line[: len(line) - len(stripped)]
        has_ts = '"timestamp"' in stripped

        new_body = (
            f'{indent}probs = torch.softmax(am_scores, dim=-1)\n'
            f'{indent}confs = probs.max(dim=-1)[0]\n'
        )
        if has_ts:
            new_body += (
                f'{indent}result_i = {{"key": key[i], "text": text_postprocessed, '
                f'"timestamp": time_stamp_postprocessed, '
                f'"confidence": round(float(confs.mean().item()), 4), '
                f'"confidence_min": round(float(confs.min().item()), 4)}}'
            )
        else:
            new_body += (
                f'{indent}result_i = {{"key": key[i], "text": text_postprocessed, '
                f'"confidence": round(float(confs.mean().item()), 4), '
                f'"confidence_min": round(float(confs.min().item()), 4)}}'
            )
        replacements.append((idx, indent, new_body))

    if not replacements:
        if logger:
            logger.warning("未找到 result_i 行，无法应用置信度 patch")
        return False

    # 从后往前替换（避免行号偏移）
    for idx, _indent, new_block in reversed(replacements):
        lines[idx] = new_block
    new_source = "\n".join(lines)

    try:
        namespace = dict(cls.inference.__globals__)
        exec(compile(new_source, "<confidence_patch>", "exec"), namespace)
        setattr(cls, "inference", namespace["inference"])
        if logger:
            logger.info(
                f"置信度 patch 已应用 ({len(replacements)} 处替换, class={cls.__name__})"
            )
        return True
    except Exception as e:
        if logger:
            logger.warning(f"置信度 patch 编译失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# 设备 & 文本工具
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# 单段转写
# ═══════════════════════════════════════════════════════════════════

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
            "reason": str,
            "confidence": float or None,
            "confidence_min": float or None,
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
            "confidence": None,
            "confidence_min": None,
        }

    try:
        # Paraformer 转写（返回 list[dict]）
        result_list = model.generate(input=wav, hotword='', use_itn=True)

        if not result_list:
            return {
                "text": "",
                "valid": False,
                "reason": "empty_result",
                "confidence": None,
                "confidence_min": None,
            }

        result = result_list[0]
        text = result.get("text", "").strip()
        confidence = result.get("confidence")
        confidence_min = result.get("confidence_min")

        # 获取时间戳信息
        sentence_info = result.get("sentence_info", [])

    except Exception as e:
        if logger:
            logger.warning(f"转写失败 {audio_path}: {e}")
        return {
            "text": "",
            "valid": False,
            "reason": f"transcribe_error: {e}",
            "confidence": None,
            "confidence_min": None,
        }

    # ── 验证逻辑 ──
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

    # 5. 置信度阈值检查（仅在文本验证通过后，且 patch 生效时有值）
    if valid:
        threshold = config.get("asr", {}).get("confidence_threshold", 0.0)
        if threshold > 0 and confidence is not None and confidence < threshold:
            valid = False
            reason = f"low_confidence: {confidence:.4f} < {threshold}"

    return {
        "text": text,
        "language": "zh",
        "duration": duration,
        "valid": valid,
        "reason": reason,
        "confidence": confidence,
        "confidence_min": confidence_min,
        "sentence_count": len(sentence_info) if sentence_info else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# 批量转写
# ═══════════════════════════════════════════════════════════════════

def transcribe_all(
    segments_meta: List[dict],
    config: dict,
    logger=None,
) -> List[dict]:
    """
    对所有段运行 ASR 转写并验证。

    参数:
        segments_meta: Stage 04/04b 输出的段列表

    返回:
        验证通过的段列表，每项附加 ASR 结果
    """
    if logger is None:
        logger = get_logger("05_asr_transcribe")

    asr_cfg = config["asr"]

    logger.info(f"=" * 60)
    logger.info(f"Stage 5: ASR 转写与验证")
    logger.info(f"模型: Paraformer-large-zh (FunASR, hub={asr_cfg.get('hub', 'hf')})")
    threshold = asr_cfg.get("confidence_threshold", 0.0)
    if threshold > 0:
        logger.info(f"置信度阈值: {threshold}（低于此值放入 fail 目录）")
    else:
        logger.info(f"置信度过滤: 未启用")
    logger.info(f"待转写: {len(segments_meta)} 段")
    logger.info(f"=" * 60)

    # 设置 hf-mirror 加速下载
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
    # 标点恢复模型
    punc_model = asr_cfg.get("punc_model")
    if punc_model:
        model_kwargs["punc_model"] = punc_model
        punc_kwargs = asr_cfg.get("punc_kwargs")
        if punc_kwargs:
            model_kwargs["punc_kwargs"] = punc_kwargs
            logger.info(f"标点模型参数: {punc_kwargs}")
        logger.info(f"标点模型: {punc_model}")
    logger.info(f"模型参数: {model_kwargs}")
    model = AutoModel(**model_kwargs)
    logger.info("Paraformer 模型已加载")

    # 应用置信度 patch
    if threshold > 0:
        patched = patch_paraformer_confidence(model, logger)
        if not patched:
            logger.warning("置信度 patch 失败，将不使用置信度过滤")
            threshold = 0.0  # 降级：不启用置信度过滤
    else:
        logger.info("置信度 patch 跳过（未启用）")

    paths = config["paths"]
    asr_audio_dir = os.path.join(paths["asr_output"], "audio")
    asr_fail_dir = os.path.join(paths["asr_output"], "fail")
    os.makedirs(asr_audio_dir, exist_ok=True)
    os.makedirs(asr_fail_dir, exist_ok=True)

    passed = []
    failed_all = []       # 记录所有失败段（含 confidence 失败）
    failed_stats = {
        "too_short": 0,
        "empty_text": 0,
        "text_too_short": 0,
        "no_chinese": 0,
        "non_linguistic": 0,
        "empty_result": 0,
        "transcribe_error": 0,
        "low_confidence": 0,
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
            # 复制音频到 ASR 输出目录（跳过源=目标的情况）
            asr_path = os.path.join(asr_audio_dir, f"{seg['segment_id']}.wav")
            if os.path.normpath(audio_path) != os.path.normpath(asr_path):
                shutil.copy2(audio_path, asr_path)
            seg["asr_path"] = asr_path
            passed.append(seg)
        else:
            reason = result.get("reason", "unknown")
            # 分类统计
            matched = False
            for key in failed_stats:
                if reason.startswith(key):
                    failed_stats[key] += 1
                    matched = True
                    break
            if not matched:
                failed_stats["other"] = failed_stats.get("other", 0) + 1

            # 置信度失败 + 文本验证通过 → 复制到 fail 目录
            if reason.startswith("low_confidence"):
                fail_path = os.path.join(asr_fail_dir, f"{seg['segment_id']}.wav")
                if os.path.normpath(audio_path) != os.path.normpath(fail_path):
                    shutil.copy2(audio_path, fail_path)
                seg["asr_path"] = fail_path
            # 文本验证失败也记录但不存音频（通常是噪声/空白，无保留价值）

            failed_all.append(seg)

    # 保存失败段元数据
    if failed_all:
        fail_meta_path = os.path.join(paths["asr_output"], "asr_failed_meta.json")
        with open(fail_meta_path, "w", encoding="utf-8") as f:
            # 只保留关键字段
            fail_export = []
            for s in failed_all:
                asr = s.get("asr", {})
                fail_export.append({
                    "segment_id": s["segment_id"],
                    "video_id": s.get("video_id", ""),
                    "duration": s.get("duration", 0),
                    "speaker": s.get("speaker", ""),
                    "snr_db": s.get("snr_db"),
                    "sfm": s.get("sfm"),
                    "hnr_db": s.get("hnr_db"),
                    "valid": asr.get("valid", False),
                    "reason": asr.get("reason", ""),
                    "text": asr.get("text", ""),
                    "confidence": asr.get("confidence"),
                    "confidence_min": asr.get("confidence_min"),
                })
            json.dump(fail_export, f, ensure_ascii=False, indent=2)
        logger.info(f"失败段元数据保存至: {fail_meta_path}")

    # 报告
    logger.info(f"有效段: {len(passed)} / 总段: {len(segments_meta)}")
    for reason, count in sorted(failed_stats.items()):
        if count > 0:
            logger.info(f"  - {reason}: {count}")
    if threshold > 0:
        conf_fails = failed_stats.get("low_confidence", 0)
        if conf_fails > 0:
            logger.info(f"  → 低置信度音频已复制到: {asr_fail_dir}")

    # 示例输出
    if passed:
        logger.info(f"转写示例（前 3 条）:")
        for p in passed[:3]:
            text = p['asr']['text'][:80]
            conf = p['asr'].get('confidence')
            conf_str = f" conf={conf:.4f}" if conf is not None else ""
            logger.info(f"  [{p['segment_id']}] {text}{conf_str}")

    return passed


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

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

    # 保存 ASR 通过段
    out_path = os.path.join(
        config["paths"]["asr_output"], "asr_passed_meta.json"
    )
    os.makedirs(config["paths"]["asr_output"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)
    logger.info(f"ASR 通过段结果保存至: {out_path}")


if __name__ == "__main__":
    main()
