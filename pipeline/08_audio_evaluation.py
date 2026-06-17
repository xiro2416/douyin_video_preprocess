"""
Stage 8: Qwen2-Audio-7B 多模态音频质量评价
=============================================
使用 Qwen2-Audio-7B-Instruct 对最终数据集的每个音频段
进行自然语言评价，输出中文评价文字。

评价维度:
  1. 语音清晰度（是否清晰可辨）
  2. 背景噪声或音乐残留（是否纯净）
  3. 录音质量（是否有失真、削波、断续等）
  4. 是否适合用于语音合成(TTS)训练
  5. 整体评分(1-5分)

输入: data/06_dataset/metadata.json + audio/*.wav
输出: data/08_audio_evaluation/evaluation_results.json
"""

import json
import os
import time
from typing import Dict, List, Optional

import librosa
import torch
from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    get_logger,
    load_config,
    setup_logger,
)


# ═══════════════════════════════════════════════════════════════
# 设备解析
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# 单段评价
# ═══════════════════════════════════════════════════════════════

def evaluate_segment(
    audio_path: str,
    text: str,
    model,
    processor,
    eval_prompt: str,
    device_str: str,
    max_new_tokens: int = 256,
    logger=None,
) -> Optional[Dict]:
    """
    对单个音频段进行 Qwen2-Audio 评价。

    参数:
        audio_path: WAV 文件路径
        text: 参考文本（ASR 转写结果）
        model: Qwen2AudioForConditionalGeneration 实例
        processor: AutoProcessor 实例
        eval_prompt: 评价模板，包含 {text} 占位符
        device_str: 推理设备字符串 ("cuda:0" / "cpu")
        max_new_tokens: 最大生成 token 数

    返回:
        {
            "segment_id": str,
            "evaluation": str,
            "error": None | str,
        }
        失败时返回 None。
    """
    if not os.path.exists(audio_path):
        if logger:
            logger.warning(f"音频文件不存在: {audio_path}")
        return None

    try:
        # 加载音频（Qwen2-Audio 使用 16kHz mono）
        wav, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

        # 构建 ChatML conversation
        prompt = eval_prompt.format(text=text)
        conversation = [
            {"role": "system", "content": "你是一个专业的语音质量评估助手。"},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]

        # 应用 chat template
        text_input = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # 预处理
        inputs = processor(
            text=text_input,
            audio=[wav],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device_str) for k, v in inputs.items()}

        # 推理
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        evaluation = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "segment_id": os.path.splitext(os.path.basename(audio_path))[0],
            "evaluation": evaluation.strip(),
            "error": None,
        }

    except Exception as e:
        if logger:
            logger.warning(f"评价失败 {audio_path}: {e}")
        return {
            "segment_id": os.path.splitext(os.path.basename(audio_path))[0],
            "evaluation": "",
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════
# 批量评价
# ═══════════════════════════════════════════════════════════════

def evaluate_all(
    metadata: List[dict],
    config: dict,
    logger=None,
) -> List[dict]:
    """
    对 metadata 中所有段运行 Qwen2-Audio 评价。

    参数:
        metadata: Stage 06 输出的 metadata 列表

    返回:
        评价结果列表，每项含 segment_id / evaluation / error
    """
    if logger is None:
        logger = get_logger("08_audio_evaluation")

    eval_cfg = config.get("audio_evaluation", {})
    enabled = eval_cfg.get("enabled", True)
    if not enabled:
        logger.info("audio_evaluation.enabled = false, 跳过")
        return []

    logger.info(f"=" * 60)
    logger.info(f"Stage 8: Qwen2-Audio 多模态音频质量评价")
    logger.info(f"待评价: {len(metadata)} 段")
    logger.info(f"=" * 60)

    # 设置 HuggingFace 镜像
    hf_endpoint = eval_cfg.get("hf_endpoint", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = hf_endpoint
    logger.info(f"HuggingFace 镜像: {hf_endpoint}")

    # 解析设备
    device = _resolve_device(config)
    device_str = f"cuda:{device.index}" if device.type == "cuda" else "cpu"
    logger.info(f"推理设备: {device_str}")

    # 加载 Qwen2-Audio 模型
    logger.info("正在加载 Qwen2-Audio-7B-Instruct（首次运行会自动下载模型）...")
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    model_id = eval_cfg.get("model", "Qwen/Qwen2-Audio-7B-Instruct")
    torch_dtype_str = eval_cfg.get("torch_dtype", "bfloat16")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": "auto"}
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    logger.info(f"模型: {model_id}")
    logger.info(f"精度: {torch_dtype_str}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    logger.info("Qwen2-Audio 模型已加载")

    # 评价 prompt
    default_prompt = (
        '请作为语音质量评估专家，评价这段音频的质量。参考文本是："{text}"。\n'
        "请从以下维度评价：\n"
        "1. 语音清晰度（是否清晰可辨）\n"
        "2. 背景噪声或音乐残留（是否纯净）\n"
        "3. 录音质量（是否有失真、削波、断续等）\n"
        "4. 是否适合用于语音合成(TTS)训练\n"
        "5. 整体评分(1-5分)\n\n"
        "请用中文给出简洁评价。"
    )
    eval_prompt = eval_cfg.get("eval_prompt", default_prompt)
    max_new_tokens = eval_cfg.get("max_new_tokens", 256)

    # 准备输出目录
    out_dir = ensure_dir(config["paths"]["audio_evaluation_output"])

    # 逐段评价
    results = []
    failed = 0
    start_time = time.time()

    # 获取数据集 audio 目录
    audio_base = os.path.join(config["paths"]["dataset"], "audio")

    for item in tqdm(metadata, desc="Qwen2-Audio 评价"):
        # 从 metadata 中获取音频路径
        rel_audio = item.get("audio", "")
        audio_path = os.path.join(config["paths"]["dataset"], rel_audio)
        text = item.get("text", "")
        segment_id = item.get("segment_id", "")

        result = evaluate_segment(
            audio_path, text, model, processor, eval_prompt,
            device_str, max_new_tokens, logger=logger,
        )

        if result is None:
            failed += 1
            continue

        # 补充 metadata 信息
        result["text"] = text
        result["duration"] = item.get("duration", 0)
        result["video_id"] = item.get("video_id", "")

        if result.get("error"):
            failed += 1
        results.append(result)

    elapsed = time.time() - start_time

    # 统计
    total = len(metadata)
    success = len(results) - failed
    logger.info(f"\n{'=' * 60}")
    logger.info(f"评价完成")
    logger.info(f"总段数: {total}")
    logger.info(f"评价成功: {success}")
    logger.info(f"失败: {failed}")
    logger.info(f"耗时: {elapsed:.1f}s ({elapsed / max(success, 1):.2f}s/段)")
    logger.info(f"{'=' * 60}")

    # 示例输出
    if results:
        logger.info(f"评价示例（前 3 条）:")
        for r in results[:3]:
            if r.get("evaluation"):
                preview = r["evaluation"][:100]
                logger.info(f"  [{r['segment_id']}] {preview}...")

    return results


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main(config=None, logger=None):
    if config is None:
        config = load_config()
    if logger is None:
        log_dir = config["paths"].get("logs", "./logs")
        logger = setup_logger("08_audio_evaluation", log_dir)

    # 读取 Stage 06 的输出
    meta_path = os.path.join(config["paths"]["dataset"], "metadata.json")
    if not os.path.exists(meta_path):
        logger.error(f"未找到数据集元数据: {meta_path}")
        logger.error("请先运行 Stage 06 生成数据集")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"读取 {len(metadata)} 条数据集元数据")

    results = evaluate_all(metadata, config, logger=logger)

    # 保存评价结果
    out_path = os.path.join(
        config["paths"]["audio_evaluation_output"],
        "evaluation_results.json",
    )
    os.makedirs(config["paths"]["audio_evaluation_output"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"评价结果已保存: {out_path} ({len(results)} 条)")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Stage 8 完成")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
