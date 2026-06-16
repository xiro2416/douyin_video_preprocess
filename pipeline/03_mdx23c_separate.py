"""
Stage 3: MDX23C 人声/背景分离（ONNX Runtime）
===============================================
使用 audio_separator 库 (ONNX Runtime 后端) 进行语音/背景分离。

MDX23C 模型来自 UVR 项目（Politrees/UVR_resources），相比 Demucs
htdemucs_ft 提供更高的分离质量（SDR 更高，背景残留更少）。

模型下载策略：
  - 优先从 HuggingFace 镜像 (hf-mirror.com) 下载到本地 pretrained_models/
  - 设置 AUDIO_SEPARATOR_MODEL_DIR 环境变量使 audio_separator 使用本地模型，
    避免触发其默认的 GitHub 下载（在中国大陆可能不稳定）

与旧 Stage 03 (Demucs) 的接口完全兼容：
  - 入口函数：separate_all(config, logger)
  - 输出格式：vocals.wav + no_vocals.wav
"""

import logging
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    get_audio_files,
    get_logger,
    load_config,
)

# ── 本地模型缓存目录（项目根目录下 pretrained_models/）──
_LOCAL_MODEL_REPO = Path(__file__).resolve().parent.parent / "pretrained_models"


def _ensure_model_local(mdx23c_cfg: dict, logger) -> Path:
    """
    确保 MDX23C 模型文件已存在于本地。

    如果本地没有，从 HuggingFace 镜像 (hf-mirror.com) 下载。
    model_repo 中的文件名可能包含子目录前缀（如 "MDX23C_models/xxx.ckpt"），
    下载时自动展开到扁平目录，以便 audio_separator 直接搜索。

    Returns:
        模型文件所在目录的 Path。
    """
    local_dir = Path(
        mdx23c_cfg.get("local_model_dir", str(_LOCAL_MODEL_REPO))
    ).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    model_filename = mdx23c_cfg["model"]  # e.g. "MDX23C-8KFFT-InstVoc_HQ_2.ckpt"
    model_full_path = local_dir / model_filename

    if model_full_path.exists():
        logger.info(f"模型文件已存在: {model_full_path}")
        return local_dir

    # ── 从 HuggingFace 镜像下载 ──
    from huggingface_hub import hf_hub_download

    hf_endpoint = mdx23c_cfg.get("hf_endpoint", "https://hf-mirror.com")
    old_endpoint = os.environ.get("HF_ENDPOINT", "")
    os.environ["HF_ENDPOINT"] = hf_endpoint

    model_repo = mdx23c_cfg.get("model_repo", "Politrees/UVR_resources")
    # HuggingFace repo 中模型文件的实际路径（可能带子目录）
    repo_filename = mdx23c_cfg.get("repo_filename", model_filename)

    logger.info(f"从 HuggingFace 下载模型 (镜像: {hf_endpoint})")
    logger.info(f"仓库: {model_repo}, 文件: {repo_filename}")

    try:
        hf_hub_download(
            repo_id=model_repo,
            filename=repo_filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        # 如果 repo 中的文件在子目录里，hf_hub_download 会在 local_dir 下
        # 创建子目录。我们需要将 .ckpt 文件移到扁平目录以便 audio_separator 找到。
        expected_path = local_dir / repo_filename
        if expected_path.exists() and not model_full_path.exists():
            expected_path.rename(model_full_path)
            # 删除可能遗留的空子目录
            parent_dir = expected_path.parent
            if parent_dir != local_dir:
                try:
                    parent_dir.rmdir()
                except OSError:
                    pass  # 如果目录非空则忽略
    except Exception as e:
        os.environ["HF_ENDPOINT"] = old_endpoint
        logger.error(f"模型下载失败: {e}")
        raise RuntimeError(
            f"无法从 {hf_endpoint} 下载 MDX23C 模型。\n"
            f"请检查: 1) 网络连接 2) HuggingFace 镜像可用性 3) 仓库中是否存在文件 {repo_filename}"
        ) from e

    # 恢复原始 HF_ENDPOINT
    if old_endpoint:
        os.environ["HF_ENDPOINT"] = old_endpoint
    elif "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]

    if not model_full_path.exists():
        raise RuntimeError(
            f"模型文件下载后仍不存在: {model_full_path}\n"
            f"请检查 repo_filename 配置是否正确。"
        )

    logger.info(f"模型下载完成: {model_full_path}")
    return local_dir


def run_mdx23c_separation(
    audio_path: str,
    output_dir: str,
    separator,
    model_name: str = "MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
    logger=None,
) -> dict | None:
    """
    对单个音频文件运行 MDX23C 分离。

    使用 audio_separator 库进行 ONNX Runtime 推理。
    输出文件重命名为标准的 vocals.wav / no_vocals.wav。

    Args:
        audio_path: 输入音频文件路径
        output_dir: 模型输出根目录
        separator: 已加载的 Separator 实例
        model_name: 模型文件名（用于构造子目录名）
        logger: 日志器

    Returns:
        {"vocals": str, "no_vocals": str} 或 None（失败时）
    """
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    model_short_name = model_name.replace(".ckpt", "").rsplit("/", 1)[-1]
    stem_dir = os.path.join(output_dir, model_short_name, basename)
    os.makedirs(stem_dir, exist_ok=True)

    vocals_path = os.path.join(stem_dir, "vocals.wav")
    no_vocals_path = os.path.join(stem_dir, "no_vocals.wav")

    # ── 跳过已处理的文件 ──
    if os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
        if logger:
            logger.debug(f"MDX23C 已处理，跳过: {basename}")
        return {"vocals": vocals_path, "no_vocals": no_vocals_path}

    if logger:
        logger.info(f"正在分离: {basename}")

    try:
        # audio_separator 将结果写入其初始化时设定的 output_dir。
        # 运行时 output_dir 是不可变的，所以分离后从 flat 目录中
        # 找到文件并移动到正确的子目录。
        output_files = separator.separate(audio_path)

        if logger:
            logger.debug(f"MDX23C 原始输出: {output_files}")

        # audio_separator 返回的是相对路径（仅文件名），
        # 补全为 output_dir 中的绝对路径
        sep_output_dir = separator.output_dir
        output_file_paths = [
            os.path.join(sep_output_dir, f) if not os.path.isabs(f) else f
            for f in output_files
        ]

        # ── 识别 vocals 和 instrumental 文件 ──
        # audio_separator 的输出命名模式："{basename}_(StemName)_{model}.wav"
        # MDX23C 2-stem 模型输出：instrumental + vocals
        vocals_file = None
        instrumental_file = None
        for fp in output_file_paths:
            fname = os.path.basename(fp)
            fname_lower = fname.lower()
            if "vocals" in fname_lower or " vocal " in fname_lower:
                vocals_file = fp
            elif "instrumental" in fname_lower or "no_vocals" in fname_lower or "no_vocal" in fname_lower:
                instrumental_file = fp

        if vocals_file is None or instrumental_file is None:
            # 回退：按位置推断（[instrumental, vocals] 是常见顺序）
            if len(output_file_paths) >= 2:
                instrumental_file = output_file_paths[0]
                vocals_file = output_file_paths[1]
                if logger:
                    logger.warning(
                        f"未能根据文件名识别输出，按位置推断: {output_file_paths}"
                    )
            else:
                raise RuntimeError(
                    f"无法识别分离输出文件"
                    f"（期望 ≥2 个，实际 {len(output_file_paths)}）"
                )

        # ── 移动到标准路径 ──
        shutil.move(vocals_file, vocals_path)
        shutil.move(instrumental_file, no_vocals_path)

        if logger:
            logger.info(f"✓ {basename}: vocals 已分离")

        return {"vocals": vocals_path, "no_vocals": no_vocals_path}

    except Exception as e:
        if logger:
            logger.error(f"MDX23C 失败 [{basename}]: {e}", exc_info=True)
        return None


def separate_all(config: dict, logger=None):
    """
    批量运行 MDX23C 人声/背景分离。

    与旧版 Demucs 入口签名完全一致：
        separate_all(config, logger)

    零输入或全部失败时抛异常终止流水线。
    """
    paths = config["paths"]
    mdx23c_cfg = config["mdx23c"]

    if logger is None:
        logger = get_logger("03_mdx23c_separate")

    # ── 输出目录 ──
    voice_sep_dir = ensure_dir(
        paths.get("voice_sep_output", paths.get("demucs_output", "./data/03_mdx23c_output"))
    )

    # ── 发现输入文件 ──
    audio_files = get_audio_files(paths["extracted_audio"])
    if not audio_files:
        raise RuntimeError(
            "Stage 03 错误：没有找到音频文件。请先运行 Stage 02 (音频提取)。"
        )

    # ── GPU 设备选择（遵循全局 gpu 配置）──
    gpu_cfg = config.get("gpu", {})
    if not gpu_cfg.get("enabled", True):
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        cvd = gpu_cfg.get("cuda_visible", "").strip()
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        device_id = gpu_cfg.get("device_id", 0)
        if torch.cuda.is_available():
            device = f"cuda:{device_id}" if torch.cuda.device_count() > device_id else "cuda"
        else:
            device = "cpu"

    model_name = mdx23c_cfg["model"]
    model_short = model_name.replace(".ckpt", "").rsplit("/", 1)[-1]

    logger.info("=" * 60)
    logger.info("Stage 3: MDX23C 人声/背景分离 (ONNX Runtime)")
    logger.info(f"模型: {model_name}")
    logger.info(f"设备: {device}")
    logger.info(
        f"segment={mdx23c_cfg.get('segment', 256)}, "
        f"overlap={mdx23c_cfg.get('overlap', 8)}"
    )
    logger.info(f"共 {len(audio_files)} 个音频")
    logger.info("=" * 60)

    # ── 1. 确保模型文件在本地 ──
    logger.info("检查模型文件...")
    local_model_dir = _ensure_model_local(mdx23c_cfg, logger)

    # ── 2. 设置环境变量使 audio_separator 使用本地模型 ──
    os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = str(local_model_dir)

    # ── 3. 初始化 Separator ──
    from audio_separator.separator import Separator

    logger.info("加载 MDX23C 模型...")

    # 抑制 audio_separator 内部过多的日志
    logging.getLogger("audio_separator").setLevel(logging.WARNING)

    separator = Separator(
        output_dir=voice_sep_dir,
        model_file_dir=str(local_model_dir),
        log_level=logging.WARNING,
        output_format="WAV",
        normalization_threshold=0.9,
        mdxc_params={
            "segment_size": mdx23c_cfg.get("segment", 256),
            "overlap": mdx23c_cfg.get("overlap", 8),
            "batch_size": 1,
        },
    )

    try:
        separator.load_model(model_filename=model_name)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise RuntimeError(f"无法加载 MDX23C 模型 {model_name}: {e}") from e

    logger.info(f"模型已加载: {model_name}")

    # ── 4. 逐个处理音频文件 ──
    results = []

    for audio_path in tqdm(audio_files, desc="MDX23C 分离"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]

        stems = run_mdx23c_separation(
            audio_path,
            voice_sep_dir,
            separator=separator,
            model_name=model_name,
            logger=logger,
        )

        if stems is None:
            logger.warning(f"✗ {basename}: 分离失败")
            continue

        results.append(stems["vocals"])
        logger.info(f"✓ {basename}")

    logger.info(f"完成: {len(results)}/{len(audio_files)} 个音频分离成功")

    if len(results) == 0 and len(audio_files) > 0:
        raise RuntimeError(
            f"Stage 03 错误：所有 {len(audio_files)} 个音频分离均失败。"
            f" 请检查模型文件是否完整（pretrained_models/ 目录）或 GPU 显存是否充足。"
        )

    return results


def main():
    config = load_config()
    from pipeline.utils import setup_logger

    logger = setup_logger("03_mdx23c_separate", config["paths"]["logs"])
    separate_all(config, logger)


if __name__ == "__main__":
    main()
