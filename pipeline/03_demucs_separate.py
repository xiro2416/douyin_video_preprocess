"""
Stage 3: Demucs 人声分离
=========================
使用 htdemucs + --two-stems vocals 分离出人声茎。
只取 vocals 部分，不额外做后处理。
"""

import os
import subprocess
import sys

from tqdm import tqdm

from pipeline.utils import (
    ensure_dir,
    get_audio_files,
    get_logger,
    load_config,
)




def run_demucs_separation(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs",
    shifts: int = 3,
    overlap: float = 0.25,
    segment: int = 5,
    logger=None,
) -> dict | None:
    """
    对单个音频运行 Demucs 分离（--two-stems vocals）。
    返回 {"vocals": path, "no_vocals": path}，失败返回 None。
    """
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model, basename)
    vocals_path = os.path.join(stem_dir, "vocals.wav")

    # 检查已完成
    if os.path.exists(vocals_path):
        if logger:
            logger.debug(f"Demucs 已处理，跳过: {basename}")
        return {"vocals": vocals_path,
                "no_vocals": os.path.join(stem_dir, "no_vocals.wav")}

    if logger:
        logger.info(f"正在分离: {basename}")

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "--shifts", str(shifts),
        "--overlap", str(overlap),
        "--segment", str(segment),
        "--clip-mode", "rescale",
        "-o", output_dir,
        audio_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=3600)
        if result.returncode != 0:
            if logger:
                stderr_text = result.stderr[-500:].decode('utf-8', errors='replace') if result.stderr else ""
                logger.error(f"Demucs 失败 [{basename}]: {stderr_text}")
            return None
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"Demucs 超时 [{basename}]")
        return None
    except Exception as e:
        if logger:
            logger.error(f"Demucs 异常 [{basename}]: {e}")
        return None

    # 确认输出
    if not os.path.exists(vocals_path):
        if logger:
            logger.error(f"Demucs 未生成 vocals [{basename}]")
        return None

    if logger:
        logger.info(f"✓ {basename}: vocals 已分离")

    return {"vocals": vocals_path,
            "no_vocals": os.path.join(stem_dir, "no_vocals.wav")}


def separate_all(config: dict, logger=None):
    """批量运行 Demucs 分离 + 降噪"""
    paths = config["paths"]
    demucs_cfg = config["demucs"]

    if logger is None:
        logger = get_logger("03_demucs_separate")

    demucs_dir = ensure_dir(paths["demucs_output"])

    audio_files = get_audio_files(paths["extracted_audio"])
    if not audio_files:
        logger.warning("没有找到音频文件，请先运行 02_extract_audio.py")
        return []

    logger.info(f"=" * 60)
    logger.info(f"Stage 3: Demucs 人声分离 + 两级降噪")
    logger.info(f"模型: {demucs_cfg['model']} (--two-stems vocals)")
    logger.info(f"shifts={demucs_cfg['shifts']}, overlap={demucs_cfg['overlap']}")
    logger.info(f"共 {len(audio_files)} 个音频")
    logger.info(f"=" * 60)

    results = []

    for audio_path in tqdm(audio_files, desc="Demucs 分离"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]

        stems = run_demucs_separation(
            audio_path, demucs_dir,
            model=demucs_cfg["model"],
            shifts=demucs_cfg["shifts"],
            overlap=demucs_cfg["overlap"],
            segment=demucs_cfg["segment"],
            logger=logger,
        )

        if stems is None:
            logger.warning(f"✗ {basename}: 分离失败")
            continue

        results.append(stems["vocals"])
        logger.info(f"✓ {basename}")

    logger.info(f"完成: {len(results)}/{len(audio_files)} 个音频分离成功")
    return results


def main():
    config = load_config()
    from pipeline.utils import setup_logger
    logger = setup_logger("03_demucs_separate", config["paths"]["logs"])
    separate_all(config, logger)


if __name__ == "__main__":
    main()
