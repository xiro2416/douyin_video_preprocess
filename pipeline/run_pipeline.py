"""
全流水线编排入口
=================
一键运行完整数据处理链路。

用法：
  # 完整流水线
  uv run python -m pipeline.run_pipeline --all

  # 单步运行（指定起始阶段）
  uv run python -m pipeline.run_pipeline --from 03

  # 从中间阶段继续
  uv run python -m pipeline.run_pipeline --from 05 --to 07

  # 仅运行 Gradio 抽检
  uv run python -m pipeline.run_pipeline --only-inspect

  # 指定博主 URL（覆盖 config.yaml）
  uv run python -m pipeline.run_pipeline --all --url "https://..."
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

# 确保包在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils import (
    ensure_dir,
    get_logger,
    load_config,
    setup_logger,
)

STAGES = {
    "01": ("下载视频", "01_download", "download_videos"),
    "02": ("提取音频", "02_extract_audio", "extract_all"),
    "03": ("Demucs 分离", "03_demucs_separate", "separate_all"),
    "04": ("说话人日志", "04_diarization", "process_all"),
    "05": ("ASR 转写", "05_asr_transcribe", "main"),
    "06": ("数据集打包", "06_dataset_build", "main"),
}


def run_stage(stage_id: str, config: dict, logger, **kwargs):
    """
    动态导入并运行指定阶段。
    每个阶段模块应包含一个接受 (config, logger) 的入口函数。
    """
    stage_name, module_name, func_name = STAGES[stage_id]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"▶ 阶段 {stage_id}: {stage_name}")
    logger.info(f"{'=' * 60}\n")

    try:
        module = __import__(
            f"pipeline.{module_name}", fromlist=[func_name]
        )
        func = getattr(module, func_name)

        # 调用函数
        result = func(config, logger=logger)

        logger.info(f"\n✓ 阶段 {stage_id} 完成\n")
        return result

    except KeyboardInterrupt:
        logger.warning(f"\n⏹ 阶段 {stage_id} 被用户中断")
        raise
    except Exception as e:
        logger.error(f"\n✗ 阶段 {stage_id} 失败: {e}", exc_info=True)
        raise


def detect_stages_to_run(config: dict) -> list:
    """
    自动检测需要运行的阶段（基于文件存在性）。
    返回需要运行的 stage_id 列表。
    """
    paths = config["paths"]
    pending = []

    # 检查 01: 是否有视频文件
    video_dir = paths["raw_videos"]
    has_videos = False
    if os.path.isdir(video_dir):
        has_videos = any(
            f.endswith((".mp4", ".mov", ".mkv"))
            for f in os.listdir(video_dir)
        )

    if not has_videos:
        pending.append("01")

    # 检查 02: 是否有音频文件
    audio_dir = paths["extracted_audio"]
    has_audio = False
    if os.path.isdir(audio_dir):
        has_audio = any(f.endswith(".wav") for f in os.listdir(audio_dir))

    if has_videos and not has_audio:
        pending.append("02")

    # 检查 03: 是否有 Demucs 输出
    demucs_dir = os.path.join(
        paths["demucs_output"], config["demucs"]["model"]
    )
    has_demucs = os.path.isdir(demucs_dir) and len(os.listdir(demucs_dir)) > 0

    if has_audio and not has_demucs:
        pending.append("03")

    # 检查 04: 是否有段元数据
    meta_04 = os.path.join(paths["diarization_output"], "segments_meta.json")
    if has_demucs and not os.path.exists(meta_04):
        pending.append("04")

    # 检查 05: ASR 结果
    asr_dir = paths["asr_output"]
    meta_05 = os.path.join(asr_dir, "asr_passed_meta.json")
    if os.path.exists(meta_04) and not os.path.exists(meta_05):
        pending.append("05")

    # 检查 06: 数据集
    meta_06 = os.path.join(paths["dataset"], "metadata.json")
    if os.path.exists(meta_05) and not os.path.exists(meta_06):
        pending.append("06")

    return pending


def auto_run(config: dict, logger):
    """自动检测并运行未完成的阶段"""
    logger.info("=" * 60)
    logger.info("自动模式：检测未完成的阶段...")
    logger.info("=" * 60)

    pending = detect_stages_to_run(config)

    if not pending:
        logger.info("所有阶段似乎已完成！运行 --all 来强制重新运行。")
        logger.info("或使用 --only-inspect 启动抽检台。")
        return

    logger.info(f"需要运行的阶段: {', '.join(pending)}")

    for stage_id in pending:
        run_stage(stage_id, config, logger)

    logger.info("\n" + "=" * 60)
    logger.info("流水线执行完毕！")
    logger.info("=" * 60)

    # 最终统计
    meta_path = os.path.join(config["paths"]["dataset"], "dataset_stats.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        logger.info(f"数据集统计: {stats.get('total_pairs', 0)} 段, "
                     f"{stats.get('total_duration_min', 0):.1f} 分钟")


def run_range(
    config: dict,
    logger,
    from_stage: str,
    to_stage: Optional[str] = None,
):
    """运行指定范围的阶段"""
    stage_ids = sorted(STAGES.keys())
    start_idx = stage_ids.index(from_stage)
    end_idx = (
        stage_ids.index(to_stage) + 1 if to_stage else len(stage_ids)
    )

    for stage_id in stage_ids[start_idx:end_idx]:
        run_stage(stage_id, config, logger)

    logger.info("\n指定范围执行完毕！")


def run_inspection(config: dict, logger):
    """仅启动 Gradio 抽检台"""
    logger.info("启动 Gradio 抽检台...")
    from pipeline.inspection_gradio import main
    main()


def main():
    parser = argparse.ArgumentParser(
        description="抖音博主语音克隆数据集全自动流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python -m pipeline.run_pipeline --all          # 完整流水线
  python -m pipeline.run_pipeline --from 03       # 从 Stage 3 开始
  python -m pipeline.run_pipeline --from 05 --to 07  # 运行 5-7 阶段
  python -m pipeline.run_pipeline --auto          # 自动检测未完成阶段
  python -m pipeline.run_pipeline --only-inspect  # 仅启动抽检台
        """
    )

    parser.add_argument(
        "--all", action="store_true",
        help="运行全部 7 个阶段（完整流水线）"
    )
    parser.add_argument(
        "--from", dest="from_stage", type=str,
        help="起始阶段编号 (01-07)"
    )
    parser.add_argument(
        "--to", dest="to_stage", type=str,
        help="结束阶段编号 (01-07)"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="自动检测并运行未完成的阶段"
    )
    parser.add_argument(
        "--only-inspect", action="store_true",
        help="仅启动 Gradio 抽检台"
    )
    parser.add_argument(
        "--url", type=str,
        help="覆盖博主抖音 URL"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径（默认 pipeline/config.yaml）"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    if args.url:
        config["blogger"]["douyin_url"] = args.url

    # 设置日志
    log_dir = ensure_dir(config["paths"]["logs"])
    logger = setup_logger("pipeline", log_dir)

    logger.info(f"语音克隆数据集流水线 v1.0.0")
    logger.info(f"博主: {config['blogger']['douyin_url']}")
    logger.info(f"工作目录: {os.getcwd()}")

    try:
        if args.only_inspect:
            run_inspection(config, logger)

        elif args.from_stage:
            run_range(config, logger, args.from_stage, args.to_stage)

        elif args.auto:
            auto_run(config, logger)

        elif args.all:
            logger.info("运行完整流水线...")
            for stage_id in sorted(STAGES.keys()):
                run_stage(stage_id, config, logger)
            logger.info("\n🎉 完整流水线已执行完毕！")

        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("\n⏹ 流水线已终止")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 流水线异常终止: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
