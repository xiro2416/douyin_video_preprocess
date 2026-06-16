"""
Stage 9: Gradio 抽样巡检台
============================
全自动流水线的最后一道防线。

功能：
  1. 对每个视频随机抽 N 段（音频 + 文本配对），交给操作员审核
  2. 操作员点击 ✓（通过）或 ✗（不合格）
  3. 如果一个视频被 ✗ 比例 > 阈值，则丢弃该视频所有段
  4. 审核结果影响最终数据集输出
  5. 自动统计通过/失败比例，生成报告

Gradio 界面：
  - 左侧显示当前段信息
  - 音频播放器
  - 转写文本展示
  - ✓ / ✗ 按钮
  - 进度条
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import gradio as gr

from pipeline.utils import get_logger, load_config


class InspectionApp:
    """
    Gradio 抽检应用程序。
    通过回调函数保存审核结果。
    """

    def __init__(
        self,
        segments_meta: List[dict],
        config: dict,
        logger=None,
    ):
        self.segments = segments_meta
        self.config = config
        self.logger = logger or get_logger("09_inspection")

        # 按视频分组
        self.by_video: Dict[str, List[dict]] = defaultdict(list)
        for seg in self.segments:
            video_id = seg.get("video_id", "unknown")
            self.by_video[video_id].append(seg)

        # 抽样策略：每个视频抽 N 段
        self.samples_per_video = config["inspection"]["sample_per_video"]
        self.samples = self._sample()

        # 状态
        self.current_idx = 0
        self.votes: Dict[str, bool] = {}  # segment_id -> True(✓) or False(✗)
        self.video_votes: Dict[str, List[bool]] = defaultdict(list)

        if self.logger:
            self.logger.info(
                f"抽检台初始化: {len(self.samples)} 段待审核 "
                f"(来自 {len(self.by_video)} 个视频)"
            )

    def _sample(self) -> List[dict]:
        """每个视频抽 N 段"""
        samples = []
        for video_id, segs in self.by_video.items():
            selected = random.sample(
                segs, min(self.samples_per_video, len(segs))
            )
            for s in selected:
                s["_video_id"] = video_id
            samples.extend(selected)
        random.shuffle(samples)
        return samples

    def get_current(self) -> Optional[dict]:
        """获取当前待审核段"""
        if self.current_idx >= len(self.samples):
            return None
        return self.samples[self.current_idx]

    def vote(self, passed: bool):
        """记录审核结果并前进到下一段"""
        seg = self.get_current()
        if seg is None:
            return

        seg_id = seg.get("segment_id", f"seg_{self.current_idx}")
        video_id = seg.get("_video_id", "unknown")

        self.votes[seg_id] = passed
        self.video_votes[video_id].append(passed)
        self.current_idx += 1

    def is_done(self) -> bool:
        return self.current_idx >= len(self.samples)

    def get_progress(self) -> str:
        return f"{self.current_idx} / {len(self.samples)}"

    def get_results(self) -> dict:
        """生成审核结果报告"""
        total = len(self.votes)
        passed = sum(1 for v in self.votes.values() if v)
        failed = total - passed

        fail_threshold = self.config["inspection"]["fail_threshold"]

        # 逐视频统计
        video_results = {}
        for video_id, votes in self.video_votes.items():
            fail_ratio = 1 - (sum(votes) / len(votes))
            video_results[video_id] = {
                "sampled": len(votes),
                "pass_count": sum(votes),
                "fail_count": len(votes) - sum(votes),
                "fail_ratio": fail_ratio,
                "discard": fail_ratio > fail_threshold,
            }

        discarded_videos = [
            vid for vid, r in video_results.items() if r["discard"]
        ]

        return {
            "total_sampled": total,
            "total_passed": passed,
            "total_failed": failed,
            "pass_rate": passed / max(total, 1),
            "videos_checked": len(self.video_votes),
            "videos_discarded": len(discarded_videos),
            "discarded_video_ids": discarded_videos,
            "video_details": video_results,
        }

    def get_audio_path(self) -> str:
        """获取当前段的音频路径"""
        seg = self.get_current()
        if seg is None:
            return ""
        return seg.get("segment_path", "")

    def get_text(self) -> str:
        """获取当前段的转写文本"""
        seg = self.get_current()
        if seg is None:
            return "（审核完成）"
        asr = seg.get("asr", {})
        if asr:
            text = asr.get("text", "")
            return f"{text}"
        return "（无 ASR 文本）"

    def get_segment_info(self) -> str:
        """获取当前段的元数据信息"""
        seg = self.get_current()
        if seg is None:
            return ""

        info = []
        info.append(f"ID: {seg.get('segment_id', '')}")
        info.append(f"视频: {seg.get('video_id', '')}")
        info.append(f"时长: {seg.get('quality', {}).get('duration_trimmed', 0):.2f}s")
        info.append(f"SNR: {seg.get('quality', {}).get('snr_db', 0):.1f}dB")
        info.append(f"视频总段数: {len(self.by_video.get(seg.get('_video_id', ''), []))}")
        return "\n".join(info)


def create_interface(config: dict = None):
    """创建 Gradio 界面"""
    if config is None:
        config = load_config()

    logger = get_logger("09_inspection")

    # 加载数据
    meta_path = os.path.join(
        config["paths"]["asr_output"], "asr_passed_meta.json"
    )

    if not os.path.exists(meta_path):
        logger.error(f"未找到数据: {meta_path}")
        # 创建占位界面
        with gr.Blocks(title="语音数据集抽检台") as demo:
            gr.Markdown(f"## 错误：未找到数据集\n请先运行完整流水线。\n\n期望路径: {meta_path}")
        return demo

    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    app = InspectionApp(segments, config, logger=logger)

    # ── Gradio UI ──────────────────────────────────────────────
    with gr.Blocks(
        title="🎤 语音数据集抽检台",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎤 语音数据集抽检台
            每个视频随机抽样审核。播放音频、查看转写文本，判断该段是否合格。

            - **✓ 通过**：语音清晰、无重叠、文本匹配、只有目标说话人
            - **✗ 不合格**：有背景音重叠、多人声、文本不匹配、语音不清晰
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                info_box = gr.Textbox(
                    label="段信息",
                    value=app.get_segment_info(),
                    lines=6,
                    interactive=False,
                )
                progress_text = gr.Textbox(
                    label="进度",
                    value=app.get_progress(),
                    interactive=False,
                )

            with gr.Column(scale=2):
                audio_player = gr.Audio(
                    value=app.get_audio_path(),
                    label="音频",
                    type="filepath",
                )
                text_display = gr.Textbox(
                    label="转写文本",
                    value=app.get_text(),
                    lines=4,
                    interactive=False,
                )

        with gr.Row():
            pass_btn = gr.Button("✓ 通过", variant="primary", size="lg")
            fail_btn = gr.Button("✗ 不合格", variant="stop", size="lg")
            skip_btn = gr.Button("跳过", size="lg")

        with gr.Row():
            result_box = gr.Textbox(
                label="审核结果",
                value="",
                lines=10,
                interactive=False,
            )
            finish_btn = gr.Button("📊 生成报告", variant="secondary")

        # ── 事件绑定 ───────────────────────────────────────────
        def handle_vote(passed: bool):
            app.vote(passed)

            if app.is_done():
                results = app.get_results()
                report = (
                    f"🎉 审核完成！\n\n"
                    f"总抽样: {results['total_sampled']}\n"
                    f"通过: {results['total_passed']}\n"
                    f"不合格: {results['total_failed']}\n"
                    f"通过率: {results['pass_rate']:.1%}\n"
                    f"审核视频数: {results['videos_checked']}\n"
                    f"丢弃视频数: {results['videos_discarded']}\n\n"
                )
                if results["discarded_video_ids"]:
                    report += "已丢弃视频:\n"
                    for vid in results["discarded_video_ids"]:
                        report += f"  - {vid}\n"

                return (
                    gr.Audio(value=None, label="音频"),
                    gr.Textbox(value="（审核完成）", label="转写文本"),
                    gr.Textbox(value=report, label="段信息"),
                    gr.Textbox(value="完成！", label="进度"),
                )

            # 更新到下一段
            return (
                gr.Audio(value=app.get_audio_path(), label="音频"),
                gr.Textbox(value=app.get_text(), label="转写文本"),
                gr.Textbox(value=app.get_segment_info(), label="段信息"),
                gr.Textbox(value=app.get_progress(), label="进度"),
            )

        pass_btn.click(
            fn=lambda: handle_vote(True),
            outputs=[audio_player, text_display, info_box, progress_text],
        )

        fail_btn.click(
            fn=lambda: handle_vote(False),
            outputs=[audio_player, text_display, info_box, progress_text],
        )

        skip_btn.click(
            fn=lambda: (
                gr.Audio(value=app.get_audio_path(), label="音频"),
                gr.Textbox(value=app.get_text(), label="转写文本"),
                gr.Textbox(value=app.get_segment_info(), label="段信息"),
                gr.Textbox(value=app.get_progress(), label="进度"),
            ),
            outputs=[audio_player, text_display, info_box, progress_text],
        )

        def finish():
            results = app.get_results()

            # 保存审核结果
            out_dir = config["paths"]["dataset"]
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "inspection_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            report = json.dumps(results, ensure_ascii=False, indent=2)
            return report

        finish_btn.click(fn=finish, outputs=[result_box])

    return demo


def main():
    """启动 Gradio 服务"""
    config = load_config()

    logger = get_logger("09_inspection")
    logger.info("启动 Gradio 抽检台...")

    demo = create_interface(config)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
