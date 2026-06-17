"""
Stage 7: 数据分析
=================
读取 Stage 04b 前后和 Stage 06 的 JSON 元数据，
输出全面的统计报告（段数、时长分布、说话人分布、文本统计等）。

分析两个关键节点：
  1. Pre-04b: segments_meta_community_pre_snr_filter.json（04b 自动备份）
     — SNR/SFM/HNR 过滤之前的说话人分离结果
  2. Post-06: data/06_dataset/metadata.json（最终数据集）
     — 经过 ASR 验证 + 响度归一化后的最终数据集

用法:
  uv run python -m pipeline.run_pipeline --from 07 --to 07
  uv run python -m pipeline.07_analysis
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from pipeline.utils import (
    ensure_dir,
    get_logger,
    load_config,
    setup_logger,
)


# ═══════════════════════════════════════════════════════════════
# 时间段分析
# ═══════════════════════════════════════════════════════════════

def _duration_stats(durations: List[float]) -> Dict:
    """计算时长统计量"""
    arr = np.array(durations)
    return {
        "count": len(arr),
        "total_seconds": float(arr.sum()),
        "total_minutes": float(arr.sum() / 60),
        "mean": float(arr.mean()) if len(arr) > 0 else 0,
        "median": float(np.median(arr)) if len(arr) > 0 else 0,
        "std": float(arr.std()) if len(arr) > 0 else 0,
        "min": float(arr.min()) if len(arr) > 0 else 0,
        "max": float(arr.max()) if len(arr) > 0 else 0,
        "p25": float(np.percentile(arr, 25)) if len(arr) > 0 else 0,
        "p75": float(np.percentile(arr, 75)) if len(arr) > 0 else 0,
        "p90": float(np.percentile(arr, 90)) if len(arr) > 0 else 0,
    }


def _duration_bins(durations: List[float]) -> Dict[str, int]:
    """时长分桶"""
    bins = {
        "<0.5s": 0,
        "0.5-1s": 0,
        "1-2s": 0,
        "2-3s": 0,
        "3-5s": 0,
        "5-10s": 0,
        "10-15s": 0,
        "15-30s": 0,
        ">=30s": 0,
    }
    for d in durations:
        if d < 0.5:
            bins["<0.5s"] += 1
        elif d < 1.0:
            bins["0.5-1s"] += 1
        elif d < 2.0:
            bins["1-2s"] += 1
        elif d < 3.0:
            bins["2-3s"] += 1
        elif d < 5.0:
            bins["3-5s"] += 1
        elif d < 10.0:
            bins["5-10s"] += 1
        elif d < 15.0:
            bins["10-15s"] += 1
        elif d < 30.0:
            bins["15-30s"] += 1
        else:
            bins[">=30s"] += 1
    return bins


def _print_table(rows: List[List[str]], header: List[str], logger):
    """打印表格（等宽列排版）"""
    if not rows:
        return
    col_widths = [
        max(len(str(row[i])) for row in [header] + rows) + 2
        for i in range(len(header))
    ]
    sep = "─" * (sum(col_widths) + len(header) - 1)
    header_line = " ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
    logger.info(sep)
    logger.info(header_line)
    logger.info(sep)
    for row in rows:
        line = " ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
        logger.info(line)
    logger.info(sep)


# ═══════════════════════════════════════════════════════════════
# 1. 分析 segments_meta（04b 前后）
# ═══════════════════════════════════════════════════════════════

def analyze_segments_meta(
    meta_path: str,
    label: str,
    logger,
) -> Optional[Dict]:
    """
    分析 segments_meta JSON（04 输出）。

    参数:
        meta_path: JSON 文件路径
        label: 显示标签，如 "04b 过滤前" / "04b 过滤后"

    返回统计字典，文件不存在返回 None。
    """
    if not os.path.exists(meta_path):
        logger.warning(f"[{label}] 文件不存在: {meta_path}")
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        logger.info(f"[{label}] 空列表")
        return {"label": label, "path": meta_path, "total": 0}

    # ── 基础统计 ──
    total = len(segments)
    durations = [s.get("duration", 0) for s in segments]
    total_dur = sum(durations)
    ds = _duration_stats(durations)

    # ── 说话人分布 ──
    speaker_counter = Counter()
    speaker_duration = defaultdict(float)
    for s in segments:
        spk = s.get("speaker", "UNKNOWN")
        speaker_counter[spk] += 1
        speaker_duration[spk] += s.get("duration", 0)

    # ── 重叠段 ──
    overlap_count = sum(1 for s in segments if s.get("is_overlap", False))
    overlap_dur = sum(
        s.get("duration", 0) for s in segments if s.get("is_overlap", False)
    )

    # ── 按 video_id 统计 ──
    video_segments = defaultdict(list)
    for s in segments:
        vid = s.get("video_id", "unknown")
        video_segments[vid].append(s.get("duration", 0))
    video_stats = {}
    for vid, durs in sorted(video_segments.items()):
        video_stats[vid] = {"segments": len(durs), "total_duration": sum(durs)}

    # ── 过滤决策分析（如果已运行 04b，filter_decision 字段存在）─╴
    filter_decisions = Counter()
    for s in segments:
        fd = s.get("filter_decision")
        if fd:
            filter_decisions[fd] += 1

    result = {
        "label": label,
        "path": meta_path,
        "total": total,
        "duration": ds,
        "duration_bins": _duration_bins(durations),
        "speaker_distribution": {
            spk: {"count": c, "total_minutes": round(speaker_duration[spk] / 60, 2)}
            for spk, c in sorted(speaker_counter.items())
        },
        "overlap": {
            "count": overlap_count,
            "total_seconds": round(overlap_dur, 2),
            "pct": round(overlap_count / total * 100, 2) if total else 0,
        },
        "filter_decisions": dict(filter_decisions),
        "video_stats": video_stats,
    }

    # ── 日志输出 ──
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 分析: {label}")
    logger.info(f"文件: {meta_path}")
    logger.info(f"{'=' * 60}")

    logger.info(f"总段数: {total}")
    logger.info(f"总时长: {ds['total_seconds']:.1f}s ({ds['total_minutes']:.1f}min)")
    logger.info(f"平均时长: {ds['mean']:.2f}s  中位数: {ds['median']:.2f}s")
    logger.info(f"范围: [{ds['min']:.2f}, {ds['max']:.2f}]")

    logger.info(f"\n─ 时长分布 ─")
    for bucket, count in sorted(result["duration_bins"].items()):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {bucket:>8s}: {count:>5d} ({pct:5.1f}%) {bar}")

    logger.info(f"\n─ 说话人分布 ─")
    spk_rows = []
    for spk, info in sorted(result["speaker_distribution"].items()):
        pct = info["count"] / total * 100
        spk_rows.append([spk, str(info["count"]), f"{pct:.1f}%",
                         f"{info['total_minutes']:.1f}min"])
    _print_table(spk_rows, ["说话人", "段数", "占比", "总时长"], logger)

    logger.info(f"\n─ 重叠段 ─")
    logger.info(f"  {result['overlap']['count']} 段 ({result['overlap']['pct']}%)")

    if filter_decisions:
        logger.info(f"\n─ 过滤决策 (filter_decision) ─")
        fd_rows = []
        for fd, cnt in sorted(filter_decisions.items()):
            fd_pct = cnt / total * 100
            fd_rows.append([fd, str(cnt), f"{fd_pct:.1f}%"])
        _print_table(fd_rows, ["决策", "段数", "占比"], logger)

    logger.info(f"\n─ 视频统计 ─")
    logger.info(f"  视频数: {len(video_stats)}")
    logger.info(f"  每视频平均段数: {total / max(len(video_stats), 1):.1f}")
    vid_rows = []
    for vid, vs in sorted(video_stats.items(), key=lambda x: -x[1]["total_duration"]):
        vid_rows.append([vid, str(vs["segments"]),
                         f"{vs['total_duration']:.1f}s",
                         f"{vs['total_duration']/60:.1f}min"])
    _print_table(vid_rows, ["视频 ID", "段数", "时长", "分钟"], logger)

    return result


# ═══════════════════════════════════════════════════════════════
# 2. 分析数据集 metadata.json（06 输出）
# ═══════════════════════════════════════════════════════════════

def analyze_dataset_meta(
    meta_path: str,
    logger,
) -> Optional[Dict]:
    """
    分析最终数据集 metadata.json（06 输出）。

    返回统计字典，文件不存在返回 None。
    """
    if not os.path.exists(meta_path):
        logger.warning(f"[最终数据集] 文件不存在: {meta_path}")
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata:
        logger.info("[最终数据集] 空列表")
        return {"label": "最终数据集", "path": meta_path, "total": 0}

    # ── 基础统计 ──
    total = len(metadata)
    durations = [m.get("duration", 0) for m in metadata]
    ds = _duration_stats(durations)

    # ── 文本统计 ──
    texts = [m.get("text", "") for m in metadata]
    text_lengths = [len(t) for t in texts]
    tl_stats = {
        "mean": float(np.mean(text_lengths)),
        "median": float(np.median(text_lengths)),
        "std": float(np.std(text_lengths)),
        "min": int(min(text_lengths)) if text_lengths else 0,
        "max": int(max(text_lengths)) if text_lengths else 0,
        "p25": float(np.percentile(text_lengths, 25)),
        "p75": float(np.percentile(text_lengths, 75)),
        "p90": float(np.percentile(text_lengths, 90)),
    }

    # ── 说话人分布 ──
    speaker_counter = Counter()
    speaker_duration = defaultdict(float)
    for m in metadata:
        spk = m.get("speaker", "unknown")
        speaker_counter[spk] += 1
        speaker_duration[spk] += m.get("duration", 0)

    # ── 按 video_id 统计 ──
    video_segments = defaultdict(list)
    for m in metadata:
        vid = m.get("video_id", "unknown")
        video_segments[vid].append({
            "duration": m.get("duration", 0),
            "text_len": len(m.get("text", "")),
        })
    video_stats = {}
    for vid, segs in sorted(video_segments.items()):
        durs = [s["duration"] for s in segs]
        video_stats[vid] = {
            "segments": len(segs),
            "total_duration": sum(durs),
            "mean_duration": float(np.mean(durs)) if durs else 0,
        }

    # ── 最长/最短段 ──
    sorted_by_dur = sorted(metadata, key=lambda x: x.get("duration", 0), reverse=True)
    longest_10 = [
        {"segment_id": m.get("segment_id", ""),
         "duration": m.get("duration", 0),
         "text_preview": m.get("text", "")[:60]}
        for m in sorted_by_dur[:10]
    ]
    shortest_10 = [
        {"segment_id": m.get("segment_id", ""),
         "duration": m.get("duration", 0),
         "text_preview": m.get("text", "")[:60]}
        for m in sorted_by_dur[-10:] if m.get("duration", 0) > 0
    ]
    # 去掉可能出现的 0 时长
    shortest_10 = [s for s in shortest_10 if s["duration"] > 0][:10]

    result = {
        "label": "最终数据集",
        "path": meta_path,
        "total": total,
        "duration": ds,
        "duration_bins": _duration_bins(durations),
        "text_length": tl_stats,
        "speaker_distribution": {
            spk: {"count": c, "total_minutes": round(speaker_duration[spk] / 60, 2)}
            for spk, c in sorted(speaker_counter.items())
        },
        "video_stats": video_stats,
        "longest_10": longest_10,
        "shortest_10": shortest_10,
    }

    # ── 日志输出 ──
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 分析: 最终数据集 (Stage 06)")
    logger.info(f"文件: {meta_path}")
    logger.info(f"{'=' * 60}")

    logger.info(f"总段数: {total}")
    logger.info(f"总时长: {ds['total_seconds']:.1f}s ({ds['total_minutes']:.1f}min)")
    logger.info(f"平均时长: {ds['mean']:.2f}s  中位数: {ds['median']:.2f}s")
    logger.info(f"范围: [{ds['min']:.2f}, {ds['max']:.2f}]s")

    logger.info(f"\n─ 时长分布 ─")
    for bucket, count in sorted(result["duration_bins"].items()):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {bucket:>8s}: {count:>5d} ({pct:5.1f}%) {bar}")

    logger.info(f"\n─ 文本长度 ─")
    logger.info(f"  平均: {tl_stats['mean']:.1f}  中位数: {tl_stats['median']:.0f}")
    logger.info(f"  范围: [{tl_stats['min']}, {tl_stats['max']}]")
    logger.info(f"  P25: {tl_stats['p25']:.0f}  P75: {tl_stats['p75']:.0f}")

    logger.info(f"\n─ 说话人分布 ─")
    spk_rows = []
    for spk, info in sorted(result["speaker_distribution"].items()):
        pct = info["count"] / total * 100
        spk_rows.append([spk, str(info["count"]), f"{pct:.1f}%",
                         f"{info['total_minutes']:.1f}min"])
    _print_table(spk_rows, ["说话人", "段数", "占比", "总时长"], logger)

    logger.info(f"\n─ 视频统计 ─")
    logger.info(f"  视频数: {len(video_stats)}")
    logger.info(f"  每视频平均段数: {total / max(len(video_stats), 1):.1f}")
    vid_rows = []
    for vid, vs in sorted(video_stats.items(), key=lambda x: -x[1]["total_duration"]):
        vid_rows.append([vid, str(vs["segments"]),
                         f"{vs['total_duration']:.1f}s",
                         f"{vs['total_duration'] / 60:.1f}min",
                         f"{vs['mean_duration']:.2f}s"])
    _print_table(vid_rows, ["视频 ID", "段数", "时长", "分钟", "平均时长"], logger)

    if longest_10:
        logger.info(f"\n─ 最长 10 段 ─")
        for s in longest_10:
            logger.info(f"  {s['duration']:6.2f}s  [{s['segment_id']}] {s['text_preview']}")

    if shortest_10:
        logger.info(f"\n─ 最短 10 段 ─")
        for s in shortest_10:
            logger.info(f"  {s['duration']:6.2f}s  [{s['segment_id']}] {s['text_preview']}")

    return result


# ═══════════════════════════════════════════════════════════════
# 3. 跨阶段对比（04b 前 → 06）
# ═══════════════════════════════════════════════════════════════

def analyze_cross_stage(
    pre_04b: Optional[Dict],
    post_06: Optional[Dict],
    logger,
) -> Optional[Dict]:
    """
    跨阶段留存分析：从 pre-04b 到最终数据集经过了多少过滤。
    """
    if pre_04b is None or post_06 is None:
        logger.warning("[跨阶段对比] 缺少前置或后置数据，跳过")
        return None

    pre_total = pre_04b.get("total", 0)
    post_total = post_06.get("total", 0)
    if pre_total == 0:
        logger.warning("[跨阶段对比] 前置数据为空，跳过")
        return None

    pre_dur = pre_04b.get("duration", {}).get("total_seconds", 0)
    post_dur = post_06.get("duration", {}).get("total_seconds", 0)

    # 说话人留存
    pre_speakers = pre_04b.get("speaker_distribution", {})
    post_speakers = post_06.get("speaker_distribution", {})

    target_before = pre_speakers.get("TARGET", {}).get("count", 0)
    target_after = post_speakers.get(
        post_06.get("label", ""), {}).get("count", 0
    ) or post_total  # 如果 post 数据集不按 speaker 分组，就用 total

    result = {
        "pre_04b": {
            "total_segments": pre_total,
            "total_minutes": round(pre_dur / 60, 2),
            "target_segments": target_before,
        },
        "post_06": {
            "total_segments": post_total,
            "total_minutes": round(post_dur / 60, 2),
        },
        "retention": {
            "segments_pct": round(post_total / pre_total * 100, 2) if pre_total else 0,
            "duration_pct": round(post_dur / pre_dur * 100, 2) if pre_dur else 0,
            "duration_lost_minutes": round((pre_dur - post_dur) / 60, 2),
        },
    }

    # 仅当 post 使用的是指定说话人名称时
    if post_speakers:
        post_target_total = sum(
            v["count"] for v in post_speakers.values()
        )
        if target_before and post_target_total:
            result["retention"]["target_segments_pct"] = round(
                post_target_total / target_before * 100, 2
            )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🔄 跨阶段留存分析")
    logger.info(f"{'=' * 60}")
    logger.info(f"               段数     | 总时长")
    logger.info(f"  04b 前:   {pre_total:>6d} 段 | {pre_dur / 60:.1f} min")
    logger.info(f"  最终:     {post_total:>6d} 段 | {post_dur / 60:.1f} min")
    logger.info(f"  ───────────────────────────────────")
    logger.info(f"  留存率:   {result['retention']['segments_pct']:>5.1f}%  | {result['retention']['duration_pct']:>5.1f}%")
    logger.info(f"  丢弃:     {pre_total - post_total:>6d} 段 | {result['retention']['duration_lost_minutes']:.1f} min")
    logger.info(f"{'=' * 60}")

    return result


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def analyze_all(config: dict, logger) -> Dict:
    """
    执行全部分析，保存结果到 JSON。
    """
    paths = config["paths"]
    diar_out = paths.get("diarization_output", "./data/04_diarization")
    dataset_dir = paths.get("dataset", "./data/06_dataset")

    # ── 1. 分析 pre-04b（备份） ──
    pre_04b_path = os.path.join(diar_out, "segments_meta_community_pre_snr_filter.json")
    pre_04b = analyze_segments_meta(pre_04b_path, "04b 过滤前", logger)

    # ── 2. 分析 post-04b（当前 segments_meta） ──
    post_04b_path = os.path.join(diar_out, "segments_meta_community.json")
    fallback_04b = os.path.join(diar_out, "segments_meta.json")
    if os.path.exists(post_04b_path):
        post_04b = analyze_segments_meta(post_04b_path, "04b 过滤后", logger)
    elif os.path.exists(fallback_04b):
        post_04b = analyze_segments_meta(fallback_04b, "04b 过滤后", logger)
    else:
        post_04b = None
        logger.warning("未找到 04b 过滤后的 segments_meta")

    # ── 3. 分析 ASR 验证结果（05 输出） ──
    asr_path = os.path.join(paths.get("asr_output", "./data/05_asr_output"),
                            "asr_passed_meta.json")
    if os.path.exists(asr_path):
        asr_result = analyze_segments_meta(asr_path, "ASR 通过", logger)
    else:
        asr_result = None
        logger.warning("未找到 ASR 通过元数据")

    # ── 4. 分析最终数据集（06 输出） ──
    dataset_path = os.path.join(dataset_dir, "metadata.json")
    dataset_result = analyze_dataset_meta(dataset_path, logger)

    # ── 5. 跨阶段对比 ──
    # pre-04b → 最终数据集
    cross_1 = analyze_cross_stage(pre_04b, dataset_result, logger)
    # ASR 通过 → 最终数据集
    cross_2 = analyze_cross_stage(asr_result, dataset_result, logger)

    # ── 汇总 ──
    summary = {
        "pre_04b": pre_04b,
        "post_04b": post_04b,
        "asr_passed": asr_result,
        "dataset": dataset_result,
        "cross_stage": {
            "pre_04b_to_dataset": cross_1,
            "asr_to_dataset": cross_2,
        },
    }

    return summary


def main(config=None, logger=None):
    if config is None:
        config = load_config()
    if logger is None:
        log_dir = config["paths"].get("logs", "./logs")
        logger = setup_logger("07_analysis", log_dir)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Stage 7: 数据分析")
    logger.info(f"{'=' * 60}")

    summary = analyze_all(config, logger)

    # ── 保存分析结果 ──
    dataset_dir = config["paths"].get("dataset", "./data/06_dataset")
    out_dir = ensure_dir(os.path.join(dataset_dir, "..", "07_analysis"))
    out_path = os.path.join(out_dir, "analysis_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\n分析报告已保存: {out_path}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Stage 7 完成")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
