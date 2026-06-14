# 抖音博主语音克隆数据集流水线

## 项目概述

从抖音视频全自动构建 TTS（文字转语音）训练数据集：

```
抖音 URL → 下载视频 → 提取音频 → Demucs 人声分离 → 跨视频说话人分离
→ ASR 转写 → 数据集打包（-26 LUFS）
```

## 环境配置

```bash
# 1. 进入项目目录
cd C:/Users/xi1/Desktop/claude_uv/voice_clone_pipeline

# 2. 安装依赖（uv 管理）
uv pip install -r requirements.txt

# 3. 安装 Whisper large-v3（首次运行自动下载）

# 4. pyannote 需要 HuggingFace 授权
# 访问 https://hf.co/pyannote/speaker-diarization-3.1 接受条款
# token 优先从环境变量 HF_TOKEN 读取，config.yaml 中 hf_token 作为 fallback

# 5. 解码 UTF-8 编码问题已修复（pipeline/utils.py）
#    强制 stdout/stderr 使用 UTF-8
```

## 流水线结构（6 个阶段）

| Stage | 模块 | 功能 | 输入 | 输出 |
|-------|------|------|------|------|
| 01 | `01_download.py` | yt-dlp 下载抖音视频 | 抖音 URL | `01_raw_videos/*.mp4` |
| 02 | `02_extract_audio.py` | ffmpeg 提取音频 (48kHz/16bit/mono) | MP4 | `02_extracted_audio/*.wav` |
| 03 | `03_demucs_separate.py` | htdemucs 人声分离 (segment=7, shifts=6) | WAV | `03_demucs_output/[model]/[video]/vocals.wav` |
| 04 | `04_diarization.py` | pyannote 说话人分离 + AHC 跨视频聚类 | vocals | `04_diarization/segments/*.wav` + `segments_meta.json` |
| 05 | `05_asr_transcribe.py` | Whisper large-v3 转写 (temperature=0.2) | 04 segments | `05_asr_output/audio/*.wav` |
| 06 | `06_dataset_build.py` | 打包 metadata.json + -26LUFS 归一化 | 05 audio | `06_dataset/audio/*.wav` |

## 运行方式

```bash
# 完整流水线
uv run python -m pipeline.run_pipeline --all

# 指定范围
uv run python -m pipeline.run_pipeline --from 04 --to 06

# 单独某个阶段
uv run python -m pipeline.run_pipeline --from 04 --to 04
```

## 目录结构（运行时生成）

```
data/
├── 01_raw_videos/           ← 原始 MP4
├── 02_extracted_audio/      ← 提取的 WAV (no normalization)
├── 03_demucs_output/        ← vocals.wav + no_vocals.wav
├── 04_diarization/          ← segments/*.wav + segments_meta.json
├── 05_asr_output/           ← audio/*.wav + asr_passed_meta.json
├── 06_dataset/              ← audio/*.wav (归一化) + metadata.json
```

## 关键参数

### Demucs (config.yaml)
- model: `htdemucs` (segment=7, shifts=6, clip-mode=rescale)
- 训练上限 7.8s，取整 7

### Stage 04: 跨视频说话人分离 (config.yaml → diarization_ad)
4 层架构: 音频预处理 → pyannote 单音频分音 → AHC 跨视频聚类 → 批量导出

- **分音模型**: `pyannote/speaker-diarization-3.1`（降级兜底: `speaker-diarization` community）
- **音频预处理**: soundfile 读取 → 降混单声道 → 16kHz 重采样 → 峰值归一化 → tensor
- **聚类算法**: 凝聚式层次聚类 (AgglomerativeClustering, average linkage, cosine distance)
- `clustering_threshold`: 0.5 — 余弦距离阈值（越小越严格，同一说话人要求更高相似度）
- `min_speakers` / `max_speakers`: 1 / 5 — 单音频说话人数量范围
- `min_segment_duration`: 0.5 — 最短语音片段（秒）
- `enable_outlier_detection`: true — 单样本孤立簇标记为 UNKNOWN
- **跨视频标签**: 最大簇 → TARGET, 其余 → OTHER_00/01..., 离群 → UNKNOWN
- **断点续传**: 每 10 个文件保存 checkpoint (`_checkpoint.json`)，中断后自动恢复
- **输出格式**: `segments_meta.json`，每段含 segment_id / speaker / start / end / duration / is_overlap / overlap_speakers / segment_path / video_id

### Whisper ASR (config.yaml)
- model: `large-v3`, temperature: 0.2（产出标点）
- 验证：文本非空、≥2字符、含中文、非纯标点

### 归一化
- **仅 Stage 06** 做归一化：峰值 → -26 LUFS
- Stage 02-05 不做任何归一化，raw 传递

## 代码修改历史

1. ✅ 删除所有 checkpoint 机制（CheckpointManager 类 + 各阶段调用）
2. ✅ 删除原 Stage 05（频谱重叠检测）
3. ✅ 删除原 Stage 05（质量滤镜），简化为直接 04→05(ASR)→06(Dataset)
4. ✅ 修复 Windows GBK 编码问题（utils.py 强制 UTF-8）
5. ✅ Demucs 参数优化：segment 5→7, shifts 3→6, clip-mode=rescale
6. ✅ 归一化统一：只在 Stage 06 做（-26 LUFS）
7. ✅ Whisper temperature: 0.0→0.2（产出标点）
8. ✅ Stage 04 重写: pyannote 3.1 + AHC 跨视频聚类 + 断点续传 (v3.0)

## 注意事项

- Windows 环境下 speechbrain 有 lazy import 崩溃问题（`04_diarization.py` 已做 importlib monkey-patch）
- pyannote 使用 soundfile 内存加载音频绕过 torchcodec 问题
- checkpoint (`_checkpoint.json`) 在流程正常结束后自动删除
- CUDA 设备字符串解析失败不影响运行（自动 fallback 到 device 0）
- config.yaml 中的 hf_token 是敏感信息，建议通过环境变量 `HF_TOKEN` 设置
