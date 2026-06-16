# 抖音博主语音克隆数据集流水线

## 项目概述

从抖音视频全自动构建 TTS（文字转语音）训练数据集：

```
抖音 URL → 下载视频 → 提取音频 → MDX23C 人声分离 → 跨视频说话人分离
→ ASR 转写 → 数据集打包（-26 LUFS）
```

## 环境配置

### 前置要求

- Python >= 3.10, uv, ffmpeg
- NVIDIA GPU + CUDA（推荐，本环境为 8× RTX 4090）

### 安装

```bash
# 1. 进入项目目录
cd /workspace/douyin_video_preprocess

# 2. 安装依赖（uv 管理）
uv pip install -r requirements.txt

# 3. 验证 CUDA
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# 4. pyannote 需要 HuggingFace 授权
#    访问 https://hf.co/pyannote/speaker-diarization-3.1 接受条款
#    token 优先从环境变量 HF_TOKEN 读取，config.yaml 中 hf_token 作为 fallback
export HF_TOKEN="your_hf_token_here"
```

## GPU 配置

在 `pipeline/config.yaml` 中集中管理 GPU 设备选择：

```yaml
gpu:
  enabled: true          # true = 使用 GPU, false = 强制 CPU
  cuda_visible: "0"      # CUDA_VISIBLE_DEVICES, 如 "0" 或 "0,1,2"
                         # 空字符串 = 不限制（使用所有可用 GPU）
  device_id: 0           # 默认 GPU 设备索引，多卡时指定推理设备
```

各 GPU 阶段均遵循上述全局配置：

| Stage | 模块 | GPU 利用率 | 配置方式 |
|-------|------|-----------|----------|
| 03 | MDX23C 人声分离 (ONNX Runtime) | 高（单卡推理） | 全局 `gpu.*` |
| 04 | pyannote 说话人分离 | 中（单卡推理） | 全局 `gpu.*` + `diarization_ad.device` |
| 05 | Paraformer-large-zh ASR | 高（单卡推理） | 全局 `gpu.*` |

> **多 GPU 调度策略**：当前每个阶段为单卡推理。如需并行处理多个视频，
> 可分别设置 `CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=1` 启动多个进程。

## 流水线结构（6 个阶段）

| Stage | 模块 | 功能 | 输入 | 输出 |
|-------|------|------|------|------|
| 01 | `01_download.py` | yt-dlp 下载抖音视频 | 抖音 URL | `01_raw_videos/*.mp4` |
| 02 | `02_extract_audio.py` | ffmpeg 提取音频 (48kHz/16bit/mono) | MP4 | `02_extracted_audio/*.wav` |
| 03 | `03_mdx23c_separate.py` | MDX23C ONNX 人声分离 (HQ) | WAV | `03_mdx23c_output/[model]/[video]/vocals.wav` |
| 04 | `04_diarization.py` | pyannote 说话人分离 + AHC 跨视频聚类 | vocals | `04_diarization/segments/*.wav` + `segments_meta.json` |
| 05 | `05_asr_transcribe.py` | Paraformer-large-zh 转写 (FunASR, hf-mirror) | 04 segments | `05_asr_output/audio/*.wav` |
| 06 | `06_dataset_build.py` | 打包 metadata.json + -26LUFS 归一化 | 05 audio | `06_dataset/audio/*.wav` |

## 运行方式

```bash
# 完整流水线
uv run python -m pipeline.run_pipeline --all

# 指定范围
uv run python -m pipeline.run_pipeline --from 04 --to 06

# 单独某个阶段
uv run python -m pipeline.run_pipeline --from 04 --to 04

# 自动检测未完成阶段
uv run python -m pipeline.run_pipeline --auto

# Shell 脚本方式（交互式菜单）
bash pipeline/run_pipeline.sh

# Shell 脚本直接运行
bash pipeline/run_pipeline.sh --all
```

## 目录结构（运行时生成）

```
data/
├── 01_raw_videos/           ← 原始 MP4
├── 02_extracted_audio/      ← 提取的 WAV (no normalization)
├── 03_mdx23c_output/        ← vocals.wav + no_vocals.wav (MDX23C)
├── 03_demucs_output/        ← (旧版 Demucs，未清理前保留)
├── 04_diarization/          ← segments/*.wav + segments_meta.json
├── 05_asr_output/           ← audio/*.wav + asr_passed_meta.json
├── 06_dataset/              ← audio/*.wav (归一化) + metadata.json
```

## 关键参数

### Stage 03: MDX23C 人声分离 (config.yaml → mdx23c)
- model: `MDX23C-8KFFT-InstVoc_HQ_2.ckpt` (UVR, ONNX Runtime)
- 使用 `audio-separator` 库，输出 2 茎: vocals.wav + no_vocals.wav
- 模型通过 HuggingFace 镜像 (hf-mirror.com) 下载到 `pretrained_models/`

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

### FunASR Paraformer ASR (config.yaml)
- model: `iic/paraformer-large-zh`（HuggingFace hub），使用 hf-mirror.com 加速下载
- `use_itn: true`（逆文本正则化，产出数字/标点）
- 验证：文本非空、≥2字符、含中文、非纯标点
- 无需额外的 VAD/标点模型，输入已是分段音频

### 归一化
- **仅 Stage 06** 做归一化：峰值 → -26 LUFS
- Stage 02-05 不做任何归一化，raw 传递

## 注意事项

- **HF Token**: pyannote 需要 HuggingFace token。设置环境变量 `HF_TOKEN`，或在 `config.yaml` 中填写 `diarization_ad.hf_token`
- **GPU 切换**: 设置 `gpu.enabled: false` 可强制全部使用 CPU 运行（速度极慢，仅调试用）
- **多 GPU**: 通过 `gpu.cuda_visible` 限制可见 GPU，`gpu.device_id` 选择推理设备
- CUDA 设备字符串解析失败不影响运行（自动 fallback 到 device 0）
