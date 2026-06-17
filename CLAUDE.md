# 抖音博主语音克隆数据集流水线

## 项目概述

从抖音视频全自动构建 TTS（文字转语音）训练数据集：

```
抖音 URL → 下载视频 → 提取音频 → Ensemble 人声分离 (Demucs+MDX23C) → 跨视频说话人分离
→ SNR+SFM+HNR 质量过滤 → ASR 转写 → 数据集打包（-26 LUFS）
→ JSON 数据分析 → Qwen2-Audio 多模态音频质量评价
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
| 03 | MDX23C 或 Ensemble 人声分离 | 高（单卡推理，双模型） | 全局 `gpu.*` |
| 04 | pyannote 说话人分离 | 中（单卡推理） | 全局 `gpu.*` + `diarization_ad.device` |
| 04b | SNR+SFM+HNR 质量过滤 | 极低（Silero VAD CPU 推理） | `snr_filter.*` |
| 05 | Paraformer-large-zh ASR | 高（单卡推理） | 全局 `gpu.*` |
| 08 | Qwen2-Audio-7B 音频评价 | 高（单卡推理，`device_map="auto"`） | 全局 `gpu.*` |

> **多 GPU 调度策略**：当前每个阶段为单卡推理。如需并行处理多个视频，
> 可分别设置 `CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=1` 启动多个进程。

## 流水线结构（8 个阶段）

| Stage | 模块 | 功能 | 输入 | 输出 |
|-------|------|------|------|------|
| 01 | `01_download.py` | yt-dlp 下载抖音视频 | 抖音 URL | `01_raw_videos/*.mp4` |
| 02 | `02_extract_audio.py` | ffmpeg 提取音频 (48kHz/16bit/mono) | MP4 | `02_extracted_audio/*.wav` |
| 03 | `03_mdx23c_separate.py` 或 `03_ensemble_separate.py` | MDX23C ONNX 或 Demucs+MDX23C Ensemble | WAV | `03_mdx23c_output/[model]/[video]/vocals.wav` 或 `03_ensemble_output/[model]/[video]/vocals.wav` |
| 04 | `04_diarization.py` | pyannote 说话人分离 + AHC 跨视频聚类 | vocals | `04_diarization/segments/*.wav` + `segments_meta.json` |
| 04b | `04b_snr_filter.py` | SNR+SFM+HNR 三维质量过滤 | 04 segments | 更新后的 segments_meta.json（丢弃段 WAV 被删除） |
| 05 | `05_asr_transcribe.py` | Paraformer-large-zh 转写 (FunASR, hf-mirror) | 04b filtered segments | `05_asr_output/audio/*.wav` |
| 06 | `06_dataset_build.py` | 打包 metadata.json + -26LUFS 归一化 | 05 audio | `06_dataset/audio/*.wav` |
| 07 | `07_analysis.py` | JSON 数据分析（段数/时长/说话人/留存率） | 04b/06 JSON | `07_analysis/analysis_report.json` |
| 08 | `08_audio_evaluation.py` | Qwen2-Audio-7B 多模态音频质量评价 | 06 audio | `08_audio_evaluation/evaluation_results.json` |

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

# 单独运行数据分析（需要 04b 或 06 的 JSON）
uv run python -m pipeline.run_pipeline --from 07

# 单独运行音频评价（需要 Stage 06 的数据集）
uv run python -m pipeline.run_pipeline --from 08

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
├── 03_ensemble_output/      ← vocals.wav + no_vocals.wav (Ensemble 融合)
├── 03_mdx23c_output/        ← vocals.wav + no_vocals.wav (MDX23C)
├── 03_demucs_output/        ← (旧版 Demucs)
├── 04_diarization/          ← segments/*.wav + segments_meta.json
├── 05_asr_output/           ← audio/*.wav + asr_passed_meta.json
├── 06_dataset/              ← audio/*.wav (归一化) + metadata.json
├── 07_analysis/             ← analysis_report.json (统计数据)
├── 08_audio_evaluation/     ← evaluation_results.json (Qwen2-Audio 评价)
```

## 关键参数

### Stage 03: Ensemble 人声分离（Demucs + MDX23C STFT 融合）

在 `config.yaml` 中设置 `ensemble.enabled: true` 后启用，自动替换 MDX23C 单模型分离。

**算法**: STFT 域软掩码平均（Wiener-filter 型）

```
M_d = |V_d|² / (|V_d|² + |N_d|² + ε)       ← Demucs 的 per-TF-bin SNR 估计
M_m = |V_m|² / (|V_m|² + |N_m|² + ε)       ← MDX23C 的 per-TF-bin SNR 估计
M_e = (w_d·M_d + w_m·M_m) / (w_d + w_m)    ← 加权平均掩码
V_e = ISTFT(M_e · STFT{mixture})            ← 用原始混合相位重建
N_e = mixture - V_e                          ← 完美重建（减法保证）
```

**为什么 ensemble 而不是后处理 denoiser**:

对 TTS 精调数据集来说，如果 ensemble 后 vocals 仍有可闻的背景音乐残留（通常源于原始视频录制时 BGM 音量过大，而非分离失败），**不推荐使用 DeepFilterNet3 / noisereduce / 去混响 等后处理工具**。原因：

1. DeepFilterNet3 是为通信降噪（DNS Challenge）设计的，目标噪声类型（风扇、街道）与音乐分离残留完全不同，对 BGM 泄漏的抑制效果有限
2. 任何神经网络后处理都会在 100% 的样本上留下自己的"声学签名"（共振峰过渡段相位畸变、弱辅音/呼吸声被压制），这些 artifact 会被 TTS 精调学会并在合成时放大
3. 原始录音的 BGM 响度过大（语音帧 SNR < 5dB）是采集质量问题，不是分离/后处理能解决的——这类视频应当被丢弃而非修复

**实际流程**:

```
原始视频 → ensemble 分离 → vocals.wav
                              ↓
    检查静音帧泄漏：人没说话时 vocals 能量 → 衡量分离质量
    检查语音帧 SNR：人说话时人声 vs 同时存在的 BGM → 衡量原始录音质量
                              ↓
    语音帧 SNR < 5dB → 放弃该视频（录音质量不足）
    静音帧泄漏 > 0.01 → 检查分离是否出错
    两者都正常 → 进入 Stage 04 说话人分离
```

- 分离质量的衡量指标是**人没说话时 vocals 的能量**（静音帧泄漏），而非全段平均RMS
- 语音帧 SNR 低是原始录音问题，不应通过后处理强修

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

### Stage 04b: SNR+SFM+HNR 质量过滤 (config.yaml → snr_filter)

TARGET 段的三维质量过滤，在 ASR 前丢弃低质量片段。

**三维指标：**

| 维度 | 算法 | 测什么 | 阈值 |
|------|------|--------|------|
| SNR | P10/P90 帧能量法（语音段最静 10% vs 最响 10%） | 人声能量 vs 噪声底 | ≥ 15 dB |
| SFM | 谐波帧频谱的几何/算术平均比 | 频谱尖峰度（谐波 vs 平谱） | ≤ 0.03 (~-20dB) |
| HNR | 自相关法谐波峰 vs 帧内噪声比 | 声带周期性（清亮 vs 气息/沙哑） | ≥ 6 dB |

- **为什么能量百分位法估算 SNR**：自然语音动态范围 40-64dB，音素间有能量低谷。P10 帧反映的正是录制环境底噪 + 分离残留。Stage 03 已将背景剥离，不必担心伴奏能量干扰。
- **为什么不用 VAD 分离语音/噪声**：Silero VAD 在连续语音段标记过激进，VAD 判为"非语音"的帧能量与语音帧几乎相同（实为气声/辅音/停顿），会导致 SNR 系统性低估至 ~0dB。
- VAD 的角色限于：**验证段确实是语音**（speech_ratio ≥ 10%）+ **为 SFM/HNR 提取有效语音帧**。
- 过滤前自动备份元数据为 `segments_meta_community_pre_snr_filter.json`，丢弃段 WAV 被删除。
- 调整阈值后可直接重跑（`uv run python -m pipeline.04b_snr_filter`），不需要重新做昂贵的 Stage 04 说话人分离。

### FunASR Paraformer ASR (config.yaml)
- model: `funasr/paraformer-zh`（HuggingFace hub），使用 hf-mirror.com 加速下载
- `punc_model: "ct-punc"` — 标点恢复模型（Paraformer 默认不输出标点，需额外加载）
- `punc_kwargs: {hub: "ms"}` — ct-punc 仅发布在 ModelScope，非 HuggingFace
- `use_itn: true`（逆文本正则化，产出数字/标点）
- 验证：文本非空、≥2字符、含中文、非纯标点
- 无需 VAD（输入已是分段音频）

### Stage 07: JSON 数据分析 (`07_analysis.py`)

分析两个关键节点的 JSON 元数据并输出结构化统计报告。

**分析对象：**

| 节点 | 输入文件 | 输出统计 |
|------|---------|---------|
| 04b 前 | `segments_meta_community_pre_snr_filter.json` | 总段数/时长、说话人分布、重叠段、过滤决策、每视频统计 |
| 04b 后 | `segments_meta_community.json` / `segments_meta.json` | 同上 + filter_decision 分布 |
| ASR 通过 | `asr_passed_meta.json` | 同上 + 文本长度统计 |
| 最终数据集 | `metadata.json` | 时长分布、文本长度、最长/最短 TOP 10、说话人/视频分布 |
| 跨阶段对比 | pre-04b ↔ 最终 / ASR ↔ 最终 | 段数/时长留存率、丢弃量 |

**输出：** `data/07_analysis/analysis_report.json`

```bash
uv run python -m pipeline.run_pipeline --from 07
```

### Stage 08: Qwen2-Audio-7B 多模态音频质量评价 (config.yaml → audio_evaluation)

使用 Qwen2-Audio-7B-Instruct 对最终数据集每段音频做自然语言评价，输出中文评价文字。

**评价维度：** 语音清晰度、背景噪声/残留、录音质量、TTS 适用性、整体评分(1-5分)

**模型：** `Qwen/Qwen2-Audio-7B-Instruct`（transformers 原生支持，>= 4.44.0），通过 hf-mirror.com 镜像下载

**推理：** 单段逐一推理（~1.3s/段），`device_map="auto"` + `torch.bfloat16`，约需 16GB VRAM

**输出：** `data/08_audio_evaluation/evaluation_results.json`

```bash
uv run python -m pipeline.run_pipeline --from 08
```

### 归一化
- **仅 Stage 06** 做归一化：峰值 → -26 LUFS
- Stage 02-05 不做任何归一化，raw 传递

## 注意事项

- **HF Token**: pyannote 需要 HuggingFace token。设置环境变量 `HF_TOKEN`，或在 `config.yaml` 中填写 `diarization_ad.hf_token`
- **GPU 切换**: 设置 `gpu.enabled: false` 可强制全部使用 CPU 运行（速度极慢，仅调试用）
- **多 GPU**: 通过 `gpu.cuda_visible` 限制可见 GPU，`gpu.device_id` 选择推理设备
- CUDA 设备字符串解析失败不影响运行（自动 fallback 到 device 0）
