# 音频数据流 (Stage 02 → 06)

## 逐阶段追踪

```
Stage 02: 音频提取
────────────────────────────────────────────────────────────────────────
  输入: 抖音 MP4 视频
  处理: ffmpeg -ar 48000 -ac 1 -acodec pcm_s16le
  输出: 48 kHz / 16-bit / mono / PCM WAV
  归一化: 无
  文件: data/02_extracted_audio/{video_id}.wav

Stage 03: Ensemble 人声分离
────────────────────────────────────────────────────────────────────────
  输入: 48 kHz WAV (02 输出)
  处理: Demucs (内部 44.1k) + MDX23C (内部 44.1k) → 统一重采样到 44.1 kHz → STFT 融合
  输出: 44.1 kHz / PCM_16 / mono  vocals.wav
  归一化: 无
  文件: data/03_ensemble_output/demucs+mdx23c_ens/{video_id}/vocals.wav

Stage 04: 跨视频说话人分离
────────────────────────────────────────────────────────────────────────
  输入: 44.1 kHz vocals.wav (03 输出)
  处理: pyannote pipeline 内部使用 16 kHz（临时重采样）
        切分时使用 raw_audio (44.1 kHz) 精确取时间窗口
        → seg_wav = raw_audio[int(start×44100) : int(end×44100)]
  输出: 44.1 kHz / float64 WAV 段（与输入帧率一致）
  归一化: 无
  文件: data/04_diarization/segments/{video_id}_SPEAKER_XX_XXXX.wav

Stage 04b: SNR+SFM+HNR 质量过滤
────────────────────────────────────────────────────────────────────────
  输入: 44.1 kHz WAV 段 (04 输出)
  处理: Silero VAD 内部使用 16 kHz（临时重采样）
        只读不写 — 丢弃段删 WAV，通过段保持原样
  输出: 44.1 kHz（通过段不变，丢弃段被删除）
  归一化: 无（完全不修改音频内容）
  文件: 更新 data/04_diarization/segments_meta_community.json

Stage 05: ASR 转写
────────────────────────────────────────────────────────────────────────
  输入: 44.1 kHz WAV 段 (04b 过滤后)
  处理: read_audio() → 内部重采样到 16 kHz → Paraformer 推理
        通过段: shutil.copy2() — 逐位复制原始文件
  输出: 44.1 kHz（bit-exact 复制，帧率不变）
  归一化: 无
  文件: data/05_asr_output/audio/{segment_id}.wav
         data/05_asr_output/asr_passed_meta.json

Stage 06: 数据集打包
────────────────────────────────────────────────────────────────────────
  输入: 44.1 kHz WAV (05 输出)
  处理: sf.read() → 保留原始采样率
        → peak normalize（防削波）
        → normalize_loudness(-26 LUFS)
        → write_audio(sr=原始采样率)
  输出: 44.1 kHz / float32 / mono / -26 LUFS
          ↑↑  仅做响度归一化，不再改变帧率  ↑↑
  文件: data/06_dataset/audio/{segment_id}.wav
         data/06_dataset/metadata.json
```

## 总结表

| Stage | 采样率 | 位深 | 归一化 | 操作类型 |
|-------|--------|------|--------|----------|
| 02 | **48 kHz** | 16-bit int | 无 | ffmpeg 提取 |
| 03 | **44.1 kHz** | 16-bit int | 无 | Ensemble 融合 |
| 04 | **44.1 kHz** | float64 | 无 | 说话人分离切段 |
| 04b | **44.1 kHz** | — | 无 | 只读不写（删低质段） |
| 05 | **44.1 kHz** | 原始格式 | 无 | bit-exact 复制 |
| 06 | **44.1 kHz** ↗ | float32 | **-26 LUFS** | 唯一归一化点 |

## 关键发现

1. **Stage 02→03 有一次降采样** (48k→44.1k)
2. **Stage 02→06 全程无 loudness normalization**，直到 Stage 06 做 -26 LUFS
3. **Stage 06 也做了降采样** (44.1k→16k) — 虽然 TTS 通常用 16-24kHz，但 dataset 输出 16kHz 是否合理取决于后续 TTS 模型需求（XTTS 用 24kHz，CosyVoice 用 16kHz）
4. **read_audio() 默认 target_sr=16000** — 所有使用 `read_audio()` 的地方（05 推理、06 打包）都静默降采样到 16kHz，但实际文件不修改（05 是 copy，06 是写入新文件）
