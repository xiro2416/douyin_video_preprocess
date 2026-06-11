#!/usr/bin/env bash
# ============================================================
# 抖音博主语音克隆数据集流水线 — Shell 入口
# ============================================================
# 依赖: uv (推荐), python >= 3.10, ffmpeg
#
# 首次使用：
#   uv pip install -r requirements.txt
#   huggingface-cli login
#
# 用法：
#   bash pipeline/run_pipeline.sh          # 交互式菜单
#   bash pipeline/run_pipeline.sh --all    # 完整流水线
#   bash pipeline/run_pipeline.sh --auto   # 自动恢复
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-uv run python}"

# ── 颜色 ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
title() { echo -e "\n${BLUE}══════════════════════════════════════════${NC}"; echo -e "${BLUE}  $*${NC}"; echo -e "${BLUE}══════════════════════════════════════════${NC}\n"; }

# ── 前置检查 ──────────────────────────────────────────────────
check_env() {
    local has_errors=0

    # Python
    if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
        error "Python 未安装"
        has_errors=1
    fi

    # uv
    if ! command -v uv &>/dev/null; then
        warn "uv 未安装，将使用 pip。推荐安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi

    # ffmpeg
    if ! command -v ffmpeg &>/dev/null; then
        error "ffmpeg 未安装。请安装:"
        error "  macOS: brew install ffmpeg"
        error "  Ubuntu: sudo apt install ffmpeg"
        error "  Windows: winget install ffmpeg 或 choco install ffmpeg"
        has_errors=1
    fi

    # yt-dlp
    if ! command -v yt-dlp &>/dev/null; then
        error "yt-dlp 未安装。请安装: pip install yt-dlp"
        has_errors=1
    fi

    if [ "$has_errors" -eq 1 ]; then
        exit 1
    fi

    # 检查 PyTorch
    info "检查 PyTorch..."
    if $PYTHON -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null; then
        :
    else
        warn "PyTorch 未安装或导入失败。运行: uv pip install torch torchaudio"
    fi

    # 检查 HuggingFace 登录
    if $PYTHON -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
        info "HuggingFace 已登录 ✓"
    else
        warn "HuggingFace 未登录。运行: huggingface-cli login"
        warn "（pyannote diarization 需要）"
    fi
}

# ── 菜单 ──────────────────────────────────────────────────────
show_menu() {
    echo ""
    echo "请选择操作:"
    echo "  1) 完整流水线 (Stage 1-6)"
    echo "  2) 自动恢复 (检测未完成的阶段)"
  echo "  3) 仅下载视频 (Stage 1)"
  echo "  4) 提取音频 (Stage 2)"
  echo "  5) Demucs 分离 (Stage 3)"
  echo "  6) 说话人日志 (Stage 4)"
  echo "  7) ASR 转写 (Stage 5)"
  echo "  8) 数据集打包 (Stage 6)"
  echo "  9) 启动抽检台"
  echo "  q) 退出"
    echo ""
    read -rp "输入选项: " choice

    case "$choice" in
        1) run_all ;;
        2) run_auto ;;
        3) run_stage 01 ;;
        4) run_stage 02 ;;
        5) run_stage 03 ;;
        6) run_stage 04 ;;
        7) run_stage 05 ;;              # 05=ASR
        8) run_stage 06 ;;              # 06=dataset
        9) run_inspect ;;
        
        q|Q) exit 0 ;;
        *) warn "无效选项"; show_menu ;;
    esac
}

# ── 运行命令 ──────────────────────────────────────────────────
run_all() {
    title "运行完整流水线 (Stage 1-6)"
    $PYTHON -m pipeline.run_pipeline --all
}

run_auto() {
    title "自动恢复模式"
    $PYTHON -m pipeline.run_pipeline --auto
}

run_stage() {
    local stage="$1"
    title "运行 Stage $stage"
    $PYTHON -m pipeline.run_pipeline --from "$stage"
}

run_inspect() {
    title "启动 Gradio 抽检台"
    info "浏览器将自动打开 http://127.0.0.1:7860"
    $PYTHON -m pipeline.run_pipeline --only-inspect
}

# ── 主入口 ────────────────────────────────────────────────────
main() {
    cd "$PROJECT_DIR"

    title "🎤 抖音博主语音克隆数据集流水线"
    echo "工作目录: $(pwd)"
    echo ""

    check_env

    if [ $# -gt 0 ]; then
        case "$1" in
            --all|all)      run_all ;;
            --auto|auto)    run_auto ;;
            --inspect)      run_inspect ;;
            --stage)        run_stage "${2:-01}" ;;
            *)              warn "未知参数: $1"; show_menu ;;
        esac
    else
        show_menu
    fi
}

main "$@"
