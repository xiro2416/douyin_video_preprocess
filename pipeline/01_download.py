"""
Stage 1: 批量下载博主全部视频
================================
基于 yt-dlp，支持增量续采、限速、随机延迟。
"""

import os
import subprocess
import time
import random

from pipeline.utils import (
    ensure_dir,
    get_logger,
    load_config,
)


# 找到 yt-dlp 命令（venv 内或系统）
def _ytdlp_cmd():
    """返回可用的 yt-dlp 命令列表，优先用 venv 内的模块。"""
    import sys as _sys
    # 方式1: python -m yt_dlp (最可靠)
    return [_sys.executable, "-m", "yt_dlp"]


def _check_chrome_running():
    """检测 Chrome 是否正在运行（Windows）。"""
    try:
        import subprocess as _sp
        r = _sp.run(["tasklist", "/FI", "IMAGENAME eq chrome.exe", "/NH"], capture_output=True, text=True, timeout=10, shell=True)
        return "chrome.exe" in r.stdout
    except Exception:
        return False


def _get_cookies_args(config: dict, logger):
    """
    尝试获取可用的 cookie 源。
    优先级：
      1. config 中指定 cookies.txt 路径
      2. browser-cookie3 自动提取
      3. --cookies-from-browser 尝试各浏览器
    返回 (args_list, source_name)
    """
    yt = _ytdlp_cmd()
    browsers = ["chrome", "edge", "brave", "opera", "firefox"]

    # 1. 如果配置中指定了 cookies 文件路径
    cfg_cookies = config.get("download", {}).get("cookies_file", "")
    if cfg_cookies and os.path.isfile(cfg_cookies):
        logger.info(f"使用配置文件指定的 cookies 文件: {cfg_cookies}")
        return (["--cookies", cfg_cookies], f"配置文件: {cfg_cookies}")

    # 2. 尝试 browser-cookie3 自动提取
    try:
        import browser_cookie3
        import http.cookiejar

        # 检查 Chrome 是否运行（browser-cookie3 需要管理员权限访问运行中的 Chrome）
        chrome_running = _check_chrome_running()
        if chrome_running:
            logger.info("检测到 Chrome 正在运行，尝试用 browser-cookie3 提取 cookies...")

        cj = http.cookiejar.CookieJar()
        cookie_count = 0
        for domain in ["douyin.com", ".douyin.com"]:
            for b in browsers:
                try:
                    loader = getattr(browser_cookie3, b, None)
                    if loader:
                        for c in loader(domain_name=domain):
                            cj.set_cookie(c)
                            cookie_count += 1
                except Exception:
                    continue

        if cookie_count > 0:
            cookie_file = os.path.join(os.path.dirname(__file__), "..", "data", "cookies.txt")
            os.makedirs(os.path.dirname(cookie_file), exist_ok=True)
            with open(cookie_file, "w", encoding="utf-8") as f:
                f.write("# Netscape HTTP Cookie File\n")
                for domain, domain_data in cj._cookies.items():
                    for path, path_data in domain_data.items():
                        for name, c in path_data.items():
                            secure = "TRUE" if c.secure else "FALSE"
                            expires = "0" if c.expires is None else str(int(c.expires))
                            f.write(f"{c.domain}\tTRUE\t{c.path}\t{secure}\t{expires}\t{c.name}\t{c.value}\n")
            return (["--cookies", cookie_file], f"browser-cookie3 ({cookie_count} cookies)")
        elif chrome_running:
            logger.warning("browser-cookie3 未找到 douyin.com 的 cookies。请确保已在 Chrome 中登录抖音。")
    except ImportError:
        pass

    # 3. 逐个尝试 --cookies-from-browser
    for browser in browsers:
        try:
            r = subprocess.run(
                yt + ["--cookies-from-browser", browser, "--cookies", os.devnull, "--skip-download", "--max-downloads", "0"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and "Could not copy" not in r.stderr:
                return (["--cookies-from-browser", browser], f"浏览器 {browser}")
        except Exception:
            continue

    # 4. 全部失败，给提示
    if _check_chrome_running():
        logger.error("=" * 60)
        logger.error("Chrome 正在运行，yt-dlp 无法读取其 cookie 数据库。")
        logger.error("请选择以下一种方式解决：")
        logger.error("  方式 A: 关闭 Chrome → 重新运行")
        logger.error("  方式 B: 在 config.yaml 中添加 cookies_file 路径")
        logger.error("     download:")
        logger.error("       cookies_file: ./data/cookies.txt")
        logger.error("     然后用 Chrome 插件导出 douyin.com 的 cookies 到该文件")
        logger.error("=" * 60)
    else:
        logger.warning("未找到可用的 cookie 源。如果抖音页面需要登录才能访问，请先在浏览器中登录抖音。")

    return ([], "无 cookie")


def download_videos(config: dict, logger=None):
    """
    下载博主所有视频。
    使用 download_archive 实现增量去重，断点续传。
    """
    cfg = config["download"]
    paths = config["paths"]
    blogger = config["blogger"]

    if logger is None:
        logger = get_logger("01_download")

    raw_dir = ensure_dir(paths["raw_videos"])
    archive_file = os.path.join(raw_dir, "downloaded.txt")

    url = blogger["douyin_url"]

    logger.info(f"=" * 60)
    logger.info(f"Stage 1: 下载博主视频")
    logger.info(f"目标 URL: {url}")
    logger.info(f"保存至: {raw_dir}")
    logger.info(f"=" * 60)

    # 获取 cookie 参数
    cookies_args, cookies_source = _get_cookies_args(config, logger)
    logger.info(f"使用 Cookie 源: {cookies_source}")

    # 先获取视频列表（不下载）
    logger.info("正在获取视频列表...")
    yt = _ytdlp_cmd()
    list_cmd = yt + [
        "--flat-playlist",
        "--dump-json",
        "--no-download",
    ] + cookies_args + [url]

    try:
        result = subprocess.run(
            list_cmd, capture_output=True, text=True, timeout=120
        )
        videos = [line for line in result.stdout.strip().split("\n") if line.strip()]
        logger.info(f"共发现 {len(videos)} 个视频")
    except Exception as e:
        logger.warning(f"获取视频列表失败: {e}")
        logger.info("将直接下载（不预先计数）")
        videos = []

    # 构建下载命令
    download_cmd = yt + cookies_args + [
        "-f", cfg["format"],
        "--limit-rate", cfg["limit_rate"],
        "-o", os.path.join(raw_dir, "%(id)s.%(ext)s"),
        "--download-archive", archive_file,
        "--no-post-overwrites",
        "--embed-metadata",
        "--extractor-args", "douyin:app_version=29.9.0",
    ]

    # 如果指定了最大下载数
    max_downloads = cfg.get("max_downloads", 0)
    if max_downloads > 0:
        download_cmd.extend(["--max-downloads", str(max_downloads)])

    download_cmd.append(url)

    logger.info("开始下载（增量模式，已下载的自动跳过）...")
    # 不打印完整命令以免泄露 cookie 路径
    logger.info(f"命令: yt-dlp {'<cookies> ' if cookies_args else ''}{'-f ...'} --limit-rate {cfg['limit_rate']} ...")

    try:
        process = subprocess.Popen(
            download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(f"[yt-dlp] {line}")

        process.wait()

        if process.returncode == 0:
            logger.info("下载完成！")
        else:
            logger.error(f"yt-dlp 退出码: {process.returncode}")

    except KeyboardInterrupt:
        logger.info("用户中断下载，已下载的视频不会重复下载")
    except Exception as e:
        logger.error(f"下载失败: {e}")

    # 统计
    from pipeline.utils import get_video_files
    downloaded = get_video_files(raw_dir)
    logger.info(f"当前目录共有 {len(downloaded)} 个视频文件")

    # 逐视频应用随机间隔，降低被限流概率
    # （yt-dlp 本身会在视频间有间隔，这里作为补充）
    return downloaded


def main():
    config = load_config()
    logger = setup_logger("01_download", config["paths"]["logs"])
    download_videos(config, logger)


if __name__ == "__main__":
    from pipeline.utils import setup_logger
    main()
