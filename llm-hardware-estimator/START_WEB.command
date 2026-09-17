#!/bin/zsh
set -u

APP_DIR="${0:A:h}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

cd "$APP_DIR" || {
  echo "启动失败：无法进入程序目录。"
  echo "按回车键关闭窗口。"
  read
  exit 1
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "启动失败：当前电脑未检测到 Python 3。"
  echo "请先安装 Python 3 后重新运行本脚本。"
  echo "按回车键关闭窗口。"
  read
  exit 1
fi

echo "LLM推理硬件需求估算工具"
echo "服务启动中..."
echo ""
echo "本机访问地址：http://127.0.0.1:${PORT}"
echo "局域网访问地址：http://<本机IP>:${PORT}"
echo ""
echo "关闭服务：在本窗口按 Control + C。"
echo ""

python3 web_app.py --host "$HOST" --port "$PORT"
STATUS=$?

echo ""
if [ "$STATUS" -ne 0 ]; then
  echo "服务已退出，退出码：$STATUS"
  echo "如提示端口被占用，可在终端执行："
  echo "PORT=8081 ./START_WEB.command"
fi
echo "按回车键关闭窗口。"
read
exit "$STATUS"
