#!/bin/bash

# 自动化训练脚本 - Anchor3DLane 项目
# 修复 PYTHONPATH 未定义问题
# 日期：2025-08-18

set -euo pipefail

# 配置参数
CONDA_ENV="lane3d"
PROJECT_DIR="/root/autodl-tmp/Anchor3DLane"  
CONFIG_FILE="$PROJECT_DIR/configs/openlane/anchor3dlane-wide.py"

echo "========================================"
echo "  Anchor3DLane 自动化训练脚本"
echo "========================================"
echo "• 激活 Conda 环境: $CONDA_ENV"
echo "• 项目目录: $PROJECT_DIR"
echo "• 配置文件: $CONFIG_FILE"
echo "----------------------------------------"

# 激活Conda环境
if ! source activate $CONDA_ENV 2>/dev/null; then
    echo "⚠️ 使用 'conda activate' 替代..."
    if ! conda activate $CONDA_ENV; then
        echo "❌ 错误：无法激活Conda环境 '$CONDA_ENV'"
        echo "请检查环境是否存在: conda env list"
        exit 1
    fi
fi

# 进入项目目录
if ! cd "$PROJECT_DIR"; then
    echo "❌ 错误：无法进入项目目录 '$PROJECT_DIR'"
    exit 1
fi
echo "✓ 进入项目目录: $(pwd)"

# 安全设置PYTHONPATH (修复点)
# 如果 PYTHONPATH 未设置，初始化为空字符串
export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
echo "✓ 设置 PYTHONPATH: $PYTHONPATH"

# 执行训练命令
echo "----------------------------------------"
echo "开始训练任务..."
echo "执行的命令: python tools/train.py $CONFIG_FILE"
echo "----------------------------------------"

start_time=$(date +%s)

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误：配置文件不存在: $CONFIG_FILE"
    echo "请检查路径是否正确"
    exit 1
fi

# 执行训练
python tools/train.py "$CONFIG_FILE" 
