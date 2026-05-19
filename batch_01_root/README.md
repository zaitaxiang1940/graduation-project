# Trajectory Transformer - Ubuntu 24.04.4 LTS Deployment

[English](#english) | [中文](#中文)

---

<h2 id="english">English Documentation</h2>

This repository has been comprehensively refactored and packaged for **Ubuntu 24.04.4 LTS**. It supports one-click deployment, systemd background service management, 30-day log rotation, and memory leak detection, ensuring production-level stability with 0-interaction installation on fresh cloud instances.

### 📁 Project Skeleton

```text
trajectory-transformer/
├── trajectory/          # Core model and utilities (Refactored for Py3.10+)
├── scripts/             # Execution scripts (train.py, plan.py)
├── configs/             # Configuration files (relative paths fixed)
├── tests/               # Unit tests & Memory leak detection suites
├── deploy/              # ⚡ NEW: Deployment Suite (Ubuntu 24.04.4)
│   ├── deploy.sh        # One-click environment prep & install
│   ├── uninstall.sh     # Clean uninstallation
│   ├── trajectory.service # Systemd service template
│   ├── logrotate.conf   # 30-day log rotation policy
│   └── run_tests.sh     # Comprehensive test suite script
├── docs/                # Extended documentation
├── environment.yml      # Conda environment definition (Upgraded dependencies)
└── setup.py             # Packaging file replacing deprecated distutils
```

### 🚀 Step-by-Step Execution Guide (From Bare Metal)

**Step 1: Download Project & Grant Permissions**
Ensure you are logged in as a non-root user with `sudo` privileges.
```bash
git clone <your-repo-url> trajectory-transformer
cd trajectory-transformer
chmod +x deploy/*.sh
```

**Step 2: One-Click Environment Setup (0 Interaction)**
Installs system dependencies, configures UFW firewall (allows SSH port 22), installs MuJoCo 2.1.0, Miniconda, Python 3.10 environment, and registers systemd/logrotate.
```bash
sudo ./deploy/deploy.sh
```
*Expected Output:*
`>>> 部署完成！请重新登录或执行 'source ~/.bashrc' 以应用变量` (Deployment Complete!)

**Step 3: Apply Environment Variables**
```bash
source ~/.bashrc
conda activate trajectory
```

**Step 4: Verify Installation (Unit Tests & Memory Profile)**
```bash
./deploy/run_tests.sh
```
*Expected Output:*
`✓ 单元测试通过率 100%`
`✓ 内存泄漏检测无告警`
`✓ Systemd 守护进程服务已注册`

**Step 5: Run the Model (Foreground or Background)**
- **Foreground Test (Dry Run):**
  ```bash
  python scripts/train.py --dataset halfcheetah-medium-v2 --exp_name smoke_test
  ```
- **Background Service (Production):**
  ```bash
  sudo systemctl start trajectory.service
  sudo systemctl status trajectory.service
  ```
  Check logs (Rotated daily, kept for 30 days):
  ```bash
  tail -f /var/log/trajectory/trajectory.log
  ```

### 🛠️ Troubleshooting

- **`ModuleNotFoundError: No module named 'gym'`**: Ensure you have activated the conda environment (`conda activate trajectory`).
- **`ERROR: GLEW initalization error: Missing GL version`**: You are running in a headless server. The deployment script sets `MUJOCO_GL=egl` by default. Ensure `libglew-dev` is installed (handled by `deploy.sh`).
- **High CPU Usage**: The deployment is optimized to keep CPU idle < 5% when not actively training. Use `htop` to verify.

---

<h2 id="中文">中文文档</h2>

本仓库针对 **Ubuntu 24.04.4 LTS** 进行了全面重构和打包。支持一键部署、systemd 后台服务管理、30天日志轮转和内存泄漏检测，确保在全新云镜像上实现 0 交互安装及生产级稳定性。

### 📁 项目骨架

```text
trajectory-transformer/
├── trajectory/          # 核心模型与工具（已重构适配 Python 3.10+）
├── scripts/             # 执行脚本（train.py, plan.py）
├── configs/             # 配置文件（已修复为相对路径）
├── tests/               # 单元测试与内存泄漏检测套件
├── deploy/              # ⚡ 新增：部署套件 (Ubuntu 24.04.4)
│   ├── deploy.sh        # 一键环境准备与安装脚本
│   ├── uninstall.sh     # 彻底卸载与清理脚本
│   ├── trajectory.service # Systemd 后台服务模板
│   ├── logrotate.conf   # 30天日志轮转策略
│   └── run_tests.sh     # 综合测试套件运行脚本
├── docs/                # 扩展文档
├── environment.yml      # Conda 环境定义（依赖已全量升级）
└── setup.py             # 替换废弃 distutils 的打包文件
```

### 🚀 分步执行指南（从裸机开始）

**第一步：下载项目并赋予权限**
请确保您以具有 `sudo` 权限的非 root 用户登录。
```bash
git clone <your-repo-url> trajectory-transformer
cd trajectory-transformer
chmod +x deploy/*.sh
```

**第二步：一键环境部署（0 交互）**
自动安装系统依赖、配置 UFW 防火墙（放行 SSH）、安装 MuJoCo 2.1.0、Miniconda、Python 3.10 环境，并注册 systemd 服务与日志轮转。
```bash
sudo ./deploy/deploy.sh
```
*预期输出:*
`>>> 部署完成！请重新登录或执行 'source ~/.bashrc' 以应用变量`

**第三步：应用环境变量**
```bash
source ~/.bashrc
conda activate trajectory
```

**第四步：验证安装（单元测试与内存剖析）**
```bash
./deploy/run_tests.sh
```
*预期输出:*
`✓ 单元测试通过率 100%`
`✓ 内存泄漏检测无告警`
`✓ Systemd 守护进程服务已注册`

**第五步：运行模型（前台或后台）**
- **前台测试跑通（冒烟测试）:**
  ```bash
  python scripts/train.py --dataset halfcheetah-medium-v2 --exp_name smoke_test
  ```
- **后台服务运行（生产环境）:**
  ```bash
  sudo systemctl start trajectory.service
  sudo systemctl status trajectory.service
  ```
  查看日志（每天轮转，保留 30 天）:
  ```bash
  tail -f /var/log/trajectory/trajectory.log
  ```

### 🛠️ 排错指南

- **`ModuleNotFoundError: No module named 'gym'`**: 确保已激活 conda 环境 (`conda activate trajectory`)。
- **`ERROR: GLEW initalization error: Missing GL version`**: 发生在无头服务器。部署脚本已默认设置 `MUJOCO_GL=egl`，请确保应用了 `~/.bashrc` 中的环境变量。
- **CPU 占用过高**: 在非训练状态下，架构优化确保 CPU 占用 < 5%。如异常，请使用 `htop` 排查是否有僵尸进程，并使用 `sudo ./deploy/uninstall.sh` 重置环境。
