# trajectory-prediction

在真实 **INTERACTION CSV** 上训练 GRU + 图注意力网络，预测周围车辆的多模态轨迹，并导出误差、恒速对照、逐帧预测及图像。项目包名为 `trajectory-prediction`，Python 模块仍为 `vehtraj`；上传到既有仓库 `LeBronKong1998/research_project`，未更改远程仓库名称。

默认 **10 Hz、1 秒历史 → 3 秒未来**，三个预测分支，最多八辆邻车。真实数据来自 InterHub 论文作者公开的 INTERACTION 路口子集：52 个 case、24,771 行记录。该子集适合开发验证，不足以代表所有复杂交通场景。合成数据仅用于单元测试/显式 `--source synthetic` 调试，训练与评估默认真实数据，缺数据会报错，绝不自动回退。

## Linux / 云环境

Python 3.11–3.12。已实测 Python 3.12 + PyTorch 2.5.1 CPU，不要求 GPU 或在线服务：

```bash
cd /workspace/trajectory-prediction
bash scripts/setup.sh
source /workspace/venvs/trajectory-prediction/bin/activate
export OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/trajectory-matplotlib
python -m vehtraj train --epochs 30 --batch-size 32 --accumulation 1 --lr 0.001 --output runs/interaction-real
python -m vehtraj evaluate --checkpoint runs/interaction-real/best.pt --output runs/interaction-real
```

`setup.sh` 创建/复用外部 venv，安装官方 CPU PyTorch、锁定依赖和本地可编辑包，下载并验证真实 CSV，执行单元测试与前后向 smoke。普通 Linux 可用 `TRAJECTORY_VENV="$PWD/.venv" bash scripts/setup.sh`，然后激活 `.venv/bin/activate`。下载需要 `raw.githubusercontent.com`；依赖需要 PyPI 与 PyTorch 官方域名，保持 TLS 和 SHA-256 验证。

## Windows

安装 Python 3.11 或 3.12 后，在项目目录执行：

```powershell
.\install.ps1
.\run.ps1 train
.\run.ps1 evaluate
```

脚本使用项目本地 `.venv`，没有原项目中的个人磁盘路径。Windows 脚本已改为通用入口，但本次只在 Linux 实测。GPU 用户可按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v251) 配置相应 2.5.1 CUDA wheel 后使用 `--device cuda`；本次未验证 CUDA。

## 自有完整数据

```bash
python -m vehtraj train --data-dir /path/to/interaction --history 30 --future 50 --output runs/full
python -m vehtraj evaluate --data-dir /path/to/interaction --history 30 --future 50 --checkpoint runs/full/best.pt --output runs/full
```

30+50 仅适用于有足够连续帧的原始录制版本；本项目默认下载的 40 帧 challenge 子集不支持。`--stride` 在训练与评估时保持一致。没有 case_id 的原始文件按整个录制文件划分，需提供足够独立文件覆盖四个集合。其他数据集格式不自动猜测。

## 输出与精度

`runs/interaction-real/` 下有：

- `best.pt`：只按验证损失选出的模型，包含数据哈希、时间窗、随机种子、选中 epoch。
- `training.json`：全部 epoch、样本数、case 划分、配置与源文件哈希。
- `evaluation.json`：minADE/minFDE、概率 top-1 ADE/FDE、位置 RMSE、1/2/3 秒误差、2 米终点命中率、速度/航向误差、恒速基线。
- `predictions.csv`：每条轨迹、每个分支、每步概率、世界坐标预测、真值和位置误差。
- `calibration.json`：校准 case 数与半径；数据不足时 `radius_m: null`，不是零误差。
- PNG/GIF：观测、邻车、真实未来、三个预测分支；没有伪造道路背景。

**minADE/minFDE 使用真实未来挑选最优分支，因此不能当作实际 top-1 准确率。** 恒速基线结果使用同一测试目标，应该同时对比。`--max-batches` 只适合流程检查；`--max-samples` 会明确标记部分评估，不能冒充完整结果。

[数据来源、论文、划分限制与指标定义](docs/DATASETS_AND_PAPERS.md) · [本次真实数据实验](reports/REAL_DATA_VALIDATION.md)

模型是目标车星形图单头注意力 + GRU 的轻量基线，没有地图、交通灯、行人、多车联合解码或碰撞约束。该路口子集只保证 case 划分不重叠，不证明物理车辆、地点或时间独立；不是官方挑战榜单结果，也不提供道路安全保证。
