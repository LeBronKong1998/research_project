# 真实数据实验记录

日期：2026-11-02。Python 3.12.14，PyTorch 2.5.1+cpu，CPU，OMP/MKL 两线程；没有 GPU 测试。

真实 CSV：INTERACTION multi-agent 子集，由 InterHub 论文作者发布，52 cases / 24,771 rows；固定源文件版本与 SHA-256 见 [data-provenance.json](data-provenance.json)。不是合成场景，不是完整 INTERACTION，也不是官方测试集。

## 执行与复现

```bash
bash scripts/setup.sh
source /workspace/venvs/trajectory-prediction/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/trajectory-matplotlib
python -m vehtraj train --device cpu --epochs 30 --batch-size 32 --accumulation 1 --hidden 64 --lr 0.001 --output runs/interaction-real --log-every 20
python -m vehtraj evaluate --checkpoint runs/interaction-real/best.pt --output runs/interaction-real --max-animations 1
```

全部 30 个 epoch 执行完成，无 `max-batches` 截断；只按验证集加权损失选择 checkpoint，本次选中 epoch 30。评估遍历全部 111 条测试轨迹，无 `max-samples` 截断；没有根据测试结果反复选择超参数。

| 划分 | case 数 | 完整目标轨迹数 |
|---|---:|---:|
| train | 31 | 282 |
| val | 5 | 33 |
| cal | 5 | 42 |
| test | 11 | 111 |

case 总计 52，完整可用目标轨迹共 468。所有输入是过去 10 帧，未来 30 帧为标签。模型与恒速基线使用相同测试目标。完整配置、每个 epoch 损失、每个 split 的 case ID 见 [training.json](training.json)。

## 误差结果

| 指标 | 三分支 GNN | 恒速基线 |
|---|---:|---:|
| oracle minADE（m） | 0.8390 | 1.3488 |
| oracle minFDE（m） | 2.2448 | 3.7228 |
| top-1 ADE（m） | 1.4035 | 1.3488 |
| top-1 FDE（m） | 3.8622 | 3.7228 |
| top-1 每轨迹位置 RMSE 均值（m） | 1.8471 | 1.7804 |
| top-1 1 秒位置误差（m） | 0.4883 | 0.4613 |
| top-1 2 秒位置误差（m） | 1.7939 | 1.7249 |
| top-1 3 秒位置误差（m） | 3.8622 | 3.7228 |
| top-1 终点 2 米内命中率 | 41.44% | 47.75% |
| 所有分支终点均超出 2 米比例 | 38.74% | 52.25% |

**解释：** 多模态候选集合覆盖有改善，但当前概率首选分支没有优于恒速；不能据此宣称模型整体精度更好。只有 52 个 case、单个路口地点且 case 相关性未知，统计量不代表跨地点、跨道路类型泛化。后续需要更多独立录制、地图特征和更好的模式概率学习，在固定验证协议下再做比较。

校准集仅 5 个 case，95% conformal 分位数要求的排名为 6，半径无界；`calibration.json` 中 radius 为 `null`，没有有限置信覆盖结论。

## 保留证据

- [evaluation.json](evaluation.json)：原始浮点结果、基线、数据哈希和评估范围。
- [calibration.json](calibration.json)：校准计数与无界半径。
- [示例轨迹图](interaction-example.png)：仅展示运动轨迹，不伪造路面地图。
- [训练权重](../checkpoints/interaction-mini.pt)：SHA-256 `3679135bf6555460fa31e859ec3bfff4d13f24cb8db4ddff4508f32e601742ea`。

已上传的权重可直接评估：

```bash
python -m vehtraj download
python -m vehtraj evaluate --checkpoint checkpoints/interaction-mini.pt --output runs/pretrained --max-animations 1
```

训练与评估的分支数、隐藏维度和预测时长由 checkpoint 恢复；评估会核验数据哈希及 history/future/stride，避免用错误配置静默产生结果。逐帧 CSV 和 GIF 在本地 `runs/interaction-real` 保留，执行上述命令可重建。

## 工程验证

`bash scripts/setup.sh` 已实际完整重跑：安装锁定依赖、可编辑安装、`pip check`、真实 CSV 下载校验、原训练版本 `pytest` **16 passed**、CPU 前向/反向 smoke 全部成功。测试覆盖重复行/非法数值/采样率拒绝、断帧拒绝、按 case 划分、历史邻车输入不依赖未来、mask 不受无效邻车影响、oracle 与 top-1 指标区分、恒速预测、无界半径 JSON 和下载器不覆盖未知文件。

Windows 入口、GitHub Actions 云端执行、CUDA 和新云任务快照恢复尚未实测；这里的测试通过指当前 Linux 实例。

## 原始数据打包与浏览界面更新

真实 CSV 现已直接纳入仓库（校验值保持不变），不再要求用户先运行下载脚本才能看到数据。`docs/data-explorer.html` 是包含同一原始 CSV 的离线交互界面，支持随机 case、播放、筛选和原始数据导出。新测试结果和浏览器交互验证见 [data-explorer-validation.json](data-explorer-validation.json)；界面截图见 [data-explorer-preview.png](data-explorer-preview.png)。该更新不改变训练数据或模型指标。

此更新验证：20 项测试通过；Chromium 逐一打开 52 个场景，随机种子、播放、逐帧、表格定位、缩放与重置、完整 CSV/单场景导出、桌面和手机布局均通过，页面无浏览器错误、无外部数据/脚本请求。完整 CSV 导出与源文件逐字节一致。只用 Git 暂存的受版本管理文件生成独立副本后，20 项测试、标准库界面重建、111 个测试目标的模型评估均通过，指标与原报告一致；不依赖原工作目录内被忽略的数据文件。
