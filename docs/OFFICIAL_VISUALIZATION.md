# 按 INTERACTION 官方流程查看地图与模型验证

## 与官方流程的对应

用户指定的 [interaction-dataset/interaction-dataset](https://github.com/interaction-dataset/interaction-dataset) 提供本地 Python/Matplotlib 可视化工具，并非必须联网使用的托管网页。官方入口是 `python/main_visualize_data.py`，默认每 100 ms 切换一个时刻。它支持 Lanelet2 或无需编译 Lanelet2 的 `map_vis_without_lanelet.py` 分支；本项目采用后者，依赖 `pyproj`。

以下官方文件固定在提交 `053213d8aeb4c6f6132d272325a11ae623bd4e35`，原样保存在 `vehtraj/official/`：

- `map_vis_without_lanelet.py`：`LL2XYProjector` 和 `draw_map_without_lanelet`。
- `dataset_types.py`：`Track` / `MotionState`。
- `tracks_vis.py`：按真实尺寸和 `psi_rad` 绘制车辆轮廓，以及按时间更新车辆。
- `dict_utils.py`、BSD 3-Clause `LICENSE` 和包含逐文件 SHA-256 的 `provenance.json`。

`vehtraj/official_viewer.py` 直接调用这些官方函数，加入 multi-agent CSV 的 `case_id` 选择、现代 Matplotlib 控件和模型预测叠加。不能直接把不同 case 中重复的 track_id 混成一条轨迹。

配套网页 `docs/data-explorer.html` 则用同样的投影坐标和官方地图线要素规则绘制到 Canvas，便于不安装 Python 的用户离线查看。它是本项目对官方流程的网页适配，不是官方维护或背书的网站。

## 地图与轨迹的坐标对齐

原始地图在 `data/interaction/maps/DR_USA_Intersection_GL.osm`，来自 InterHub 作者与该 CSV 同目录分发的同名地图，不是官方工具里的 `.TestScenarioForScripts` 测试地图。源文件、独立的 `.osm_xy`、来源和校验值均随工程提供。

完全采用官方默认投影：WGS84、UTM zone 31、`lat_origin=0.0`、`lon_origin=0.0`，再减去投影后的原点。此数据中的 OSM 经纬度是用于恢复局部坐标的参考表达，不能将它解释为路口的真实地理位置。

583 个节点逐一与作者提供的 `.osm_xy` 对照，最大差为 `0.000006875183` m；163 条地图 way 中 10 个 `traffic_sign` 按官方规则不绘制。浏览器中的黑色路缘、白色车道线/停止线、蓝色虚拟边界均来自真实地图标签。没有拟合平移、旋转或缩放来强行对齐。

## 直接打开统一离线界面

双击 `docs/data-explorer.html`，在“查看内容”中切换：

- **原始数据**：全部 52 个 case，播放所有实际参与者。
- **验证集预测**：33 个实际验证目标，带训练/验证损失曲线及每步预测误差。
- **测试集预测**：111 个实际测试目标，显示同一权重的完整评估结果和恒速对照。

蓝色是历史，绿色是真实未来，彩色虚线是模型分支，粗粉线为**模型概率 top-1**。逐帧播放时，粉色点是该时刻预测，粉色连线对应预测与真值的位移误差。界面可查看全分支或仅 top-1，并逐一选择 case 和目标样本。原始 CSV、地图坐标和预测结果均内嵌，不需要外部服务器或 CDN。

完整集合指标与当前单条轨迹指标分开显示。minADE/minFDE 是分别选择最优分支的 oracle 指标，不代表概率 top-1。显示真实未来是离线验证操作；推理仅接收历史帧。

## 在训练和评估中自动生成同一界面

```bash
source /workspace/venvs/trajectory-prediction/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/trajectory-matplotlib
python -m vehtraj train --epochs 30 --batch-size 32 --accumulation 1 --lr 0.001 --output runs/official-trained
# 训练结束自动生成 runs/official-trained/val-review.html 与 val-review.json。
python -m vehtraj evaluate --checkpoint runs/official-trained/best.pt --split test --output runs/official-test --max-animations 1
# 评估生成 test-review.html/json、逐步 CSV、指标和使用官方地图渲染的 PNG/GIF。
```

训练选择 best checkpoint 后，用它对完整验证集进行实际推理，再生成验证界面；不是把训练标签复制成预测。`--max-batches` 截断训练会在界面明确标注。`evaluate --max-samples` 限制目标数时，界面明确标注子集，不能当作完整集合。

验证现有随工程权重，也可以执行：

```bash
python -m vehtraj evaluate --checkpoint checkpoints/interaction-mini.pt --split val --training-log reports/training.json --output runs/validation --max-animations 1
python -m vehtraj evaluate --checkpoint checkpoints/interaction-mini.pt --split test --training-log reports/training.json --output runs/test --max-animations 1
```

## 官方 Matplotlib 控件界面

有图形桌面时，下列命令显示官方地图绘制函数与播放/暂停/前后帧控件，并叠加本次真实验证预测：

```bash
python -m vehtraj visualize --review reports/val-review.json --sample-index 9
```

本项目默认 csv/map 已匹配。更换原始数据时必须显式提供匹配的 `--csv` 与 `--map`，且 review 的数据哈希、车辆与帧号必须一致。地图、case 或预测身份不匹配会报错，不静默画到另一条轨迹上。

无图形桌面时可保存：

```bash
python -m vehtraj visualize --review reports/val-review.json --sample-index 9 --frame 40 --save-png runs/validation.png --save-gif runs/validation.gif
```

提供的 PNG/GIF 都实际调用官方 `draw_map_without_lanelet`、`update_objects_plot` 和车辆轮廓函数生成。行人/自行车没有原始尺寸，适配层使用点符号，不虚构物理尺寸。

## 重新构建与验证

```bash
python scripts/prepare_official_map.py
python scripts/build_data_explorer.py
python -m pytest -q
# 可选浏览器回归，另需 Playwright 与本机 Chromium：
python tools/verify_data_explorer.py --chromium /usr/bin/chromium
```

地图 `.map.json` 是官方投影得到的缓存，默认随工程提供；标准库页面构建入口仍然无需训练环境。地图改变后必须在安装 pyproj 的环境重建缓存。已有错误哈希的地图或 CSV 不会被下载器覆盖。

## 已执行的验证与边界

本次完整执行 30 轮 CPU 训练；选中 epoch 30，权重 SHA-256 与随工程权重相同。浏览器检查全部 52 个原始 case、33 个验证目标、111 个测试目标的身份与末帧误差；Matplotlib 回放和误差连线也经过验证。详细证据见 `reports/data-explorer-validation.json`。

当前地图仅用于显示，并未作为 GNN 输入。模型结构、测试精度、数据划分和仅单路口子集的限制保持原样；不能将有地图的画面解释为模型已学会车道约束。当前云端没有图形桌面，已实际验证的是官方函数渲染、控件程序化切帧、PNG/GIF 和浏览器交互，未测试实体桌面窗口。
