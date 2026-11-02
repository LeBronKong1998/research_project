# 官方地图与模型验证结果

本次按用户指定的 [官方工具](https://github.com/interaction-dataset/interaction-dataset) 实际复用其无 Lanelet2 的 `pyproj` 投影、地图绘制和车辆轮廓函数。版本固定为 `053213d8aeb4c6f6132d272325a11ae623bd4e35`，BSD 通知和源文件哈希随工程保存。操作说明见 [官方可视化指南](../docs/OFFICIAL_VISUALIZATION.md)。

- 真实地图：`DR_USA_Intersection_GL.osm`，与 CSV 同名；583 节点、163 way。
- 投影：官方原点 (0,0)，WGS84 UTM 31；逐节点对照作者 `.osm_xy`，最大偏差 `0.000006875183` 米，无坐标拟合。
- 完整执行 30 轮 CPU 训练；按验证损失选择 epoch 30。权重 SHA-256 `3679135bf6555460fa31e859ec3bfff4d13f24cb8db4ddff4508f32e601742ea`，与已提供的 `checkpoints/interaction-mini.pt` 完全相同。
- 同一网页模板展示原始数据、33 个验证目标、111 个测试目标；不是将标签复制为预测。每个目标都带原始 case/track/frame 身份、世界坐标、概率与逐帧误差。
- 训练后自动生成验证集地图页面，评估后自动生成所选 val/test 集合地图页面及官方 PNG/GIF。

| 集合 | 目标数 | Top-1 ADE（m） | Top-1 FDE（m） | 恒速 ADE（m） | 恒速 FDE（m） |
|---|---:|---:|---:|---:|---:|
| 验证集 | 33 | 1.1417 | 3.1028 | 1.2341 | 3.3650 |
| 测试集 | 111 | 1.4035 | 3.8622 | 1.3488 | 3.7228 |

测试集首选预测仍略差于恒速，不能以验证集改善推断测试泛化提升；本次主要完成官方地图上的检查与展示。地图未进入当前模型输入，原有单路口子集与 case 独立性限制不变。

## 可查看的实际输出

- [统一离线界面](../docs/data-explorer.html)：上方选择“验证集预测”或“测试集预测”。
- [模型验证网页截图](model-validation-preview.png) / [手机截图](model-validation-mobile.png)。
- [官方 Matplotlib 验证图](official-model-validation.png) / [验证动画](official-model-validation.gif)。
- [官方 Matplotlib 测试图](official-model-test.png) / [测试动画](official-model-test.gif)。
- [验证集实际推理记录](val-review.json) / [测试集实际推理记录](test-review.json)。

网页截图固定展示验证样本 9（case 16 / track 4），示例本身 top-1 终点误差约 9.123 米，不能将它的 oracle 最优分支误差冒充实际首选预测；所有目标均可在界面中选择。

## 验证证据

当前依赖安装脚本完整重跑通过（含 pyproj、地图投影、可编辑包、原始数据完整性、界面构建、pip check、27 项测试及 CPU 前后向 smoke）。浏览器核查全部 52 个原始场景、33 个验证目标和 111 个测试目标的显示、身份和末帧误差；地图开关、播放、逐帧、目标选择、缩放、CSV 导出与桌面/手机布局通过，无浏览器异常或外部数据请求。详细机器可读记录见 [data-explorer-validation.json](data-explorer-validation.json)。

官方 Matplotlib 函数渲染、控件程序化切帧、误差向量与 PNG/GIF 已实测；当前云端无实体图形桌面，未声称已测试本地桌面窗口。浏览器播放界面已实测。CUDA 未测试。

只从 Git 受版本管理文件导出的独立副本也已通过 27 项测试、页面重建、官方 PNG 渲染，并逐值复现 33 个验证目标与 111 个测试目标的预测和误差，不依赖原工作目录的未上传文件。
