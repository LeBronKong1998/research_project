# 数据来源、论文与实验协议

## 实际采用的数据

INTERACTION：Zhan et al., **INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps**, 2019, [arXiv:1910.03088](https://arxiv.org/abs/1910.03088)。[官方数据主页](https://interaction-dataset.com/)、[官方解析工具](https://github.com/interaction-dataset/interaction-dataset)。官方代码的 `DELTA_TIMESTAMP_MS=100` 支持 10 Hz 采样约定。官方工具内 `.TestScenarioForScripts` 是脚本测试场景，本项目不把它作为真实实验数据。

Jiang et al., **A naturalistic trajectory dataset with dense interaction for autonomous driving**, *Scientific Data* 12, 1084 (2025), [DOI:10.1038/s41597-025-05344-7](https://doi.org/10.1038/s41597-025-05344-7)。其 [作者仓库 InterHub](https://github.com/zxc-tju/InterHub) 明确将 `data/0_origin_datasets/interaction_multi` 称为原始 INTERACTION 的真实 mini 子集。本项目直接下载该子集 CSV，不运行场景生成器、不用论文中的事件标签代替轨迹。

可复现下载源固定到作者仓库提交 `1d8a764554017618447247e5c55eff4b4ab50de4`：

- 路径：`data/0_origin_datasets/interaction_multi/train/DR_USA_Intersection_GL_train.csv`
- SHA-256：`c59094abd09f3638268b916a8f24a1b83458ace02d2d8dd4d6861082401b6970`
- 1,535,255 bytes，24,771 行，52 个 case；不是完整 INTERACTION，也不是全部 InterHub。
- 原始列：`case_id,track_id,frame_id,timestamp_ms,agent_type,x,y,vx,vy,psi_rad,length,width`。
- 世界坐标为米，速度米/秒，航向弧度，时间毫秒；10 Hz，每个 case 最长 40 帧。
- 原始 CSV 已随工程存入 `data/interaction` 并纳入 GitHub；源文件逐字节保留，来源、校验值和上游许可通知同时提供。使用须遵守 INTERACTION 与发布方的数据条款，作者代码许可不自动替代原数据许可。

## 文献对实现的影响

- INTERACTION 与 InterHub：采用真实多车路口轨迹、显式身份/时间列，按 case 分组；默认 10 帧历史、30 帧预测。这与 multi-agent 版本的 4 秒样本兼容，不能沿用原脚本的 30+50 帧。
- Li et al., **GRIP: Graph-based Interaction-aware Trajectory Prediction**, ITSC 2019；增强版 **GRIP++**, [arXiv:1907.07792](https://arxiv.org/abs/1907.07792)、[作者实现](https://github.com/xincoder/GRIP)：车辆间交互应通过图结构编码。此项目使用目标车星形图上的单头注意力和 GRU，属于轻量基线，不宣称完整复现 GRIP。
- Liang et al., **Learning Lane Graph Representations for Motion Forecasting (LaneGCN)**, ECCV 2020, [arXiv:2007.13732](https://arxiv.org/abs/2007.13732)、[作者实现](https://github.com/uber-research/LaneGCN)：地图和车道拓扑对复杂场景有价值；本版本没有读取地图，不能宣称具备相应能力。
- Deo & Trivedi, **Convolutional Social Pooling for Vehicle Trajectory Prediction**, CVPR Workshops 2018, [arXiv:1805.06771](https://arxiv.org/abs/1805.06771)：多模态轨迹需要区分预测分支与实际可选择的轨迹，因此同时报告 best-of-K 与 top-1，不能把 oracle minADE 当作部署误差。

本次在线实际核查了 INTERACTION 官方工具、InterHub 作者说明及真实 CSV。arXiv/期刊全文站点受当前环境网络策略限制；以上论文链接是可追溯参考，不表示本次已完整下载和逐篇复现实验。新增算法是本项目的基线实现。

## 数据划分与限制

`SHA256(split_group)[:8] % 100` 固定划分：train 0–64，val 65–74，cal 75–87，test 88–99。同一 case 内所有车辆/窗口保持在同一集合；没有 case_id 的连续原始文件整个文件为一组，避免随机切窗导致明显泄漏。真实文件名和 case_id 组成 group，因此请勿在实验中重命名 CSV。`training.json` 保存具体分组与文件哈希；评估拒绝与训练不一致的数据、时间窗口或 stride。

该 mini 子集来自官方 train 文件，在其中另建自定义划分，**不是官方挑战测试协议**。不同 case 可能来自同一地点、相关时间或同一物理车辆；case_id 内 track_id 不能证明跨 case 独立。只保证 case 不重叠，不保证地点、时间、物理车辆独立，不把结果称作跨场景泛化或排行榜成绩。完整研究应使用官方原始 train/val/test 元数据，或增加独立录制文件/地点，再评估分布迁移。不要合并镜像重复 CSV 或官方 train/val 数据后将结果当作官方成绩。

输入只用历史帧，邻车按历史最后一帧距离选取；最多八辆邻车，40 米图掩码；不插补断帧。重复 case/track/frame、非有限状态与非 10 Hz 时间戳直接报错。完整未来只用于目标标签，不参与邻车筛选。只保留 `car/vehicle`，没有地图、交通灯与行人建模。

## 指标解释

- `minADE/minFDE`：三个分支中最优的平均/终点位置误差，单位米；两项分别取最小，最优分支可能不同。
- `top1ADE/top1FDE`：仅使用模型概率最大的分支，反映不看真实未来时可选择的轨迹误差。
- `top1PositionRMSE`：每条轨迹欧氏位置误差的 RMS，再对轨迹平均；不是全样本全时刻池化的 RMSE。
- `top1Displacement@1s/2s/3s`：相应时刻 top-1 位置误差。
- `top1EndpointAccuracy2m`：top-1 终点落在真值 2 米内的轨迹比例；不是分类准确率。
- `missRate2m`：所有分支终点均超过 2 米的比例，属于 oracle 多模态指标。
- 速度误差米/秒，航向圆周误差弧度。恒速基线只使用最后一个历史位置和速度，使用同一测试目标。
- conformal 校准按 case 聚合最大误差。少于 19 个校准 case 时 95% 半径无界，JSON 为 `null`，不输出虚假的有限置信区间。case 相关性未知，即使样本足够也不能保证实际道路 95% 覆盖。

## 扩展数据

完整 INTERACTION 更适合多路口、环岛、汇入泛化；highD / inD / exiD 分别适合高速、路口、出入口，但需申请许可，CSV 列、采样率和坐标定义不同。NGSIM 可公开获取 CSV，但有已知测量噪声；不能把非因果平滑产生的速度用作在线输入而不披露未来泄漏。本版本不提供未经验证的 highD/exiD/inD 占位开关；仅承诺已实现并实测的 INTERACTION 解析器。
