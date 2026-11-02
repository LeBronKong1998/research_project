# 随工程提供的真实原始数据

`DR_USA_Intersection_GL_train.csv` 是 **InterHub 作者公开提供的 INTERACTION multi-agent mini 子集原始文件**。本工程逐字节保留该 CSV，没有生成轨迹、删除参与者或插补缺失值。

- 原始出处：[InterHub 固定提交中的数据目录](https://github.com/zxc-tju/InterHub/tree/1d8a764554017618447247e5c55eff4b4ab50de4/data/0_origin_datasets/interaction_multi/train)
- 文件大小：1,535,255 bytes
- SHA-256：`c59094abd09f3638268b916a8f24a1b83458ace02d2d8dd4d6861082401b6970`
- 52 个 case，755 条 case/track 参与者轨迹，24,771 行记录。
- 类型：24,415 行 `car`，356 行 `pedestrian/bicycle`。行人/自行车的 `psi_rad/length/width` 在源数据中为空，保持为空。模型训练仅使用 car/vehicle，数据浏览器展示全部原始记录。
- 10 Hz，每个场景最长 40 帧。坐标为米、速度米/秒、时间毫秒、航向弧度。

克隆仓库会直接获得原始 CSV，无需下载脚本才能运行。`python -m vehtraj download` 现在用于校验这个内置文件，只有缺文件时才重新从固定源下载，不覆盖内容不同的文件。

双击工程中的 `docs/data-explorer.html` 即可离线查看真实运动与数据表。页面也内嵌完全相同的 CSV，可导出原文件或当前 case。

## 出处与许可保留

来源论文：Jiang et al., *A naturalistic trajectory dataset with dense interaction for autonomous driving*, Scientific Data 12, 1084 (2025), DOI [10.1038/s41597-025-05344-7](https://doi.org/10.1038/s41597-025-05344-7)。原始 INTERACTION 论文：Zhan et al., 2019, [arXiv:1910.03088](https://arxiv.org/abs/1910.03088)。

随附 `UPSTREAM_LICENSE.txt` 原样保留 InterHub 上游仓库的 MIT 许可通知（Copyright 2024 Xiaocong Zhao）。本工程不重新给原始 INTERACTION 数据授予许可；原数据仍归原作者，使用时也须遵守 [INTERACTION 发布方](https://interaction-dataset.com/) 适用条款。该文件不是完整 INTERACTION 数据集；其他数据集或完整版不在本工程的再发布范围内。

精确来源、固定提交和文件校验值见 `provenance.json`。
