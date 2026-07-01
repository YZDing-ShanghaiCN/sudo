# MegaPose 模块说明

本目录是 PoseEstimation 项目里的 MegaPose 模块，只介绍我这边整理后的单任务调用方式，不覆盖整个 MegaPose 官方项目，也不展开复杂的环境部署。

## 1. 目录结构

MegaPose 按 task directory 组织数据。每次只处理一个任务目录，例如：

```text
megapose_data/examples/<task_name>/
├── image_rgb.png
├── camera_data.json
├── inputs/
│   └── object_data.json
├── meshes/
│   └── <object_label>/...
├── outputs/
│   └── object_data.json
└── visualizations/
    ├── all_results.png
    ├── contour_overlay.png
    ├── detections.png
    └── mesh_overlay.png
```

说明：

- `image_rgb.png`：输入 RGB 图像。
- `camera_data.json`：相机内参。
- `inputs/object_data.json`：输入的目标信息，通常包含类别名、检测框或初始位姿信息。
- `meshes/`：当前任务用到的 CAD / mesh 文件。
- `outputs/object_data.json`：MegaPose 推理结果，保存 6D pose 输出。
- `visualizations/`：推理和渲染可视化结果。

## 2. 输入准备

在运行脚本前，先把要处理的任务整理到 `megapose_data/examples/<task_name>/` 下，并确认这个目录里至少有：

- `image_rgb.png`
- `camera_data.json`
- `inputs/object_data.json`
- `meshes/`

如果你已经有一个自定义任务，比如 `barbecue-sauce`，就直接按这个目录名处理，不要把多个 example 混在一次运行里。

## 3. 单任务推理

我这边的默认调用方式是一次只跑一个 task directory。最常用的入口是：

```bash
cd /home/user/Desktop/PoseEstimation/megapose6d
python -m megapose.scripts.run_inference_on_example barbecue-sauce --vis-detections --run-inference --vis-outputs
```

把 `barbecue-sauce` 换成你自己的 `<task_name>` 即可。这个脚本会：

1. 读取 `megapose_data/examples/<task_name>/image_rgb.png`
2. 读取 `camera_data.json`
3. 读取 `inputs/object_data.json`
4. 读取 `meshes/` 里的 mesh 文件
5. 生成推理结果到 `outputs/object_data.json`
6. 生成可视化到 `visualizations/`

如果你只想先看检测框，可以只开 `--vis-detections`；如果只想生成 pose 可视化，可以只开 `--vis-outputs`。

## 4. 结果分析脚本

结果分析脚本的作用是：读取

```text
<task_dir>/outputs/object_data.json
```

然后在同一个任务目录下输出分析结果文件，方便快速检查 MegaPose 的位姿估计是否稳定、是否存在离群结果。

我这边当前对应的单任务分析脚本是：

```bash
python megapose_data/examples/megapose/scripts/analyze_pose_consistency.py \
  --task-dir megapose_data/examples/barbecue-sauce \
  --label <object_label> \
  --unit m
```

这个脚本会在同一个 task directory 里写出：

- `outputs/pose_consistency_analysis.json`
- `outputs/pose_consistency_table.csv`

其中 `--label` 需要和 `outputs/object_data.json` 里的目标类别一致；如果你的任务只有一个物体类别，就直接填那个类别名。

## 5. 输出说明

推理完成后，通常会得到两类结果：

- `outputs/object_data.json`：MegaPose 的 6D pose 结果。
- `visualizations/`：用于人工检查结果的图像。

分析完成后，会额外得到：

- `outputs/pose_consistency_analysis.json`：汇总统计结果。
- `outputs/pose_consistency_table.csv`：逐条记录，便于后续筛查。

## 6. 常见问题

- 这个流程是按单个任务目录设计的，不是批量遍历所有 example。
- 如果 `outputs/object_data.json` 不存在，先确认推理步骤已经执行完，并且目录名写对了。
- 如果可视化文件没生成，检查是否加了 `--vis-detections` 或 `--vis-outputs`。
- 如果分析脚本报 `--label` 不匹配，说明任务里的目标类别名和你传入的标签不一致。

## 7. 备注

本 README 只服务于 PoseEstimation 项目里的 MegaPose 子模块。其他模型目录各自有自己的输入输出约定，这里不展开。