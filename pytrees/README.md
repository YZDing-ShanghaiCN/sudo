# py_trees 纯 Python 行为树学习项目

这个项目用纯 Python 和 `py_trees` 模拟一个移动机器人完成“寻找水杯并递送给用户”的任务。不依赖 ROS、ROS 2、Nav2、Gazebo 或任何真实硬件。

## 安装与运行

```bash
cd pytrees_demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --scenario recovery
pytest -q
```

可用场景：

```bash
python main.py --scenario normal
python main.py --scenario recovery
python main.py --scenario low-battery
python main.py --scenario grasp-failure
python main.py --scenario object-dropped
```

`main.py` 默认每 0.5 秒 Tick 一次。调试时可加快：

```bash
python main.py --scenario recovery --tick-period 0
```

## 项目结构

```text
pytrees_demo/
├── README.md
├── requirements.txt
├── main.py
├── robot_state.py
├── tree_builder.py
├── behaviours/
│   ├── __init__.py
│   ├── base.py
│   ├── conditions.py
│   ├── navigation.py
│   ├── manipulation.py
│   └── monitoring.py
└── tests/
    ├── conftest.py
    └── test_behaviours.py
```

## 行为树结构

```text
递送任务：Sequence(memory=True)
├── 系统准备：Sequence(memory=False)
│   ├── 检查电量
│   └── 检查传感器状态
├── 获得目标位置：Selector(memory=True)
│   ├── 使用已有记忆：Sequence(memory=False)
│   │   ├── 检查记忆中是否有目标位置
│   │   └── 验证目标位置是否可靠
│   └── 搜索目标物体
├── 到达目标附近：Selector(memory=True)
│   ├── 正常导航
│   └── 导航恢复：Sequence(memory=True)
│       ├── 清除旧路径
│       ├── 重新规划
│       └── 再次导航
├── 有限次数重试抓取：Retry(num_failures=3)
│   └── 抓取物体：Sequence(memory=True)
│       ├── 执行抓取
│       └── 验证抓取结果
├── 递送过程监控：Parallel(SuccessOnAll)
│   ├── 导航到用户位置
│   ├── 监控电量
│   └── 监控物体是否仍被抓住
└── 完成交付：Sequence(memory=True)
    ├── 放置或交付物体
    └── 验证任务完成
```

## 核心概念

### Tick

行为树不是从头到尾只执行一次，而是周期性 Tick。每次 Tick 从根节点开始访问当前应执行的分支。动作节点如果尚未完成，会返回 `RUNNING`，下一次 Tick 继续推进模拟进度。

本项目中，`NavigateToTarget`、`NavigateToUser`、`SearchForObject`、`DeliverObject` 都需要多个 Tick 才能完成。

### 状态

- `SUCCESS`：节点完成，例如电量足够、导航到达、交付验证通过。
- `FAILURE`：节点失败，例如低电量、导航路径被阻塞、水杯掉落。
- `RUNNING`：节点还在执行，例如导航进度只完成一部分。
- `INVALID`：节点当前没有运行，或之前运行中但被父节点切换分支中断。

### 生命周期

所有自定义节点都继承 `behaviours/base.py` 中的 `RobotBehaviour`。

- `setup()`：行为树 setup 时执行一次，真实机器人中常用于创建 action client、订阅器或硬件句柄。
- `initialise()`：节点从非 `RUNNING` 状态进入执行时调用，适合重置局部计数器。
- `update()`：每次 Tick 访问到该节点都会调用，是条件判断和动作推进的主要位置。
- `terminate()`：节点成功、失败或被中断时调用；被中断时通常会收到 `INVALID`。

### Sequence

`Sequence` 按顺序执行子节点。任意子节点 `FAILURE`，整个 Sequence 立即 `FAILURE`；所有子节点 `SUCCESS`，它才 `SUCCESS`。

例如 `系统准备` 中，`检查电量` 失败后不会继续检查传感器。

### Selector

`Selector` 按顺序尝试子节点。遇到第一个 `SUCCESS` 或 `RUNNING` 的分支就停止，适合备用方案和失败恢复。

例如 `获得目标位置` 先尝试使用已有记忆；如果没有可靠目标位置，则切换到 `搜索目标物体`。`到达目标附近` 先尝试正常导航；失败后切换到恢复分支。

### Parallel

`递送过程监控` 使用：

```python
py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
```

含义是所有子节点都成功时 Parallel 才返回 `SUCCESS`。如果任意监控节点返回 `FAILURE`，Parallel 立即失败，父节点也会失败。

导航到用户位置是主动作；电量监控和物体监控是安全条件。它们适合放在 Parallel 中，因为每次 Tick 都应同时推进导航并检查风险。

### Decorator

Decorator 包装一个子节点并改变其行为。本项目使用 `Retry` 包装 `抓取物体`，最多容忍 3 次失败。`normal` 和 `recovery` 场景中第一次抓取失败、第二次成功；`grasp-failure` 场景中持续失败，最终整个任务失败。

### Blackboard

Blackboard 类似行为树中的共享任务状态，不同节点可以通过它交换信息。节点会创建 client，并注册 key 的访问权限：

```python
self.blackboard = py_trees.blackboard.Client(name=name)
self.blackboard.register_key("target_location", py_trees.common.Access.WRITE)
self.blackboard.register_key("target_location", py_trees.common.Access.READ)
```

本项目共享这些 key：

```text
target_location
target_confidence
navigation_failure_reason
recovery_count
object_grasped
task_progress
```

例子：`SearchForObject` 写入 `target_location` 和 `target_confidence`；`NavigateToTarget` 读取 `target_location`；`ReplanPath` 写入 `recovery_count`。

### memory 参数

`Sequence(memory=True)`：如果某个子节点返回 `RUNNING`，下一次 Tick 会从该子节点继续，不再重新 Tick 已经成功的前序子节点。根节点 `递送任务` 使用这种方式，任务推进后不会每轮从系统准备重新开始。

`Sequence(memory=False)`：每次 Tick 都从第一个子节点开始。`系统准备` 使用这种方式，便于演示无记忆 Sequence 的顺序检查。

`Selector(memory=True)`：一旦某个分支返回 `RUNNING`，下一次 Tick 会继续该分支。`获得目标位置` 和 `到达目标附近` 使用这种方式；开始搜索或导航恢复后，不会每轮重新从第一个备用分支开始。

`Selector(memory=False)`：每次 Tick 都从第一个子节点重新尝试，适合“每轮都重新检查最高优先级条件”的场景。本项目没有把恢复 Selector 设为无记忆，因为那会让恢复流程不够直观。

## 场景说明

- `normal`：目标位置已知，正常导航成功；第一次抓取失败，Retry 后成功；最终递送成功。
- `recovery`：目标位置未知，需要搜索；第一次正常导航失败，清除旧路径并重新规划后成功。
- `low-battery`：初始电量低于阈值，`系统准备` 立即失败。
- `grasp-failure`：抓取持续失败，Retry 达到上限后任务失败。
- `object-dropped`：递送途中水杯掉落，Parallel 中的物体监控失败，任务失败。

## 观察输出

每次 Tick 会打印：

- Tick 编号
- 根节点状态
- `py_trees.display.unicode_tree(root, show_status=True)` 的 Unicode 树
- 每个节点的 status 和 `feedback_message`
- 关键 Blackboard 数据
- 当前 `SimulatedRobotState`

## 从模拟迁移到真实机器人

迁移到真实机器人时，行为树结构可以保留，主要替换动作和监控节点内部实现：

- `NavigateToTarget`、`NavigateToUser`：替换为 ROS 2 action client、Nav2 goal 发送与结果检查。
- `SearchForObject`：替换为视觉检测、目标定位或感知服务调用。
- `GraspObject`、`DeliverObject`：替换为机械臂、夹爪或末端执行器控制接口。
- `CheckBattery`、`CheckSensors`、`MonitorBattery`、`MonitorObject`：替换为真实传感器、诊断话题、状态估计或硬件反馈。

`Blackboard` 仍可保留，用来传递目标位姿、导航失败原因、恢复次数、抓取状态和任务进度。
