# MuJoCo Gym Envs

这是一个基于 **MuJoCo + Gymnasium + PyTorch** 的强化学习练习项目。

项目重点是搭出一套比较标准、容易扩展的 RL 代码结构：

- `models/` 放 MuJoCo XML 物理模型；
- `envs/` 放 Gymnasium 环境；
- `agents/` 放策略网络；
- `training/` 放训练配置、PPO 算法、checkpoint 工具；
- `scripts/` 只放常用命令入口；
- `scripts/legacy/` 保留旧版单任务脚本，方便对照。

---

## 1. 项目结构

```text
gym_envs/
├── agents/
│   ├── ppo_agent.py                  # PPO Actor-Critic 网络
│   └── sac_agent.py                  # SAC Actor / 双 Q 网络
├── envs/
│   ├── cart_reach_env.py             # 1D 小车 reach
│   ├── block_reach_3d_env.py         # 3D 物块 reach
│   ├── cart_pole_balance_env.py      # 小车杆近似倒立平衡
│   ├── cart_pole_recover_env.py      # 小车杆中等角度恢复
│   ├── cart_pole_wide_recover_env.py # 小车杆大角度恢复
│   ├── cart_pole_swing_up_env.py     # 从下方甩杆
│   ├── cart_pole_final_env.py        # 甩上去并让小球回到中心停留 2s
│   ├── two_link_arm_reach_env.py     # 二连杆机械臂末端 reach
│   ├── two_link_arm_random_reach_env.py
│   └── two_link_arm_gripper_ball_reach_env.py
├── models/
│   ├── cart_reach.xml
│   ├── block_reach_3d.xml
│   ├── cart_pole_balance.xml
│   ├── two_link_arm_reach.xml
│   └── two_link_arm_gripper_ball.xml
├── training/
│   ├── checkpoints.py                # 加载 checkpoint
│   ├── configs.py                    # PPO 任务配置
│   ├── ppo.py                        # 通用 PPO 训练逻辑
│   ├── sac_configs.py                # SAC 任务配置
│   └── sac.py                        # 通用 SAC 训练逻辑
├── scripts/
│   ├── train.py                      # PPO 训练入口
│   ├── render.py                     # PPO 可视化入口
│   ├── train_sac.py                  # SAC 训练入口
│   ├── render_sac.py                 # SAC 可视化入口
│   ├── eval_cart_pole_final.py       # 最终任务评估脚本
│   └── legacy/                       # 旧版单任务脚本
├── checkpoints/                      # 训练出的模型，不建议提交
├── logs/                             # 评估 CSV / 日志，不建议提交
└── readme.md
```

以后新增任务时，推荐流程是：

1. 在 `models/` 里加 XML；
2. 在 `envs/` 里加 Gymnasium Env；
3. 在 `training/configs.py` 里注册一个任务配置；
4. 直接用 `python scripts/train.py --task 任务名` 或 `python scripts/train_sac.py --task 任务名` 训练。

---

## 2. 环境依赖

```bash
conda create -n mujoco_rl python=3.11 -y
conda activate mujoco_rl
pip install mujoco gymnasium torch numpy
```

检查依赖：

```bash
python -c "import mujoco, gymnasium, torch, numpy; print('ok')"
```

---

## 3. 当前任务

当前已经注册的任务名：

```text
cart_reach
block_reach_3d
cart_pole_balance
cart_pole_recover
cart_pole_wide_recover
cart_pole_swing_up
cart_pole_final
two_link_arm_reach
two_link_arm_random_reach
two_link_arm_gripper_ball_reach
```

小车杆课程学习难度：

```text
cart_pole_balance
→ cart_pole_recover
→ cart_pole_wide_recover
→ cart_pole_swing_up
→ cart_pole_final
```

含义：

- `cart_pole_balance`：从接近倒立开始，只练平衡；
- `cart_pole_recover`：从更大扰动恢复到倒立；
- `cart_pole_wide_recover`：从约 ±60 度恢复；
- `cart_pole_swing_up`：从下方开始，把杆子甩上去；
- `cart_pole_final`：甩上去后，让小球逐渐来到世界中心，并连续停留 2 秒。
- `two_link_arm_reach`：二维二连杆机械臂，前方工作区 reach。
- `two_link_arm_random_reach`：更大初始姿态和更大目标区域的二连杆 reach，默认从 `two_link_arm_reach` checkpoint 初始化。
- `two_link_arm_gripper_ball_reach`：二连杆末端带打开的小夹爪，目标是让夹爪框住黄色小球。

---

## 4. 训练命令

PPO 训练入口：

```bash
python scripts/train.py --task cart_reach
python scripts/train.py --task block_reach_3d
python scripts/train.py --task cart_pole_balance
python scripts/train.py --task cart_pole_recover
python scripts/train.py --task cart_pole_wide_recover
python scripts/train.py --task cart_pole_swing_up
python scripts/train.py --task cart_pole_final
python scripts/train.py --task two_link_arm_reach
python scripts/train.py --task two_link_arm_random_reach
python scripts/train.py --task two_link_arm_gripper_ball_reach
```

小车杆推荐按难度顺序训练：

```bash
python scripts/train.py --task cart_pole_balance
python scripts/train.py --task cart_pole_recover
python scripts/train.py --task cart_pole_wide_recover
python scripts/train.py --task cart_pole_swing_up
python scripts/train.py --task cart_pole_final
```

机械臂推荐按难度顺序训练：

```bash
python scripts/train.py --task two_link_arm_reach
python scripts/train.py --task two_link_arm_random_reach
python scripts/train.py --task two_link_arm_gripper_ball_reach
```

后一个任务会默认从前一个难度的 checkpoint 初始化，例如：

```text
cart_pole_recover      <- checkpoints/ppo_cart_pole_balance_torch.pt
cart_pole_wide_recover <- checkpoints/ppo_cart_pole_recover_torch.pt
cart_pole_swing_up     <- checkpoints/ppo_cart_pole_wide_recover_torch.pt
cart_pole_final        <- checkpoints/ppo_cart_pole_swing_up_torch.pt
```

如果想从随机网络开始：

```bash
python scripts/train.py --task cart_pole_final --no-init-from
```

如果想指定某个 checkpoint 作为起点：

```bash
python scripts/train.py --task cart_pole_final --init-from checkpoints/your_model.pt
```

如果想临时改训练轮数：

```bash
python scripts/train.py --task cart_pole_final --total-updates 500
```

网络规模：

```text
简单任务默认使用 64-64 MLP。
swing-up 和 final 默认使用 128-128-128 MLP。
```

SAC 训练入口：

```bash
python scripts/train_sac.py --task cart_reach
python scripts/train_sac.py --task block_reach_3d
python scripts/train_sac.py --task two_link_arm_reach
python scripts/train_sac.py --task two_link_arm_random_reach
python scripts/train_sac.py --task two_link_arm_gripper_ball_reach
python scripts/train_sac.py --task cart_pole_balance
python scripts/train_sac.py --task cart_pole_recover
python scripts/train_sac.py --task cart_pole_wide_recover
python scripts/train_sac.py --task cart_pole_swing_up
python scripts/train_sac.py --task cart_pole_final
```

临时改 SAC 训练步数：

```bash
python scripts/train_sac.py --task two_link_arm_gripper_ball_reach --total-steps 300000
```

SAC 会保存到 `checkpoints/sac_*.pt`，不会覆盖 PPO 的 `ppo_*.pt`。

---

## 5. 可视化命令

所有任务都用同一个可视化入口：

```bash
python scripts/render.py --task cart_reach
python scripts/render.py --task block_reach_3d
python scripts/render.py --task cart_pole_balance
python scripts/render.py --task cart_pole_recover
python scripts/render.py --task cart_pole_wide_recover
python scripts/render.py --task cart_pole_swing_up
python scripts/render.py --task cart_pole_final
python scripts/render.py --task two_link_arm_reach
python scripts/render.py --task two_link_arm_random_reach
python scripts/render.py --task two_link_arm_gripper_ball_reach
```

指定模型：

```bash
python scripts/render.py --task cart_pole_final --model checkpoints/ppo_cart_pole_final_torch.pt
```

SAC 可视化入口：

```bash
python scripts/render_sac.py --task cart_reach
python scripts/render_sac.py --task block_reach_3d
python scripts/render_sac.py --task two_link_arm_reach
python scripts/render_sac.py --task two_link_arm_random_reach
python scripts/render_sac.py --task two_link_arm_gripper_ball_reach
python scripts/render_sac.py --task cart_pole_final
```

---

## 6. 最终模型评估

最终任务建议用专门评估脚本，它会统计成功率、平均回报、停留时间等指标。

```bash
python scripts/eval_cart_pole_final.py --episodes 200
```

打开 viewer 评估：

```bash
python scripts/eval_cart_pole_final.py --episodes 20 --render
```

保存每回合结果：

```bash
python scripts/eval_cart_pole_final.py --episodes 50 --csv logs/final_eval.csv
```

---

## 7. Observation 设计

`cart_reach` 的 observation 是 4 维：

```python
obs = [
    x / 0.8,
    x_dot / 3.0,
    target_x / 0.8,
    error / 0.8,
]
```

`cart_reach` 的 reward 现在没有无条件存活奖励，避免小车停在目标附近但不触发成功：

```python
reward = -6.0 * abs_error
reward += 12.0 * progress
reward -= 0.05 * x_dot**2
reward -= 0.001 * force**2
```

成功条件：

```python
success = abs_error < 0.04 and abs(x_dot) < 0.35
```

如果小车停在成功圈边缘外不继续靠近，会额外扣一点分；成功时额外给 `+20.0`。

`block_reach_3d` 的 observation 是 12 维：

```python
obs = [
    pos / 0.8,
    vel / 3.0,
    target_pos / 0.8,
    error / 0.8,
]
```

其中 `pos`、`vel`、`target_pos`、`error` 都是 3 维向量，所以一共是 `3 + 3 + 3 + 3 = 12` 维。

小车杆任务的 observation 会包含小车位置、速度、杆角度误差、角速度等信息，具体以对应 `envs/cart_pole_*.py` 为准。

`two_link_arm_reach` 和 `two_link_arm_random_reach` 的 observation 都是 12 维：

```python
obs = [
    cos(q1), sin(q1),
    cos(q2), sin(q2),
    qvel / 6.0,
    end_effector_xy / 0.78,
    target_xy / 0.78,
    error_xy / 0.78,
]
```

动作是 2 维连续 torque：

```python
action = [shoulder_action, elbow_action]
torque = 3.0 * action
```

`two_link_arm_reach` 是第一阶段课程学习：

```text
初始姿态：大致朝前，小范围随机；
目标位置：前方工作区随机；
成功条件：末端距离目标小于 0.08。
```

reward 保持直观，但加入了“距离变近”的 shaping：

```python
reward = -10.0 * distance
reward += 20.0 * progress
reward -= 0.001 * torque_square_sum
if distance < 0.20:
    reward -= 0.10 * end_effector_speed
    reward -= 0.03 * action_delta_square_sum
```

成功条件：

```python
success = distance < 0.09
```

如果成功，额外给 `+30.0`，并结束当前 episode。超时还没成功会额外扣 `-10.0`。这个版本没有“目标附近每步加分”，所以策略停在成功圈外会继续被距离项扣分。若末端停在成功圈边缘外且没有继续靠近，会额外扣一点分，避免卡边缘。

`two_link_arm_random_reach` 使用同一套 reward，但扩大了初始姿态和目标采样范围。默认会优先从自己的 checkpoint 续训；如果还没有自己的 checkpoint，再从第一阶段 checkpoint 初始化。这个第二阶段先用中等随机范围，目标是把成功率推到接近 100%，稳定后再加第三阶段全空间随机。

这个任务的 PPO batch 比默认更大：

```text
steps_per_update = 4096
minibatch_size = 512
entropy_coef = 0.001
learning_rate = 2e-4
```

这样每次更新看到的轨迹更多，连续控制任务的梯度会稳一点，但单个 update 会更慢。

`two_link_arm_gripper_ball_reach` 是夹爪任务第一阶段。它还不要求闭合夹爪，也不要求搬运，只要求打开的二指夹爪框住黄色小球。绿色点是下一阶段搬运目标的预告。observation 是 14 维：

```python
obs = [
    cos(q1), sin(q1),
    cos(q2), sin(q2),
    qvel / 6.0,
    gripper_center_xy / 0.8,
    ball_xy / 0.8,
    gripper_to_ball_xy / 0.8,
    ball_xy_in_gripper_frame,
]
```

reward 的核心是：

```python
reward = -8.0 * gripper_ball_distance
reward += 20.0 * progress
reward -= 1.5 * abs(ball_local_y)
reward -= 0.8 * max(abs(ball_local_x) - 0.08, 0.0)
reward -= 0.001 * torque_square_sum
```

如果小球位于两根红色手指之间，会得到额外奖励。成功条件是夹爪中心距离小球小于 `0.085`，并且小球在夹爪开口内：

```python
abs(ball_local_y) < 0.050
abs(ball_local_x) < 0.080
```

这个任务默认只从自己的 checkpoint 续训；如果没有自己的 checkpoint，就从随机初始化开始，避免不同 observation 维度的 reach checkpoint 把策略带歪。

---

## 8. PPO 代码位置

现在 PPO 不再散落在每个训练脚本里，主要看这几个文件：

```text
agents/ppo_agent.py       # 策略网络和值函数网络
training/configs.py       # 每个任务的超参数和 checkpoint 名称
training/ppo.py           # GAE、PPO loss、训练循环
training/checkpoints.py   # 加载模型
```

奖励函数在对应环境文件里，例如：

```text
envs/cart_pole_final_env.py
envs/cart_pole_swing_up_env.py
envs/cart_pole_balance_env.py
envs/two_link_arm_reach_env.py
envs/two_link_arm_gripper_ball_reach_env.py
```

如果想调任务行为，优先改环境里的 reward；如果想调训练稳定性，再改 `training/configs.py` 里的 PPO 超参数。

---

## 9. 常见问题

如果出现：

```text
ModuleNotFoundError: No module named 'envs'
```

请确认在项目根目录运行：

```bash
cd gym_envs
python scripts/train.py --task cart_pole_final
```

如果 MuJoCo viewer 没有弹出，可以先测试：

```bash
python -m mujoco.viewer
```

如果是在远程服务器、WSL 或无图形界面环境中运行，可能需要配置 OpenGL / EGL。

---

## 10. 后续扩展方向

可以继续按这个结构加更复杂任务：

- 二连杆机械臂 reach；
- 二连杆机械臂 push block；
- 加载 Panda / UR5e 等真实机械臂模型；
- 多阶段 curriculum learning；
- 更系统的评估和曲线日志。
