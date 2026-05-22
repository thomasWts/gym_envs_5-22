# MuJoCo Reach RL

这是一个基于 **MuJoCo + Gymnasium + PyTorch** 的强化学习入门项目。

当前包含两个 reach 任务：

- `CartReachEnv`：小车只能沿 x 方向运动，控制水平力到达绿色目标点；
- `BlockReach3DEnv`：物块可以沿 x/y/z 三个方向运动，控制三维力到达绿色目标点。

这个项目的目的不是追求复杂任务，而是建立一个标准的 MuJoCo RL 项目结构：

- 使用 MuJoCo 构建物理仿真模型
- 使用 Gymnasium 封装标准 RL 环境
- 使用 PyTorch 从零实现一个简单 PPO 算法训练策略
- 使用 MuJoCo viewer 可视化训练后的策略

---

## 1. 项目结构

```text
mujoco/
├── agents/
│   ├── __init__.py
│   └── ppo_agent.py             # PyTorch Actor-Critic 网络
├── envs/
│   ├── cart_reach_env.py        # 1D 小车 reach 环境
│   └── block_reach_3d_env.py    # 3D 物块 reach 环境
├── models/
│   ├── cart_reach.xml           # 1D 小车 MuJoCo 模型
│   └── block_reach_3d.xml       # 3D 物块 MuJoCo 模型
├── scripts/
│   ├── train_cart_reach_ppo.py
│   ├── render_cart_reach_ppo.py
│   ├── train_block_reach_3d_ppo.py
│   └── render_block_reach_3d_ppo.py
├── checkpoints/
│   ├── ppo_cart_reach_torch.pt
│   └── ppo_block_reach_3d_torch.pt
├── logs/
├── README.md
└── .gitignore
```

其中最重要的是：

```text
envs/cart_reach_env.py
```

这个文件定义了标准 Gymnasium 环境：

```python
class CartReachEnv(gym.Env):
    def reset(self, seed=None, options=None):
        ...

    def step(self, action):
        ...

    def render(self):
        ...

    def close(self):
        ...
```

---

## 2. 环境依赖

建议使用 conda 创建单独环境。

```bash
conda create -n mujoco_rl python=3.11 -y
conda activate mujoco_rl
```

安装依赖：

```bash
pip install mujoco gymnasium torch numpy
```

如果你是在 Windows PowerShell 中运行，也是同样的命令：

```powershell
conda create -n mujoco_rl python=3.11 -y
conda activate mujoco_rl
pip install mujoco gymnasium torch numpy
```

检查 MuJoCo 是否安装成功：

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

---

## 3. 当前任务说明

当前环境叫：

```python
CartReachEnv
BlockReach3DEnv
```

任务目标：

```text
让控制对象移动到随机生成的绿色目标点附近，并尽量停住。
```

小车只能沿 x 方向运动。每个 episode 开始时：

- 小车初始位置随机
- 目标点位置随机
- 策略输出一个连续动作
- 动作被映射成施加在小车上的水平力

3D 物块任务中，动作是三维连续力：

```python
action = [ax, ay, az]
force = force_limit * action
```

其中 `action` 每一维都在 `[-1, 1]`。

---

## 4. Observation 设计

1D 小车 observation 是 4 维：

```python
obs = [
    x / 0.8,
    x_dot / 3.0,
    target_x / 0.8,
    error / 0.8,
]
```

含义如下：

| 变量 | 含义 |
|---|---|
| `x / 0.8` | 小车当前位置归一化 |
| `x_dot / 3.0` | 小车速度归一化 |
| `target_x / 0.8` | 目标点位置归一化 |
| `error / 0.8` | 目标点与小车当前位置的误差 |

其中：

```python
error = target_x - x
```

3D 物块 observation 是 12 维：

```python
obs = [
    pos / 0.8,
    vel / 3.0,
    target_pos / 0.8,
    error / 0.8,
]
```

其中 `pos`、`vel`、`target_pos`、`error` 都是 3 维向量。

---

## 5. Action 设计

当前 action 是 1 维连续动作：

```python
action ∈ [-1, 1]
```

环境中会将它映射成水平力：

```python
force = force_limit * action
```

其中：

```python
force_limit = 10.0
```

也就是说：

```text
action = -1  →  向左施加最大力
action =  0  →  不施加力
action =  1  →  向右施加最大力
```

---

## 6. Reward 设计

当前 reward 的主要目标是：

1. 小车越接近目标点越好；
2. 小车速度越小越好；
3. 控制力越小越好；
4. 到达目标并稳定下来时给额外奖励；
5. 小车接近边界时给惩罚并结束 episode。

核心形式：

```python
reward = 1.0
reward -= 8.0 * error ** 2
reward -= 0.10 * x_dot ** 2
reward -= 0.001 * force ** 2
```

成功条件：

```python
success = abs(error) < 0.03 and abs(x_dot) < 0.20
```

如果成功：

```python
reward += 10.0
terminated = True
```

如果小车出界：

```python
reward -= 20.0
terminated = True
```

---

## 7. 训练 PPO

运行：

```bash
python scripts/train_cart_reach_ppo.py
```

Windows PowerShell：

```powershell
python .\scripts\train_cart_reach_ppo.py
```

训练脚本会使用 PyTorch 实现 PPO，主要包含：

- `ActorCritic` 网络；
- Gaussian policy + `tanh` 动作限幅；
- GAE advantage 估计；
- PPO clipped policy loss；
- value loss 和 entropy bonus。

训练完成后会保存模型：

```text
ppo_cart_reach_torch.pt
```

训练 3D 物块任务：

```bash
python scripts/train_block_reach_3d_ppo.py
```

训练完成后会保存：

```text
ppo_block_reach_3d_torch.pt
```

---

## 8. 展示训练好的策略

训练完成后运行：

```bash
python scripts/render_cart_reach_ppo.py
```

Windows PowerShell：

```powershell
python .\scripts\render_cart_reach_ppo.py
```

展示 3D 物块任务：

```bash
python scripts/render_block_reach_3d_ppo.py
```

运行后会打开 MuJoCo viewer。

你应该能看到：

- 蓝色小车在导轨上运动；
- 绿色点表示目标位置；
- 小车会尝试移动到绿色目标点附近；
- 到达目标后会自动 reset 到下一轮。

---

## 9. 常见问题

### 9.1 找不到 `envs.cart_reach_env`

如果出现：

```text
ModuleNotFoundError: No module named 'envs'
```

请确认你是在项目根目录运行脚本：

```bash
cd gym_envs
python scripts/train_cart_reach_ppo.py
```

不要进入 `envs/` 文件夹里面运行。

---

### 9.2 MuJoCo viewer 没有弹出

先测试：

```bash
python -m mujoco.viewer
```

如果 viewer 可以打开，说明 MuJoCo 安装正常。

如果是在远程服务器、WSL 或无图形界面环境中运行，可能需要配置 OpenGL / EGL，这部分可以先跳过，在本地机器上运行可视化。

---

### 9.3 策略训练后效果不好

可以尝试增加训练步数：

```python
model.learn(total_timesteps=300_000)
```

也可以调整 reward，例如增大接近目标的惩罚项：

```python
reward -= 12.0 * error ** 2
```

或者放宽成功条件：

```python
success = abs(error) < 0.05 and abs(x_dot) < 0.30
```

---

## 10. 后续扩展方向

当前任务只是最简单的小车到达目标点。后面可以逐步升级：

### 10.1 Cart Reach

```text
小车移动到目标点
```

这是当前任务。

---

### 10.2 Cart-Pole Balance

```text
小车连接一根杆子，杆子初始接近倒立位置。
目标是控制小车，让杆子保持不倒。
```

需要加入：

```text
rod angle
rod angular velocity
```

到 observation 中。

---

### 10.3 Cart-Pole Swing-Up

```text
杆子初始自然下垂。
目标是控制小车左右运动，把杆子甩到上方并稳定住。
```

这个任务明显更难，需要重新设计 reward：

- 鼓励小球高度升高；
- 接近上方时给额外奖励；
- 小车出界时终止；
- 控制力不能太大。

---

### 10.4 Two-Link Arm Reach

```text
二连杆机械臂末端到达目标点。
```

可以使用：

```text
joint angle
joint velocity
end-effector position
target position
```

作为 observation。

---

### 10.5 Two-Link Arm Push

```text
二连杆机械臂推动一个小方块到目标区域。
```

这个任务开始涉及接触，需要关注：

- geom 之间的 contact；
- friction；
- mass；
- reward shaping；
- 目标物体位置。

---

### 10.6 Panda / UR5e Manipulation

之后可以加载 MuJoCo Menagerie 中的真实机械臂模型，例如：

- Franka Panda
- UR5e
- xArm7
- Unitree Go2

这一步适合在熟悉自定义 Gymnasium 环境之后再做。

---

## 11. 当前学习重点

这个项目最重要的不是让小车任务本身有多复杂，而是掌握 MuJoCo RL 的标准结构：

```text
MuJoCo XML
→ Gymnasium Env
→ observation_space / action_space
→ reset()
→ step()
→ reward
→ terminated / truncated
→ PPO train
→ render policy
```

之后无论做小车倒立摆、二连杆机械臂、推方块还是 Panda 机械臂，基本结构都是一样的。

---

## 12. 运行命令总结

训练：

```bash
python scripts/train_cart_reach_ppo.py
python scripts/train_block_reach_3d_ppo.py
```

展示：

```bash
python scripts/render_cart_reach_ppo.py
python scripts/render_block_reach_3d_ppo.py
```

检查 MuJoCo：

```bash
python -m mujoco.viewer
```

检查环境依赖：

```bash
python -c "import mujoco, gymnasium, torch, numpy; print('ok')"
```
