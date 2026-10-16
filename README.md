# RobotDecision

让 OpenAI 的决策接口（`POST /v1/decisions`，只有 `gpt-6-luna` 能用）看俯视 Gemini 335、腕部 Gemini 305、侧面 UGREEN 三路画面，每一步从预设动作里选一个，驱动 AR5 臂和因时灵巧手（拇指加食指当二指夹爪用）抓起白板上的蓝色套筒。

运行时不检测套筒、不算坐标：模型只看画了箭头和绿环的图做选择题；步长、安全检查、合手后判断夹没夹住都是固定规则。

> [!IMPORTANT]
> **用户认定的方案是 `auto_loop.py`**：抓取朝向抄 VR 示教，没有旋转选项。`auto_loop2.py`、`auto_loop3.py`、`closed_loop.py` 是对照或早期版本，结论见下文，不要拿来替代它。

下面的命令除非另写了 `cd`，都在本目录下、用 conda 环境 `laya` 跑，写法和 [工作区说明](../../../工作区说明.md) 一致。

> [!NOTE]
> **放在 GitHub 上要知道的**：这个仓库原本是一个更大工作区里的项目（`<工作区>/算法/Jev/RobotDecision/`），单独克隆跑不起来。
> - 依赖仓库外的 `atom_robot`（AR5 驱动和相机驱动，在原工作区 `数据_硬件/SO101/`，没有公开），以及 [VisionPro_teleop_DexterousHand](https://github.com/Vector-xxr/VisionPro_teleop_DexterousHand)（要克隆到 `<工作区>/数据_硬件/灵巧手遥操/`，本项目导入它 `VR设备/` 下的 `ar5_poe_ik.py`、`inspire_hand6.py`，回位用它的 `home_arm.py`）。找它们的方式是从文件位置往上找工作区根，或者设环境变量 `YUHANG_WS`。
> - 运行记录和标定照片在 `~/atom_episodes/`，不在仓库里；调模型要自己的 OpenAI 密钥（`~/.config/openai/key`）。
> - 文中指向 `../../../工作区说明.md`、`../RobotJev` 等的链接是原工作区里的其他项目，在这里打不开（RobotJev 也已公开：[Vector-xxr/RobotJev](https://github.com/Vector-xxr/RobotJev)）。

## 认定方案：auto_loop.py

1. **起步**（固定动作，不问模型）：从起始位竖直抬到 z340，再把位置和朝向一起插值到 `--start`（默认法兰 (380, −300, 330) mm、示教抓取朝向）。
2. **每一步先问位置题**（腕部图 + 绿环）：套筒看不到 / 在环外 / 在环内，决定走哪一段：

   | 段 | 给模型看的图 | 选项 | 步长 |
   |---|---|---|---|
   | 粗定位（腕部看不到） | 俯视 + 四个字母箭头 | 选箭头；或「俯视也看不到」→ 停 | 20 mm |
   | 进环前（在环外） | 俯视 + 腕部，两张图画同字母箭头 | 选箭头 | 15 mm |
   | 环内 | 腕部（环 + 中心点 + 箭头）+ 侧面 | 箭头 / 下降 / 合手 | 平移 5 mm，下降 8 mm |

3. 每道选择题正序、倒序各问一次，一致才执行；不一致不动、重拍重问；同一位姿 3 轮都不一致就取多数票。字母和方向每步随机配。
4. **合手**后看手指读数：被挡住 = 夹住。空夹 → 张手、降 5 mm 再试，最多 4 次。模型要降但已到法兰下限 → 在下限试合。夹住 → 每步抬 10 mm、共 8 步，途中手指不再被挡就停。
5. **收尾**：夹着就原路放下、张手；抬到起步高度 → 回起步点 → 倒着走起步路径回起始位。

安全检查只用物理判据，不画工作空间框：逆解可达、回代误差 <1 mm、软限位（留 0.05 rad）、单步关节 ≤10°、法兰不低于下限（默认 274 mm = 示教抓取高度 284 − 10）。连续 3 步被拒就停并报原因；最多 150 步。

真机记录（`~/atom_episodes/robotdecision_auto/` 下；秒数取 `log.json` 的 `seconds`，含放回和回位）：

| 运行目录 | 说明 | 步数 | 秒 | 合手 |
|---|---|---|---|---|
| `20261013_141540` | 第一次真机 | 107 | 255 | 1 次夹住 |
| `20261014_093856` | 按用户要求从标定起复现：重拍回放和网格，三项标定和 10-13 一致，代码没改 | 62 | 188 | 1 次夹住 |
| `20261014_094534` | 夹住后抬到法兰 z327 时下一步关节要跳 50.8°，连续被拒 3 次停下，放回后回位 | 57 | 176 | 1 次夹住 |
| `20261014_095442` | 原版再跑 | 81 | 237 | 1 次夹住 |
| `20261015_112134` | 旧驱动（vel_ar5）最后一次 | 87 | 234 | 1 次夹住 |
| `20261015_150901` | **换成 atom_robot 驱动后的真机验收** | 102 | 262 | 1 次夹住（食指 562 / 拇指 731） |

验收那次调了 185 次接口，输入 token 共约 17 万，单次延迟中位 0.57 s；102 步里 35 步是重问。

### 其他版本

| 脚本 | 是什么 | 结论 |
|---|---|---|
| `auto_loop2.py` | 去示教：起始朝向；进环后模型在「调位置 / 调姿态」里选，调姿态时选六种旋转；靠手指受力判断碰板 | 不用示教也能抓，但要合手 2~3 次（`robotdecision_auto2/20261013_164636` 2 次，`20261014_085807` 3 次） |
| `auto_loop3.py` | 能力测试：不分阶段，每步一道 20 选 1，6 张图（含两路深度） | 失败：120 步没下降、没合手、没转过；正倒序一致只有 19/120。选项一多一致性就崩 |
| `closed_loop.py` | 人在回路的早期版：每步按回车才执行 | 10-13 第三次真机夹住（`robotdecision_loop/20261013_135450`，40 步） |

## 跑之前

**依赖的别处代码**，按文件位置往上找工作区根自动找到（单独拷走时设环境变量 `YUHANG_WS`）：
- `数据_硬件/SO101/atom_robot`：AR5 驱动、相机驱动；
- `数据_硬件/灵巧手遥操/VR设备`：直接 import 它的 `ar5_poe_ik.py`（自己的逆解）和 `inspire_hand6.py`（手的 Modbus 驱动）；
- `~/atom_episodes/robotdecision_grid/` 下至少一个不带 `_mock` 的目录，否则连 `--dry-run` 都起不来。见下面「换机器时要改的两处代码」第 1 条。

**1. 臂在起始位**（和 atom_robot 配置里的 `home_joints` 每个关节差 ≤3°，否则报「臂不在起始位」退出）：

```bash
cd ~/yuhang_workspace/数据_硬件/灵巧手遥操/VR设备
PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python home_arm.py
```

`home_arm.py` 走关节空间 MoveAbsJ，不管路上有什么。臂停在白板上方低处（手可能挨着套筒）时，先用 `rescue_home.py`：竖直抬起 → 保持朝向平移回 (450, 0) 上方 → 转回起始朝向 → 回起始位。

```bash
cd ~/yuhang_workspace/算法/Jev/RobotDecision
PYTHONPATH= ~/miniforge3/envs/laya/bin/python rescue_home.py        # 只算路径、不动臂；但会连真臂和真手读当前位姿
PYTHONPATH= ~/miniforge3/envs/laya/bin/python rescue_home.py --go   # 真动
```

`rescue_home.py` 是给 `auto_loop2.py`（手保持起始朝向）写的，从 `auto_loop.py` 的示教朝向回来没验证过，先不加 `--go` 看它算出的路径。

**2. 套筒**立在白板右半边（脚本开跑时的提示原话），白板上别放别的东西：运行时不检测障碍。成功的 5 次合手时法兰在 x 371~411、y −203~−248 mm；偏左会出问题，见「已知问题」。

**3. 三路相机**，路径写在 `replay_capture.py` 的 `CAM_TOP` / `CAM_WRIST` / `CAM_SIDE`，按 USB 口认（`/dev/v4l/by-path/`）：
- 俯视 335（`…usb-0:5:1.4-video-index0`）、腕部 305（`…usb-0:6:1.4-video-index0`）加 `v4l2:` 前缀，明确走 V4L2。atom_robot 默认把 Orbbec 改走 OBSENSOR，而腕部 305 走 OBSENSOR 一帧不出；像素标定也是按 V4L2 的 640×480 画面做的；
- 侧面 UGREEN（`pci-0000:10:00.4-usb-0:2:1.0-video-index0`）：序列号都是 SN0001，只能按口认；
- 换了 USB 口就改这三个常量。

**4. 手和臂**：手走网口 Modbus TCP `tcp://192.168.11.210:6000`（`replay_capture.py` 的 `HAND_PORT`），臂 `192.168.2.160`（atom_robot 的 `config.py`）。连接顺序是相机 → 手 → 臂，前两样连不上，臂不会上电。

**5. 密钥**放在 `~/.config/openai/key`（权限 600），要能连上 OpenAI。

**6. 开跑**：

```bash
cd ~/yuhang_workspace/算法/Jev/RobotDecision
PYTHONPATH= ~/miniforge3/envs/laya/bin/python auto_loop.py --dry-run   # 离线演算起步路径，不连硬件、不联网
PYTHONPATH= ~/miniforge3/envs/laya/bin/python auto_loop.py             # 真跑
```

- 要在有图形界面的终端里跑：会开 OpenCV 窗口「RobotDecision closed loop」，三格是俯视 / 腕部 / 侧面，下面一行是当前动作。开跑前看一眼侧面那格是不是全黑。
- 提示后**输入 `yes` 回车**才开始。别用 `echo yes |` 喂：标准输入不是终端时，q 急停不起作用。
- 运行中**按 `q` 急停**：走完当前一小步就停，臂停在原地，手保持当前状态（夹着的不会松）。别按 Ctrl+C；结束时断开臂可能要约 1 分钟，这时也别按。
- 参数：`--start X Y Z`（起步点，mm）、`--floor`（法兰下限，mm）、`--speed`（MoveAbsJ 速度参数，默认 100）、`--out`（默认 `~/atom_episodes/robotdecision_auto`）。

## 换机器时要改的两处代码

10-15 用户定：这两处在本机不改，换机器、重写相关代码时一并处理。

**1. `near_tests.py` 一被导入就去找网格照片目录，找不到就报错**

- 现象：`near_tests.py` 第 32 行在文件顶层执行
  `GRID = sorted(...(Path.home() / "atom_episodes/robotdecision_grid").glob("2026*") ...)[-1]`。
  新机器上没有这个目录（或者只有 `_mock` 目录）时列表为空，`[-1]` 抛 `IndexError`。
- 影响：`closed_loop.py` 导入了 `near_tests`，而 `auto_loop*`、`closed_loop`、`calib_start`、`axis_look`、`touch_test`、`rescue_home`、`run_with_model` 都导入了 `closed_loop`，所以这些脚本全都起不来，连 `--dry-run` 也不行。
- 抓取本身用不到这批照片：`closed_loop.py` 从 `near_tests` 只借了提问文字（`A1_TEXT`、`A1_Q`、`A2_RINGS_TEXT`、`A2_RINGS_Q`）、内圈半径 `INNER_R` 和 `img_url()`。`GRID` 只在 `near_tests.py` 自己的实验函数里用（第 61、153、220 行）。
- 改法：把找目录挪进真正用到它的函数里，比如写成函数 `grid_dir()`，第一次调用时再找；找不到时报一句说清楚缺哪个目录。改完后，没有这个目录时 `auto_loop.py --dry-run` 也应该能跑。抓取行为不变。
- 不改代码的临时办法：把本机的 `~/atom_episodes/robotdecision_grid/20261014_092937` 一起拷过去（取的是排序最后、不带 `_mock` 的那个）。

**2. `auto_loop.py` 调接口出错时直接退出，臂停在半路**

- 现象：每一步都要调 OpenAI 的决策接口。`auto_loop.py` 只对限流重试。碰上服务端出错（503）、连不上、超时，异常会被最外层的 `except Exception`（第 424 行）接住，结果记成「出错」，然后**跳过收尾**直接断开：臂停在原地（可能低低地挨着套筒，也可能正夹着套筒悬在半空），手保持当前状态，要人用 `rescue_home.py` 或手动处理。
- 真碰到过：10-13 `auto_loop2.py` 换位置重跑到第 55 步，决策接口回了 503（「Decision inference is unavailable」），臂停在套筒旁的低处。
- 改法：照搬 `auto_loop2.py` 的做法（`ApiDown` 和 `make_decider` 里的 `robust`，约第 246~265 行；收尾在第 661 行起）：
  - 把 `CL.make_decider` 包一层，捕获 `InternalServerError`、`APIConnectionError`、`APITimeoutError`，分别等 3、6、12、24、30、30 秒重试，约 2 分钟；
  - 还不行就抛 `ApiDown`，主循环接住后按正常流程收尾：夹着就放下、张手、回起始位。
- 只在出错时才走到这段，正常抓取的每一步不受影响。改完用模拟运行验证：在假模型里故意抛这几种异常，看能不能重试成功，以及重试用完后能不能正常收尾。

## 标定链

`auto_loop.py` 用到的标定量都是代码里的常量，没有单独的标定文件：

| 量 | 现值 | 存在哪 | 从哪来 | 什么时候重标 |
|---|---|---|---|---|
| 抓取朝向、法兰下限 | 下限 274 mm | `data/traj_20261013_101346.csv` 的 t=48.1 s（`replay_capture.py` 的 `DEFAULT_TRAJ`、`DEFAULT_T_GRASP`） | 10-13 VR 示教 | 手在法兰上的装法、手指、白板高度变了 |
| 腕部绿环 | 中心 (304, 228)，半径 110 px | `eval_decisions.py` 的 `WRIST_CIRCLE` | `replay_capture.py` 回放（`robotdecision_replay/20261013_103555`）抓取位那帧里套筒的像素 | 腕部 305 相对手动过；换了大小不同的物体 |
| 腕部箭头方向 | 2×2 矩阵，px/mm | `eval_decisions.py` 的 `WRIST_J`（算出 `WRIST_VEC`） | `grid_capture.py` 网格（`robotdecision_grid/20261013_132038`）+50 mm 那层拟合 | 腕部 305 相对手动过；抓取朝向变了 |
| 俯视箭头起点 | 6 对「法兰 xy ↔ 手背白盒子像素」 | `eval_decisions.py` 的 `CAL_PTS`（拟合成 `TOP_MAP`） | 回放照片里目测 | 俯视 335 动过 |
| 起始位 | `home_joints` | atom_robot 的 `config.py` | — | 改了它就重跑 `--dry-run` 看起步路径 |

- 仓库里只有拍照的脚本（`replay_capture.py`、`grid_capture.py`）。从照片量像素、拟合 `WRIST_J` 那一步当时用的是临时脚本和目测，没进仓库；重标要自己算，再改上表的常量。
- 套筒换个位置不用重标：运行时不检测套筒，标定量的是相机和手。但示教朝向下能抓的范围有限，见「已知问题」。
- 10-14 复现前重拍过回放（`robotdecision_replay/20261014_091440`）和网格（`robotdecision_grid/20261014_092937`），按当时的核对记录三项都和 10-13 一致，常量没改（本次没重算）。
- `calib_start.py` 是给去示教版准备的「起始朝向」标定。**真机从来没完整产出过 `calib_start.json`**：10-13 15:00 那次 9 次合手全空，只在输出目录写了 `calib_start_无环.json`；之后两次没跑完。仓库里也没有这个文件。`auto_loop.py` 不读它；`auto_loop2.py` 也不读，用的是写在自己文件里的 `J0`、`PINCH_PX`、`PIVOT0`。

## 脚本地图

「动臂」指会连真臂并让它动；「联网」指会调 OpenAI 接口（花钱）。有 `--dry-run`（离线演算）的脚本加上它都不连硬件、不联网；有 `--mock`（模拟运行）的用假臂、假手、假相机、假模型。

**主流程**

| 脚本 | 做什么 | 动臂 | 联网 |
|---|---|---|---|
| `auto_loop.py` | 认定方案，见上 | 是 | 是 |
| `auto_loop2.py` | 去示教版，起始朝向 + 旋转选项 + 碰板检测 | 是 | 是 |
| `auto_loop3.py` | 不分阶段的能力测试 | 是 | 是 |
| `closed_loop.py` | 人在回路，每步回车确认 | 是 | 是 |

**标定和数据采集**（都会动臂，都不联网）

| 脚本 | 做什么 |
|---|---|
| `replay_capture.py` | 回放示教轨迹，逐点停稳拍三路图：空板走到抓取位 → 人放套筒 → 倒退逐点拍 → 走回抓取位 → 合手 → 抬起逐点拍 → 放回 → 回起点 |
| `grid_capture.py` | 在示教抓取位上方 +50 / +80 mm 两层走 5×5 网格（±20 mm）逐点拍，用来标腕部箭头 |
| `calib_start.py` | 起始朝向下的标定（量下限、网格、转腕、3×3 合手），为去示教版准备，没完整跑成过 |
| `axis_look.py` | 抬到白板中间 z400，绕末端 x/y/z 轴各正反转一下拍照给人看 |
| `touch_test.py` | 空中合手拍捏合点；再慢慢降到指尖碰板，看手指受力能不能测出碰板 |

**离线评测**（都不动臂；不加 `--dry` 就联网）

| 脚本 | 做什么 |
|---|---|
| `eval_decisions.py` | 用回放 / 网格照片测各版问法，标签由正解算；结果写到 `eval_runs/<时间>/`（`report.md`、`results.csv`、`responses/` 原样存的每个回答，`--resume` 时复用） |
| `near_tests.py` | 只给腕部图的两个小实验：A1 近处方向、A2 防提前合手 |

`eval_runs/` 每个目录是一轮评测：

| 目录 | 哪一轮 | 关键结论 |
|---|---|---|
| `20261013_104928` | v1 冒烟：两张图直接 10 选 1（原图 / 镜像 / 去掉说明），30 个请求 | 原图正序只对 21%，基本在猜；「看得见套筒」判对 100% |
| `20261013_113309` | v2 方向改名、按手的开合给选项；v2arrows 俯视图上画字母箭头 | v2 正序 9%；加箭头后朝右的帧 83%，镜像后朝左的帧也 83% |
| `20261013_114440` | v3 冒烟：箭头 + 腕部绿圈 + 「填满圈」「夹着」两道是非题 | 选动作正序 68%、镜像 88%；两道是非题按 0.5 判对 100% |
| `20261013_114810` | v3 全量：回放 199 帧 + 镜像 93 帧，292 个请求 | 合手改由「填满圈」触发后，原图 85%、镜像 75% |
| `20261013_123143` | 第二次回放上比 v3 和 v4（加侧面图） | v3 58%、v4 53%，侧面图没帮上忙 |
| `20261013_132531` | 网格（套筒正上方 5~8 cm）上比 v3 和 v5（腕部也画箭头） | v3 6%、v5 35%；网格帧全都不该合手，「填满圈」只判对 58% / 76% → 会提前合手 |
| `20261013_132603` | v5 在第二次回放上 | 74% |
| `near_20261013_133013` | `near_tests.py` 拆题 | A1「哪根箭头从圈指向套筒」正序 77%、镜像 72%；A2 加虚线内圈、按 ≥0.2 合手：该合 8/10、误报 0/72（单圈按 0.5：10/10 但误报 30/72）→ 拆成小题比一道综合题好 |
| `_preview/` | `--dry` 出的预览图 | — |

**工具**

| 脚本 | 做什么 | 动臂 | 联网 |
|---|---|---|---|
| `make_video.py <运行目录>` | 把每步存的 `*_panel.jpg` 按 `log.json` 的时间拼成 `interface.mp4`，内容就是窗口画面，时长等于真实时长 | 否 | 否 |
| `play_video.py <mp4>` | 本机没装播放器，用 OpenCV 窗口放：空格暂停，← / → 退进 5 秒，q 或 Esc 退出 | 否 | 否 |
| `record_window.py --out <mp4>` | 按窗口位置截屏录像；被别的窗口挡住会录进去，所以改用 `make_video.py` | 否 | 否 |
| `depth_v4l2.py [输出目录]` | 用纯 V4L2 读两台 Orbbec 的深度（不对齐彩色），开 10 秒报帧率、存彩色化深度图；`auto_loop3.py` 用它 | 否（但会开相机） | 否 |
| `run_with_model.py --model <名>` | 原样跑 `auto_loop.py`，只把问模型换成通用接口；写好了但没测过 | 是 | 是 |
| `rescue_home.py` | 从白板上方安全回起始位；`--go` 才动 | 加 `--go` 才动 | 否 |

`play_video.py`、`depth_v4l2.py` 没有参数解析，`--help` 会被当成路径（`depth_v4l2.py` 会去开相机），别这么用。

## 输出在哪

运行记录都在 `~/atom_episodes/` 下，不在仓库里（迁移时必须带上，见工作区说明第 6 节）。模拟运行的目录名带 `_mock`。

| 目录 | 谁写的 | 里面有什么 |
|---|---|---|
| `robotdecision_auto/` | `auto_loop.py` | 每步 `NNN_top/wrist/side.jpg`（原图）、`NNN_panel.jpg`（窗口画面）、`api_NNN_<题>.json`（每次接口回答）；`log.json`（结果、步数、秒数、合手次数、逐步记录）；`make_video.py` 生成的 `interface.mp4` |
| `robotdecision_auto2/`、`robotdecision_auto3/` | `auto_loop2.py`、`auto_loop3.py` | 同上；auto3 还有深度 `.npy`、旋转参考图 |
| `robotdecision_loop/` | `closed_loop.py` | 同上（`log.json` 是逐步列表） |
| `robotdecision_replay/` | `replay_capture.py` | `top/` `wrist/` `side/` 三路图，`meta.csv`（每张图的关节、法兰位姿、手指读数），`plan.json` |
| `robotdecision_grid/` | `grid_capture.py` | 同 replay |
| `robotdecision_calib/` | `calib_start.py` | `img/`、`meta.json` |
| `robotdecision_axis/`、`robotdecision_touch/` | `axis_look.py`、`touch_test.py` | 照片 + `log.json` |
| `robotdecision_rotval/` | 不是仓库脚本写的 | 10-13 旋转选项的线下检验，脚本 `rot_val.py` 就在那个目录里 |

仓库里的 `data/` 放运行要用的小数据，免得依赖别的项目的日志目录：
- `traj_20261013_101346.csv`：10-13 VR 示教轨迹（`avp_servoj_teleop.py --traj-hz` 录的，约 10 Hz、704 行；列为时间、离合、7 关节指令和实测、手 6 指指令和实测）。原件在 `VR设备/logs/`。`replay_capture`、`grid_capture`、`closed_loop` 沿它走；`auto_loop` 只取 t=48.1 s 的抓取朝向和高度。
- `robotdecision_axis_20261013_152719/`：`axis_look.py` 10-13 真机拍的水平姿态和六种转法 × 三路相机。`auto_loop2`、`auto_loop3` 的旋转参考图。

`logs/`、`_rokae_log_/` 是珞石 SDK 自己写的运行日志，不进 git。

## 离线检查

下面每条都实际跑过（2026-10-15），不连硬件、不联网。在本目录下执行：

```bash
cd ~/yuhang_workspace/算法/Jev/RobotDecision
export PYTHONPATH=; PY=~/miniforge3/envs/laya/bin/python

$PY -m py_compile *.py                                # 没有输出 = 17 个脚本语法都对
$PY auto_loop.py --dry-run
$PY auto_loop.py --mock --out "$(mktemp -d)"           # 约 20 秒
$PY closed_loop.py --mock --out "$(mktemp -d)"
$PY replay_capture.py --mock --out "$(mktemp -d)"      # 约 1 分钟
$PY make_video.py <上面 auto_loop 打印的记录目录> --out "$(mktemp -d)/mock.mp4"
$PY eval_decisions.py --dry                            # 要有 ~/atom_episodes/robotdecision_replay/20261013_103555
```

加 `--out` 是为了不往 `~/atom_episodes` 里塞模拟运行的目录。正常结果：

| 命令 | 正常输出 |
|---|---|
| `auto_loop.py --dry-run` | `起步：竖直抬到 z340，再转 90°、移到 (380, -300, 330)，共 48 步；单步关节最多 3.5°`，`法兰下限 274mm；…`，`--dry-run：没连硬件、没调 API。` |
| `auto_loop.py --mock` | `AR5Env running in MOCK mode -- no hardware is being driven`；第一次合手「空夹」、第二次「夹住了」，抬 8 步后 `结果：完成（N 步，…，合手 2 次）`、`已回到起始位。`（这次 37 步） |
| `closed_loop.py --mock` | 最后 `已回到起始位。` 和 `共 N 步：执行 …` |
| `replay_capture.py --mock` | `已拍 200 组`（空板 92、抓取位 1、倒退 92、合手 1、抬起 13、放回 1） |
| `make_video.py` | `37 步、18 秒（1 倍速）→ …    运行结果：完成` |
| `eval_decisions.py --dry` | `200 帧，参与评估 199 帧；目标 G = [393.0, -289.0, 284.0] mm`、`请求数 279`，再打印一个请求，不发出去 |

其他脚本的 `--dry-run` 也都跑过、正常结束：`auto_loop2.py`、`auto_loop3.py`、`grid_capture.py`、`calib_start.py`、`axis_look.py`、`touch_test.py`。

> [!WARNING]
> `near_tests.py --dry` 和 `eval_decisions.py --dry --variants v2…/v3…` 会把预览图写进 `eval_runs/_preview/`，那里的图在 git 里，会被覆盖。`eval_decisions.py`、`near_tests.py` 不加 `--dry`、`run_with_model.py --selftest` 都会调接口花钱。

## 已知问题和限制

- **决策接口只接受 `gpt-6-luna`**：astra / sol / 6.1-sol 都回 404。`run_with_model.py` 改走通用接口，写好了但没测。
- **示教朝向下第 7 关节余量小**：套筒偏左（合手时法兰 y > −200 mm 左右）时，往西（+y）或往上走会换逆解支、被单步 ≤10° 拒掉。离线从起步点算：法兰 (410, −240, 274) 余量 6.3°，(410, −198, 274) 4.4°，(410, −170, 274) 3.1°；往西在 z330 走到 y≈−180、在 z274 走到 y≈−140 就过不去。`20261014_094534` 就是在 (410, −198) 夹住后抬到 z327，下一步要跳 50.8°，连续被拒停下。认定方案没有「绕竖直轴让开」的补救（`auto_loop2.py` 有）。`replay_capture.py`、`home_arm.py` 打印的关节编号从 0 数，这里的第 7 关节在那里是 `j6`。
- **抬起途中靠手指读数判「还夹着」不可靠**：手指被挡停以后，即使物体滑掉，读数也不回落。10-14 回放（`robotdecision_replay/20261014_091440`）合手读数 541/730，抬到 z427 时套筒已经留在板上，读数还是 539/728。
- **侧面相机判断不了指尖高度**：10-13 换过三种问法都不随高度变。所以环内的「合手」实际上几乎都是「降到法兰下限 → 试合 → 看手指读数」，上面 5 次成功合手时法兰都在 z274.4~274.8。
- **侧面 UGREEN 时好时坏**：10-15 下午出过全黑画面，是设备自身问题。黑了就拔插一次。
- **不要同时启动 RobotJev**：它会让腕部 305 走 OBSENSOR，弄得连 V4L2 也不出帧，只能物理重插。
- **俯视箭头起点仍和示教有关**：`TOP_MAP` 来自回放示教时拍的照片。它属于相机标定，换成任意臂位姿拍的照片也能重标。
- **接口出错没有重试**：接口回 503 或断网时直接退出，臂停在半路。见「换机器时要改的两处代码」第 2 条。
- **重问很多**：正倒序答案常不一致，验收那次 102 步里 35 步是重问。

## 相关文档

- [工作区说明](../../../工作区说明.md)：工作区结构、conda 环境、约定、迁移。
- [RobotJev](../RobotJev/README.md)：本项目的前身（Laya / PlayJev + 现场测量的抓取流程）。
- [atom_robot](../../../数据_硬件/SO101/atom_robot)：AR5 驱动和相机驱动，全工作区只有这一份。
- [VR 遥操](../../../数据_硬件/灵巧手遥操/VR设备/README.md)：示教轨迹（`avp_servoj_teleop.py --traj-hz`）和回位脚本 `home_arm.py` 的来源；本项目直接 import 它的 `ar5_poe_ik.py`、`inspire_hand6.py`。
