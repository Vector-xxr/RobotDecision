# Decisions 离线评测 20261013_104928
数据 /home/crp-5070ti-01/atom_episodes/robotdecision_replay/20261013_103555，模型 gpt-6-luna，目标 G=[393.0, -289.0, 284.0]

## base（14 帧）
- 动作正确率：正序 21%、倒序 29%；两遍一致 29%；一致且对 21%
- 按正确动作分：
  - done（3）对 0%；答成 close_hand×2、open_hand×1
  - close_hand（3）对 0%；答成 move_left×3
  - lower（2）对 0%；答成 close_hand×1、move_left×1
  - raise（2）对 0%；答成 close_hand×1、move_up×1
  - target_not_visible（2）对 50%；答成 target_not_visible×1、move_up×1
  - move_right（2）对 100%；答成 move_right×2
- confidence 中位 0.23；答错时中位 0.17（11 次答错）
- predicate sleeve_visible：按 0.5 判对 100%；真为是 12 帧概率中位 1.00，真为否 2 帧中位 0.00
- predicate between_fingers：按 0.5 判对 50%；真为是 3 帧概率中位 0.05，真为否 11 帧中位 0.00
- predicate lifted：按 0.5 判对 79%；真为是 3 帧概率中位 0.01，真为否 11 帧中位 0.00
- 延迟 p50 564 ms、p95 934 ms、最大 1046 ms；输入 token 中位 1864

## mirror（8 帧）
- 动作正确率：正序 25%、倒序 25%；两遍一致 75%；一致且对 25%
- 按正确动作分：
  - close_hand（2）对 100%；答成 close_hand×2
  - raise（2）对 0%；答成 move_up×2
  - lower（2）对 0%；答成 move_right×1、close_hand×1
  - move_left（2）对 0%；答成 move_right×1、move_up×1
- confidence 中位 0.27；答错时中位 0.24（6 次答错）
- predicate sleeve_visible：按 0.5 判对 100%；真为是 8 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate between_fingers：按 0.5 判对 100%；真为是 2 帧概率中位 0.60，真为否 6 帧中位 0.00
- predicate lifted：按 0.5 判对 100%；真为是 0 帧概率中位 nan，真为否 8 帧中位 0.00
- 延迟 p50 880 ms、p95 1300 ms、最大 1333 ms；输入 token 中位 1864

## nocontext（8 帧）
- 动作正确率：正序 25%、倒序 12%；两遍一致 12%；一致且对 12%
- 按正确动作分：
  - close_hand（2）对 0%；答成 lower×2
  - lower（2）对 100%；答成 lower×2
  - move_right（2）对 0%；答成 move_up×2
  - raise（2）对 0%；答成 move_up×2
- confidence 中位 0.15；答错时中位 0.17（6 次答错）
- predicate sleeve_visible：按 0.5 判对 75%；真为是 8 帧概率中位 0.98，真为否 0 帧中位 nan
- predicate between_fingers：按 0.5 判对 75%；真为是 2 帧概率中位 0.70，真为否 6 帧中位 0.06
- predicate lifted：按 0.5 判对 100%；真为是 0 帧概率中位 nan，真为否 8 帧中位 0.03
- 延迟 p50 727 ms、p95 1156 ms、最大 1314 ms；输入 token 中位 1703

共 30 个请求，输入 54632 token ≈ $0.005