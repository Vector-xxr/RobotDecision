# Decisions 离线评测 20261013_123143
数据 /home/crp-5070ti-01/atom_episodes/robotdecision_replay/20261013_122101，模型 gpt-6-luna，目标 G=[393.0, -289.0, 284.0]

## v3（19 帧，其中选动作的 19 帧）
- 选择题本身：正序 47%、倒序 53%；两遍一致 53%；一致且对 26%
- **策略（合手由「填满圈」触发）**：出动作 19/19，对 58%
  - east（8）对 25%；出成 arrow_B×2、arrow_D×2、arrow_C×2、target_not_visible×2
  - close_hand（5）对 100%；出成 close_hand×5
  - lower_or_close（3）对 100%；出成 close_hand×3
  - lower（3）对 33%；出成 close_hand×1、arrow_B×1、lower×1
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - east（8）对 25%；答成 arrow_B×2、arrow_D×2、arrow_C×2、target_not_visible×2
  - close_hand（5）对 40%；答成 lower×3、close_hand×2
  - lower_or_close（3）对 100%；答成 close_hand×2、lower×1
  - lower（3）对 67%；答成 lower×2、arrow_B×1
- predicate sleeve_visible：按 0.5 判对 95%；真为是 19 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate fills_circle：按 0.5 判对 94%；真为是 5 帧概率中位 0.95，真为否 11 帧中位 0.00
- 延迟 p50 784 ms、p95 2085 ms；输入 token 中位 1681

## v4（19 帧，其中选动作的 19 帧）
- 选择题本身：正序 37%、倒序 32%；两遍一致 74%；一致且对 21%
- **策略（合手由「填满圈」触发）**：出动作 19/19，对 53%
  - east（8）对 0%；出成 arrow_D×3、arrow_B×2、arrow_C×2、arrow_A×1
  - close_hand（5）对 100%；出成 close_hand×5
  - lower_or_close（3）对 100%；出成 close_hand×3
  - lower（3）对 67%；出成 lower×1、close_hand×1、arrow_C×1
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - east（8）对 0%；答成 arrow_B×2、arrow_C×2、arrow_D×2、close_hand×1
  - close_hand（5）对 20%；答成 lower×4、close_hand×1
  - lower_or_close（3）对 100%；答成 lower×2、close_hand×1
  - lower（3）对 100%；答成 lower×2、arrow_C×1
- predicate sleeve_visible：按 0.5 判对 100%；真为是 19 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate fills_circle：按 0.5 判对 94%；真为是 5 帧概率中位 0.89，真为否 11 帧中位 0.00
- 延迟 p50 791 ms、p95 2243 ms；输入 token 中位 2047

共 38 个请求，输入 70832 token ≈ $0.007