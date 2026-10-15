# Decisions 离线评测 20261013_132603
数据 /home/crp-5070ti-01/atom_episodes/robotdecision_replay/20261013_122101，模型 gpt-6-luna，目标 G=[393.0, -289.0, 284.0]

## v5（19 帧，其中选动作的 19 帧）
- 选择题本身：正序 68%、倒序 79%；两遍一致 58%；一致且对 53%
- **策略（合手由「填满圈」触发）**：出动作 19/19，对 74%
  - east（8）对 50%；出成 arrow_A×5、arrow_B×2、target_not_visible×1
  - close_hand（5）对 100%；出成 close_hand×5
  - lower_or_close（3）对 100%；出成 close_hand×3
  - lower（3）对 67%；出成 close_hand×1、arrow_A×1、arrow_C×1
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - east（8）对 50%；答成 arrow_A×5、arrow_B×2、target_not_visible×1
  - close_hand（5）对 80%；答成 close_hand×4、lower×1
  - lower_or_close（3）对 100%；答成 close_hand×3
  - lower（3）对 67%；答成 close_hand×1、arrow_A×1、arrow_C×1
- predicate sleeve_visible：按 0.5 判对 95%；真为是 19 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate fills_circle：按 0.5 判对 94%；真为是 5 帧概率中位 0.84，真为否 11 帧中位 0.00
- 延迟 p50 804 ms、p95 1410 ms；输入 token 中位 1761

共 19 个请求，输入 33459 token ≈ $0.003