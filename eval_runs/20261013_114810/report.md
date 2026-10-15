# Decisions 离线评测 20261013_114810
数据 /home/crp-5070ti-01/atom_episodes/robotdecision_replay/20261013_103555，模型 gpt-6-luna，目标 G=[393.0, -289.0, 284.0]

## v3（199 帧，其中选动作的 185 帧）
- 选择题本身：正序 84%、倒序 80%；两遍一致 79%；一致且对 74%
- **策略（合手由「填满圈」触发）**：出动作 185/185，对 85%
  - target_not_visible（92）对 91%；出成 target_not_visible×84、arrow_C×4、arrow_D×1、arrow_B×1
  - east（82）对 78%；出成 arrow_B×26、arrow_C×21、arrow_A×18、arrow_D×14
  - close_hand（5）对 100%；出成 close_hand×5
  - lower_or_close（3）对 100%；出成 close_hand×3
  - lower（3）对 33%；出成 lower×1、close_hand×1、arrow_B×1
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - target_not_visible（92）对 91%；答成 target_not_visible×84、arrow_C×4、arrow_D×1、arrow_B×1
  - east（82）对 78%；答成 arrow_B×26、arrow_C×21、arrow_A×18、arrow_D×14
  - close_hand（5）对 40%；答成 lower×3、close_hand×2
  - lower_or_close（3）对 100%；答成 close_hand×2、lower×1
  - lower（3）对 67%；答成 lower×2、arrow_B×1
- predicate sleeve_visible：按 0.5 判对 96%；真为是 107 帧概率中位 1.00，真为否 92 帧中位 0.00
- predicate fills_circle：按 0.5 判对 99%；真为是 5 帧概率中位 0.91，真为否 177 帧中位 0.00
- predicate holding：按 0.5 判对 100%；真为是 14 帧概率中位 0.92，真为否 0 帧中位 nan
- 延迟 p50 594 ms、p95 1571 ms；输入 token 中位 1681

## v3_mirror（93 帧，其中选动作的 93 帧）
- 选择题本身：正序 74%、倒序 77%；两遍一致 73%；一致且对 65%
- **策略（合手由「填满圈」触发）**：出动作 93/93，对 75%
  - west（82）对 74%；出成 arrow_C×29、arrow_A×18、arrow_B×17、arrow_D×14
  - close_hand（5）对 100%；出成 close_hand×5
  - lower_or_close（3）对 100%；出成 close_hand×3
  - lower（3）对 33%；出成 close_hand×1、arrow_A×1、lower×1
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - west（82）对 74%；答成 arrow_C×29、arrow_A×18、arrow_B×17、arrow_D×14
  - close_hand（5）对 80%；答成 close_hand×4、lower×1
  - lower_or_close（3）对 100%；答成 close_hand×3
  - lower（3）对 33%；答成 close_hand×1、arrow_A×1、lower×1
- predicate sleeve_visible：按 0.5 判对 95%；真为是 93 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate fills_circle：按 0.5 判对 99%；真为是 5 帧概率中位 0.94，真为否 85 帧中位 0.00
- 延迟 p50 574 ms、p95 1538 ms；输入 token 中位 1681

共 292 个请求，输入 483922 token ≈ $0.048