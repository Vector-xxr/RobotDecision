# Decisions 离线评测 20261013_132531
数据 /home/crp-5070ti-01/atom_episodes/robotdecision_grid/20261013_132038，模型 gpt-6-luna，目标 G=[393.0, -289.0, 284.0]

## v3（50 帧，其中选动作的 50 帧）
- 选择题本身：正序 10%、倒序 18%；两遍一致 42%；一致且对 4%
- **策略（合手由「填满圈」触发）**：出动作 49/50，对 6%
  - east（16）对 6%；出成 lower×6、arrow_C×5、close_hand×2、arrow_D×2
  - north（11）对 0%；出成 close_hand×9、lower×2
  - south（10）对 0%；出成 lower×4、close_hand×2、arrow_A×2、arrow_C×2
  - west（6）对 0%；出成 close_hand×4、lower×2
  - lower（6）对 33%；出成 close_hand×4、lower×2
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - east（16）对 6%；答成 lower×5、arrow_C×5、close_hand×3、arrow_D×2
  - north（12）对 0%；答成 lower×9、close_hand×3
  - south（10）对 10%；答成 lower×5、arrow_A×3、arrow_C×2
  - west（6）对 0%；答成 lower×3、close_hand×3
  - lower（6）对 50%；答成 close_hand×3、lower×3
- predicate sleeve_visible：按 0.5 判对 100%；真为是 50 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate fills_circle：按 0.5 判对 58%；真为是 0 帧概率中位 nan，真为否 50 帧中位 0.18
- 延迟 p50 885 ms、p95 2011 ms；输入 token 中位 1681

## v5（50 帧，其中选动作的 50 帧）
- 选择题本身：正序 26%、倒序 36%；两遍一致 40%；一致且对 6%
- **策略（合手由「填满圈」触发）**：出动作 46/50，对 35%
  - east（15）对 47%；出成 arrow_A×6、arrow_C×4、lower×3、arrow_B×2
  - north（10）对 20%；出成 close_hand×7、arrow_B×2、lower×1
  - south（10）对 50%；出成 arrow_A×6、lower×2、arrow_D×1、arrow_C×1
  - lower（6）对 33%；出成 close_hand×2、lower×2、arrow_C×1、arrow_A×1
  - west（5）对 0%；出成 close_hand×3、lower×1、arrow_A×1
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - east（16）对 44%；答成 arrow_A×6、arrow_C×4、close_hand×2、lower×2
  - north（12）对 8%；答成 close_hand×9、lower×2、arrow_B×1
  - south（10）对 40%；答成 arrow_A×6、lower×2、close_hand×1、arrow_C×1
  - west（6）对 0%；答成 close_hand×5、arrow_A×1
  - lower（6）对 17%；答成 close_hand×3、lower×1、arrow_C×1、arrow_A×1
- predicate sleeve_visible：按 0.5 判对 100%；真为是 50 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate fills_circle：按 0.5 判对 76%；真为是 0 帧概率中位 nan，真为否 50 帧中位 0.00
- 延迟 p50 583 ms、p95 1089 ms；输入 token 中位 1761

共 100 个请求，输入 172100 token ≈ $0.017