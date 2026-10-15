# Decisions 离线评测 20261013_113309
数据 /home/crp-5070ti-01/atom_episodes/robotdecision_replay/20261013_103555，模型 gpt-6-luna，目标 G=[393.0, -289.0, 284.0]

## v2（23 帧）
- 动作正确率：正序 9%、倒序 35%；两遍一致 9%；一致且对 4%
- 按正确动作分：
  - east（6）对 17%；答成 move_north×5、move_east×1
  - close_hand（4）对 0%；答成 move_north×2、move_east×1、lower×1
  - lower（4）对 0%；答成 move_north×3、move_east×1
  - raise（4）对 25%；答成 move_north×3、raise×1
  - done（3）对 0%；答成 raise×2、open_hand×1
  - target_not_visible（2）对 0%；答成 move_north×2
- confidence 中位 0.42；答错时中位 0.44（21 次答错）
- predicate sleeve_visible：按 0.5 判对 100%；真为是 21 帧概率中位 1.00，真为否 2 帧中位 0.00
- predicate between_fingers：按 0.5 判对 65%；真为是 4 帧概率中位 0.01，真为否 19 帧中位 0.00
- predicate lifted：按 0.5 判对 87%；真为是 3 帧概率中位 0.00，真为否 20 帧中位 0.00
- 延迟 p50 591 ms、p95 1227 ms、最大 1372 ms；输入 token 中位 1831

## v2arrows（23 帧）
- 动作正确率：正序 35%、倒序 35%；两遍一致 70%；一致且对 30%
- 按正确动作分：
  - east（6）对 83%；答成 arrow_B×2、arrow_C×2、target_not_visible×1、arrow_A×1
  - close_hand（4）对 0%；答成 arrow_B×2、arrow_C×1、arrow_A×1
  - lower（4）对 0%；答成 arrow_C×2、close_hand×1、arrow_B×1
  - raise（4）对 25%；答成 arrow_A×3、raise×1
  - done（3）对 0%；答成 raise×2、open_hand×1
  - target_not_visible（2）对 100%；答成 target_not_visible×2
- confidence 中位 0.40；答错时中位 0.28（15 次答错）
- predicate sleeve_visible：按 0.5 判对 96%；真为是 21 帧概率中位 1.00，真为否 2 帧中位 0.00
- predicate between_fingers：按 0.5 判对 65%；真为是 4 帧概率中位 0.03，真为否 19 帧中位 0.00
- predicate lifted：按 0.5 判对 87%；真为是 3 帧概率中位 0.00，真为否 20 帧中位 0.00
- 延迟 p50 583 ms、p95 615 ms、最大 626 ms；输入 token 中位 1820

## v2arrows_mirror（17 帧）
- 动作正确率：正序 35%、倒序 47%；两遍一致 76%；一致且对 35%
- 按正确动作分：
  - west（6）对 83%；答成 arrow_D×2、arrow_C×2、target_not_visible×1、arrow_A×1
  - close_hand（4）对 25%；答成 lower×1、close_hand×1、arrow_C×1、arrow_B×1
  - lower（4）对 0%；答成 close_hand×2、arrow_A×2
  - raise（3）对 0%；答成 arrow_B×1、arrow_C×1、arrow_D×1
- confidence 中位 0.28；答错时中位 0.21（11 次答错）
- predicate sleeve_visible：按 0.5 判对 94%；真为是 17 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate between_fingers：按 0.5 判对 82%；真为是 4 帧概率中位 0.94，真为否 13 帧中位 0.00
- predicate lifted：按 0.5 判对 100%；真为是 0 帧概率中位 nan，真为否 17 帧中位 0.00
- 延迟 p50 597 ms、p95 628 ms、最大 657 ms；输入 token 中位 1820

共 63 个请求，输入 112813 token ≈ $0.011