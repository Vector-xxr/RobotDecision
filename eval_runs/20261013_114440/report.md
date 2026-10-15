# Decisions 离线评测 20261013_114440
数据 /home/crp-5070ti-01/atom_episodes/robotdecision_replay/20261013_103555，模型 gpt-6-luna，目标 G=[393.0, -289.0, 284.0]

## v3（23 帧，其中选动作的 19 帧）
- 动作正确率：正序 68%、倒序 63%；两遍一致 79%；一致且对 53%
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - east（9）对 78%；答成 arrow_A×4、arrow_C×3、target_not_visible×1、arrow_B×1
  - close_hand（4）对 50%；答成 close_hand×2、lower×2
  - target_not_visible（2）对 100%；答成 target_not_visible×2
  - lower_or_close（2）对 100%；答成 close_hand×2
  - lower（2）对 0%；答成 arrow_B×1、arrow_C×1
- predicate sleeve_visible：按 0.5 判对 96%；真为是 21 帧概率中位 1.00，真为否 2 帧中位 0.00
- predicate fills_circle：按 0.5 判对 100%；真为是 4 帧概率中位 0.90，真为否 13 帧中位 0.00
- predicate holding：按 0.5 判对 100%；真为是 4 帧概率中位 0.92，真为否 0 帧中位 nan
- 延迟 p50 831 ms、p95 2169 ms；输入 token 中位 1681

## v3_mirror（17 帧，其中选动作的 17 帧）
- 动作正确率：正序 88%、倒序 82%；两遍一致 82%；一致且对 76%
- 按正确动作分（方向按俯视画面：east=右、west=左）：
  - west（9）对 89%；答成 arrow_D×3、arrow_C×3、target_not_visible×1、arrow_A×1
  - close_hand（4）对 75%；答成 close_hand×3、lower×1
  - lower_or_close（2）对 100%；答成 close_hand×2
  - lower（2）对 100%；答成 lower×2
- predicate sleeve_visible：按 0.5 判对 94%；真为是 17 帧概率中位 1.00，真为否 0 帧中位 nan
- predicate fills_circle：按 0.5 判对 100%；真为是 4 帧概率中位 0.93，真为否 11 帧中位 0.00
- 延迟 p50 836 ms、p95 1038 ms；输入 token 中位 1681

共 40 个请求，输入 65260 token ≈ $0.007