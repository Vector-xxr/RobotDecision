# 近处小实验 near_20261013_133013

## A1（88 帧）：正序 77%、倒序 78%、两遍一致且对 73%
  - 回放 centered（10）对 70%；答成 centered×7、arrow_C×1、arrow_A×1
  - 回放 east（17）对 100%；答成 arrow_B×5、arrow_C×5、arrow_D×4
  - 回放 north（8）对 75%；答成 centered×4、arrow_C×2、arrow_A×1
  - 回放 not_visible（1）对 0%；答成 arrow_B×1
  - 回放 west（2）对 100%；答成 centered×1、arrow_C×1
  - 网格 centered（2）对 0%；答成 arrow_A×2
  - 网格 east（16）对 100%；答成 arrow_D×8、arrow_A×4、arrow_B×3
  - 网格 north（12）对 33%；答成 centered×8、arrow_D×2、arrow_B×2
  - 网格 south（12）对 92%；答成 arrow_B×4、arrow_D×3、arrow_C×2
  - 网格 west（8）对 62%；答成 centered×4、arrow_A×3、arrow_C×1
  - 按水平偏移：0mm 50%(4)  5mm 62%(8)  10mm 83%(18)  15mm 70%(10)  20mm 75%(24)  25mm 100%(2)  30mm 80%(10)  35mm 100%(2)  40mm 100%(4)  50mm 100%(4)  55mm 50%(2)

## A1_mirror（50 帧）：正序 72%、倒序 68%、两遍一致且对 62%
  - 网格 centered（2）对 50%；答成 centered×1、arrow_A×1
  - 网格 east（16）对 75%；答成 arrow_A×6、arrow_C×5、arrow_B×3
  - 网格 north（12）对 42%；答成 centered×9、arrow_A×2、arrow_C×1
  - 网格 south（12）对 100%；答成 arrow_A×8、arrow_B×2、arrow_D×1
  - 网格 west（8）对 75%；答成 arrow_A×4、centered×2、arrow_D×1
  - 按水平偏移：0mm 50%(2)  10mm 88%(8)  15mm 50%(8)  20mm 71%(24)  30mm 88%(8)

## A2_single：该合手 10 帧判是 10（概率中位 0.95）；不该合手 72 帧误报 30（概率中位 0.21）
  - 误报按高度：+30mm×2  +50mm×17  +80mm×11

## A2_rings：该合手 10 帧判是 5（概率中位 0.52）；不该合手 72 帧误报 0（概率中位 0.00）
  - 误报按高度：

共 302 个请求