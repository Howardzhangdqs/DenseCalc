# 基于递归划分框架的人群密度估计

## 简介

本项目提出了一种全新的人群计数范式，通过逐层细分图像区域并进行目标数量估计，以提高人群计数的准确性和效率。该方法结合了区域分割与逐步细化的思想，能够自适应地对密集区域进行更精细的处理，而对稀疏区域保持较低的计算开销。

## 算法流程

1. **输入图像处理**
   - 读取输入图像，并将其切分为若干个大小相等的 patch。

2. **目标数量估计**
   - 对每个 patch 内的目标数量进行估计：
     - 如果目标数量小于阈值 $N$，则将该 patch 交由计数器（counter）进行精确计数。
     - 如果目标数量大于或等于 $N$，则将该patch按照2x2切分为四个更小的patch，并对每个小patch重复目标数量估计的过程。

3. **终止条件**
   - 当 patch 内的目标数量小于 $N$，或者 patch 的大小小于 $16\times 16$ 像素时，停止进一步切分并交由计数器进行计数。

4. **结果整合**
   - 将所有 patch 及其子 patch 的计数结果相加，得到最终的人群总数。