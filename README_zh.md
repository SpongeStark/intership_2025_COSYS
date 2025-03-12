# COSYS 实习：基于图像处理的人眼视觉在对“对比度”感知的建模研究

- 杨凯
- 索邦大学
- 计算机图像 M2

[TOC]

# 第一阶段：参考文献

## 边缘检测 VS 边界检测

## 数据标注

- BSDS500
- BIPEDv2

## 边缘检测网络结构

### DexiNed

[notebook](./DexiNed/main.ipynb)

### PiDiNet

[notebook](./PiDiNet/pidinet-master/index.ipynb)

### EDTER

别急，还在写！！！

# 第二阶段：加入可见度

结合Lucas的代码，通过滤波的方法得到能见度，作为数据训，并改进神经网络模型以适应对应数据，进行训练