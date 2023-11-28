# STC-NiMi

##file description
Poisson_spike_1Hz  生成发放序列

STC Poisson_spike 频率一一对应 

weak、strong 刺激时间

## 网络规模
20个兴奋性的神经元、5个抑制性的神经元
### 神经元模型
三房室的神经元，以后可以尝试更多房室的神经元
### 突触连接 
兴奋性连接、抑制性连接

连接规则：
兴奋性神经元之间的连接矩阵21*60

(1)1 代表外部输入

(2)每三行和不能为0，代表每个神经元都能接受输入

(3)每三行的连接概率分别为1/4、1/2、1/4
### 突触可塑性模型 STC