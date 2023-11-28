import numpy as np
import matplotlib.pyplot as plt


class LIF():
    def __init__(self, ts, T):
        self.I_init = 0
        self.T = T  # 总时间 单位 ms
        self.ts = ts  # 步长 单位 ms
        self.Num = int(T / ts)  # 总的时间点数
        self.t = np.arange(0, T, ts)  # 时间序列
        self.V = np.zeros_like(self.t)  # 电压
        self.I = np.zeros_like(self.t)  # 突触电流
        self.spikeT = []  # 发生spike的时候，记录时间点
        self.spikeT_i = np.zeros_like(self.t)  # 记录每个时间点是否发生spike,发生值为1，反之为0
        self.spikeF = np.zeros_like(self.t)  # 发放率
        self.spike_number = 0  # 1s内发生的总的spike次数
        self.buyingqi = False  # 是否不应期
        self.buyingqi_time = 0  # 不应期计时
        self.buyingqi_threshold = 50

        # 初始化各个值
        self.V[0] = -70
        self.Rm = 10
        self.Em = -65
        self.tao = 10  #
        self.v_th = -15  # 阈值
        self.t_slider = 20  # 滑窗
        self.spike_num = 0  # 滑窗内发生spike的次数

    # def euler(self):
    #     for i in range(self.Num-1):
    #         self.V[i + 1] = self.V[i] + self.ts * ((-self.V[i] + self.Em + (self.I[i + 1] + self.I_init) * self.Rm) / self.tao)
    #         if self.V[i + 1] > self.v_th and self.t[i] % self.Pre_time == 0:
    #             # print(self.t[i])
    #             self.V[i + 1] = self.Em
    #             self.spikeT.append(self.t[i])
    #     for i in range(self.Num - 1):
    #         for j in range(len(self.spikeT)):
    #             if self.t[i] < self.t[j] < self.t[i+self.t_slider]:
    #                 self.spike_num += self.spike_num
    #         self.spikeF[i] = self.spike_num/self.t_slider
    #         self.spike_num = 0
    #     return 1

    # 欧拉计算 and self.t[i]
    def euler_i(self, i):

        # 判断是否不应期
        if self.buyingqi:
            self.V[i + 1] = self.Em
            self.buyingqi_time += 1
            if self.buyingqi_time == self.buyingqi_threshold:
                self.buyingqi = False
                self.buyingqi_time = 0
        else:
            # 计算电压
            self.V[i + 1] = self.V[i] + self.ts * (
                    (-self.V[i] + self.Em + (self.I[i] + self.I_init) * self.Rm) / self.tao)
            # 限定电压
            if self.V[i + 1] < -70:
                self.V[i + 1] = self.Em
            elif self.V[i + 1] > 10:
                self.V[i + 1] = 10
            # 判断是否应产生spike
            if self.V[i + 1] > self.v_th:
                self.spike_number += 1
                self.spikeT_i[i] = 1
                self.spikeT.append(self.t[i])
                self.buyingqi_time = 0
                self.buyingqi = True

        # 计算发放率
        for spike_j in range(len(self.spikeT)):
            t_temp = i - self.t_slider
            if t_temp < 0:
                t_temp = 0
            if self.t[t_temp] <= self.spikeT[spike_j] <= self.t[i]:
                self.spike_num += 1
        self.spikeF[i] = self.spike_num / self.t_slider
        self.spike_num = 0

        return 1

    # 画图函数
    def draw(self):
        plt.figure()
        plt.title('LIF(v-t)')
        plt.xlabel('t(ms)')
        plt.ylabel('v(mV)')
        plt.plot(self.t, self.V)
        plt.show()

class Exc_LIF(LIF):
    def __init__(self, ts, T, I_init, IntervalTime):
        LIF.__init__(self, ts, T)
        self.I_init = I_init  # 注入刺激电流
        self.buyingqi_threshold = IntervalTime  # 不应期阈值


class Inh_LIF(LIF):
    def __init__(self, ts, T, I_init, IntervalTime):
        LIF.__init__(self, ts, T)
        self.I_init = I_init  # 注入刺激电流
        self.buyingqi_threshold = IntervalTime  # 不应期阈值


if __name__ == '__main__':
    I = 110  # 刺激电流
    h = 1.0  # 步长
    T = 1000   # 时间范围

    lif = Exc_LIF(h, T, 10, 20)
    # 调用欧拉计算函数
    for t in range(lif.Num - 1):
        lif.euler_i(t)
    # 调用画图函数
    lif.draw()
    # print(lif.spike_number)
    # print(len(lif.spikeT))


