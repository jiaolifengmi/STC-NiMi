import numpy as np
import matplotlib.pyplot as plt


class HH():
    def __init__(self, I_init, ts, T):
        # 初始化各个值
        self.I_init = I_init  # 刺激电流
        self.Num = int(T/ts)  # 总点数
        self.ts = ts  # 步长
        self.t = np.arange(0, T, ts)  # 时间序列
        self.V = np.zeros_like(self.t)  # 电压
        self.I = np.zeros_like(self.t)  # 突触电流
        self.m = np.zeros_like(self.t)
        self.n = np.zeros_like(self.t)
        self.h = np.zeros_like(self.t)
        self.am = np.zeros_like(self.t)
        self.bm = np.zeros_like(self.t)
        self.ah = np.zeros_like(self.t)
        self.bh = np.zeros_like(self.t)
        self.an = np.zeros_like(self.t)
        self.bn = np.zeros_like(self.t)
        self.spikeT = []  # spike time

        self.V[0] = -70
        self.am[0] = 0.1 * ((self.V[0] + 40) / (1 - np.exp(-(self.V[0] + 40) / 10)))
        self.bm[0] = 4 * np.exp(-(self.V[0] + 65) / 18)
        self.ah[0] = 0.07 * np.exp(-(self.V[0] + 65) / 20)
        self.bh[0] = 1 / (np.exp(-(self.V[0] + 35) / 10) + 1)
        self.an[0] = 0.01 * ((self.V[0] + 55) / (1 - np.exp(-(self.V[0] + 55) / 10)))
        self.bn[0] = 0.125 * np.exp(-(self.V[0] + 65) / 80)

        self.m[0] = self.am[0] / (self.am[0] + self.bm[0])
        self.n[0] = self.an[0] / (self.an[0] + self.bn[0])
        self.h[0] = self.ah[0] / (self.ah[0] + self.bh[0])

        self.C_m = 1
        self.E_na = 50
        self.E_k = -77
        self.E_l = -54.4
        self.g_na = 120
        self.g_k = 36
        self.g_l = 0.3
        self.inh_v_yu = -15  # 阈值
        self.exc_v_yu = 15

    # 按循环进行欧拉计算
    def euler(self):
        # 依次迭代，欧拉计算
        for i in range(self.Num-1):
            self.V[i + 1] = self.V[i] + self.ts * ((-self.g_l * (self.V[i] - self.E_l)
                                          - self.g_na * (self.m[i] ** 3) * self.h[i] * (self.V[i] - self.E_na)
                                          - self.g_k * (self.n[i] ** 4) * (self.V[i] - self.E_k) + self.I_init + self.I[i])
                                         / self.C_m)

            self.am[i + 1] = 0.1 * (self.V[i + 1] + 40) / (1 - np.exp(-(self.V[i + 1] + 40) / 10))
            self.bm[i + 1] = 4 * np.exp(-(self.V[i + 1] + 65) / 18)
            self.ah[i + 1] = 0.07 * np.exp(-(self.V[i + 1] + 65) / 20)
            self.bh[i + 1] = 1 / (np.exp(-(self.V[i + 1] + 35) / 10) + 1)
            self.an[i + 1] = 0.01 * ((self.V[i + 1] + 55) / (1 - np.exp(-(self.V[i + 1] + 55) / 10)))
            self.bn[i + 1] = 0.125 * np.exp(-(self.V[i + 1] + 65) / 80)

            self.m[i + 1] = self.m[i] + (self.am[i + 1] * (1 - self.m[i]) - self.bm[i + 1] * self.m[i]) * self.ts
            self.h[i + 1] = self.h[i] + (self.ah[i+1] * (1 - self.h[i]) - self.bh[i+1] * self.h[i])*self.ts
            self.n[i + 1] = self.n[i] + (self.an[i+1] * (1 - self.n[i]) - self.bn[i+1] * self.n[i])*self.ts

            # 判断是否发生spike
            if self.I_init == 0:
                if i != 0 and self.V[i] > self.V[i-1] and self.V[i] > self.V[i+1] and self.V[i] > self.inh_v_yu:
                    self.spikeT.append(self.t[i])
            else:
                if i != 0 and self.V[i] > self.V[i-1] and self.V[i] > self.V[i+1] and self.V[i] > self.exc_v_yu:
                    self.spikeT.append(self.t[i])

        return self.t, self.V, self.spikeT

    # 欧拉计算
    def euler_i(self, i):
        self.V[i + 1] = self.V[i] + self.ts * ((-self.g_l * (self.V[i] - self.E_l)
                                                - self.g_na * (self.m[i] ** 3) * self.h[i] * (self.V[i] - self.E_na)
                                                - self.g_k * (self.n[i] ** 4) * (self.V[i] - self.E_k) + self.I_init +
                                                self.I[i])
                                               / self.C_m)

        self.am[i + 1] = 0.1 * (self.V[i + 1] + 40) / (1 - np.exp(-(self.V[i + 1] + 40) / 10))
        self.bm[i + 1] = 4 * np.exp(-(self.V[i + 1] + 65) / 18)
        self.ah[i + 1] = 0.07 * np.exp(-(self.V[i + 1] + 65) / 20)
        self.bh[i + 1] = 1 / (np.exp(-(self.V[i + 1] + 35) / 10) + 1)
        self.an[i + 1] = 0.01 * ((self.V[i + 1] + 55) / (1 - np.exp(-(self.V[i + 1] + 55) / 10)))
        self.bn[i + 1] = 0.125 * np.exp(-(self.V[i + 1] + 65) / 80)

        self.m[i + 1] = self.m[i] + (self.am[i + 1] * (1 - self.m[i]) - self.bm[i + 1] * self.m[i]) * self.ts
        self.h[i + 1] = self.h[i] + (self.ah[i + 1] * (1 - self.h[i]) - self.bh[i + 1] * self.h[i]) * self.ts
        self.n[i + 1] = self.n[i] + (self.an[i + 1] * (1 - self.n[i]) - self.bn[i + 1] * self.n[i]) * self.ts

        # 判断是否发生spike
        if self.I_init == 0:
            if i != 0 and self.V[i] > self.V[i - 1] and self.V[i] > self.V[i + 1] and self.V[i] > self.inh_v_yu:
                self.spikeT.append(self.t[i])
        else:
            if i != 0 and self.V[i] > self.V[i - 1] and self.V[i] > self.V[i + 1] and self.V[i] > self.exc_v_yu:
                self.spikeT.append(self.t[i])

    # 画图函数
    def draw(self):
        plt.figure()
        plt.title('HH(v-t)')
        plt.xlabel('t(ms)')
        plt.ylabel('v(mV)')
        plt.plot(self.t, self.V)
        plt.show()

if __name__ == '__main__':
    I = 10  # 刺激电流
    h = 0.01  # 步长
    T = 1000   # 时间范围

    hh = HH(I, h, T)
    # 调用欧拉计算函数
    hh.euler()
    # 调用画图函数
    hh.draw()


