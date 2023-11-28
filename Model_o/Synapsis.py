import numpy as np
import matplotlib.pyplot as plt


class Synapsis():
    def __init__(self, E_syn, t_s, ts, T):
        self.E_syn = E_syn  # E_syn抑制性的为-80 兴奋性的为0
        self.t_s = t_s  # 发生spike的时间点序列
        self.Num = int(T / ts)  # 总点数
        self.ts = ts  # 步长
        self.t = np.arange(0, T, ts)  # 时间序列
        self.g = np.zeros_like(self.t)  # 点电导
        self.gl = np.zeros_like(self.t)  # 积累电导
        self.g_max = np.ones_like(self.t)*0.1  # 权重
        self.change_rate = np.zeros_like(self.t)  # 权重的变化率
        self.u = np.zeros_like(self.t)
        self.v = np.zeros_like(self.t)
        self.x = np.zeros_like(self.t)

        # 初始化各个值
        self.tao = 5
        self.g_syn = 1
        self.a = 0.01
        self.b = 0.001
        self.A_LTD = 0.0008
        self.A_LTP = 0.0014
        self.LTD_th = -70  # 发生LTD的阈值
        self.LTP_th = -49  # 发生LTP的阈值
        self.tao_u = 10
        self.tao_v = 7
        self.tao_x = 15
        self.g_max_change = 0

    # 计算所有时间点的电导
    def calculate_g(self):
        if self.t_s == []:
            return 0
        for j in range(len(self.t_s)):
            for i in range(self.Num-1):
                self.g[i] = self.g_syn * ((self.t[i] - self.t_s[j]) / self.tao) * np.exp(-(self.t[i] - self.t_s[j]) / self.tao)
                if self.g[i] < 0:
                    self.g[i] = 0
                self.gl[i] += self.g[i]
        return 1

    # 计算单个时间点的电导
    def calculate_g_i(self, i):
        if self.t_s == []:
            return 0
        for j in range(len(self.t_s)):
            self.g[i] = self.g_syn * ((self.t[i] - self.t_s[j]) / self.tao) * np.exp(-(self.t[i] - self.t_s[j]) / self.tao)
            if self.g[i] < 0:
                self.g[i] = 0
            self.gl[i] += self.g[i]
        return 1

    # Hebbian rule
    def heb(self, i, pre_f, post_f):
        if self.t_s == []:
            return 0
        # 计算 G_max
        """if i % 20 == 0:
            print(pre_f, post_f)"""
        self.g_max[i + 1] = self.g_max[i] + self.a * (pre_f * post_f - self.b)
        if self.g_max[i + 1] > 2:
            self.g_max[i + 1] = 2
        if self.g_max[i + 1] < 0:
            self.g_max[i + 1] = 0
        # 计算电导
        for j in range(len(self.t_s)):
            self.g[i] = self.g_max[i] * ((self.t[i] - self.t_s[j]) / self.tao) * np.exp(-(self.t[i] - self.t_s[j]) / self.tao)
            if self.g[i] < 0:
                self.g[i] = 0
            self.gl[i] += self.g[i]
        return 1

    # STDP中的R函数
    def R(self, x):
        if x < 0:
            return 0
        else:
            return x

    # STDP
    def stdp(self, i, pre_spikeT_i, post_v):
        if self.t_s == []:
            return 0
        # 计算 G_max
        # u(t) = u(t-1) + dt(V(t-1)-u(t-1))/tao_u
        self.x[i+1] = self.x[i] + self.ts * pre_spikeT_i / self.tao_x
        self.u[i+1] = self.u[i] + self.ts * (post_v - self.u[i]) / self.tao_u
        self.v[i+1] = self.v[i] + self.ts * (post_v - self.v[i]) / self.tao_v
        self.g_max_change = -self.A_LTD * pre_spikeT_i * self.R(self.u[i] - self.LTD_th) + \
                            self.A_LTP * self.x[i] * self.R(post_v - self.LTP_th) * self.R(self.v[i] - self.LTD_th)

        self.change_rate[i] = self.g_max_change

        self.g_max[i + 1] = self.g_max[i] + self.ts * self.g_max_change


        # 计算电导
        for j in range(len(self.t_s)):
            self.g[i] = self.g_max[i] * ((self.t[i] - self.t_s[j]) / self.tao) * np.exp(
                -(self.t[i] - self.t_s[j]) / self.tao)
            if self.g[i] < 0:
                self.g[i] = 0
            self.gl[i] += self.g[i]
        return 1



    # 绘制电导随时间的变化图
    def draw(self):
        plt.figure()
        plt.title('g-t')
        plt.xlabel('t(ms)')
        plt.ylabel('g')
        plt.plot(self.t, self.gl)
        plt.show()
