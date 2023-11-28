import HH_model as HH
import LIF_model as LIF
import Synapsis as Syn
import matplotlib.pyplot as plt

class examples():
    def __init__(self, type_str, h, T):
        self.type = type_str
        self.h = h  # 步长
        self.T = T  # 时间范围
        self.Num = int(self.T / self.h)

    def run(self):
        if self.type == 1:
            self.run_example1()
        elif self.type == 2:
            self.run_example2()
        elif self.type == 3:
            self.run_example3()
        elif self.type == 4:
            self.run_example4()
        elif self.type == 5:
            self.run_example5()
        else:
            self.run_example6()

    # 兴奋 -> 抑制
    def run_example1(self):
        # 前兴奋性神经元
        front_Neuron = LIF.LIF(10, self.h, self.T)
        # front_Neuron = HH.HH(10, self.h, self.T)

        # 后抑制性神经元
        behind_Neuron = LIF.LIF(0, self.h, self.T)
        # behind_Neuron = HH.HH(0, self.h, self.T)

        # 兴奋性连接
        syn = Syn.Synapsis(0, [], self.h, self.T)

        for i in range(self.Num-1):
            front_Neuron.euler_i(i)
            syn.t_s = front_Neuron.spikeT
            syn.calculate_g_i(i)
            behind_Neuron.I[i] = -syn.gl[i] * (behind_Neuron.V[i] - syn.E_syn)
            behind_Neuron.euler_i(i)

        # 绘图
        fig, ax = plt.subplots(3, 1)
        plt.subplots_adjust(hspace=1)
        ax[0].plot(front_Neuron.t, front_Neuron.V)
        ax[0].set_title('E1_v')
        ax[0].set_ylabel('v(mV)')

        ax[1].plot(behind_Neuron.t, behind_Neuron.V)
        ax[1].set_title('I1_v')
        ax[1].set_ylabel('v(mV)')

        ax[2].plot(syn.t, syn.gl)
        ax[2].set_title('conductance')
        ax[2].set_xlabel('t(ms)')
        ax[2].set_ylabel('g(s/m)')

        plt.show()

    # 兴奋 -> 抑制 -> 兴奋
    def run_example2(self):
        # 前兴奋性神经元
        front_Neuron = LIF.LIF(10, self.h, self.T)
        # front_Neuron = HH.HH(10, self.h, self.T)

        # 后1抑制性神经元
        behind_Neuron1 = LIF.LIF(0, self.h, self.T)
        # behind_Neuron1 = HH.HH(0, self.h, self.T)

        # 后2兴奋性神经元
        behind_Neuron2 = LIF.LIF(10, self.h, self.T)
        # behind_Neuron2 = HH.HH(10, self.h, self.T)

        # 兴奋性连接(E1-I1)
        syn1 = Syn.Synapsis(0, [], self.h, self.T)

        # 抑制性连接(I1-E2)
        syn2 = Syn.Synapsis(-80, [], self.h, self.T)


        # 循环，依次进行spike_T、电导、电流、电压的计算
        for i in range(self.Num - 1):
            front_Neuron.euler_i(i)
            syn1.t_s = front_Neuron.spikeT
            syn1.calculate_g_i(i)
            behind_Neuron1.I[i] = -syn1.gl[i] * (behind_Neuron1.V[i] - syn1.E_syn)
            behind_Neuron1.euler_i(i)

            syn2.t_s = behind_Neuron1.spikeT
            syn2.calculate_g_i(i)
            behind_Neuron2.I[i] = -syn2.gl[i] * (behind_Neuron2.V[i] - syn2.E_syn)
            behind_Neuron2.euler_i(i)

        # 绘图
        fig, ax = plt.subplots(5, 1)
        plt.subplots_adjust(hspace=1)
        ax[0].plot(front_Neuron.t, front_Neuron.V)
        ax[0].set_title('E1_v')
        ax[0].set_ylabel('v(mV)')

        ax[1].plot(behind_Neuron1.t, behind_Neuron1.V)
        ax[1].set_title('I1_v')
        ax[1].set_ylabel('v(mV)')

        ax[2].plot(behind_Neuron2.t, behind_Neuron2.V)
        ax[2].set_title('E2_v')
        ax[2].set_ylabel('v(mV)')

        ax[3].plot(syn1.t, syn1.gl)
        ax[3].set_title('Excitability-conductance')
        ax[3].set_ylabel('g(s/m)')

        ax[4].plot(syn2.t, syn2.gl)
        ax[4].set_title('Inhibitory-conductance')
        ax[4].set_xlabel('t(ms)')
        ax[4].set_ylabel('g(s/m)')

        plt.show()

    # 兴奋 <-> 抑制
    def run_example3(self):
        # 前兴奋性神经元
        front_Neuron = LIF.LIF(10, self.h, self.T)
        # front_Neuron = HH.HH(10, self.h, self.T)

        # 后抑制性神经元
        behind_Neuron = LIF.LIF(0, self.h, self.T)
        # behind_Neuron = HH.HH(0, self.h, self.T)

        # 兴奋性连接
        syn1 = Syn.Synapsis(0, [], self.h, self.T)

        # 抑制性连接
        syn2 = Syn.Synapsis(-80, [], self.h, self.T)

        # 循环，依次进行spike_T、电导、电流、电压的计算
        for i in range(self.Num-1):
            syn2.t_s = behind_Neuron.spikeT
            syn2.calculate_g_i(i)
            front_Neuron.I[i] = -syn2.gl[i] * (front_Neuron.V[i] - syn2.E_syn)
            front_Neuron.euler_i(i)

            syn1.t_s = front_Neuron.spikeT
            syn1.calculate_g_i(i)
            behind_Neuron.I[i] = -syn1.gl[i] * (behind_Neuron.V[i] - syn1.E_syn)
            behind_Neuron.euler_i(i)

        # 绘图
        fig, ax = plt.subplots(4, 1)
        plt.subplots_adjust(hspace=1)
        ax[0].plot(front_Neuron.t, front_Neuron.V)
        ax[0].set_title('E1_v')
        ax[0].set_ylabel('v(mV)')

        ax[1].plot(behind_Neuron.t, behind_Neuron.V)
        ax[1].set_title('I1_v')
        ax[1].set_ylabel('v(mV)')

        ax[2].plot(syn1.t, syn1.gl)
        ax[2].set_title('Excitability-conductance')
        ax[2].set_ylabel('g(s/m)')

        ax[3].plot(syn2.t, syn2.gl)
        ax[3].set_title('Inhibitory-conductance')
        ax[3].set_xlabel('t(ms)')
        ax[3].set_ylabel('g(s/m)')

        plt.show()

    # 兴奋（《-》）<-> 抑制（《-》）
    def run_example4(self):
        # 前兴奋性神经元
        front_Neuron = LIF.LIF(10, self.h, self.T)
        # front_Neuron = HH.HH(10, self.h, self.T)

        # 后抑制性神经元
        behind_Neuron = LIF.LIF(0, self.h, self.T)
        # behind_Neuron = HH.HH(0, self.h, self.T)

        # 兴奋性神经元-自连接
        syn1 = Syn.Synapsis(0, [], self.h, self.T)

        # 兴奋性神经元-抑制性连接
        syn2 = Syn.Synapsis(-80, [], self.h, self.T)

        # 抑制性神经元-自连接
        syn3 = Syn.Synapsis(-80, [], self.h, self.T)

        # 抑制性神经元-兴奋性连接
        syn4 = Syn.Synapsis(0, [], self.h, self.T)

        # 循环，依次进行spike_T、电导、电流、电压的计算
        for i in range(self.Num-1):
            # 计算两个突触叠加的电流
            syn1.t_s = front_Neuron.spikeT
            syn1.calculate_g_i(i)
            front_I_temp = -syn1.gl[i] * (front_Neuron.V[i] - syn1.E_syn)

            syn2.t_s = behind_Neuron.spikeT
            syn2.calculate_g_i(i)
            front_Neuron.I[i] = -syn2.gl[i] * (front_Neuron.V[i] - syn2.E_syn) + front_I_temp

            front_Neuron.euler_i(i)

            # 计算两个突触叠加的电流
            syn3.t_s = behind_Neuron.spikeT
            syn3.calculate_g_i(i)
            behind_I_temp = -syn3.gl[i] * (behind_Neuron.V[i] - syn3.E_syn)

            syn4.t_s = front_Neuron.spikeT
            syn4.calculate_g_i(i)
            behind_Neuron.I[i] = -syn4.gl[i] * (behind_Neuron.V[i] - syn4.E_syn) + behind_I_temp

            behind_Neuron.euler_i(i)

        # 绘图
        fig, ax = plt.subplots(6, 1)
        plt.subplots_adjust(hspace=1)
        ax[0].plot(front_Neuron.t, front_Neuron.V)
        ax[0].set_title('E1_v')
        ax[0].set_ylabel('v(mV)')

        ax[1].plot(behind_Neuron.t, behind_Neuron.V)
        ax[1].set_title('I1_v')
        ax[1].set_ylabel('v(mV)')

        ax[2].plot(syn1.t, syn1.gl)
        ax[2].set_title('self-Excitability-conductance')
        ax[2].set_ylabel('g(s/m)')

        ax[3].plot(syn2.t, syn2.gl)
        ax[3].set_title('Inhibitory-conductance')
        ax[3].set_xlabel('t(ms)')
        ax[3].set_ylabel('g(s/m)')

        ax[4].plot(syn3.t, syn3.gl)
        ax[4].set_title('self-Inhibitory-conductance')
        ax[4].set_ylabel('g(s/m)')

        ax[5].plot(syn4.t, syn4.gl)
        ax[5].set_title('Excitability-conductance')
        ax[5].set_xlabel('t(ms)')
        ax[5].set_ylabel('g(s/m)')

        plt.show()

    # 兴奋 ->(可塑性) 兴奋
    def run_example5(self):
        # 前兴奋性神经元
        pre_neuron = LIF.LIF(10, self.h, self.T)
        # pre_neuron = HH.HH(10, self.h, self.T)

        # 后兴奋性神经元
        post_neuron = LIF.LIF(0, self.h, self.T)
        # post_neuron = HH.HH(0, self.h, self.T)

        # 可塑性连接
        syn = Syn.Synapsis(0, [], self.h, self.T)

        for i in range(self.Num-1):
            pre_neuron.euler_i(i)
            syn.t_s = pre_neuron.spikeT
            # syn.heb(i, pre_neuron.spikeF[i], post_neuron.spikeF[i])
            syn.stdp(i, pre_neuron.spikeT_i[i], post_neuron.V[i])
            post_neuron.I[i] = -syn.gl[i] * (post_neuron.V[i] - syn.E_syn)
            post_neuron.euler_i(i)

        # 绘图
        fig, ax = plt.subplots(4, 1)
        plt.subplots_adjust(hspace=1)
        ax[0].plot(pre_neuron.t, pre_neuron.V)
        ax[0].set_title('E1_v')
        ax[0].set_ylabel('v(mV)')

        ax[1].plot(post_neuron.t, post_neuron.V)
        ax[1].set_title('E2_v')
        ax[1].set_ylabel('v(mV)')

        ax[2].plot(syn.t, syn.gl)
        ax[2].set_title('Conductance')
        ax[2].set_ylabel('g(s/m)')

        ax[3].plot(syn.t, syn.v)
        # ax[3].set_title('G_max')
        ax[3].set_xlabel('t(ms)')
        # ax[3].set_ylabel('g_max')

        plt.show()

        return 1

    # 兴奋（《-》可塑性）
    def run_example6(self):
        # 兴奋性神经元
        neuron = LIF.LIF(10, self.h, self.T)
        # neuron = HH.HH(10, self.h, self.T)

        # 可塑性连接
        syn = Syn.Synapsis(0, [], self.h, self.T)

        for i in range(self.Num-1):
            neuron.euler_i(i)
            syn.t_s = neuron.spikeT
            # syn.heb(i, neuron.spikeF[i], neuron.spikeF[i])
            syn.stdp(i, neuron.spikeT_i[i], neuron.V[i])
            neuron.I[i] = -syn.gl[i] * (neuron.V[i] - syn.E_syn)
            neuron.euler_i(i)

        # 绘图
        fig, ax = plt.subplots(3, 1)
        plt.subplots_adjust(hspace=1)
        ax[0].plot(neuron.t, neuron.V)
        ax[0].set_title('E1_v')
        ax[0].set_ylabel('v(mV)')

        ax[1].plot(syn.t, syn.gl)
        ax[1].set_title('conductance')
        ax[1].set_ylabel('g(s/m)')

        ax[2].plot(syn.t, syn.g_max)
        ax[2].set_title('G_max')
        ax[2].set_xlabel('t(ms)')
        ax[2].set_ylabel('g_max')

        plt.show()

        return 1

if __name__ == '__main__':
    example = examples(5, 1.0, 1000)
    example.run()
