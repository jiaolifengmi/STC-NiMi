## [1] Tegnér J, Compte A, Wang X J. The dynamical stability of reverberatory neural circuits[J]. Biological cybernetics, 2002, 87(5): 471-481.
## [2] Wang X J, Tegnér J, Constantinidis C, et al. Division of labor among distinct subtypes of inhibitory neurons in a cortical microcircuit of working memory[J]. Proceedings of the National Academy of Sciences of the United States of America, 2004, 101(5): 1368-1373.
## [1]和[2]的离子通道类型一样，公式基本一样，只有K-A的p gate tau_a不同，公式以[1]为准，但是[1]中CAN channel的表达式有误，所以CAN channel的表达式以[2]为准。


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import Poisson_spike_100Hz
import Poisson_spike_low_Frequency
from brokenaxes import brokenaxes


## soma model##
class Soma():
    def __init__(self):
        self.g_na = 55  ## mS/cm**2   55
        self.g_l = 0.05  ## mS/cm**2  0.05
        self.g_k = 15  ## mS/cm**2   15
        self.g_ca = 1.5  ## mS/cm**2   1.5
        self.g_can = 0.025  ## mS/cm**2  0.025

        self.alpha_ca = 0.000667  ## um(ms uA)**(-1)*cm**2   0.000667
        self.tau_ca = 240  ## ms  240
        self.alpha_can = 0.0056  ## (ms(mM)**2)*(-1)  0.0056
        self.beta_can = 0.002  ## ms*(-1)    0.002

        self.m = 0.0
        self.h = 0.999
        self.n = 0.001
        self.m_can = 0.0

        self.V_s = -75
        self.V_L = -70  ##Reversal potential
        self.V_na = 55  ##Reversal potential
        self.V_k = -80  ##Reversal potential
        self.V_Ca = 120  ##Reversal potential
        self.V_can = -20  ## mV
        self.Ca = 0.0

        self.Phi_h = 4.0
        self.Phi_n = 4.0

    def alpha_m(self, V_s_):
        alpha = (-0.1 * (V_s_ + 31)) / (np.exp(-0.1 * (V_s_ + 31)) - 1)
        return alpha

    def beta_m(self, V_s_):
        beta = 4 * np.exp(-(V_s_ + 56) / 18.0)
        return beta

    def alpha_h(self, V_s_):
        alpha = 0.07 * np.exp(-(V_s_ + 47) / 20)
        return alpha

    def beta_h(self, V_s_):
        beta = 1.0 / (1.0 + np.exp(-0.1 * (V_s_ + 17)))
        return beta

    def alpha_n(self, V_s_):
        alpha = (-0.01 * (V_s_ + 34)) / (np.exp(-0.1 * (V_s_ + 34)) - 1)
        return alpha

    def beta_n(self, V_s_):
        beta = 0.125 * np.exp(-(V_s_ + 44) / 80)
        return beta

    def m_inf(self, V_s_):
        m_inf = self.alpha_m(V_s_) / (self.alpha_m(V_s_) + self.beta_m(V_s_))
        return m_inf

    def m_tau(self, V_s_):
        m_tau = 1.0 / (self.alpha_m(V_s_) + self.beta_m(V_s_))
        return m_tau

    def h_inf(self, V_s_):
        h_inf = self.alpha_h(V_s_) / (self.alpha_h(V_s_) + self.beta_h(V_s_))
        return h_inf

    def h_tau(self, V_s_):
        h_tau = 1.0 / (self.alpha_h(V_s_) + self.beta_h(V_s_))
        return h_tau

    def n_inf(self, V_s_):
        n_inf = self.alpha_n(V_s_) / (self.alpha_n(V_s_) + self.beta_n(V_s_))
        return n_inf

    def n_tau(self, V_s_):
        n_tau = 1.0 / (self.alpha_n(V_s_) + self.beta_n(V_s_))
        return n_tau

    def m_inf_ca(self, V_s_):
        m_inf = 1 / (1 + np.exp(-(V_s_ + 20) / 9))
        return m_inf

    def m_inf_can(self, Ca):
        m_inf_can = (self.alpha_can * Ca * Ca) / (self.alpha_can * Ca * Ca + self.beta_can)
        return m_inf_can

    def m_tau_can(self, Ca):
        tau_can = 1 / (self.alpha_can * Ca * Ca + self.beta_can)
        return tau_can

    def I_Leak(self, V_s_):
        I_leak = self.g_l * (V_s_ - self.V_L)
        return I_leak

    def I_na(self, V_s_):
        I_na = self.g_na * self.m_inf(V_s_) ** 3 * self.h * (V_s_ - self.V_na)
        return I_na

    def I_k(self, V_s_):
        I_k = self.g_k * self.n ** 4 * (V_s_ - self.V_k)
        return I_k

    def I_Ca(self, V_s_):
        I_ca = self.g_ca * self.m_inf_ca(V_s_) ** 2 * (V_s_ - self.V_Ca)
        return I_ca

    def I_Can(self, V_s_):  ##slow calcium-dependent cationic current
        I_can = self.g_can * self.m_can ** 2 * (V_s_ - self.V_can)
        return I_can

    def Ca_dt(self, V_s_):
        Ca = self.Ca + (-self.alpha_ca * self.I_Ca(V_s_) - self.Ca / self.tau_ca) * step
        return Ca

    def h_dt(self, V_s_):
        h = self.h + (self.Phi_h * (self.h_inf(V_s_) - self.h) / self.h_tau(V_s_)) * step
        return h

    def n_dt(self, V_s_):
        n = self.n + (self.Phi_n * (self.n_inf(V_s_) - self.n) / self.n_tau(V_s_)) * step
        return n

    def m_can_dt(self, Ca):
        m_can = self.m_can + ((self.m_inf_can(Ca) - self.m_can) / self.m_tau_can(Ca)) * step
        return m_can

## dendritic model##
class dendritic1():  ##proximal
    def __init__(self):
        self.g_l = 0.05  ## mS/cm**2  0.05
        self.g_nap = 0.15  ## mS/cm**2  0.15
        self.g_ks = 2.0  ## mS/cm**2   2.0

        self.h = 0.999
        self.q = 0
        self.r = 0.5

        self.V_d_1 = -75
        self.V_L = -70  ## Reversal potential
        self.V_Nap = 55  ## mV
        self.V_KS = -80  ## mV

    def alpha_nap(self, V_d_1_):
        alpha = 0.001 * np.exp(((-85) - V_d_1_) / 30)
        return alpha

    def beta_nap(self, V_d_1_):
        beta = 0.0034 / (np.exp((-17 - V_d_1_) / 10) + 1)
        return beta

    def m_inf_nap(self, V_d_1_):
        m_inf = 1 / (1 + (np.exp(-(V_d_1_ + 55.7) / 7.7)))
        return m_inf

    def q_inf(self, V_d_1_):
        inf = 1 / (1 + np.exp(-(V_d_1_ + 34) / 6.5))
        return inf

    def q_tau(self, V_d_1_):
        tau = 8 / (np.exp(-(V_d_1_ + 55) / 30) + np.exp((V_d_1_ + 55) / 30))
        return tau

    def r_inf(self, V_d_1_):
        inf = 1 / (1 + np.exp((V_d_1_ + 65) / 6.6))
        return inf

    def r_tau(self, V_d_1_):
        tau = 100 + 100 / (1 + np.exp(-(V_d_1_ + 65) / 6.8))
        return tau

    def I_Leak(self, V_d_1_):
        I_leak = self.g_l * (V_d_1_ - self.V_L)
        return I_leak

    def I_Nap(self, V_d_1_):
        I_nap = self.g_nap * self.m_inf_nap(V_d_1_) ** 3 * self.h * (V_d_1_ - self.V_Nap)
        return I_nap

    def I_KS(self, V_d_1_):
        I_ks = self.g_ks * self.q * self.r * (V_d_1_ - self.V_KS)
        return I_ks

    def h_dt(self, V_d_1_):
        h = self.h + (self.alpha_nap(V_d_1_) * (1 - self.h) - self.beta_nap(V_d_1_) * self.h) * step
        return h

    def q_dt(self, V_d_1_):
        q = self.q + ((self.q_inf(V_d_1_) - self.q) / self.q_tau(V_d_1_)) * step
        return q

    def r_dt(self, V_d_1_):
        r = self.r + ((self.r_inf(V_d_1_) - self.r) / self.r_tau(V_d_1_)) * step
        return r

class dendritic2():  ##distal
    def __init__(self):
        self.g_l = 0.05  ## mS/cm**2  0.05
        self.g_ca = 0.25  ## mS/cm**2  0.25
        self.g_a = 1.0  ## mS/cm**2   1.0

        self.V_d_2 = -75
        self.V_L = -70  ##Reversal potential
        self.V_Ca = 120  ##Reversal potential
        self.V_A = -80  ##Reversal potential

        self.a = 0
        self.b = 0.0

        self.Ca = 0.0
        self.alpha_ca = 0.002  ## um(ms uA)**(-1)*cm**2  0.002
        self.tau_ca = 80  ## ms  80

    def m_inf_ca(self, V_d_2_):
        m_inf = 1 / (1 + np.exp(-(V_d_2_ + 20) / 9))
        return m_inf

    def a_inf(self, V_d_2_):
        inf = 1 / (1 + np.exp(-(V_d_2_ + 0) / 8.5))
        return inf

    def a_tau(self, V_d_2_):
        tau = 0.37 + 1 / (np.exp((V_d_2_ + 46) / 5) + np.exp(-(V_d_2_ + 238) / 37.5))
        return tau

    def b_inf(self, V_d_2_):
        inf = 1 / (1 + np.exp((V_d_2_ + 78) / 6))
        return inf

    def b_tau(self, V_d_2_):
        tau = 19 + 1 / (np.exp((V_d_2_ + 46) / 5) + np.exp((V_d_2_ + 238) / (-37.5)))
        return tau

    def I_Leak(self, V_d_2_):
        I_leak = self.g_l * (V_d_2_ - self.V_L)
        return I_leak

    def I_Ca(self, V_d_2_):
        I_ca = self.g_ca * self.m_inf_ca(V_d_2_) ** 2 * (V_d_2_ - self.V_Ca)
        return I_ca

    def I_A(self, V_d_2_):
        I_a = self.g_a * self.a ** 4 * self.b * (V_d_2_ - self.V_A)
        return I_a

    def Ca_dt(self, V_d_2_):
        Ca = self.Ca + (-self.alpha_ca * self.I_Ca(V_d_2_) - self.Ca / self.tau_ca) * step
        return Ca

    def a_dt(self, V_d_2_):
        a = self.a + ((self.a_inf(V_d_2_) - self.a) / self.a_tau(V_d_2_)) * step
        return a

    def b_dt(self, V_d_2_):
        b = self.b + ((self.b_inf(V_d_2_) - self.b) / self.b_tau(V_d_2_)) * step
        return b

## compartment connection##
class current_balance_multi_compart():
    def __init__(self):
        self.g_c1 = 0.75  ##ms/cm**2  0.75
        self.g_c2 = 0.25  ##ms/cm**2  0.25
        self.Cm = 1.0  ##uF/cm**2
        self.P1 = 0.5
        self.P2 = 0.3

        self.I_syn_s = 0
        self.I_syn_d1 = 0
        self.I_syn_d2 = 0

        self.soma = Soma()
        self.dend1 = dendritic1()
        self.dend2 = dendritic2()

        self.High_CREB = 1
        self.Low_CREB = 0

    def V_update(self, t_i, V_s_pre, V_d_1_pre, V_d_2_pre):

        self.soma.m = self.soma.m_inf(V_s_pre)
        self.soma.h = self.soma.h_dt(V_s_pre)
        self.soma.n = self.soma.n_dt(V_s_pre)
        self.soma.m_can = self.soma.m_can_dt(self.soma.Ca)
        self.soma.Ca = self.soma.Ca_dt(V_s_pre)
        self.soma.V_s = self.soma.V_s + ((-self.soma.I_na(V_s_pre) - self.soma.I_k(V_s_pre) - self.soma.I_Ca(V_s_pre) - self.soma.I_Leak(V_s_pre) - self.soma.I_Can(V_s_pre) - self.g_c1 * (self.soma.V_s - self.dend1.V_d_1) / self.P1 + self.I_syn_s) / self.Cm) * step
        ## dendritic 1
        self.dend1.h = self.dend1.h_dt(V_d_1_pre)
        self.dend1.q = self.dend1.q_dt(V_d_1_pre)
        self.dend1.r = self.dend1.r_dt(V_d_1_pre)
        self.dend1.V_d_1 = self.dend1.V_d_1 + ((-self.dend1.I_Nap(V_d_1_pre) - self.dend1.I_KS(V_d_1_pre) - self.dend1.I_Leak(V_d_1_pre)- self.g_c1 * (self.dend1.V_d_1 - V_s_pre) / self.P2 - self.g_c2 * (V_d_1_pre - V_d_2_pre) / self.P2+ self.I_syn_d1) / self.Cm) * step
        ## dendritic 2
        self.dend2.a = self.dend2.a_dt(V_d_2_pre)
        self.dend2.b = self.dend2.b_dt(V_d_2_pre)
        self.dend2.Ca = self.dend2.Ca_dt(V_d_2_pre)
        self.dend2.V_d_2 = self.dend2.V_d_2 + ((- self.dend2.I_A(V_d_2_pre) - self.dend2.I_Ca(V_d_2_pre) - self.dend2.I_Leak(V_d_2_pre)- self.g_c2 * (self.dend2.V_d_2 - V_d_1_pre) / (1 - self.P1 - self.P2) + self.I_syn_d2) / self.Cm) * step
        return self.soma.V_s, self.dend1.V_d_1, self.dend2.V_d_2, self.dend2.Ca

## synapse model##
class synapases():
    def __init__(self):
        self.E_syn_AMPA = 0.0
        self.V_Ca = 0
        self.tau_ca = 50
        self.G_AMPA = 0.23
        self.G_NMDA = 0.3
        self.tau_ampa = 1.6
        self.tau_nmda = 1.6

        self.mg = 1  ## magnesium concentration

    def I_NMDA(self, t_i, V_d_, g_list_nmda,preneuron_spikes_time):  ##计算突触电流
        g_syn = self.alphaf_g_syn_nmda(t_i,preneuron_spikes_time)
        I_nmda = - self.G_NMDA * (g_syn) * self.H_2002(V_d_)
        g_list_nmda.append(g_syn * self.B(V_d_))
        return I_nmda

    def I_AMPA(self, t_i, V_d_, g_list_ampa,preneuron_spikes_time,z):  ##计算突触电流
        I_ampa = -(self.G_AMPA*z * self.alphaf_g_syn_ampa(t_i, g_list_ampa,preneuron_spikes_time) * (V_d_ - self.E_syn_AMPA))
        return I_ampa

    def H_2002(self, V_d_):  ## 2002-PNAS-A unified model of NMDA receptor-dependent bidirectional synaptic plasticity
        hv = self.B(V_d_) * (V_d_ - self.V_Ca)
        return hv

    def B(self, V_d_):  ## 2002-PNAS-A unified model of NMDA receptor-dependent bidirectional synaptic plasticity
        bv = 1 / (1 + np.exp(-0.062 * (V_d_ - 60)) * (self.mg / 3.57))
        return bv

    def NMDA_Ca(self, t_i, Ca_pre, V_d_, g_list_nmda,preneuron_spikes_time):
        I_nmda_ = self.I_NMDA(t_i, V_d_, g_list_nmda,preneuron_spikes_time)
        Ca_level = (I_nmda_ - (1 / self.tau_ca) * Ca_pre) * step + Ca_pre
        return Ca_level, I_nmda_

    def alphaf_g_syn_nmda(self, t_i,preneuron_spikes_time):  ##alpha函数电导求和公式
        g_syn_t = 0
        for s in range(len(preneuron_spikes_time)):
            g_syn_t += self.g_syn_nmda(t_i, preneuron_spikes_time[s])  ##第s个preneuron spike time
        return g_syn_t

    def g_syn_nmda(self, t_i, preneuron_spike_time_no_s):  ##单个电导值实时计算
        if t_i * step - preneuron_spike_time_no_s < 0:
            return 0
        else:
            g_syn = ((t_i * step - preneuron_spike_time_no_s) / self.tau_nmda) * np.exp(
                - ((t_i * step - preneuron_spike_time_no_s) / self.tau_nmda))
        return g_syn

    def alphaf_g_syn_ampa(self, t_i, g_list_ampa,preneuron_spikes_time):  ##alpha函数电导求和公式
        g_syn_t = 0
        for s in range(len(preneuron_spikes_time)):
            g_syn_t += self.g_syn_ampa(t_i, preneuron_spikes_time[s])  ##第s个preneuron spike time
        g_list_ampa.append(g_syn_t)
        return g_syn_t

    def g_syn_ampa(self, t_i, preneuron_spike_time_no_s):  ##单个电导值实时计算
        if (t_i * step) - preneuron_spike_time_no_s < 0:
            return 0
        else:
            g_syn = ((t_i * step - preneuron_spike_time_no_s) / self.tau_ampa) * np.exp(
                - ((t_i * step - preneuron_spike_time_no_s) / self.tau_ampa))
        return g_syn

class STC_model():
    def __init__(self):
        # time parameter
        self.dt = 1  ##ms
        self.dt_stc = 0.1  ##s
        self.calcium_time_window = 500  ## calcium time window
        self.tag_prp_check_window = self.dt_stc * 1000  ##ms  step size of tag and prp

        # Weight change
        self.zh = 2
        self.zl = 0.5
        self.beta_z = 0.1
        self.tau_y = 0.15
        self.y_synapse = 0

        # PRP
        self.prp_tau_rise = 80  ## 900s
        self.prp_tau_decay = 9000
        self.prp_trigger = 1

        # Tag
        self.lambda_tag = 10.0  ## 是常数，是需要设置的，需要手工调节，目的是避免因Tag过小而y过小，进而导致z的变化过小
        self.alpha_tag = 0.0007  ## s-1
        self.beta_tag_synapse = 0
        self.beta_tag_LTP = 1.0  ## s-1
        self.beta_tag_LTD = 0.2  ## s-1

        ##calcium threshold for spine
        self.Ca0_synapse = 0.01
        self.Ca1_synapse = 0.2
        self.Ca0_dendritic = 0.025

        ## Flag for LTP or LTD
        self.Flag_synapse = 0

        ## Dopamine
        self.dopamine_novel = 1
        self.dopamine_nonovel = 0

    def Flag_beta_tag_synapse(self, synapse_Ca_NMDA):
        if synapse_Ca_NMDA < self.Ca0_synapse:
            self.Flag_synapse = 0  ##None
            self.beta_tag_synapse = 0
        elif synapse_Ca_NMDA > self.Ca1_synapse:
            self.Flag_synapse = 1  ##LTP
            self.beta_tag_synapse = self.beta_tag_LTP
        else:
            self.Flag_synapse = -1  ##LTD
            self.beta_tag_synapse = self.beta_tag_LTD
        return self.Flag_synapse, self.beta_tag_synapse

    def prp_x(self, dendritic_Ca_total_pre, prp_x1_pre, prp_x2_pre):
        if dendritic_Ca_total_pre > self.Ca0_dendritic:
            prp_x1_pre = prp_x1_pre + self.prp_trigger
            prp_x2_pre = prp_x2_pre + self.prp_trigger
        return prp_x1_pre, prp_x2_pre

    def PRP(self, dendritic_Ca_total_pre, prp_x1_pre, prp_x2_pre):
        prp_x1_pre_new, prp_x2_pre_new = self.prp_x(dendritic_Ca_total_pre, prp_x1_pre, prp_x2_pre)
        prp_x1_now = prp_x1_pre_new + self.dt_stc * (-prp_x1_pre_new / self.prp_tau_rise)
        prp_x2_now = prp_x2_pre_new + self.dt_stc * (-prp_x2_pre_new / self.prp_tau_decay)
        prp_now = (prp_x1_now - prp_x2_now) / (self.prp_tau_rise - self.prp_tau_decay)
        return prp_now,prp_x1_pre_new,prp_x1_now,prp_x2_pre_new,prp_x2_now

    def Tag_synapse(self, synapse_Ca_NMDA, tag_synapse_pre, dopamine):
        if dopamine == self.dopamine_novel:
            self.Ca0_dendritic = 0.05
        Flag_synapse, beta_tag_synapse = self.Flag_beta_tag_synapse(synapse_Ca_NMDA)
        tag_synapse_now = tag_synapse_pre + self.dt_stc * (
                    -self.alpha_tag * tag_synapse_pre + beta_tag_synapse * (Flag_synapse - tag_synapse_pre))
        return tag_synapse_now, Flag_synapse

    def y_synapse_dt(self, prp_pre, tag_synapse_pre):
        if prp_pre > 0:
            self.y_synapse = self.y_synapse + self.dt_stc * (prp_pre * tag_synapse_pre / self.tau_y)
        else:
            self.y_synapse = self.lambda_tag * tag_synapse_pre
        return self.y_synapse

    def z_synapse(self, prp_pre, tag_synapse_pre):
        y_synapse = self.y_synapse_dt(prp_pre, tag_synapse_pre)
        z = ((1 - self.zl) * self.zh * np.exp(self.beta_z * y_synapse) + self.zl * (self.zh - 1) * np.exp(
            self.beta_z * (-y_synapse))) / ((1 - self.zl) * np.exp(self.beta_z * y_synapse) + (self.zh - 1) * np.exp(
            self.beta_z * (-y_synapse)))
        return z, y_synapse

if __name__ == '__main__':
    tic = time.time()
    ##------------------initialize neuron and synapse model------------------##
    voltage = current_balance_multi_compart()
    synapse_weakHFS = synapases();synapse_strongHFS = synapases()

    STC_1 = STC_model()  ## weakHFS
    STC_2 = STC_model()  ## strongHFS
    ## -----------------------calculation preparation------------------##
    # calculation data
    step = step_neuron = 0.05
    thirty_min = 30*60*1000  ##30min后给第二个刺激
    ten_min = 10*60*1000  ##strong stimu 间隔时间
    time_neuron = 5*60*60*1000  ##模拟总时长:ms
    # time_neuron = 4000  ##ms
    t_neuron = np.arange(0,time_neuron,step)
    time_STC = np.arange(0, time_neuron, STC_1.dt)
    tag_prp_time = np.arange(0, np.ceil(time_neuron/STC_1.tag_prp_check_window), STC_1.dt)
    ## -----------------------pre_neuron_train parameter------------------##
    time_spike_train_one_weakHFS = 200  ## ms  一次刺激时长  200
    time_spike_train_one_strongHFS = 1000  ## ms  一次刺激时长  1000
    repeat_times = 1  ##为1则有刺激，为0则无刺激
    repeat_interval = 1000
    repeat_interval_weakHFS = 1000  # ms  重复给予刺激的间隔时间  1000
    repeat_interval_strongHFS = ten_min
    stimu_times = 1   ##900  重复给予刺激的次数
    stimu_Num_weakHFS = 1  ##一次实验包括了几次随机分布
    stimu_Num_strongHFS = 3  ##一次实验包括了几次随机分布
    trail_No = 10  ## 第几次实验（1-10）
    preneuron_firing_rate_weakHFS = 100  ##Hz
    preneuron_spikes_train_weakHFS, preneuron_spikes_time_weakHFS = Poisson_spike_100Hz.poisson_spike_train_H(trail_No-1,stimu_Num_weakHFS,time_spike_train_one_weakHFS,preneuron_firing_rate_weakHFS,repeat_times,repeat_interval)
    preneuron_firing_rate_strongHFS = 100  ##Hz
    preneuron_spikes_train_strongHFS, preneuron_spikes_time_strongHFS = Poisson_spike_100Hz.poisson_spike_train_H(trail_No-1,stimu_Num_strongHFS,time_spike_train_one_strongHFS,preneuron_firing_rate_strongHFS,repeat_times,repeat_interval)

    print("weakHFS number of pulses : " + str(np.shape(preneuron_spikes_time_weakHFS)))  ## (1, 20)
    print("strongHFS number of pulses : " + str(np.shape(preneuron_spikes_time_strongHFS)))  ## (3, 20)

    ## ------------------ initialize the storage matrix ------------------##
    V_s = -75.0 * np.ones(len(t_neuron));    V_d_1 = -75.0 * np.ones(len(t_neuron));   V_d_2 = -75.0 * np.ones(len(t_neuron))

    ## Ca
    soma_Ca_channel = np.zeros(len(t_neuron));   dend2_Ca_channel = np.zeros(len(t_neuron))
    synapse_Ca_NMDA_weakHFS = np.zeros(len(t_neuron));   synapse_Ca_NMDA_strongHFS = np.zeros(len(t_neuron))


    g_list_nmda_weakHFS = [0];                g_list_ampa_weakHFS = [0]
    g_list_nmda_strongHFS = [0];         g_list_ampa_strongHFS = [0]

    ## STC
    ## synapse 1 : weakHFS
    tag_synapse_1 = np.zeros(int(np.ceil(len(time_STC)/STC_1.tag_prp_check_window)))
    tag_flag_synapse_1 = np.zeros(int(np.ceil(len(time_STC)/STC_1.tag_prp_check_window)))
    z_synapse_1 = np.ones(int(np.ceil(len(time_STC) / STC_1.tag_prp_check_window)))
    y_synapse_1 = np.zeros(int(np.ceil(len(time_STC) / STC_1.tag_prp_check_window)))


    ## synapse 2 : strongHFS
    tag_synapse_2 = np.zeros(int(np.ceil(len(time_STC)/STC_2.tag_prp_check_window)))
    tag_flag_synapse_2 = np.zeros(int(np.ceil(len(time_STC)/STC_2.tag_prp_check_window)))
    z_synapse_2 = np.ones(int(np.ceil(len(time_STC) / STC_2.tag_prp_check_window)))
    y_synapse_2 = np.zeros(int(np.ceil(len(time_STC) / STC_2.tag_prp_check_window)))
    prp_x1 = np.zeros(int(np.ceil(len(time_STC) / STC_1.tag_prp_check_window)))
    prp_x2 = np.zeros(int(np.ceil(len(time_STC) / STC_1.tag_prp_check_window)))
    prp = np.zeros(int(np.ceil(len(time_STC) / STC_2.tag_prp_check_window)))


    ## Ca浓度平均值
    dend2_Ca_channel_average = np.zeros(len(time_STC))
    synapse_Ca_NMDA_strongHFS_average = np.zeros(len(time_STC))
    synapse_Ca_NMDA_weakHFS_average = np.zeros(len(time_STC))
    ## Tag_flag 绘图
    time_1_hour = np.arange(0, time_neuron)
    firing_rate = np.zeros(time_neuron)  ##1ms步长
    spike_time = []

    # t_neuron_sub = np.arange(0, np.ceil(time_neuron / STC_1.tag_prp_check_window), STC_1.dt)
    # synapse_Ca_NMDA_strongHFS_sub = np.zeros(int(np.ceil(len(time_STC) / STC_2.tag_prp_check_window)))
    # synapse_Ca_NMDA_weakHFS_sub = np.zeros(int(np.ceil(len(time_STC) / STC_2.tag_prp_check_window)))
    # dend2_Ca_channel_sub = np.zeros(int(np.ceil(len(time_STC) / STC_2.tag_prp_check_window)))

    ## ----------calculation function of voltage and current----------- ##
    def calculation():
        print("*************start***************")
        time_ = 1
        V_calcutalte_completed = 0
        I_nmda_weakHFS_pre = I_nmda_weakHFS_now = 0
        I_ampa_weakHFS_pre = I_ampa_weakHFS_now = 0
        I_nmda_strongHFS_pre = I_nmda_strongHFS_now = 0
        I_ampa_strongHFS_pre = I_ampa_strongHFS_now = 0
        ## start iterative calculation（ run_time/step times）
        for t_i in range(STC_1.calcium_time_window,len(t_neuron)):
            num_strong = int((t_i*step-STC_1.calcium_time_window - thirty_min - (stimu_Num_weakHFS) * (repeat_interval_weakHFS))/ten_min)
            if t_i * step < (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + STC_1.calcium_time_window*2:
                time_ = 1
                count = int((t_i * step - STC_1.calcium_time_window) / repeat_interval_weakHFS)  ##0-n次刺激,repeat_interval = 1000ms/1次
                if count >= stimu_Num_weakHFS:
                    count = stimu_Num_weakHFS - 1
                    print("count: " + str(count))
                    print("=========================== error ==================================")
                ### weakHFS
                V_s[t_i],V_d_1[t_i],V_d_2[t_i],dend2_Ca_channel[t_i] = voltage.V_update(t_i,V_s[t_i-1],V_d_1[t_i-1],V_d_2[t_i-1])
                synapse_Ca_NMDA_weakHFS[t_i],I_nmda_weakHFS_now = synapse_weakHFS.NMDA_Ca(t_i,synapse_Ca_NMDA_weakHFS[t_i - 1],V_d_2[t_i - 1],g_list_nmda_weakHFS,(np.array(preneuron_spikes_time_weakHFS[count]) + repeat_interval_weakHFS * count + STC_1.calcium_time_window).tolist())
                I_ampa_weakHFS_now = synapse_weakHFS.I_AMPA(t_i,V_d_2[t_i],g_list_ampa_weakHFS,(np.array(preneuron_spikes_time_weakHFS[count]) + repeat_interval_weakHFS * count + STC_1.calcium_time_window).tolist(),z_synapse_1[int(np.ceil((t_i*step) / STC_1.tag_prp_check_window))-1])
                voltage.I_syn_d2 = I_nmda_weakHFS_now + I_ampa_weakHFS_now + I_nmda_strongHFS_now + I_ampa_strongHFS_now
                I_ampa_weakHFS_pre = I_ampa_weakHFS_now
                I_nmda_weakHFS_pre = I_nmda_weakHFS_now
            elif num_strong in np.arange(0, stimu_Num_strongHFS).tolist() \
                    and (t_i * step - num_strong * ten_min - STC_1.calcium_time_window - thirty_min - stimu_Num_weakHFS * repeat_interval_weakHFS) < repeat_interval * 2\
                    and t_i * step > thirty_min + STC_1.calcium_time_window + repeat_interval_weakHFS * stimu_Num_weakHFS + 100:  ##30min 后给另一个刺激
                time_ = 2
                count = int((t_i * step - STC_1.calcium_time_window - thirty_min - stimu_Num_weakHFS * repeat_interval_weakHFS) / ten_min)  ##0-n次刺激,repeat_interval_strongHFS = 10min/1次
                if count >= stimu_Num_strongHFS:
                    count = stimu_Num_strongHFS -1
                    print("count: " + str(count))
                    print("=========================== error ==================================")
                ## strongHFS
                V_s[t_i],V_d_1[t_i],V_d_2[t_i],dend2_Ca_channel[t_i] = voltage.V_update(t_i,V_s[t_i-1],V_d_1[t_i-1],V_d_2[t_i-1])
                synapse_Ca_NMDA_strongHFS[t_i],I_nmda_strongHFS_now = synapse_strongHFS.NMDA_Ca(t_i,synapse_Ca_NMDA_strongHFS[t_i - 1],V_d_2[t_i - 1],g_list_nmda_strongHFS,(np.array(preneuron_spikes_time_strongHFS[count])+ thirty_min + ten_min * count +STC_1.calcium_time_window + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)).tolist())
                I_ampa_strongHFS_now = synapse_strongHFS.I_AMPA(t_i,V_d_2[t_i],g_list_ampa_strongHFS,(np.array(preneuron_spikes_time_strongHFS[int(count)])+ thirty_min + ten_min * count + STC_1.calcium_time_window + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)).tolist(),z_synapse_2[int(np.ceil((t_i*step) / STC_1.tag_prp_check_window))-1])
                voltage.I_syn_d2 = I_nmda_weakHFS_now + I_ampa_weakHFS_now + I_nmda_strongHFS_now + I_ampa_strongHFS_now
                I_ampa_strongHFS_pre = I_ampa_strongHFS_now
                I_nmda_strongHFS_pre = I_nmda_strongHFS_now
            else:
                if V_calcutalte_completed == 0:
                    V_calcutalte_completed = 1
                    time_ = 3
                    V_s[t_i::],V_d_1[t_i::],V_d_2[t_i::],dend2_Ca_channel[t_i::] = V_s[t_i-1],V_d_1[t_i-1],V_d_2[t_i-1],dend2_Ca_channel[t_i-1]
                    synapse_Ca_NMDA_weakHFS[t_i::],I_nmda_weakHFS_now = synapse_Ca_NMDA_weakHFS[t_i-1],I_nmda_weakHFS_pre
                    synapse_Ca_NMDA_strongHFS[t_i::], I_nmda_strongHFS_now = synapse_Ca_NMDA_strongHFS[t_i - 1], I_nmda_strongHFS_pre
                    I_nmda_weakHFS_pre = I_nmda_weakHFS_now
                    I_nmda_strongHFS_pre = I_nmda_strongHFS_now
            if len(spike_time) == 0:
                spike_time_pre = 0
            else:
                spike_time_pre = spike_time[-1]
            if V_s[t_i-1] >= V_s[t_i-2] and V_s[t_i-1]  >= 0 and V_s[t_i-2] <= 0 and (((t_i - 1) * step - spike_time_pre) >= 5):  ##给刺激的1000ms
                spike_time.append((t_i-1) * step_neuron)
            if np.mod(t_i*step, STC_1.tag_prp_check_window) == 0.0:  # tag——prp时间窗长度,更新Tag/PRP/y/z
                print("current time (ms):      " + str(t_i * step) + "     total time (ms) : " + str(time_neuron))
                dend2_Ca_channel_average[int(t_i*step)] = sum(dend2_Ca_channel[int(t_i-STC_1.tag_prp_check_window/step):t_i]) / int(STC_1.tag_prp_check_window / step_neuron)
                synapse_Ca_NMDA_strongHFS_average[int(t_i*step)] = sum(synapse_Ca_NMDA_strongHFS[int(t_i-STC_1.tag_prp_check_window/step):t_i]) / int(STC_1.tag_prp_check_window / step_neuron)
                synapse_Ca_NMDA_weakHFS_average[int(t_i*step)] = sum(synapse_Ca_NMDA_weakHFS[int(t_i-STC_1.tag_prp_check_window/step):t_i]) / int(STC_1.tag_prp_check_window / step_neuron)
                print(z_synapse_1[int(np.ceil((t_i*step) / STC_1.tag_prp_check_window))-1])
                print(z_synapse_2[int(np.ceil((t_i*step) / STC_1.tag_prp_check_window))-1])
                print('syn_strongHFS、syn_weakHFS、dend')
                print(synapse_Ca_NMDA_strongHFS_average[int(t_i*step)])
                print(synapse_Ca_NMDA_weakHFS_average[int(t_i * step)])
                print(dend2_Ca_channel_average[int(t_i * step)])
                print('---------------------------')
                j = int(np.ceil((t_i*step) / STC_1.tag_prp_check_window))
                # dend2_Ca_channel_sub[j] = dend2_Ca_channel[t_i]
                # synapse_Ca_NMDA_strongHFS_sub[j] = synapse_Ca_NMDA_strongHFS[t_i]
                # synapse_Ca_NMDA_weakHFS_sub[j] = synapse_Ca_NMDA_weakHFS[t_i]
                ## synapse 1 : weakHFS      synapse 2 : strongHFS
                tag_synapse_2[j], tag_flag_synapse_2[j] = STC_2.Tag_synapse(synapse_Ca_NMDA_strongHFS_average[int(t_i*step)], tag_synapse_2[j - 1],STC_2.dopamine_nonovel)
                tag_synapse_1[j], tag_flag_synapse_1[j] = STC_1.Tag_synapse(synapse_Ca_NMDA_weakHFS_average[int(t_i*step)],tag_synapse_1[j - 1], STC_1.dopamine_nonovel)
                prp[j],prp_x1[j-1],prp_x1[j],prp_x2[j-1],prp_x2[j] = STC_1.PRP(dend2_Ca_channel_average[int(t_i * step)],prp_x1[j - 1], prp_x2[j - 1])
                z_synapse_1[j], y_synapse_1[j] = STC_1.z_synapse(prp[j], tag_synapse_1[j])
                z_synapse_2[j], y_synapse_2[j] = STC_2.z_synapse(prp[j], tag_synapse_2[j])
            ## 计算发放率
            if t_i * step >= 1000 and np.mod(t_i * step - STC_1.calcium_time_window, 1000) == 0.0:
                if time_ == 1:
                    time_spike_train_one = time_spike_train_one_weakHFS
                elif time_ == 2:
                    time_spike_train_one = time_spike_train_one_strongHFS
                else:
                    time_spike_train_one = time_spike_train_one_weakHFS
                firing_rate_value = len(spike_time)  ##
                firing_rate[int(t_i * step - 1000):int(t_i * step)] = firing_rate_value * (1000 / time_spike_train_one)
                if t_i * step == 1500:
                    f = open("cross-capture1_weakHFS+strongHFS_10/firing rate.txt", "a")
                    f.write("firing rate -- spike time (weak HFS) : " + str(spike_time))
                    f.write("\n")
                    f.write("firing_rate_value * (1000/time_spike_train_one) (weak HFS) : " + str(firing_rate_value * (1000/time_spike_train_one)))
                    f.write("\n")
                    f.write("PRP: " + str(prp))
                    f.write("\n")
                    f.close()
                spike_time.clear()
                print("当前时间：  " + str(np.round(t_i * step)) + "ms" + "       进度： " + str(np.round(t_i * step / time_neuron * 100)) + "%")

        print("**************end****************")

        toc = time.time()
        print("Biological time : " + str(time_neuron / 1000) + "s" + "   calculating time : " + str(round(toc - tic, 3)) + "s")
        f = open("cross-capture1_weakHFS+strongHFS_10/运行时间.txt", "a")
        f.write("Biological time : " + str(time_neuron / 1000) + "s" + "   calculating time : " + str(
            round(toc - tic, 3)) + "s")
        f.write("\n")
        f.close()

    ## -------------start iteration calculation from here------------- ##
    calculation()

    resultH5 = h5py.File('cross-capture1_weakHFS+strongHFS_10.hdf5', 'w')
    resultH5.create_dataset(name='V_s', data=(np.array(V_s[0:int((STC_1.calcium_time_window*2 + stimu_Num_weakHFS * repeat_interval + thirty_min + int((stimu_Num_strongHFS - 1) * repeat_interval_strongHFS))/step)])).tolist())
    resultH5.create_dataset(name='V_d_1', data=(np.array(V_d_1[0:int((STC_1.calcium_time_window*2 + stimu_Num_weakHFS * repeat_interval + thirty_min + int((stimu_Num_strongHFS - 1) * repeat_interval_strongHFS))/step)])).tolist())
    resultH5.create_dataset(name='V_d_2', data=(np.array(V_d_2[0:int((STC_1.calcium_time_window*2 + stimu_Num_weakHFS * repeat_interval + thirty_min + int((stimu_Num_strongHFS - 1) * repeat_interval_strongHFS))/step)])).tolist())
    resultH5.create_dataset(name='soma_Ca_channel', data=(np.array(soma_Ca_channel[0:int((STC_1.calcium_time_window*2 + stimu_Num_weakHFS * repeat_interval + thirty_min + int((stimu_Num_strongHFS - 1) * repeat_interval_strongHFS))/step)])).tolist())
    resultH5.create_dataset(name='dend_Ca_channel', data=(np.array(dend2_Ca_channel[0:int((STC_1.calcium_time_window*2 + stimu_Num_weakHFS * repeat_interval + thirty_min + int((stimu_Num_strongHFS - 1) * repeat_interval_strongHFS))/step)])).tolist())
    resultH5.create_dataset(name='synapse_Ca_NMDA_strongHFS', data=(np.array(synapse_Ca_NMDA_strongHFS[0:int((STC_1.calcium_time_window*2 + stimu_Num_weakHFS * repeat_interval + thirty_min + int((stimu_Num_strongHFS - 1) * repeat_interval_strongHFS))/step)])).tolist())
    resultH5.create_dataset(name='synapse_Ca_NMDA_weakHFS', data=(np.array(synapse_Ca_NMDA_weakHFS[0:int((STC_1.calcium_time_window*2 + stimu_Num_weakHFS * repeat_interval + thirty_min + int((stimu_Num_strongHFS - 1) * repeat_interval_strongHFS))/step)])).tolist())
    # resultH5.create_dataset(name='dend_Ca_channel_sub', data=np.round(np.array(dend2_Ca_channel_sub),2).tolist())
    # resultH5.create_dataset(name='synapse_Ca_NMDA_strongHFS_sub', data=np.round(np.array(synapse_Ca_NMDA_strongHFS_sub),2).tolist())
    # resultH5.create_dataset(name='synapse_Ca_NMDA_weakHFS_sub', data=np.round(np.array(synapse_Ca_NMDA_weakHFS_sub),2).tolist())
    resultH5.create_dataset(name='z_synapse_strongHFS', data=(np.array(z_synapse_1)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='z_synapse_weakHFS', data=(np.array(z_synapse_2)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='prp', data=(np.array(prp)).tolist())
    resultH5.create_dataset(name='tag_synapse_strongHFS', data=(np.array(tag_synapse_1)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='tag_synapse_weakHFS', data=(np.array(tag_synapse_2)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='y_synapse_strongHFS', data=(np.array(y_synapse_1)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='y_synapse_weakHFS', data=(np.array(y_synapse_2)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='tag_flag_strongHFS', data=(np.array(tag_flag_synapse_1)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='tag_flag_weakHFS', data=(np.array(tag_flag_synapse_2)).tolist())### strong 和 weak的z、tag、prp、y数据存反了，后期再进行训练需要更改一下
    resultH5.create_dataset(name='firing_rate_list', data=(np.array(firing_rate)).tolist())
    resultH5.close()



    # # Visualization
    # fig = plt.figure(figsize=(6, 5))
    # ax3 = fig.add_subplot(111)
    # time_scale_tag_prp = (1000 / STC_1.tag_prp_check_window) * 60
    # ax4 = ax3.twinx()
    # lns66 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_2, label='z synapse(strong HFS stimuli)',color='black')  ##familiar
    # lns6 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_1, '--', label='z synapse(weak HFS stimuli)',color='black')  ##novel
    # lns33 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_2, label='tag synapse(strong HFS stimuli)',color='red')
    # lns3 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_1, '--', label='tag synapse(weak HFS stimuli)', color='red')
    # lns4 = ax4.plot(tag_prp_time / time_scale_tag_prp, prp*1000, label='PRP', color='b')
    # # lns5 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_1, label='y_synapse_strongHFS stimuli', color='y')
    # # lns55 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_2, '--', label='y_synapse_weakHFS stimuli',color='y')
    # ax4.set_ylabel("Amount(tag PRP)")
    # ax3.set_xlabel("Time (min)")
    # ax3.set_ylabel('z synapse', color='black')
    # lns = lns3 + lns33 + lns4 + lns6 + lns66
    # labs = [l.get_label() for l in lns]
    # ax3.legend(lns, labs, loc=1)
    # plt.title(" synaptic weight changes")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-1.jpg',bbox_inches = 'tight')
    # plt.close()
    # # # plt.show()
    #
    # # Visualization
    # fig = plt.figure(figsize=(6, 5))
    # ax3 = fig.add_subplot(211)
    # time_scale_tag_prp = (1000 / STC_1.tag_prp_check_window) * 60
    # ax4 = ax3.twinx()
    # lns66 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_2, label='z synapse(strong HFS stimuli)',color='black')  ##familiar
    # lns6 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_1, '--', label='z synapse(weak HFS stimuli)',color='black')  ##novel
    # lns33 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_2, label='tag synapse(strong HFS stimuli)',color='red')
    # lns3 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_1, '--', label='tag synapse(weak HFS stimuli)', color='red')
    # lns4 = ax4.plot(tag_prp_time / time_scale_tag_prp, prp*1000, label='PRP', color='b')
    # # lns5 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_1, label='y_synapse_strongHFS stimuli', color='y')
    # # lns55 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_2, '--', label='y_synapse_weakHFS stimuli',color='y')
    # ax4.set_ylabel("Amount(tag PRP)")
    # ax3.set_xlabel("Time (min)")
    # ax3.set_ylabel('z synapse', color='black')
    # lns = lns3 + lns33 + lns4 +  lns6 + lns66
    # labs = [l.get_label() for l in lns]
    # ax3.legend(lns, labs, loc=1)
    # plt.title(" synaptic weight changes")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-1_1_prp1000.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    # fig = plt.figure(figsize=(6, 5))
    # ax3 = fig.add_subplot(211)
    # time_scale_tag_prp = (1000 / STC_1.tag_prp_check_window) * 60
    # ax4 = ax3.twinx()
    # lns66 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_2, label='z synapse(strong HFS stimuli)',color='black')  ##familiar
    # lns6 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_1, '--', label='z synapse(weak HFS stimuli)',color='black')  ##novel
    # lns33 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_2, label='tag synapse(strong HFS stimuli)',color='red')
    # lns3 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_1, '--', label='tag synapse(weak HFS stimuli)', color='red')
    # lns4 = ax4.plot(tag_prp_time / time_scale_tag_prp, prp*100, label='PRP', color='b')
    # # lns5 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_1, label='y_synapse_strongHFS stimuli', color='y')
    # # lns55 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_2, '--', label='y_synapse_weakHFS stimuli',color='y')
    # ax4.set_ylabel("Amount(tag PRP)")
    # ax3.set_xlabel("Time (min)")
    # ax3.set_ylabel('z synapse', color='black')
    # lns = lns3 + lns33 + lns4 +  lns6 + lns66
    # labs = [l.get_label() for l in lns]
    # ax3.legend(lns, labs, loc=1)
    # plt.title(" synaptic weight changes")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-1_2_prp100.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    # fig = plt.figure(figsize=(6, 5))
    # ax3 = fig.add_subplot(211)
    # time_scale_tag_prp = (1000 / STC_1.tag_prp_check_window) * 60
    # ax4 = ax3.twinx()
    # lns66 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_2, label='z synapse(strong HFS stimuli)',color='black')  ##familiar
    # lns6 = ax3.plot(tag_prp_time / time_scale_tag_prp, z_synapse_1, '--', label='z synapse(weak HFS stimuli)',color='black')  ##novel
    # lns33 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_2, label='tag synapse(strong HFS stimuli)',color='red')
    # lns3 = ax4.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_1, '--', label='tag synapse(weak HFS stimuli)', color='red')
    # lns4 = ax4.plot(tag_prp_time / time_scale_tag_prp, prp*10, label='PRP', color='b')
    # # lns5 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_1, label='y_synapse_strongHFS stimuli', color='y')
    # # lns55 = ax4.plot(tag_prp_time / time_scale_tag_prp, y_synapse_2, '--', label='y_synapse_weakHFS stimuli',color='y')
    # ax4.set_ylabel("Amount(tag PRP)")
    # ax3.set_xlabel("Time (min)")
    # ax3.set_ylabel('z synapse', color='black')
    # lns = lns3 + lns33 + lns4 +  lns6 + lns66
    # labs = [l.get_label() for l in lns]
    # ax3.legend(lns, labs, loc=1)
    # plt.title(" synaptic weight changes")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-1_3_prp10.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    # fig = plt.figure(figsize=(6, 5))
    # ax1 = fig.add_subplot(211)
    # ax2 = ax1.twinx()
    # time_scale_tag_prp = (1000 / STC_1.tag_prp_check_window) * 60
    # lns1 = ax1.plot(tag_prp_time / time_scale_tag_prp, z_synapse_1, label='z synapse', color='black')
    # lns2 = ax2.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_1, label='tag synapse', color='red')
    # lns3 = ax2.plot(tag_prp_time / time_scale_tag_prp, prp*100, label='PRP', color='b')
    # # lns4 = ax2.plot(tag_prp_time / time_scale_tag_prp, y_synapse_1, label='y_synapse', color='y')
    # ax1.set_xlabel("Time (min)")
    # ax1.set_ylabel('z synapse(weak HFS)', color='black')
    # ax2.set_ylabel("Amount(tag PRP)")
    # # ax1.set_ybound(0.9, 1.55)
    # lns = lns2 + lns3 + lns1
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=1)
    # plt.title("synaptic weight changes(synapse1 weak HFS stimuli)")
    # plt.tight_layout(0.1)
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # time_scale_tag_prp = (1000 / STC_1.tag_prp_check_window) * 60
    # lns1 = ax1.plot(tag_prp_time / time_scale_tag_prp, z_synapse_2, label='z synapse',color='black')
    # lns2 = ax2.plot(tag_prp_time / time_scale_tag_prp, tag_synapse_2, label='tag synapse', color='red')
    # lns3 = ax2.plot(tag_prp_time / time_scale_tag_prp, prp*100, '--', label='PRP', color='b')
    # # lns4 = ax2.plot(tag_prp_time / time_scale_tag_prp, y_synapse_2, label='y synapse', color='y')
    # ax1.set_xlabel("Time (min)")
    # ax1.set_ylabel('z synapse(strong HFS)', color='black')
    # ax2.set_ylabel("Amount(tag PRP)")
    # # ax1.set_ylim(0.9, 1.55)
    # lns = lns2 + lns3 + lns1
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=1)
    # plt.title("synaptic weight changes(synapse2 strong HFS stimuli)")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-3_prp100.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    # ## After the calculation，save the result as H5 file
    #
    #
    # step = 0.05
    # dend2_Ca_channel_average = np.zeros(5 * 60 * 60 * 1000)
    # synapse_Ca_NMDA_weakHFS_average = np.zeros(5 * 60 * 60 * 1000)
    # synapse_Ca_NMDA_strongHFS_average = np.zeros(5 * 60 * 60 * 1000)
    #
    # for i in range(STC_1.calcium_time_window + stimu_Num_weakHFS * repeat_interval_weakHFS + thirty_min + (stimu_Num_strongHFS-1) * repeat_interval_strongHFS + time_spike_train_one_strongHFS + repeat_interval):
    #     if i < STC_1.tag_prp_check_window + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + repeat_interval \
    #         or (i >=  STC_1.tag_prp_check_window + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + thirty_min):
    #         if i >= 100:
    #             dend2_Ca_channel_average[i] = sum(dend2_Ca_channel[int((i - STC_1.tag_prp_check_window) / step):int(i / step)]) / int(STC_1.tag_prp_check_window / step)
    #             synapse_Ca_NMDA_weakHFS_average[i] = sum(synapse_Ca_NMDA_weakHFS[int((i - STC_1.tag_prp_check_window) / step):int(i / step)]) / int(STC_1.tag_prp_check_window / step)
    #             synapse_Ca_NMDA_strongHFS_average[i] = sum(synapse_Ca_NMDA_strongHFS[int((i - STC_1.tag_prp_check_window) / step):int(i / step)]) / int(STC_1.tag_prp_check_window / step)
    #         else:
    #             dend2_Ca_channel_average[i] = sum(dend2_Ca_channel[0:int(i / step)]) / int(STC_1.tag_prp_check_window / step)
    #             synapse_Ca_NMDA_weakHFS_average[i] = sum(synapse_Ca_NMDA_weakHFS[0:int(i / step)]) / int(STC_1.tag_prp_check_window / step)
    #             synapse_Ca_NMDA_strongHFS_average[i] = sum(synapse_Ca_NMDA_strongHFS[0:int(i / step)]) / int(STC_1.tag_prp_check_window / step)
    #
    # t_average = np.arange(0, len(dend2_Ca_channel_average), 1)
    # fig = plt.figure(figsize=(5.8, 4.0))
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(t_average[0:1500] - STC_1.calcium_time_window, synapse_Ca_NMDA_strongHFS_average[0:1500],
    #                 label=' synapse Ca NMDA-strong HFS stimuli ', color="r")
    # lns2 = ax1.plot(t_average[0:1500] - STC_1.calcium_time_window, synapse_Ca_NMDA_weakHFS_average[0:1500],
    #                 label='  synapse Ca NMDA-weak HFS stimuli ', color="b")
    # lns3 = ax2.plot(t_average[0:1500] - STC_1.calcium_time_window, dend2_Ca_channel_average[0:1500], '--',
    #                 label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(100ms average)")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # ax1 = fig.add_subplot(211)
    # ax2 = ax1.twinx()
    # for i in range(len(V_d_2[0:int(1500/step)])-1):
    #     if i >0 and V_d_2[i] >= V_d_2[i - 1]  and V_d_2[i] >= V_d_2[i+1] and V_d_2[i] >= -40:
    #         plt.axvline(x = (i*step-STC_1.calcium_time_window),ls = '--',c = 'y',linewidth = 1.0)
    # lns1 = ax1.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window,
    #                 synapse_Ca_NMDA_strongHFS[0:int(1500 / step)],
    #                 label=' synapse Ca NMDA-strong HFS stimuli ', color="r")
    # lns2 = ax1.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window,
    #                 synapse_Ca_NMDA_weakHFS[0:int(1500 / step)],
    #                 label=' synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window, dend2_Ca_channel[0:int(1500 / step)], '--',
    #                 label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-Ca_average_weakHFS_1.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    #
    # fig = plt.figure(figsize=(5.8, 4.0))
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 synapse_Ca_NMDA_strongHFS_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS):int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + 1500)],
    #                 label=' synapse Ca NMDA-strong HFS stimuli ', color="r")
    # lns2 = ax1.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 synapse_Ca_NMDA_weakHFS_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS):int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + 1500)],
    #                 label='  synapse Ca NMDA-weak HFS stimuli ', color="b")
    # lns3 = ax2.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 dend2_Ca_channel_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS):int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + 1500)], '--',
    #                 label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(100ms average)")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # ax1 = fig.add_subplot(211)
    # ax2 = ax1.twinx()
    # for i in range(len(V_s[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + 1500) / step)]) - 1):
    #     if i > 0 and V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step)] >= V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) - 1] and V_s[
    #         i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step)] >= V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + 1] and V_s[
    #         i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step)] >= -40:
    #         plt.axvline(x=(i * step - STC_1.calcium_time_window), ls='--', c='y', linewidth=1.0)
    # lns1 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_strongHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)],
    #     label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_weakHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)],
    #     label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     dend2_Ca_channel[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], '--',
    #     label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca_NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-Ca_average_strongHFS_1.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    # fig = plt.figure(figsize=(5.8, 4.0))
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 synapse_Ca_NMDA_strongHFS_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)+ten_min:int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) +ten_min + 1500)],
    #                 label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 synapse_Ca_NMDA_weakHFS_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min:int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min + 1500)],
    #                 label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 dend2_Ca_channel_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min:int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min + 1500)], '--',
    #                 label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(100ms average)")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # ax1 = fig.add_subplot(211)
    # ax2 = ax1.twinx()
    # for i in range(len(V_s[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step):int(((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) + 1500) / step)]) - 1):
    #     if i > 0 and V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step)] >= V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) - 1] and V_s[
    #         i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step)] >= V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) + 1] and V_s[
    #         i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step)] >= -40:
    #         plt.axvline(x=(i * step - STC_1.calcium_time_window), ls='--', c='y', linewidth=1.0)
    # lns1 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_strongHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) + int(1500 / step)],
    #     label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_weakHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) + int(1500 / step)],
    #     label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     dend2_Ca_channel[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) + int(1500 / step)], '--',
    #     label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca_NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-Ca_average_strongHFS_2.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    # fig = plt.figure(figsize=(5.8, 4.0))
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 synapse_Ca_NMDA_strongHFS_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)+ten_min*2:int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) +ten_min*2 + 1500)],
    #                 label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 synapse_Ca_NMDA_weakHFS_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2:int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2 + 1500)],
    #                 label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(t_average[thirty_min:int(thirty_min + 1500)] - thirty_min - 500,
    #                 dend2_Ca_channel_average[thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2:int(thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2 + 1500)], '--',
    #                 label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(100ms average)")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # ax1 = fig.add_subplot(211)
    # ax2 = ax1.twinx()
    # for i in range(len(V_s[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step):int(((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) + 1500) / step)]) - 1):
    #     if i > 0 and V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step)] >= V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) - 1] and V_s[
    #         i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step)] >= V_s[i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) + 1] and V_s[
    #         i + int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step)] >= -40:
    #         plt.axvline(x=(i * step - STC_1.calcium_time_window), ls='--', c='y', linewidth=1.0)
    # lns1 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_strongHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) + int(1500 / step)],
    #     label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_weakHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) + int(1500 / step)],
    #     label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     dend2_Ca_channel[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) + int(1500 / step)], '--',
    #     label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca_NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration")
    # ax1.grid()
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-Ca_average_strongHFS_3.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    # fig = plt.figure(figsize=(10, 5))
    # # membrane potention   ::  neuron with high CREB level
    # time_scale = 1000 * 60
    # bax = brokenaxes(xlims=((-2, 70), (270, 302)), hspace=.05, despine=True)
    # bax.plot(time_1_hour / time_scale, firing_rate)
    # bax.set_ylabel("firing rate (Hz)")
    # bax.set_xlabel("time (min)")
    # bax.set_title("postneuron firing rate")
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-8.jpg')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax1 = fig.add_subplot(212)
    # plt.plot(t_neuron[0:int(1500 / step)] - 500, V_s[0:int(1500 / step)], label=' soma', color="blue")
    # plt.plot(t_neuron[0:int(1500 / step)] - 500, V_d_1[0:int(1500 / step)], "--", label=' dend1', color="red")
    # plt.plot(t_neuron[0:int(1500 / step)] - 500, V_d_2[0:int(1500 / step)], "--", label=' dend2', color="black")
    # ax1.set_ylabel("Voltage (mV)")
    # ax1.set_xlabel("time (ms)")
    # plt.title("postneuron spike train")
    # plt.xlim(0, 240)
    # plt.legend(loc = 1)
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-9.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    #
    # fig = plt.figure(figsize=(6, 4.5))
    # ax1 = fig.add_subplot(211)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window,
    #                 synapse_Ca_NMDA_strongHFS[0:int(1500 / step)],
    #                 label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window,
    #                 synapse_Ca_NMDA_weakHFS[0:int(1500 / step)],
    #                 label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window, dend2_Ca_channel[0:int(1500 / step)], '--',
    #                 label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca_NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(weak HFS)")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-15.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    # fig = plt.figure(figsize=(6, 4.5))
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_strongHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)],
    #     label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_weakHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)],
    #     label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     dend2_Ca_channel[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], '--',
    #     label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(strong HFS)")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-16_1.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    # fig = plt.figure(figsize=(6, 4.5))
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_strongHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) + int(1500 / step)],
    #     label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_weakHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) + int(1500 / step)],
    #     label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     dend2_Ca_channel[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min) / step) + int(1500 / step)], '--',
    #     label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(strong HFS)")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-16_2.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    # fig = plt.figure(figsize=(6, 4.5))
    # ax1 = fig.add_subplot(212)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_strongHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) + int(1500 / step)],
    #     label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     synapse_Ca_NMDA_weakHFS[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) + int(1500 / step)],
    #     label='  synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     dend2_Ca_channel[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS) + ten_min*2) / step) + int(1500 / step)], '--',
    #     label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration(strong HFS)")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-16_3.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    #
    # fig = plt.figure(figsize=(6, 4.5))
    # ax1 = fig.add_subplot(211)
    # ax2 = ax1.twinx()
    # lns1 = ax1.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window,
    #                 synapse_Ca_NMDA_strongHFS[0:int(1500 / step)],
    #                 label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # lns2 = ax1.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window,
    #                 synapse_Ca_NMDA_weakHFS[0:int(1500 / step)],
    #                 label=' synapse Ca NMDA-weak HFS stimuli', color="b")
    # lns3 = ax2.plot(t_neuron[0:int(1500 / step)] - STC_1.calcium_time_window, dend2_Ca_channel[0:int(1500 / step)], '--',
    #                 label=' dend Ca channel', color="black")
    # plt.xlim(0, 1000)
    # ax1.set_xlabel("time (ms)")
    # ax1.set_ylabel("synapse Ca NMDA (μmol/L)", color="black")
    # ax2.set_ylabel('dend Ca channel (μmol/L)', color="black")
    # plt.title("changes in calcium concentration")
    # plt.tight_layout(0.1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-17.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    # # fig = plt.figure(figsize=(10, 5))
    # # bax = brokenaxes(xlims=((-2, 70), (270, 302)), hspace=.05, despine=True)
    # # time_scale_sub = (1000 / STC_1.tag_prp_check_window) * 60
    # # lns1 = bax.plot(t_neuron_sub / time_scale_sub, synapse_Ca_NMDA_strongHFS_sub,
    # #                 label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # # lns2 = bax.plot(t_neuron_sub / time_scale_sub, synapse_Ca_NMDA_weakHFS_sub,
    # #                 label=' synapse Ca NMDA-weak HFS stimuli', color="b")
    # # lns3 = bax.plot(t_neuron_sub / time_scale_sub, dend2_Ca_channel_sub, '--',
    # #                 label=' dend Ca channel',
    # #                 color="black")
    # # bax.set_xlabel("time (min)")
    # # bax.set_ylabel(" Ca concentration (μmol/L)")
    # # bax.legend(loc=1)
    # # bax.set_title("changes in calcium concentration")
    # # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-18.jpg')
    # # # plt.show()
    #
    # fig = plt.figure(figsize=(10, 5))
    # # membrane potention   ::  neuron with high CREB level
    # time_scale = 1000 * 60
    # bax = brokenaxes(xlims=((-2, 70), (270, 302)), hspace=.05, despine=True)
    # bax.plot(time_1_hour / time_scale, firing_rate)
    # bax.set_ylabel("firing rate (Hz)")
    # bax.set_xlabel("time (min)")
    # bax.set_title("postneuron firing rate")
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-19.jpg')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax1 = fig.add_subplot(212)
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_s[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], label=' soma', color="blue")
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_d_1[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], "--", label=' dend1', color="red")
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_d_2[int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], "--", label=' dend2', color="black")
    # ax1.set_ylabel("Voltage (mV)")
    # ax1.set_xlabel("time (ms)")
    # plt.title("postneuron spike train")
    # plt.tight_layout(0.1)
    # plt.xlim(0, 1000)
    # plt.legend(loc = 1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-20.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax1 = fig.add_subplot(212)
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_s[int((thirty_min + repeat_interval_strongHFS + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int(( thirty_min + repeat_interval_strongHFS+ (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], label=' soma', color="blue")
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_d_1[int((thirty_min + repeat_interval_strongHFS + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + repeat_interval_strongHFS + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], "--", label=' dend1', color="red")
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_d_2[int((thirty_min + repeat_interval_strongHFS + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + repeat_interval_strongHFS + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], "--", label=' dend2', color="black")
    # ax1.set_ylabel("Voltage (mV)")
    # ax1.set_xlabel("time (ms)")
    # plt.title("postneuron spike train")
    # plt.tight_layout(0.1)
    # plt.xlim(0, 1000)
    # plt.legend(loc = 1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-20_2.jpg',bbox_inches = 'tight')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax1 = fig.add_subplot(212)
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_s[int((thirty_min + repeat_interval_strongHFS*2 + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int(( thirty_min + repeat_interval_strongHFS*2+ (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], label=' soma', color="blue")
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_d_1[int((thirty_min + repeat_interval_strongHFS*2 + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + repeat_interval_strongHFS*2 + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], "--", label=' dend1', color="red")
    # plt.plot(
    #     t_neuron[int(thirty_min / step):int(thirty_min / step) + int(1500 / step)] - thirty_min - STC_1.calcium_time_window,
    #     V_d_2[int((thirty_min + repeat_interval_strongHFS*2 + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step):int((thirty_min + repeat_interval_strongHFS*2 + (stimu_Num_weakHFS) * (repeat_interval_weakHFS)) / step) + int(1500 / step)], "--", label=' dend2', color="black")
    # ax1.set_ylabel("Voltage (mV)")
    # ax1.set_xlabel("time (ms)")
    # plt.title("postneuron spike train")
    # plt.tight_layout(0.1)
    # plt.xlim(0, 1000)
    # plt.legend(loc = 1)
    # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1-20_3.jpg',bbox_inches = 'tight')
    # plt.close()
    # # plt.show()
    #
    # # fig = plt.figure(figsize=(10, 5))
    # # bax = brokenaxes(xlims=((-2, 70), (270, 302)), hspace=.05, despine=True)
    # # lns1 = bax.plot(t_neuron / (60 * 1000), synapse_Ca_NMDA_strongHFS,
    # #                 label=' synapse Ca NMDA-strong HFS stimuli', color="r")
    # # lns2 = bax.plot(t_neuron / (60 * 1000), synapse_Ca_NMDA_weakHFS,
    # #                 label=' synapse Ca NMDA-weak HFS stimuli', color="b")
    # # lns3 = bax.plot(t_neuron / (60 * 1000), dend2_Ca_channel, '--', label=' dend Ca channel',
    # #                 color="black")
    # # bax.set_xlabel("time (min)")
    # # bax.set_ylabel(" Ca concentration (μmol/L)")
    # # bax.legend(loc=1)
    # # plt.title("changes in calcium concentration")
    # # plt.savefig('cross-capture1_weakHFS+strongHFS_10/1_7_1_1.jpg',bbox_inches = 'tight')
    # # plt.close()
    # plt.show()